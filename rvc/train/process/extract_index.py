import os
import sys
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Parse command line arguments
exp_dir = str(sys.argv[1])
index_algorithm = str(sys.argv[2])

try:
    feature_dir = os.path.join(exp_dir, f"extracted")
    model_name = os.path.basename(exp_dir)

    if not os.path.exists(feature_dir):
        print(
            f"Feature to generate index file not found at {feature_dir}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)

    index_filename_added = f"{model_name}.index"
    index_filepath_added = os.path.join(exp_dir, index_filename_added)

    if os.path.exists(index_filepath_added):
        pass
    else:
        npys = []
        listdir_res = sorted(os.listdir(feature_dir))

        for name in listdir_res:
            file_path = os.path.join(feature_dir, name)
            phone = np.load(file_path)
            npys.append(phone)

        big_npy = np.concatenate(npys, axis=0)

        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 2e5 and (
            index_algorithm == "Auto" or index_algorithm == "KMeans"
        ):
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * cpu_count(),
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )

        n_frames = big_npy.shape[0]
        dim = big_npy.shape[1]  # 768 for contentvec / spin

        # Normalize all embeddings so that inner product == cosine similarity.
        # Speaker embedding models (ContentVec, SPIN) encode phonetic identity
        # as direction, not magnitude — cosine similarity is the correct metric.
        norms = np.linalg.norm(big_npy, axis=1, keepdims=True)
        big_npy_norm = (big_npy / (norms + 1e-8)).astype("float32")

        # Choose index type based on dataset size:
        #   • IndexFlatIP  – exact cosine, no quantization error, best quality.
        #     Used when n_frames < 20k (covers ~99% of RVC fine-tune use cases).
        #   • IVFFlat + METRIC_INNER_PRODUCT – approximate cosine for very large
        #     datasets where exact search would be too slow at inference time.
        #     nprobe set to 10% of clusters (vs the old nprobe=1) for good recall.
        if n_frames < 20000:
            index_added = faiss.IndexFlatIP(dim)
            print(f"Using IndexFlatIP (exact cosine) for {n_frames} frames.")
        else:
            n_ivf = min(int(16 * np.sqrt(n_frames)), n_frames // 39)
            quantizer = faiss.IndexFlatIP(dim)
            index_added = faiss.IndexIVFFlat(quantizer, dim, n_ivf, faiss.METRIC_INNER_PRODUCT)
            index_ivf = faiss.extract_index_ivf(index_added)
            # nprobe=10% of clusters gives much better recall than the old nprobe=1
            # with only a small latency cost per inference frame.
            index_ivf.nprobe = max(1, n_ivf // 10)
            index_added.train(big_npy_norm)
            print(f"Using IVFFlat/IP (n_ivf={n_ivf}, nprobe={index_ivf.nprobe}) for {n_frames} frames.")

        batch_size_add = 8192
        for i in range(0, n_frames, batch_size_add):
            index_added.add(big_npy_norm[i : i + batch_size_add])

        faiss.write_index(index_added, index_filepath_added)
        print(f"Saved index file '{index_filepath_added}'")

except Exception as error:
    print(f"An error occurred extracting the index: {error}")
    print(
        "If you are running this code in a virtual environment, make sure you have enough GPU available to generate the Index file."
    )
