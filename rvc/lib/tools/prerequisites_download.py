import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests 

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources" # Might change in future
pc_nsf_hifigan_base_url = (
    "https://huggingface.co/shiromiya/RIFT-SVC/resolve/main/"
    "Vocoders/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
)

pretraineds_hifigan_list = []

vocoders_list = [
    (
        "vocoders/",
        [
            (
                "model.ckpt",
                "pc_nsf_hifigan_44.1k_hop512_128bin.pth",
            )
        ],
        pc_nsf_hifigan_base_url,
    )
]

smartcutter_list = [
    (
        "smartcutter/",
        [
            "v3_model_32000.pth",
            "v3_model_40000.pth",
            "v3_model_48000.pth",
        ],
        "https://huggingface.co/Codename0/SmartCutter/resolve/main",
    )
]

models_list = [
    ("predictors/", ["rmvpe.pt"]),
    ("predictors/", ["fcpe_ddsp.pt"], "https://huggingface.co/Codename0/codename-rvc-fork-4-assets/resolve/main/f0_predictors")
]

embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin_v1", ["pytorch_model.bin", "config.json"], "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin"),
    ("embedders/spin_v2", ["pytorch_model.bin", "config.json"], "https://huggingface.co/dr87/spinv2_rvc/resolve/main"),
]

executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "embedders/spin_v1": "rvc/models/embedders/spin_v1/",
    "embedders/spin_v2": "rvc/models/embedders/spin_v2/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
    "smartcutter/": "rvc/models/smartcutter/",
    "vocoders/": "rvc/models/vocoders/",
}


def resolve_file_names(file_spec):
    """Return the remote and local names for a download entry."""
    if isinstance(file_spec, (tuple, list)):
        return file_spec
    return file_spec, file_spec


def has_missing_files(file_list):
    """Check whether at least one mapped file is absent locally."""
    for entry in file_list:
        remote_folder, files = entry[:2]
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file_spec in files:
            _, local_name = resolve_file_names(file_spec)
            if not os.path.exists(os.path.join(local_folder, local_name)):
                return True
    return False


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    Supports optional third element (custom base URL) in the tuple.
    File entries may be strings or (remote name, local name) pairs.
    """
    total_size = 0
    for entry in file_list:
        if len(entry) == 2:
            remote_folder, files = entry
            base_url = url_base
        else:
            remote_folder, files, base_url = entry

        local_folder = folder_mapping_list.get(remote_folder, "")
        for file_spec in files:
            remote_name, local_name = resolve_file_names(file_spec)
            destination_path = os.path.join(local_folder, local_name)
            if not os.path.exists(destination_path):
                # Construct URL depending on whether it's using the shared base or custom one
                if base_url == url_base:
                    url = f"{base_url}/{remote_folder}{remote_name}"
                else:
                    url = f"{base_url}/{remote_name}"
                response = requests.head(url, allow_redirects=True)
                response.raise_for_status()
                total_size += int(response.headers.get("content-length", 0))
    return total_size



def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    partial_path = f"{destination_path}.part"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    block_size = 1024
    try:
        with open(partial_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                global_bar.update(len(data))
        os.replace(partial_path, destination_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    Supports optional third element (custom base URL) in the tuple.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for entry in file_mapping_list:
            if len(entry) == 2:
                remote_folder, file_list = entry
                base_url = url_base
            else:
                remote_folder, file_list, base_url = entry

            local_folder = folder_mapping_list.get(remote_folder, "")
            for file_spec in file_list:
                remote_name, local_name = resolve_file_names(file_spec)
                destination_path = os.path.join(local_folder, local_name)
                if not os.path.exists(destination_path):
                    if base_url == url_base:
                        url = f"{base_url}/{remote_folder}{remote_name}"
                    else:
                        url = f"{base_url}/{remote_name}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for entry in pretrained_list:
        if len(entry) == 3:
            folder, files, url = entry
        else:
            folder, files = entry
            url = None

        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]

        if f0_files:
            f0_list.append((folder, f0_files, url) if url else (folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files, url) if url else (folder, non_f0_files))

    return f0_list, non_f0_list

pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
    smartcutter,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0

    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
        total_size += get_file_size_if_missing(vocoders_list)

    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)

    if smartcutter:
        total_size += get_file_size_if_missing(smartcutter_list)

    total_size += get_file_size_if_missing(pretraineds_hifigan)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
    smartcutter,
):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
        smartcutter,
    )

    download_required = (
        (
            models
            and (
                has_missing_files(models_list)
                or has_missing_files(embedders_list)
                or has_missing_files(vocoders_list)
            )
        )
        or (exe and os.name == "nt" and has_missing_files(executables_list))
        or (smartcutter and has_missing_files(smartcutter_list))
        or (
            pretraineds_hifigan
            and has_missing_files(pretraineds_hifigan_list)
        )
    )

    if download_required:
        with tqdm(
            total=total_size or None,
            unit="iB",
            unit_scale=True,
            desc="Downloading all files",
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
                download_mapping_files(vocoders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if smartcutter:
                download_mapping_files(smartcutter_list, global_bar)
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
    else:
        pass
