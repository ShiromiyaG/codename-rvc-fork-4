import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json
import numpy as np
import soundfile as sf

config = Config()
current_directory = os.getcwd()


def _validate_sample_rate(sample_rate) -> int:
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid sample rate: {sample_rate!r}") from error
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    return sample_rate


def generate_config(
    sample_rate: int,
    model_path: str,
    vocoder_arch: str,
    f0_min: float = 30.0,
    f0_max: float = 1600.0,
):
    sample_rate = _validate_sample_rate(sample_rate)
    config_path = os.path.join("rvc", "configs", vocoder_arch, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)
        print(f"Config saved at {config_save_path}")
    else:
        with open(config_path, "r", encoding="utf-8") as source:
            recommended_config = json.load(source)
        with open(config_save_path, "r", encoding="utf-8") as existing:
            saved_config = json.load(existing)
        recommended_architecture = recommended_config.get("architecture", "Mel-VITS")
        saved_architecture = saved_config.get("architecture", "Mel-VITS")
        config_changed = False
        if recommended_architecture != saved_architecture:
            preserved_speakers = saved_config.get("model", {}).get("spk_embed_dim")
            saved_config = recommended_config
            if preserved_speakers is not None:
                saved_config["model"]["spk_embed_dim"] = preserved_speakers
            print(
                f"Changed acoustic architecture in {config_save_path}: "
                f"{saved_architecture} -> {recommended_architecture}"
            )
            config_changed = True
        current_segment_size = int(saved_config["train"].get("segment_size", 0))
        recommended_segment_size = int(
            recommended_config["train"]["segment_size"]
        )
        if current_segment_size < recommended_segment_size:
            saved_config["train"]["segment_size"] = recommended_segment_size
            config_changed = True
            print(
                "Updated acoustic segment size in "
                f"{config_save_path}: {current_segment_size} -> "
                f"{recommended_segment_size}"
            )
        recommended_fp16 = bool(recommended_config["train"].get("fp16_run", True))
        if saved_config["train"].get("fp16_run") != recommended_fp16:
            saved_config["train"]["fp16_run"] = recommended_fp16
            config_changed = True
            print(
                f"Updated acoustic default precision in {config_save_path}: "
                f"{'fp16' if recommended_fp16 else 'fp32'}"
            )
        for section in ("train", "data", "model", "vocoder"):
            saved_section = saved_config.setdefault(section, {})
            for key, value in recommended_config.get(section, {}).items():
                if key not in saved_section:
                    saved_section[key] = value
                    config_changed = True
        if config_changed:
            with open(config_save_path, "w", encoding="utf-8") as output:
                json.dump(saved_config, output, indent=4)
        else:
            print(f"Config file already exists at {config_save_path}")
    with open(config_save_path, "r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    saved_config.setdefault("data", {})["f0_min"] = float(f0_min)
    saved_config["data"]["f0_max"] = float(f0_max)
    with open(config_save_path, "w", encoding="utf-8") as output:
        json.dump(saved_config, output, indent=4)

def generate_filelist(
    model_path: str, sample_rate: int, include_mutes: int = 2, embedder_model: str = "contentvec", vocoder_arch: str = "melvits"
):
    sample_rate = _validate_sample_rate(sample_rate)
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = sorted(os.listdir(gt_wavs_dir), key=lambda x: x.split(".")[0])
    feature_files = sorted(os.listdir(feature_dir), key=lambda x: x.split(".")[0])

    f0_files = sorted(os.listdir(f0_dir), key=lambda x: x.split(".")[0])
    f0nsf_files = sorted(os.listdir(f0nsf_dir), key=lambda x: x.split(".")[0])

    options = []

    if embedder_model == "contentvec":
        mute_folder = "mute"
    elif embedder_model == "spin_v1":
        mute_folder = "mute_spin_v1"
    else:
        mute_folder = "mute_spin_v2"

    mute_base_path = os.path.join(current_directory, "logs", mute_folder)

    sids = []
    
    vocoder_arch = vocoder_arch

    for gt_wavs_file, feature_file, f0_file, f0nsf_file in zip(gt_wavs_files, feature_files, f0_files, f0nsf_files, strict=True):
        sid = gt_wavs_file.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, gt_wavs_file)}|{os.path.join(feature_dir, feature_file)}|{os.path.join(f0_dir, f0_file)}|{os.path.join(f0nsf_dir, f0nsf_file)}|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(
            mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
        )
        mute_feature_path = os.path.join(
            mute_base_path, f"extracted", "mute.npy"
        )
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # Build a native 44.1 kHz silence sample instead of relying on legacy
        # sample-rate-specific assets.
        for directory in (
            os.path.dirname(mute_audio_path),
            os.path.dirname(mute_feature_path),
            os.path.dirname(mute_f0_path),
            os.path.dirname(mute_f0nsf_path),
        ):
            os.makedirs(directory, exist_ok=True)
        duration_seconds = 2
        if not os.path.isfile(mute_audio_path):
            sf.write(
                mute_audio_path,
                np.zeros(sample_rate * duration_seconds, dtype=np.float32),
                sample_rate,
                subtype="FLOAT",
            )
        if not os.path.isfile(mute_feature_path):
            np.save(
                mute_feature_path,
                np.zeros((duration_seconds * 50, 768), dtype=np.float16),
                allow_pickle=False,
            )
        if not os.path.isfile(mute_f0_path):
            np.save(
                mute_f0_path,
                np.zeros(duration_seconds * 100, dtype=np.uint8),
                allow_pickle=False,
            )
        if not os.path.isfile(mute_f0nsf_path):
            np.save(
                mute_f0nsf_path,
                np.zeros(duration_seconds * 100, dtype=np.float32),
                allow_pickle=False,
            )

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data["speakers_id"] = len(sids)
    data["vocoder_architecture"] = vocoder_arch

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    model_config["model"]["spk_embed_dim"] = max(1, len(sids))
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=4)

    shuffle(options)


    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
