import os


def pretrained_selector(vocoder, sample_rate):
    root = os.path.join("rvc", "models", "pretraineds")
    filename_g = f"f0G{str(sample_rate)[:2]}k.pth"
    filename_d = f"f0D{str(sample_rate)[:2]}k.pth"
    normalized = str(vocoder).lower().replace("_", "-")
    folders = [normalized]
    # Older installations store the standard RVC pretrain pair under
    # `hifi-gan`, while the UI now names the external vocoder pc-NSF-HiFiGAN.
    if normalized in {"pc-nsf-hifigan", "pc-nsf-hifi-gan"}:
        folders.append("hifi-gan")
    folders.append("")

    def find(filename):
        for folder in folders:
            candidate = os.path.join(root, folder, filename)
            if os.path.exists(candidate):
                return candidate
        return ""

    return find(filename_g), find(filename_d)
