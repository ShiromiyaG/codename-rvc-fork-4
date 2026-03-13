"""
Extrai small model do checkpoint G do pretrain.

Uso:
  python extract_pretrain.py <G_checkpoint.pth> <output.pth> [--name "Model Name"]
"""

import os
import sys
import json
import argparse
import torch


def build_config_list(config_dict: dict, n_speakers: int) -> list:
    model = config_dict["model"]
    data = config_dict["data"]
    train = config_dict["train"]

    return [
        data["filter_length"] // 2 + 1,
        train["segment_size"] // data["hop_length"],
        model["inter_channels"],
        model["hidden_channels"],
        model["filter_channels"],
        model["n_heads"],
        model["n_layers"],
        model["kernel_size"],
        model["p_dropout"],
        model["resblock"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        n_speakers,
        model["gin_channels"],
        data["sample_rate"],
    ]


def extract_pretrain(g_checkpoint_path, output_path, model_name="ChouwaGAN Pretrain"):
    device = torch.device("cpu")

    print(f"Loading checkpoint: {g_checkpoint_path}")
    checkpoint = torch.load(g_checkpoint_path, map_location=device, weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        # Strip enc_q keys (posterior encoder not needed for inference)
        if "enc_q" in k:
            continue
        # Strip torch.compile prefix
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)
        cleaned[k] = v
    print(f"  Keys: {len(cleaned)}")

    # ─── Detectar parâmetros do checkpoint ───────────────────────────
    # Speakers
    if "emb_g.weight" in cleaned:
        n_speakers = cleaned["emb_g.weight"].shape[0]
    else:
        n_speakers = 1

    # version: control text_enc_hidden_dim in inference
    if "enc_p.emb_phone.weight" in cleaned:
        emb_channels = cleaned["enc_p.emb_phone.weight"].shape[1]
        version = "v2" if emb_channels == 768 else "v1"
    else:
        emb_channels = 768
        version = "v2"

    # Detect VITS version from model architecture
    # Check in original state_dict since enc_q keys were stripped from cleaned
    if any("enc_q.blocks" in k for k in state_dict):
        vits_version = "mod"
    elif any("flow.flows" in k and "self_attn" in k for k in cleaned):
        vits_version = "v2"
    else:
        vits_version = "v1"

    # Vocoder detection from decoder architecture keys
    if any("dec.backbone" in k for k in cleaned):
        vocoder = "ChouwaGAN"
    elif any("dec.conformers" in k for k in cleaned):
        vocoder = "RingFormer_v2"  # v1 vs v2 is config-level, default to v2
    elif any("dec.downsample_blocks" in k for k in cleaned):
        vocoder = "RefineGAN"
    elif any("dec.har_convs" in k for k in cleaned):
        vocoder = "PCPH-GAN"
    else:
        vocoder = "HiFi-GAN"

    print(f"  Speakers: {n_speakers}")
    print(f"  version: {version} (emb_channels={emb_channels})")
    print(f"  vits_version: {vits_version}")
    print(f"  Vocoder: {vocoder}")

    # ─── Config ──────────────────────────────────────────────────────
    model_dir = os.path.dirname(g_checkpoint_path)
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config_list = build_config_list(config_dict, n_speakers)
    sr = config_list[-1]

    print(f"  SR: {sr}")
    print(f"  Config ({len(config_list)} elements): OK")

    # ─── Metadados ───────────────────────────────────────────────────
    epoch = checkpoint.get("epoch", 0)
    filename = os.path.basename(g_checkpoint_path)
    try:
        step = int(filename.split("_")[-1].split(".")[0])
    except (ValueError, IndexError):
        step = 0

    total_params = sum(v.numel() for v in cleaned.values())
    size_mb = sum(v.numel() * v.element_size() for v in cleaned.values()) / 1024 / 1024
    print(f"  Step: {step}, Params: {total_params:,} ({size_mb:.1f} MB)")

    # ─── Precision-aware conversion ──────────────────────────────────
    # Flow and encoder weights are kept in FP32 to avoid precision loss
    # that compounds during normalizing flow reverse transforms.
    # Only decoder and embedding weights are converted to FP16.
    opt_cleaned = {}
    fp32_prefixes = ("flow.", "enc_p.", "dec.")
    for k, v in cleaned.items():
        if v.dtype == torch.float32 and not k.startswith(fp32_prefixes):
            opt_cleaned[k] = v.half()
        else:
            opt_cleaned[k] = v

    size_out = sum(v.numel() * v.element_size() for v in opt_cleaned.values()) / 1024 / 1024
    n_fp32 = sum(1 for v in opt_cleaned.values() if v.dtype == torch.float32)
    n_fp16 = sum(1 for v in opt_cleaned.values() if v.dtype == torch.float16)
    print(f"  Size: {size_out:.1f} MB ({n_fp32} FP32 keys, {n_fp16} FP16 keys)")

    # ─── Small model ─────────────────────────────────────────────────
    small_model = {
        "weight": opt_cleaned,
        "config": config_list,
        "info": f"{model_name} | {vocoder} | {version} | {sr}Hz | {step}s",
        "sr": sr,
        "f0": 1,
        "version": version,
        "vits_version": vits_version,
        "vocoder": vocoder,
        "epoch": epoch,
        "step": step,
        "emb_channels": emb_channels,
    }

    # ─── Save ──────────────────────────────────────────────────────
    print(f"\nSaving to: {output_path}")
    torch.save(small_model, output_path)

    final_size = os.path.getsize(output_path) / 1024 / 1024
    original_size = os.path.getsize(g_checkpoint_path) / 1024 / 1024
    print(f"  {original_size:.1f} MB → {final_size:.1f} MB ({(1 - final_size/original_size)*100:.0f}% reduction)")

    # ─── Verification ─────────────────────────────────────────────────
    print("\n─── Verification ───")
    v = torch.load(output_path, map_location="cpu", weights_only=False)
    print(f"  config[-1] (sr):  {v['config'][-1]}")
    print(f"  config[-3] (spk): {v['config'][-3]}")
    print(f"  version:          {v['version']}")
    print(f"  vits_version:     {v.get('vits_version', 'v1')}")
    print(f"  vocoder:          {v['vocoder']}")
    print(f"  emb_channels:     {v.get('emb_channels', 'N/A')}")

    # Simulate what infer.py does:
    test_version = v["version"]
    test_dim = 768 if test_version == "v2" else 256
    print(f"  text_enc_hidden_dim (simulated): {test_dim}")
    print(f"  → Matches emb_phone shape: {'✅' if test_dim == emb_channels else '❌'}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--name", type=str, default="ChouwaGAN Pretrain")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Not found: {args.checkpoint}")
        sys.exit(1)

    extract_pretrain(args.checkpoint, args.output, args.name)


if __name__ == "__main__":
    main()