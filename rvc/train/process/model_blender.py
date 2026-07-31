"""Weight interpolation for compatible acoustic checkpoints."""

from __future__ import annotations

import os
from collections import OrderedDict

import torch


def _weights(checkpoint):
    if "weight" in checkpoint:
        return checkpoint["weight"]
    if "model" in checkpoint:
        return {
            key: value
            for key, value in checkpoint["model"].items()
            if not key.startswith("enc_q.")
        }
    raise ValueError("Checkpoint has neither 'weight' nor 'model'")


def model_blender(name, path1, path2, ratio):
    try:
        first = torch.load(path1, map_location="cpu", weights_only=True)
        second = torch.load(path2, map_location="cpu", weights_only=True)
        architecture = first.get("architecture")
        if architecture not in {"Mel-VITS", "Hybrid-FSQ"} or second.get("architecture") != architecture:
            return "Both checkpoints must use the same supported architecture.", None
        if first.get("model_config") != second.get("model_config"):
            return f"{architecture} model configurations differ.", None

        first_weights = _weights(first)
        second_weights = _weights(second)
        if first_weights.keys() != second_weights.keys():
            return f"{architecture} checkpoint parameter sets differ.", None

        blended = OrderedDict()
        for key in first_weights:
            left, right = first_weights[key], second_weights[key]
            if left.shape != right.shape:
                return f"Shape mismatch for {key}: {left.shape} != {right.shape}", None
            blended[key] = (
                ratio * left.float() + (1.0 - ratio) * right.float()
            ).half()

        message = f"{architecture} models blended with alpha {ratio}."
        output = OrderedDict(first)
        output["weight"] = blended
        output["info"] = message
        output_path = os.path.join("logs", f"{name}.pth")
        torch.save(output, output_path)
        return message, output_path
    except Exception as error:
        return str(error), None
