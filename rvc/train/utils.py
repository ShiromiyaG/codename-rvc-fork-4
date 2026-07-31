"""Small training utilities shared by the Mel-VITS loader and trainer."""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import soundfile as sf
import torch


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = HParams(**value) if isinstance(value, dict) else value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


def load_config_from_json(path: str) -> HParams:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return HParams(**json.load(handle))
    except FileNotFoundError:
        print(
            f"Model config not found at {path}. Run preprocessing and extraction first."
        )
        raise SystemExit(1)


def load_filepaths_and_text(filename: str, split: str = "|"):
    with open(filename, encoding="utf-8") as handle:
        return [line.strip().split(split) for line in handle if line.strip()]


def load_wav_to_torch(path: str):
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    tensor = torch.from_numpy(data)
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=1)
    return tensor, sample_rate


def save_checkpoint(
    model,
    optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: str,
    gradscaler=None,
    extra=None,
) -> None:
    module = model.module if hasattr(model, "module") else model
    payload = {
        "model": module.state_dict(),
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    if gradscaler is not None:
        payload["gradscaler"] = gradscaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


def load_checkpoint(path, model, optimizer=None, strict_load=True, return_extra=False):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(payload["model"], strict=strict_load)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    result = (
        model,
        optimizer,
        payload.get("learning_rate", 0.0),
        payload["iteration"],
        payload.get("gradscaler", {}),
    )
    if return_extra:
        return result + (payload,)
    return result


def latest_checkpoint_path(directory: str, regex: str = "G_*.pth"):
    candidates = glob.glob(os.path.join(directory, regex))
    if not candidates:
        return None

    def step(path: str) -> int:
        match = re.search(r"(\d+)(?=\.pth$)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)
