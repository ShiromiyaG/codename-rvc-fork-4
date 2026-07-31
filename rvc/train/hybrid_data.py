"""Stationary targets and compact batches for the Hybrid-FSQ acoustic model."""

from __future__ import annotations

import json
import hashlib
import math
import mmap
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from rvc.train.data_utils import (
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)


_SLICE_RE = re.compile(r"^(?P<speaker>\d+)_(?P<source>\d+)_(?P<slice>\d+)$")


def slice_identity(audio_path: str) -> tuple[str, str, int]:
    """Return speaker, stable source id and sequential slice index."""
    stem = Path(audio_path).stem
    match = _SLICE_RE.match(stem)
    if match:
        speaker = match.group("speaker")
        return speaker, f"{speaker}_{match.group('source')}", int(match.group("slice"))
    speaker = stem.split("_", 1)[0]
    return speaker, stem, 0


def split_entries_by_source(entries, ratio: float, seed: int):
    """Split complete source recordings, preventing adjacent-slice leakage."""
    if ratio <= 0 or len(entries) < 3:
        return entries, []
    import random

    sources = defaultdict(list)
    for entry in tqdm(entries, desc="Hybrid source split", unit="file"):
        if "mute" not in Path(entry[0]).name.lower():
            _, source, _ = slice_identity(entry[0])
            speaker = str(entry[4])
            sources[(speaker, source)].append(entry)
    by_speaker = defaultdict(list)
    for (speaker, source), source_entries in sources.items():
        by_speaker[speaker].append((source, source_entries))
    validation_paths = set()
    rng = random.Random(seed)
    for speaker_sources in by_speaker.values():
        if len(speaker_sources) < 2:
            continue
        rng.shuffle(speaker_sources)
        count = min(
            len(speaker_sources) - 1,
            max(1, round(len(speaker_sources) * ratio)),
        )
        for _, source_entries in speaker_sources[:count]:
            validation_paths.update(entry[0] for entry in source_entries)
    train = [entry for entry in entries if entry[0] not in validation_paths]
    validation = [entry for entry in entries if entry[0] in validation_paths]
    return train, validation


def orthonormal_dct(rows: int, columns: int) -> torch.Tensor:
    n = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(columns, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi / rows * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / rows)
    if columns > 1:
        basis[:, 1:] *= math.sqrt(2.0 / rows)
    return basis


def temporal_lowpass(value: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Reflect-padded low-pass; accepts [C,T] or [B,C,T]."""
    squeeze = value.ndim == 2
    if squeeze:
        value = value.unsqueeze(0)
    radius = kernel_size // 2
    if value.size(-1) <= radius:
        result = value
    else:
        padded = F.pad(value, (radius, radius), mode="reflect")
        result = F.avg_pool1d(padded, kernel_size, stride=1)
    return result.squeeze(0) if squeeze else result


def coarse_mel(value: torch.Tensor) -> torch.Tensor:
    """Stationary coarse target: temporal blur plus a small spectral blur."""
    temporal = temporal_lowpass(value, 5)
    padded = F.pad(temporal.unsqueeze(0), (0, 0, 1, 1), mode="replicate")
    return F.avg_pool2d(padded, (3, 1), stride=1).squeeze(0)


def _mel_cache_paths(path: str) -> tuple[Path, Path]:
    base = Path(os.path.splitext(path)[0])
    return Path(f"{base}.mel.npy"), Path(f"{base}.mel.pt")


def _load_cached_mel(
    path: str, start: int | None = None, stop: int | None = None
) -> torch.Tensor:
    numpy_path, torch_path = _mel_cache_paths(path)
    if numpy_path.is_file():
        mapped = np.load(numpy_path, mmap_mode="r", allow_pickle=False)
        selected = mapped if start is None else mapped[:, start:stop]
        # Copy only the selected pages; tensors backed directly by a read-only
        # mmap produce undefined behavior if an op attempts to mutate them.
        return torch.from_numpy(np.array(selected, dtype=np.float32, copy=True))
    if not torch_path.is_file():
        raise FileNotFoundError(
            f"Missing mel cache {torch_path}. Open the training tab once after "
            "feature extraction, or remove hybrid_stats.pt to rebuild statistics."
        )
    mel = torch.load(torch_path, map_location="cpu", weights_only=True).float()
    return mel if start is None else mel[:, start:stop]


def _migrate_one_mel_cache(args):
    audio_path, remove_legacy = args
    numpy_path, torch_path = _mel_cache_paths(audio_path)
    if numpy_path.is_file():
        if remove_legacy and torch_path.is_file():
            torch_path.unlink()
        return False
    if not torch_path.is_file():
        return False
    mel = torch.load(torch_path, map_location="cpu", weights_only=True)
    temporary = Path(f"{numpy_path}.temporary.npy")
    np.save(temporary, mel.float().numpy(), allow_pickle=False)
    os.replace(temporary, numpy_path)
    if remove_legacy:
        torch_path.unlink()
    return True


def migrate_mel_caches_to_mmap(
    entries, workers: int = 4, remove_legacy: bool = True
) -> int:
    """Convert derived `.mel.pt` files to crop-readable NumPy mmap caches."""
    pending = []
    seen = set()
    for entry in entries:
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        numpy_path, torch_path = _mel_cache_paths(entry[0])
        if torch_path.is_file() or not numpy_path.is_file():
            pending.append((entry[0], remove_legacy))
    if not pending:
        return 0
    converted = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = executor.map(_migrate_one_mel_cache, pending)
        for changed in tqdm(
            results,
            total=len(pending),
            desc="Hybrid mmap mel cache",
            unit="file",
        ):
            converted += int(changed)
    return converted


def _packed_fingerprint(entries) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update("\0".join(map(str, entry)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_hybrid_packed_cache(entries, experiment_dir: str | Path):
    cache_dir = Path(experiment_dir) / "hybrid_packed_cache"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("version") != 1
        or manifest.get("fingerprint") != _packed_fingerprint(entries)
    ):
        return None
    required = ("mel.bin", "phone.bin", "pitch.bin", "pitchf.bin")
    if not all((cache_dir / name).is_file() for name in required):
        return None
    manifest["cache_dir"] = str(cache_dir)
    return manifest


def build_hybrid_packed_cache(
    entries,
    experiment_dir: str | Path,
    remove_individual_mels: bool = True,
):
    """Pack random-access training tensors into four persistent mmap files."""
    existing = load_hybrid_packed_cache(entries, experiment_dir)
    if existing is not None:
        return existing
    experiment_dir = Path(experiment_dir)
    final_dir = experiment_dir / "hybrid_packed_cache"
    building_dir = experiment_dir / "hybrid_packed_cache.building"
    if building_dir.exists():
        shutil.rmtree(building_dir)
    building_dir.mkdir(parents=True)
    records = {}
    mel_offset = phone_offset = pitch_offset = pitchf_offset = 0
    mel_file = open(building_dir / "mel.bin", "wb")
    phone_file = open(building_dir / "phone.bin", "wb")
    pitch_file = open(building_dir / "pitch.bin", "wb")
    pitchf_file = open(building_dir / "pitchf.bin", "wb")
    try:
        unique_entries = {}
        for entry in entries:
            unique_entries.setdefault(entry[0], entry)
        for audio_path, entry in tqdm(
            unique_entries.items(),
            desc="Hybrid packed cache",
            unit="file",
        ):
            mel = _load_cached_mel(audio_path).numpy().T
            phone = np.load(entry[1], mmap_mode="r", allow_pickle=False)
            pitch = np.load(entry[2], mmap_mode="r", allow_pickle=False)
            pitchf = np.load(entry[3], mmap_mode="r", allow_pickle=False)
            mel = np.ascontiguousarray(mel, dtype=np.float32)
            phone = np.ascontiguousarray(phone, dtype=np.float16)
            pitch = np.ascontiguousarray(pitch, dtype=np.uint8)
            pitchf = np.ascontiguousarray(pitchf, dtype=np.float32)
            mel.tofile(mel_file)
            phone.tofile(phone_file)
            pitch.tofile(pitch_file)
            pitchf.tofile(pitchf_file)
            records[audio_path] = [
                mel_offset,
                int(mel.shape[0]),
                phone_offset,
                int(phone.shape[0]),
                pitch_offset,
                int(pitch.shape[0]),
                pitchf_offset,
                int(pitchf.shape[0]),
            ]
            mel_offset += int(mel.size)
            phone_offset += int(phone.shape[0])
            pitch_offset += int(pitch.size)
            pitchf_offset += int(pitchf.size)
    finally:
        mel_file.close()
        phone_file.close()
        pitch_file.close()
        pitchf_file.close()
    manifest = {
        "version": 1,
        "fingerprint": _packed_fingerprint(entries),
        "mel_channels": 128,
        "phone_channels": 768,
        "records": records,
    }
    with open(building_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(building_dir, final_dir)
    if remove_individual_mels:
        for audio_path in tqdm(
            records, desc="Removing replaced mel files", unit="file"
        ):
            numpy_path, torch_path = _mel_cache_paths(audio_path)
            numpy_path.unlink(missing_ok=True)
            torch_path.unlink(missing_ok=True)
    return load_hybrid_packed_cache(entries, experiment_dir)


class PackedLocalityBatchSampler:
    """Shuffle large physical blocks while keeping each batch sequential.

    The packed files are much larger than RAM. Fully random sample order turns
    every mmap access into an unrelated page fault; this sampler reduces that
    to roughly one seek per locality block without fixing the epoch order.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        locality_batches: int = 64,
        seed: int = 1234,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.locality_batches = max(1, int(locality_batches))
        self.seed = int(seed)
        self.epoch = 0
        records = (
            dataset.packed_cache.get("records", {})
            if getattr(dataset, "packed_cache", None) is not None
            else {}
        )
        self.physical_order = sorted(
            range(len(dataset)),
            key=lambda index: records.get(
                dataset.audiopaths_and_text[index][0],
                (index,),
            )[0],
        )
        global_batches = math.ceil(len(dataset) / max(1, self.batch_size))
        self.batches_per_rank = math.ceil(global_batches / self.num_replicas)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_rank

    def __iter__(self):
        block_size = self.batch_size * self.locality_batches
        blocks = [
            self.physical_order[start : start + block_size]
            for start in range(0, len(self.physical_order), block_size)
        ]
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(blocks), generator=generator).tolist()
        batches = []
        for block_index in order:
            block = blocks[block_index]
            for start in range(0, len(block), self.batch_size):
                batch = block[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    needed = self.batch_size - len(batch)
                    repeats = math.ceil(needed / max(1, len(block)))
                    batch += (block * repeats)[:needed]
                batches.append(batch)
        required = self.batches_per_rank * self.num_replicas
        if len(batches) < required:
            batches += batches[: required - len(batches)]
        return iter(batches[self.rank:required:self.num_replicas])


def build_hybrid_statistics(
    entries,
    experiment_dir: str | Path,
    mel_channels: int = 128,
    shrinkage_tau: float = 20.0,
    parent_slices: int = 2,
) -> dict:
    """Build train-only normalization and parent/speaker DCT statistics.

    A pc-NSF mel cache is normally produced by the loader.  If it is absent,
    callers should warm it through ``TextAudioLoaderMultiNSFsid`` first.
    """
    experiment_dir = Path(experiment_dir)
    output = experiment_dir / "hybrid_stats.pt"
    if output.is_file():
        return torch.load(output, map_location="cpu", weights_only=False)

    count = 0
    total = torch.zeros(mel_channels, dtype=torch.float64)
    total_sq = torch.zeros(mel_channels, dtype=torch.float64)
    for entry in tqdm(entries, desc="Hybrid mel statistics", unit="file"):
        mel = _load_cached_mel(entry[0])
        total += mel.double().sum(-1)
        total_sq += mel.double().square().sum(-1)
        count += mel.size(-1)
    mean = (total / max(1, count)).float()
    variance = total_sq / max(1, count) - mean.double().square()
    std = variance.clamp_min(1e-6).sqrt().float()
    basis = orthonormal_dct(mel_channels, 8)

    parent_values = defaultdict(list)
    parent_members = defaultdict(list)
    for entry in tqdm(entries, desc="Hybrid parent styles", unit="file"):
        mel = (_load_cached_mel(entry[0]) - mean[:, None]) / std[:, None]
        residual = mel - coarse_mel(mel)
        energy = mel.exp().mean(0)
        threshold = torch.quantile(energy, 0.2)
        mask = energy > threshold
        pooled = residual[:, mask].mean(-1) if mask.any() else residual.mean(-1)
        coeff = basis.T @ pooled
        _, source, slice_index = slice_identity(entry[0])
        speaker = str(entry[4])
        parent = f"{speaker}|{source}:{slice_index // max(1, parent_slices)}"
        parent_values[parent].append(coeff)
        parent_members[parent].append(Path(entry[0]).name)

    parent_coeff = {
        key: torch.stack(values).mean(0) for key, values in parent_values.items()
    }
    speaker_parents = defaultdict(list)
    for parent, coeff in parent_coeff.items():
        speaker_parents[parent.split("|", 1)[0]].append(coeff)
    dataset_mean = torch.stack(list(parent_coeff.values())).mean(0)
    speaker_coeff = {}
    for speaker, values in speaker_parents.items():
        empirical = torch.stack(values).mean(0)
        n = float(len(values))
        speaker_coeff[speaker] = (
            n / (n + shrinkage_tau) * empirical
            + shrinkage_tau / (n + shrinkage_tau) * dataset_mean
        )

    centered_parent = {}
    file_to_parent = {}
    for parent, coeff in parent_coeff.items():
        speaker = parent.split("|", 1)[0]
        centered_parent[parent] = coeff - speaker_coeff[speaker]
        for filename in parent_members[parent]:
            file_to_parent[filename] = parent

    # Estimate fixed p99.5 per-bin caps from a bounded deterministic subset.
    cap_samples = {"global": [], "slow": [], "fast": []}
    stride = max(1, len(entries) // 4096)
    for entry in tqdm(entries[::stride], desc="Hybrid residual caps", unit="file"):
        mel = (_load_cached_mel(entry[0]) - mean[:, None]) / std[:, None]
        base = coarse_mel(mel)
        speaker = str(entry[4])
        parent = file_to_parent[Path(entry[0]).name]
        global_target = (basis @ centered_parent[parent])[:, None]
        local = mel - base - (basis @ speaker_coeff[speaker])[:, None] - global_target
        slow = temporal_lowpass(local, 5)
        slow = slow - slow.mean(-1, keepdim=True)
        fast = local - slow
        step = max(1, mel.size(-1) // 16)
        cap_samples["global"].append(global_target.abs().T)
        cap_samples["slow"].append(slow[:, ::step].abs().T)
        cap_samples["fast"].append(fast[:, ::step].abs().T)
    caps = {}
    for key, chunks in cap_samples.items():
        values = torch.cat(chunks, dim=0)
        cap = torch.quantile(values, 0.995, dim=0).clamp(0.2, 2.0)
        cap = temporal_lowpass(cap.view(1, -1), 5).view(-1)
        caps[key] = cap

    stats = {
        "version": 1,
        "mel_mean": mean,
        "mel_std": std,
        "dct_basis": basis,
        "speaker_coeff": speaker_coeff,
        "centered_parent_coeff": centered_parent,
        "file_to_parent": file_to_parent,
        "caps": caps,
    }
    torch.save(stats, output)
    manifest = {
        "version": 1,
        "files": {
            filename: {"parent_style_id": parent}
            for filename, parent in file_to_parent.items()
        },
    }
    with open(experiment_dir / "hybrid_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return stats


def attach_evaluation_parents(stats: dict, entries) -> dict:
    """Add validation parent targets using frozen train-only normalization."""
    grouped = defaultdict(list)
    members = defaultdict(list)
    mean = stats["mel_mean"]
    std = stats["mel_std"]
    basis = stats["dct_basis"]
    for entry in entries:
        filename = Path(entry[0]).name
        if filename in stats["file_to_parent"]:
            continue
        mel = (_load_cached_mel(entry[0]) - mean[:, None]) / std[:, None]
        residual = mel - coarse_mel(mel)
        energy = mel.exp().mean(0)
        mask = energy > torch.quantile(energy, 0.2)
        pooled = residual[:, mask].mean(-1) if mask.any() else residual.mean(-1)
        _, source, slice_index = slice_identity(entry[0])
        speaker = str(entry[4])
        parent = f"validation:{source}:{slice_index // 2}"
        grouped[parent].append(basis.T @ pooled)
        members[parent].append((filename, speaker))
    for parent, values in grouped.items():
        coefficient = torch.stack(values).mean(0)
        speaker = members[parent][0][1]
        centered = coefficient - stats["speaker_coeff"].get(
            speaker, torch.zeros_like(coefficient)
        )
        stats["centered_parent_coeff"][parent] = centered
        for filename, _ in members[parent]:
            stats["file_to_parent"][filename] = parent
    return stats


class HybridFSQDataset(TextAudioLoaderMultiNSFsid):
    """Loader that crops before the model and returns stationary targets."""

    def __init__(
        self,
        *args,
        stats: dict,
        segment_frames: int = 256,
        load_waveform: bool = False,
        waveform_items: int = 0,
        packed_cache=None,
        **kwargs,
    ):
        self.stats = stats
        self.segment_frames = segment_frames
        self.load_waveform = load_waveform
        self.waveform_items = max(0, int(waveform_items))
        self.packed_cache = packed_cache
        self._packed_arrays = None
        super().__init__(*args, **kwargs)

    def _filter(self):
        """Hybrid batches are cropped to a fixed size; avoid 75k sf.info calls."""
        self.audiopaths_and_text = [
            list(entry) for entry in self.audiopaths_and_text
        ]
        self.lengths = [self.segment_frames] * len(self.audiopaths_and_text)

    def _open_packed_cache(self):
        if self._packed_arrays is not None:
            return self._packed_arrays
        cache_dir = Path(self.packed_cache["cache_dir"])
        self._packed_arrays = (
            np.memmap(cache_dir / "mel.bin", mode="r", dtype=np.float32),
            np.memmap(cache_dir / "phone.bin", mode="r", dtype=np.float16).reshape(
                -1, int(self.packed_cache["phone_channels"])
            ),
            np.memmap(cache_dir / "pitch.bin", mode="r", dtype=np.uint8),
            np.memmap(cache_dir / "pitchf.bin", mode="r", dtype=np.float32),
        )
        for mapped in self._packed_arrays:
            try:
                mapped._mmap.madvise(mmap.MADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
        return self._packed_arrays

    @staticmethod
    def _condition_crop(
        entry,
        mel_frames,
        start,
        stop,
        phone_source=None,
        pitch_source=None,
        pitchf_source=None,
    ):
        """Read only conditioning rows that contribute to the requested crop."""
        if phone_source is None:
            phone_source = np.load(entry[1], mmap_mode="r", allow_pickle=False)
            pitch_source = np.load(entry[2], mmap_mode="r", allow_pickle=False)
            pitchf_source = np.load(entry[3], mmap_mode="r", allow_pickle=False)
        source_frames = min(
            phone_source.shape[0] * 2,
            pitch_source.shape[0],
            pitchf_source.shape[0],
        )
        output_index = torch.arange(start, stop, dtype=torch.float64)
        linear_position = (
            (output_index + 0.5) * source_frames / max(1, mel_frames) - 0.5
        ).clamp(0, max(0, source_frames - 1))
        left = linear_position.floor().long()
        right = (left + 1).clamp_max(max(0, source_frames - 1))
        weight = (linear_position - left).float()

        phone_left = torch.from_numpy(
            np.asarray(phone_source[(left.numpy() // 2)], dtype=np.float32)
        )
        phone_right = torch.from_numpy(
            np.asarray(phone_source[(right.numpy() // 2)], dtype=np.float32)
        )
        phone = phone_left + (phone_right - phone_left) * weight[:, None]

        pitchf_left = torch.from_numpy(
            np.asarray(pitchf_source[left.numpy()], dtype=np.float32)
        )
        pitchf_right = torch.from_numpy(
            np.asarray(pitchf_source[right.numpy()], dtype=np.float32)
        )
        pitchf = pitchf_left + (pitchf_right - pitchf_left) * weight

        nearest = torch.floor(
            output_index * source_frames / max(1, mel_frames)
        ).long().clamp(0, max(0, source_frames - 1))
        pitch = torch.from_numpy(
            np.asarray(pitch_source[nearest.numpy()], dtype=np.int64)
        )
        return phone, pitch, pitchf

    def get_audio_text_pair(self, entry, load_waveform=None):
        # The Hybrid-FSQ objective is entirely mel-domain.  Calling the base
        # loader here used to decode every FLAC/WAV only to discard it in the
        # training loop, which dominated random I/O on large datasets.
        record = (
            self.packed_cache["records"].get(entry[0])
            if self.packed_cache is not None
            else None
        )
        numpy_mel, torch_mel = _mel_cache_paths(entry[0])
        full_spec = None
        if record is not None:
            mel_frames = int(record[1])
        elif numpy_mel.is_file():
            mel_frames = int(
                np.load(numpy_mel, mmap_mode="r", allow_pickle=False).shape[-1]
            )
        else:
            full_spec = torch.load(
                torch_mel, map_location="cpu", weights_only=True
            ).float()
            mel_frames = full_spec.size(-1)
        sid = self.get_sid(entry[4])
        frames = mel_frames
        wanted = min(self.segment_frames, frames)
        start = (
            torch.randint(0, frames - wanted + 1, ()).item()
            if self.augment and frames > wanted
            else 0
        )
        stop = start + wanted
        if record is not None:
            mel_map, phone_map, pitch_map, pitchf_map = self._open_packed_cache()
            mel_offset, _, phone_offset, phone_length, pitch_offset, pitch_length, pitchf_offset, pitchf_length = record
            selected_mel = mel_map[
                mel_offset + start * 128 : mel_offset + stop * 128
            ].reshape(stop - start, 128)
            spec = torch.from_numpy(
                np.array(selected_mel.T, dtype=np.float32, copy=True)
            )
            phone_source = phone_map[
                phone_offset : phone_offset + phone_length
            ]
            pitch_source = pitch_map[
                pitch_offset : pitch_offset + pitch_length
            ]
            pitchf_source = pitchf_map[
                pitchf_offset : pitchf_offset + pitchf_length
            ]
            phone, pitch, pitchf = self._condition_crop(
                entry,
                mel_frames,
                start,
                stop,
                phone_source,
                pitch_source,
                pitchf_source,
            )
        else:
            spec = (
                full_spec[:, start:stop]
                if full_spec is not None
                else _load_cached_mel(entry[0], start, stop)
            )
            phone, pitch, pitchf = self._condition_crop(
                entry, mel_frames, start, stop
            )
        if load_waveform is None:
            load_waveform = self.load_waveform
        if load_waveform:
            from rvc.train.utils import load_wav_to_torch

            waveform, sample_rate = load_wav_to_torch(entry[0])
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"{sample_rate} SR doesn't match target {self.sample_rate} SR"
                )
            wav = waveform.unsqueeze(0)[
                :, start * self.hop_length : stop * self.hop_length
            ]
        else:
            # Keep the established collate contract without allocating or
            # transferring a waveform that the acoustic objective never uses.
            wav = torch.empty(1, 0, dtype=torch.float32)

        mean = self.stats["mel_mean"][:, None]
        std = self.stats["mel_std"][:, None]
        mel = (spec - mean) / std
        coarse = coarse_mel(mel)
        speaker = str(entry[4])
        parent = self.stats["file_to_parent"].get(Path(entry[0]).name)
        speaker_coeff = self.stats["speaker_coeff"].get(
            speaker, torch.zeros(8)
        )
        global_coeff = self.stats["centered_parent_coeff"].get(
            parent, torch.zeros(8)
        )
        basis = self.stats["dct_basis"]
        speaker_bias = (basis @ speaker_coeff)[:, None]
        global_target = (basis @ global_coeff)[:, None].expand_as(mel)
        base_target = coarse + speaker_bias
        local = mel - base_target - global_target
        slow = temporal_lowpass(local, 5)
        slow = slow - slow.mean(-1, keepdim=True)
        voiced = (pitchf > 0).float()[None]
        slow = slow * (0.25 + 0.75 * voiced)
        fast = local - slow
        return (
            mel,
            wav,
            phone,
            pitch,
            pitchf,
            sid,
            base_target,
            global_target,
            slow,
            fast,
            global_coeff,
        )

    def __getitem__(self, index):
        load_waveform = self.load_waveform and index < self.waveform_items
        return self.get_audio_text_pair(
            self.audiopaths_and_text[index], load_waveform=load_waveform
        )


class HybridFSQCollate(TextAudioCollateMultiNSFsid):
    def __call__(self, batch):
        base = super().__call__([row[:6] for row in batch])
        order = torch.argsort(
            torch.tensor([row[0].size(-1) for row in batch]), descending=True
        )
        max_len = base[4].size(-1)

        def pad_field(index: int):
            output = torch.zeros(
                len(batch), batch[0][index].size(0), max_len, dtype=torch.float32
            )
            for destination, source_index in enumerate(order.tolist()):
                value = batch[source_index][index]
                output[destination, :, : value.size(-1)] = value
            return output

        coeff = torch.stack([batch[index][10] for index in order.tolist()])
        return base + (
            pad_field(6),
            pad_field(7),
            pad_field(8),
            pad_field(9),
            coeff,
        )
