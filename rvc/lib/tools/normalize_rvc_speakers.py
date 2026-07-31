"""Normalize a folder-per-speaker dataset for the RVC preprocessor.

The RVC preprocessor treats every directory containing audio as an independent
speaker and requires contiguous numeric directory prefixes. This utility moves
nested audio to its top-level speaker directory, assigns contiguous IDs, and
writes a manifest describing every move.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"}
MANIFEST_NAME = "rvc_dataset_manifest.tsv"


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS


def normalized_directories(root: Path):
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    )
    parsed = []
    for directory in directories:
        match = re.fullmatch(r"(\d+)_(.+)", directory.name)
        if match is None:
            return None
        parsed.append((int(match.group(1)), directory))
    if [speaker_id for speaker_id, _ in parsed] != list(range(len(parsed))):
        return None
    return parsed


def build_plan(root: Path):
    speakers = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    )
    if not speakers:
        raise ValueError(f"No speaker directories found in {root}")
    root_audio = [path for path in root.iterdir() if is_audio(path)]
    if root_audio:
        raise ValueError(
            "Audio files at dataset root are ambiguous in a multi-speaker "
            f"dataset: {root_audio[0]}"
        )

    directory_moves = []
    audio_moves = []
    occupied_destinations = set()
    for speaker_id, source_directory in enumerate(speakers):
        original_name = source_directory.name
        clean_name = re.sub(r"^\d+_", "", original_name)
        target_directory = root / f"{speaker_id}_{clean_name}"
        if target_directory.exists() and target_directory != source_directory:
            raise FileExistsError(f"Target speaker directory exists: {target_directory}")
        directory_moves.append((source_directory, target_directory, speaker_id))

        nested_audio = sorted(
            (
                path
                for path in source_directory.rglob("*")
                if is_audio(path) and path.parent != source_directory
            ),
            key=lambda path: natural_key(path.relative_to(source_directory).as_posix()),
        )
        for index, source_audio in enumerate(nested_audio):
            safe_name = re.sub(
                r"[^A-Za-z0-9._-]+", "_", source_audio.name
            ).strip("._")
            if not safe_name:
                safe_name = f"audio{source_audio.suffix.casefold()}"
            destination_name = f"rvc_flat_{index:06d}_{safe_name}"
            destination_before_rename = source_directory / destination_name
            destination_after_rename = target_directory / destination_name
            key = str(destination_before_rename).casefold()
            if destination_before_rename.exists() or key in occupied_destinations:
                raise FileExistsError(
                    f"Flattened audio destination collision: {destination_before_rename}"
                )
            occupied_destinations.add(key)
            audio_moves.append(
                (
                    source_audio,
                    destination_before_rename,
                    destination_after_rename,
                    speaker_id,
                    original_name,
                )
            )
    return directory_moves, audio_moves


def apply_plan(root: Path, directory_moves, audio_moves) -> Path:
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists; refusing to overwrite: {manifest_path}"
        )

    manifest_rows = []
    for source, temporary_destination, final_destination, speaker_id, name in audio_moves:
        source_relative = source.relative_to(root).as_posix()
        temporary_destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(temporary_destination)
        manifest_rows.append(
            (
                "audio",
                speaker_id,
                name,
                source_relative,
                final_destination.relative_to(root).as_posix(),
            )
        )

    for source, destination, speaker_id in directory_moves:
        source_relative = source.relative_to(root).as_posix()
        if source != destination:
            source.rename(destination)
        manifest_rows.append(
            (
                "speaker_directory",
                speaker_id,
                source.name,
                source_relative,
                destination.relative_to(root).as_posix(),
            )
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ("kind", "speaker_id", "original_speaker", "source", "destination")
        )
        writer.writerows(manifest_rows)
    return manifest_path


def validate(root: Path):
    speakers = normalized_directories(root)
    if speakers is None:
        raise ValueError("Speaker directories do not have contiguous numeric IDs")
    nested_audio = [
        path
        for _, directory in speakers
        for path in directory.rglob("*")
        if is_audio(path) and path.parent != directory
    ]
    if nested_audio:
        raise ValueError(f"Nested audio remains after normalization: {nested_audio[0]}")
    audio_counts = [
        sum(1 for path in directory.iterdir() if is_audio(path))
        for _, directory in speakers
    ]
    if any(count == 0 for count in audio_counts):
        empty_index = audio_counts.index(0)
        raise ValueError(f"Speaker {empty_index} has no audio")
    return speakers, audio_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned moves. Without this flag, perform a dry run.",
    )
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    existing = normalized_directories(root)
    if existing is not None:
        speakers, counts = validate(root)
        print(
            f"Dataset is already normalized: {len(speakers)} speakers, "
            f"{sum(counts)} audio files."
        )
        return

    directory_moves, audio_moves = build_plan(root)
    print(
        f"Plan: {len(directory_moves)} speaker directories, "
        f"{len(audio_moves)} nested audio moves."
    )
    if not args.apply:
        print("Dry run only. Pass --apply to normalize the dataset.")
        return

    manifest = apply_plan(root, directory_moves, audio_moves)
    speakers, counts = validate(root)
    print(
        f"Normalized {len(speakers)} speakers and {sum(counts)} audio files. "
        f"Manifest: {manifest}"
    )


if __name__ == "__main__":
    main()
