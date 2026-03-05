#!/usr/bin/env python3
"""
Migrate pretrained models and training logs from FireflyGAN → ChouwaGAN.

This script updates:
  1. .pth checkpoint files: renames "vocoder" and "vocoder_architecture" fields
  2. model_info.json files: renames "vocoder_architecture" field
  3. config.json files: no content changes needed (they don't store the vocoder name)

Usage:
    python EXTRAS/migrate_firefly_to_chouwa.py [--scan-dir DIR] [--dry-run]

By default, scans the project root (logs/, rvc/models/) for .pth and model_info.json files.
Use --dry-run to preview changes without writing anything.
"""

import argparse
import json
import os
import sys

# Attempt torch import for .pth handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


OLD_VOCODER = "FireflyGAN"
NEW_VOCODER = "ChouwaGAN"

OLD_ARCH = "firefly_gan"
NEW_ARCH = "chouwa_gan"


def migrate_pth(filepath: str, dry_run: bool = False) -> bool:
    """Update vocoder fields inside a .pth checkpoint. Returns True if modified."""
    if not HAS_TORCH:
        print(f"  [SKIP] {filepath} — torch not available, cannot load .pth files")
        return False

    try:
        ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return False

    modified = False

    # Fix "vocoder" field
    if ckpt.get("vocoder") == OLD_VOCODER:
        if dry_run:
            print(f"  [DRY-RUN] Would rename vocoder '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
        else:
            ckpt["vocoder"] = NEW_VOCODER
            print(f"  [OK] Renamed vocoder '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
        modified = True

    # Fix "vocoder_architecture" field
    if ckpt.get("vocoder_architecture") == OLD_VOCODER:
        if dry_run:
            print(f"  [DRY-RUN] Would rename vocoder_architecture '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
        else:
            ckpt["vocoder_architecture"] = NEW_VOCODER
            print(f"  [OK] Renamed vocoder_architecture '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
        modified = True

    if modified and not dry_run:
        torch.save(ckpt, filepath)
        print(f"  [SAVED] {filepath}")

    return modified


def migrate_json(filepath: str, dry_run: bool = False) -> bool:
    """Update vocoder fields inside a JSON file (model_info.json). Returns True if modified."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return False

    modified = False

    for key in ["vocoder_architecture", "vocoder", "vocoder_v2"]:
        if data.get(key) == OLD_VOCODER:
            if dry_run:
                print(f"  [DRY-RUN] Would rename {key} '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
            else:
                data[key] = NEW_VOCODER
                print(f"  [OK] Renamed {key} '{OLD_VOCODER}' → '{NEW_VOCODER}' in {filepath}")
            modified = True

    if modified and not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"  [SAVED] {filepath}")

    return modified


def find_files(scan_dir: str, extensions: list[str]) -> list[str]:
    """Recursively find files matching given extensions, skipping env/ and __pycache__/."""
    results = []
    skip_dirs = {"env", "__pycache__", ".git", "node_modules"}
    for root, dirs, files in os.walk(scan_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                results.append(os.path.join(root, f))
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate FireflyGAN → ChouwaGAN in pretrained models and configs."
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default=None,
        help="Directory to scan (default: project root, scanning logs/ and rvc/models/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    args = parser.parse_args()

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.scan_dir:
        scan_dirs = [os.path.abspath(args.scan_dir)]
    else:
        scan_dirs = [
            os.path.join(project_root, "logs"),
            os.path.join(project_root, "rvc", "models"),
        ]

    print(f"{'=' * 60}")
    print(f"  FireflyGAN → ChouwaGAN Migration Script")
    print(f"  Mode: {'DRY RUN (no files will be modified)' if args.dry_run else 'LIVE (files will be modified)'}")
    print(f"{'=' * 60}")

    total_modified = 0

    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            print(f"\n[WARN] Directory not found: {scan_dir}, skipping.")
            continue

        print(f"\nScanning: {scan_dir}")
        print("-" * 40)

        # Process .pth files
        pth_files = find_files(scan_dir, [".pth"])
        if pth_files:
            print(f"\nFound {len(pth_files)} .pth file(s):")
            for pth in pth_files:
                rel = os.path.relpath(pth, project_root)
                result = migrate_pth(pth, dry_run=args.dry_run)
                if result:
                    total_modified += 1
                elif HAS_TORCH:
                    print(f"  [SKIP] {rel} — no FireflyGAN references found")

        # Process JSON files (model_info.json, training presets, etc.)
        json_files = find_files(scan_dir, [".json"])
        if json_files:
            print(f"\nFound {len(json_files)} .json file(s):")
            for jf in json_files:
                if migrate_json(jf, dry_run=args.dry_run):
                    total_modified += 1
                else:
                    rel = os.path.relpath(jf, project_root)
                    print(f"  [SKIP] {rel} — no FireflyGAN references found")

    print(f"\n{'=' * 60}")
    print(f"  Done! {total_modified} file(s) {'would be' if args.dry_run else ''} modified.")
    if args.dry_run and total_modified > 0:
        print(f"  Run without --dry-run to apply changes.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
