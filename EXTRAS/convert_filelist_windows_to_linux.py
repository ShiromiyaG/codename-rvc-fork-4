#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


def normalize_windows_path(path: str) -> str:
    return path.replace("\\", "/")


def normalize_windows_root(root: str | None) -> str | None:
    if not root:
        return None
    normalized = normalize_windows_path(root).rstrip("/")
    return normalized.lower()


def looks_like_path(token: str) -> bool:
    return (
        "\\" in token
        or "/" in token
        or bool(re.match(r"^[A-Za-z]:", token.strip()))
    )


def convert_path_token(token: str, linux_root: str, windows_root: str | None) -> str:
    original = token.strip()
    if not original or not looks_like_path(original):
        return token

    path = normalize_windows_path(original)
    linux_root = linux_root.rstrip("/")

    if windows_root:
        low_path = path.lower()
        if low_path.startswith(windows_root):
            suffix = path[len(windows_root) :].lstrip("/")
            converted = f"{linux_root}/{suffix}" if suffix else linux_root
            return token.replace(original, converted)

    logs_idx = path.lower().find("/logs/")
    if logs_idx != -1:
        converted = f"{linux_root}{path[logs_idx:]}"
        return token.replace(original, converted)

    if WINDOWS_DRIVE_RE.match(path):
        converted = "/" + path.split(":", 1)[1].lstrip("/")
        return token.replace(original, converted)

    return token.replace(original, path)


def convert_file(file_path: Path, linux_root: str, windows_root: str | None, in_place: bool) -> tuple[Path, int]:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    changed_lines = 0
    output_lines = []

    for line in lines:
        if not line.strip():
            output_lines.append(line)
            continue

        parts = line.split("|")
        new_parts = [convert_path_token(part, linux_root, windows_root) for part in parts]
        new_line = "|".join(new_parts)

        if new_line != line:
            changed_lines += 1

        output_lines.append(new_line)

    output_text = "\n".join(output_lines)
    if content.endswith("\n"):
        output_text += "\n"

    if in_place:
        target = file_path
    else:
        target = file_path.with_name(f"{file_path.stem}.linux{file_path.suffix}")

    target.write_text(output_text, encoding="utf-8")
    return target, changed_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte filelist.txt com caminhos Windows para Linux."
    )
    parser.add_argument(
        "filelists",
        nargs="+",
        help="Um ou mais caminhos de filelist.txt para converter.",
    )
    parser.add_argument(
        "--linux-root",
        default=str(Path.cwd()),
        help="Raiz Linux do projeto (padrão: diretório atual).",
    )
    parser.add_argument(
        "--windows-root",
        default=None,
        help="Prefixo Windows a substituir (ex: F:\\AI Cover\\codename-rvc-fork-4).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Sobrescreve os arquivos originais em vez de criar *.linux.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    linux_root = str(Path(args.linux_root).resolve())
    windows_root = normalize_windows_root(args.windows_root)

    for file_name in args.filelists:
        file_path = Path(file_name)
        if not file_path.exists():
            print(f"[ERRO] Arquivo não encontrado: {file_path}")
            continue

        target, changed = convert_file(
            file_path=file_path,
            linux_root=linux_root,
            windows_root=windows_root,
            in_place=args.in_place,
        )
        print(f"[OK] {file_path} -> {target} ({changed} linhas alteradas)")


if __name__ == "__main__":
    main()
