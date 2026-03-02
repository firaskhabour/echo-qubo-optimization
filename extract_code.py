"""
extract_code.py
===============

Repo dump for review:
1) Full project tree from root, with summarisation rules:
   - Do NOT list JSON/NPZ one-by-one anywhere.
   - Instead, show JSON/NPZ summary lines as part of the tree in any folder containing them.
   - data/seeds: show full tree for first seed only; abbreviate remaining seed folders.
   - .venv appears but is not expanded.
2) Content export:
   - .py/.yaml/.yml full content
   - CSV: only export header for CSV files under results/ (and only those)
   - No JSON/NPZ content anywhere

Usage:
    python extract_code.py --root . --output project_dump.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


NO_EXPAND_DIRS = {".venv", "venv"}

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

EXPORT_SOURCE_EXTS = {".py", ".yaml", ".yml"}
CSV_EXT = ".csv"

SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


@dataclass
class BinSummary:
    json_count: int = 0
    json_example: Optional[str] = None
    npz_count: int = 0
    npz_example: Optional[str] = None


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def should_no_expand_dir(dirname: str) -> bool:
    return dirname in NO_EXPAND_DIRS


def summarize_json_npz(dir_path: Path) -> BinSummary:
    s = BinSummary()
    if not dir_path.exists() or not dir_path.is_dir():
        return s
    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf == ".json":
            s.json_count += 1
            if s.json_example is None:
                s.json_example = f.name
        elif suf == ".npz":
            s.npz_count += 1
            if s.npz_example is None:
                s.npz_example = f.name
    return s


def read_csv_header_line(csv_path: Path) -> str:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return "(empty csv)"
            return ",".join(header)
    except Exception as e:
        return f"(error reading csv header: {e})"


def collect_seed_dirs(seeds_root: Path) -> List[Path]:
    if not seeds_root.exists() or not seeds_root.is_dir():
        return []
    seed_dirs = []
    for p in seeds_root.iterdir():
        if not p.is_dir():
            continue
        m = SEED_DIR_RE.match(p.name)
        if m:
            seed_dirs.append(p)
    seed_dirs.sort(key=lambda x: int(SEED_DIR_RE.match(x.name).group(1)))  # type: ignore
    return seed_dirs


def build_tree(root: Path) -> str:
    lines: List[str] = []
    root = root.resolve()

    def list_entries(dir_path: Path) -> List[Path]:
        entries = []
        for p in dir_path.iterdir():
            if p.name in EXCLUDE_DIRS:
                continue
            # Never list JSON/NPZ individually in tree
            if p.is_file() and p.suffix.lower() in {".json", ".npz"}:
                continue
            entries.append(p)
        entries.sort(key=lambda x: (x.is_file(), x.name.lower()))
        return entries

    def add_bin_summaries(dir_path: Path, prefix: str) -> None:
        s = summarize_json_npz(dir_path)
        if s.json_count > 0:
            lines.append(prefix + f"JSON files (count={s.json_count}, example={s.json_example})")
        if s.npz_count > 0:
            lines.append(prefix + f"NPZ files (count={s.npz_count}, example={s.npz_example})")

    def recurse(dir_path: Path, prefix: str = "", within_seed: bool = False) -> None:
        # Special handling: data/seeds
        if dir_path.name == "seeds" and dir_path.parent.name == "data":
            seed_dirs = collect_seed_dirs(dir_path)
            other_entries = [e for e in list_entries(dir_path) if e not in seed_dirs]

            # Non-seed entries
            for i, entry in enumerate(other_entries):
                is_last = (i == len(other_entries) - 1) and (len(seed_dirs) == 0)
                connector = "└── " if is_last else "├── "
                if entry.is_dir():
                    lines.append(prefix + connector + entry.name + "/")
                    extension = "    " if is_last else "│   "
                    recurse(entry, prefix + extension, within_seed=False)
                else:
                    lines.append(prefix + connector + entry.name)

            # Seed dirs: first full, rest abbreviated
            if seed_dirs:
                first = seed_dirs[0]
                connector = "└── " if len(other_entries) == 0 else "├── "
                lines.append(prefix + connector + first.name + "/")
                extension = "    " if connector == "└── " else "│   "
                recurse(first, prefix + extension, within_seed=True)

                if len(seed_dirs) > 1:
                    rest = seed_dirs[1:]
                    names = [p.name for p in rest]
                    if len(names) <= 3:
                        short = ", ".join(names)
                    else:
                        short = f"{names[0]}, {names[1]}, ..., {names[-1]}"
                    lines.append(prefix + "└── " + f"(other seeds) {short}")

            # Add summaries for JSON/NPZ directly under data/seeds (rare, but consistent)
            add_bin_summaries(dir_path, prefix)
            return

        entries = list_entries(dir_path)

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "

            if entry.is_dir() and should_no_expand_dir(entry.name):
                lines.append(prefix + connector + entry.name + "/")
                continue

            if entry.is_dir():
                lines.append(prefix + connector + entry.name + "/")
                extension = "    " if is_last else "│   "
                recurse(entry, prefix + extension, within_seed=within_seed or (within_seed))
            else:
                lines.append(prefix + connector + entry.name)

        # After listing visible entries, append JSON/NPZ summary lines for this directory
        add_bin_summaries(dir_path, prefix)

    lines.append(root.name + "/")
    recurse(root, prefix="")
    return "\n".join(lines)


def collect_content_files(root: Path) -> List[Path]:
    """
    Content export rules:
      - Full content for .py/.yaml/.yml anywhere (except excluded/no-expand)
      - CSV header ONLY for CSV under results/
      - No CSV content outside results/
    """
    root = root.resolve()
    results_dir = root / "results"

    out: List[Path] = []

    for p in root.rglob("*"):
        if is_excluded(p):
            continue
        if any(part in NO_EXPAND_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue

        suf = p.suffix.lower()
        if suf in EXPORT_SOURCE_EXTS:
            out.append(p)
        elif suf == CSV_EXT:
            # Only include CSV header in content if it's under results/
            try:
                p.relative_to(results_dir)
                out.append(p)
            except Exception:
                pass

    out.sort(key=lambda x: str(x).lower())
    return out


def write_dump(project_root: Path, output_file: Path) -> None:
    project_root = project_root.resolve()
    tree_str = build_tree(project_root)
    files = collect_content_files(project_root)

    with output_file.open("w", encoding="utf-8") as out:
        out.write("=" * 110 + "\n")
        out.write("PROJECT DUMP (TREE + SELECTED CONTENT)\n")
        out.write("=" * 110 + "\n")
        out.write(f"Root:      {project_root}\n")
        out.write(f"Generated: {datetime.now()}\n")
        out.write("Content export:\n")
        out.write("  - .py/.yaml/.yml full content\n")
        out.write("  - CSV header only for CSV under results/\n")
        out.write("  - JSON/NPZ content omitted (summarised in tree only)\n")
        out.write("=" * 110 + "\n\n")

        out.write("FOLDER TREE\n")
        out.write("-" * 110 + "\n")
        out.write(tree_str + "\n\n")
        out.write("=" * 110 + "\n\n")

        out.write("FILE CONTENT\n")
        out.write("-" * 110 + "\n")

        for fp in files:
            rel = fp.relative_to(project_root)
            out.write("\n" + "#" * 110 + "\n")
            out.write(f"# FILE: {rel}\n")
            out.write("#" * 110 + "\n\n")

            suf = fp.suffix.lower()
            if suf == CSV_EXT:
                out.write("CSV HEADER ONLY:\n")
                out.write(read_csv_header_line(fp) + "\n")
                continue

            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                out.write(f"(error reading file: {e})\n")
                continue

            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")

    print(f"[OK] Wrote dump: {output_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump project tree + selected file content for review.")
    ap.add_argument("--root", default=".", help="Project root (default: .)")
    ap.add_argument("--output", default="project_dump.txt", help="Output file name")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    write_dump(root, out)


if __name__ == "__main__":
    main()