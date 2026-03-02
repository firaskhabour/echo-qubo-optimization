"""
Refactor helper: build a CLEAN src_code/ package from selected src/ files,
renaming key files and rewriting imports/paths/results filenames.

- Does NOT modify src/
- Only writes into src_code/

Run:
    python src/refactor_to_src_code.py --dry-run
    python src/refactor_to_src_code.py --apply --overwrite-dst
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


NEW_SRC_DIRNAME = "src_code"

# ---------------------------------------------------------------------
# 1) Curated file set: only what a new user needs
# ---------------------------------------------------------------------
# Map: src/<filename> -> (dst_subdir, dst_filename)
COPY_PLAN: Dict[str, Tuple[str, str]] = {
    # Generators
    "generate_data.py": ("generators", "generate_data.py"),
    "generate_seeds_batch.py": ("generators", "generate_seeds_batch.py"),
    "build_qubo.py": ("generators", "build_qubo.py"),

    # Solvers
    "solve_classical.py": ("solvers", "solve_classical.py"),
    "solve_qubo_gurobi_exact.py": ("solvers", "solve_qubo_gurobi_exact.py"),
    "spectral_landscape_navigation.py": ("solvers", "echo_optimizer.py"),

    # Runners
    "run_experiments.py": ("runners", "run_baseline_full.py"),
    "run_sln_full.py": ("runners", "run_echo_full.py"),

    # Utils (keep only if referenced)
    "audit_solution.py": ("utils", "audit_solution.py"),
    "preflight_validation.py": ("utils", "preflight_validation.py"),
    "extract_parameters.py": ("utils", "extract_parameters.py"),
    "check_qubo.py": ("utils", "check_qubo.py"),
    "check_npz.py": ("utils", "check_npz.py"),
}

# ---------------------------------------------------------------------
# 2) Text rewrite rules (applied ONLY to files inside src_code/)
# ---------------------------------------------------------------------
TEXT_REWRITE_RULES: List[Tuple[re.Pattern, str]] = [
    # sys.path points to src_code (some runner scripts do this)
    (re.compile(r"sys\.path\.insert\(\s*0\s*,\s*['\"]src['\"]\s*\)"),
     "sys.path.insert(0, 'src_code')"),

    # Old imports of spectral module -> new module path
    (re.compile(r"from\s+spectral_landscape_navigation\s+import\s+spectral_landscape_navigation"),
     "from solvers.echo_optimizer import spectral_landscape_navigation"),
    (re.compile(r"import\s+spectral_landscape_navigation\b"),
     "from solvers import echo_optimizer  # noqa: F401"),

    # If code imported solve files as top-level modules, point to package folders
    (re.compile(r"from\s+solve_classical\s+import\s+"),
     "from solvers.solve_classical import "),
    (re.compile(r"from\s+solve_qubo_gurobi_exact\s+import\s+"),
     "from solvers.solve_qubo_gurobi_exact import "),

    (re.compile(r"from\s+build_qubo\s+import\s+"),
     "from generators.build_qubo import "),
    (re.compile(r"from\s+generate_data\s+import\s+"),
     "from generators.generate_data import "),
    (re.compile(r"from\s+generate_seeds_batch\s+import\s+"),
     "from generators.generate_seeds_batch import "),

    # Results file naming
    (re.compile(r"results/results_master\.csv"),
     "results/baseline_full_results.csv"),
    (re.compile(r"\bresults_master\.csv\b"),
     "baseline_full_results.csv"),

    (re.compile(r"results/sln_full_results\.csv"),
     "results/echo_full_results.csv"),
    (re.compile(r"\bsln_full_results\.csv\b"),
     "echo_full_results.csv"),
]

REWRITE_EXTENSIONS = {".py", ".md", ".txt"}


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_remove_tree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def copy_selected_files(src_root: Path, dst_root: Path) -> List[Tuple[Path, Path]]:
    copied: List[Tuple[Path, Path]] = []
    for src_name, (subdir, dst_name) in COPY_PLAN.items():
        s = src_root / src_name
        d = dst_root / subdir / dst_name
        if not s.exists():
            print(f"[WARN] Missing in src/, skipping: {src_name}")
            continue
        mkdir(d.parent)
        shutil.copy2(s, d)
        copied.append((s, d))
        print(f"[OK] Copied: {s.name} -> {d.relative_to(dst_root)}")
    return copied


def rewrite_text_in_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    updated = original
    for pattern, repl in TEXT_REWRITE_RULES:
        updated = pattern.sub(repl, updated)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def rewrite_all(dst_root: Path) -> None:
    changed_count = 0
    scanned_count = 0
    for p in dst_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in REWRITE_EXTENSIONS:
            scanned_count += 1
            if rewrite_text_in_file(p):
                changed_count += 1
    print(f"[DONE] Scanned {scanned_count} files; changed {changed_count} files.")


def strip_big_comment_blocks(path: Path) -> None:
    """
    Minimal removal of noisy header commentary while avoiding code damage.
    Only removes very large triple-quoted blocks at top of file.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Remove ONLY if the file starts with a huge """...""" docstring > 30 lines
    if txt.lstrip().startswith('"""'):
        end = txt.find('"""', 3)
        if end != -1:
            header = txt[:end+3]
            if header.count("\n") > 30:
                new_txt = txt[end+3:].lstrip()
                path.write_text(new_txt, encoding="utf-8")
                print(f"[OK] Trimmed oversized header docstring in: {path.name}")


def optional_cleanup_comments(dst_root: Path) -> None:
    # Apply only to the big offender (echo optimizer / long runners), if you want
    for rel in [
        Path("solvers/echo_optimizer.py"),
        Path("runners/run_echo_full.py"),
        Path("runners/run_baseline_full.py"),
    ]:
        p = dst_root / rel
        if p.exists():
            strip_big_comment_blocks(p)


def write_init_files(dst_root: Path) -> None:
    # Turn folders into importable packages
    for sub in ["generators", "solvers", "runners", "utils"]:
        pkg_dir = dst_root / sub
        pkg_dir.mkdir(parents=True, exist_ok=True)   # <-- FIX: ensure directory exists
        initp = pkg_dir / "__init__.py"
        if not initp.exists():
            initp.write_text("", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without writing anything")
    ap.add_argument("--apply", action="store_true", help="Perform build of src_code/")
    ap.add_argument("--overwrite-dst", action="store_true", help="Delete existing src_code/ then recreate")
    ap.add_argument("--no-comment-clean", action="store_true", help="Skip trimming oversized header docstrings")
    args = ap.parse_args()

    project_root = Path.cwd()
    src_root = project_root / "src"
    dst_root = project_root / NEW_SRC_DIRNAME

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Using src:     {src_root}")
    print(f"[INFO] Target dst:    {dst_root}")

    if not src_root.exists():
        raise FileNotFoundError(f"Missing {src_root}. Run from project root (where src/ exists).")

    if args.dry_run or not args.apply:
        print(f"[DRY] Would (re)create: {dst_root}")
        print("[DRY] Would copy these files:")
        for k, (sub, v) in COPY_PLAN.items():
            present = (src_root / k).exists()
            print(f"  - {k:<35} -> {sub}/{v:<25} present_in_src={present}")
        print("\nNext: run with --apply (optionally --overwrite-dst).\n")
        return

    # APPLY
    if dst_root.exists():
        if not args.overwrite_dst:
            raise FileExistsError(
                f"{dst_root} already exists. Use --overwrite-dst to recreate it."
            )
        safe_remove_tree(dst_root)

    mkdir(dst_root)
    write_init_files(dst_root)

    copy_selected_files(src_root, dst_root)
    rewrite_all(dst_root)

    if not args.no_comment_clean:
        optional_cleanup_comments(dst_root)

    print("\n[OK] Clean refactor complete.")
    print("Original src/ untouched.")
    print("Use new scripts from src_code/runners/ ...\n")


if __name__ == "__main__":
    main()