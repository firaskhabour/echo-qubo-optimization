#!/usr/bin/env python
# Start_ECHO_Experiment.py  —  place at project root, run with:
#     python Start_ECHO_Experiment.py
"""
ECHO Experiment Launcher
========================
Single entry point for all experiment pipelines.

MAIN MENU
  1. Run Full Pipeline          (runs Option 2 then Option 3 in sequence)
  2. Run Primary QUBO Pipeline  (Generate instances → Greedy → SA → Gurobi → ECHO-SA)
  3. Run Benchmark QUBO Families (Max-Cut, Portfolio) → SA → ECHO-SA
  Q. Quit

Option 2 calls the two existing runner scripts unchanged:
    src_code/runners/run_baseline_full.py   (greedy + SA + Gurobi)
    src_code/runners/run_echo_full.py       (SA-ECHO vs SA baseline)

Option 3 calls:
    src_code/runners/run_benchmarks.py      (Max-Cut + Portfolio SA + SA-ECHO)

Dependency checks are performed before each option so the user receives
a clear message and a recovery prompt rather than a crash.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY   = sys.executable

# Existing runner scripts  (never modified)
_BASELINE_SCRIPT = ROOT / "src_code" / "runners" / "run_baseline_full.py"

# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

# Required packages: (import_name, pip_name, min_version_display)
# These must be present for any pipeline to run.
# gurobipy is optional — checked separately per option.
_REQUIRED_PACKAGES = [
    ("numpy",  "numpy",  "1.24"),
    ("pandas", "pandas", "1.5"),
    ("yaml",   "pyyaml", "6.0"),
    ("tqdm",   "tqdm",   "4.64"),
]

def _get_version(import_name: str, pip_name: str) -> str:
    """Return installed version string, or '' if not found."""
    try:
        import importlib.metadata
        return importlib.metadata.version(pip_name)
    except Exception:
        pass
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return ""


def _check_requirements() -> bool:
    """
    Check every required package and print a full status table.

    For each package:
      [ok]      name  vX.Y.Z   — installed
      [missing] name           — not importable

    If all packages are present  → "All X libraries found and installed."
    If any are missing           → offer to install each one by one via pip.

    Returns True only when every required package is available after
    any installs.  Gurobi is optional and checked per-option, not here.
    """
    print()
    print("  Checking required libraries...")
    _hr("-")

    statuses = []   # (import_name, pip_name, min_ver, installed, version)
    for import_name, pip_name, min_ver in _REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            ver = _get_version(import_name, pip_name)
            statuses.append((import_name, pip_name, min_ver, True,  ver))
        except ImportError:
            statuses.append((import_name, pip_name, min_ver, False, ""))

    for import_name, pip_name, min_ver, installed, ver in statuses:
        if installed:
            ver_str = f"  v{ver}" if ver and ver != "?" else ""
            print(f"  [ok]      {pip_name:<12}{ver_str}")
        else:
            print(f"  [missing] {pip_name:<12}  (required >= {min_ver})")

    _hr("-")

    missing = [(n, p, v) for n, p, v, ok, _ in statuses if not ok]

    if not missing:
        print(f"  All {len(statuses)} libraries found and installed.")
        return True

    print(f"  {len(missing)} of {len(statuses)} libraries are missing.")
    print()
    print("  You can install everything at once with:")
    print("      pip install -r requirements.txt")
    print()

    for import_name, pip_name, min_ver in missing:
        ch = input(f"  Install '{pip_name}>={min_ver}' now? [y/n]: ").strip().lower()
        if ch == "y":
            print(f"  Installing {pip_name}...")
            result = subprocess.run(
                [PY, "-m", "pip", "install", f"{pip_name}>={min_ver}"],
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"\n  [error] Failed to install {pip_name}.")
                print(f"  Please run:  pip install {pip_name}>={min_ver}")
                print("  Then restart this script.")
                return False
            # Verify it now imports
            try:
                __import__(import_name)
                print(f"  [ok] {pip_name} installed successfully.")
            except ImportError:
                print(f"  [error] {pip_name} installed but still not importable.")
                print(f"  Try: pip install --upgrade {pip_name}")
                return False
        else:
            print(f"\n  Cannot continue without '{pip_name}'.")
            print(f"  Please run:  pip install {pip_name}>={min_ver}")
            print("  Then restart this script.")
            return False

    # All missing packages were just installed — print final confirmation
    print()
    print(f"  All {len(statuses)} libraries now installed.")
    return True

# Remaining runner paths (defined after ROOT is set above)
_ECHO_SCRIPT  = ROOT / "src_code" / "runners" / "run_echo_full.py"
_BENCH_SCRIPT = ROOT / "src_code" / "runners" / "run_benchmarks.py"

# Key result files produced by Option 2
_BASELINE_CSV = ROOT / "results" / "baseline_full_results.csv"
_ECHO_CSV     = ROOT / "results" / "echo_full_results.csv"
_ECHO_SCRIPT     = ROOT / "src_code" / "runners" / "run_echo_full.py"

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _hr(ch: str = "=", width: int = 60) -> None:
    print(ch * width)

def _ask(prompt: str, valid: list[str]) -> str:
    vl = [v.lower() for v in valid]
    while True:
        raw = input(f"\n{prompt} ").strip().lower()
        if raw in vl:
            return raw
        print(f"  Please enter one of: {' / '.join(valid)}")


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run(script: Path, extra_flags: list[str] = ()) -> bool:
    """
    Run *script* as a subprocess from the project root, streaming output live.
    Returns True on success (exit code 0).
    """
    cmd    = [PY, str(script)] + list(extra_flags)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_scripts_for_option(option: str) -> bool:
    """
    Check that the runner scripts required for *option* exist.
    option: "2" | "3" | "1"  (matches menu choices)

    Option 3 needs only run_benchmarks.py.
    Options 1 and 2 need run_baseline_full.py + run_echo_full.py.
    Option 1 needs all three.
    """
    if option == "3":
        required = [("run_benchmarks.py",   _BENCH_SCRIPT)]
    elif option == "2":
        required = [("run_baseline_full.py", _BASELINE_SCRIPT),
                    ("run_echo_full.py",     _ECHO_SCRIPT)]
    else:  # option "1" — full pipeline
        required = [("run_baseline_full.py", _BASELINE_SCRIPT),
                    ("run_echo_full.py",     _ECHO_SCRIPT),
                    ("run_benchmarks.py",    _BENCH_SCRIPT)]

    ok = True
    for name, path in required:
        if path.exists():
            print(f"  [ok]      {name}")
        else:
            print(f"  [missing] {name}  (expected: {path.relative_to(ROOT)})")
            ok = False

    return ok


def _check_gurobi_optional() -> None:
    """
    Print Gurobi availability status (Options 1 and 2 only).
    If missing, give clear download and licence instructions.
    Gurobi is optional — its absence skips the exact-solver baseline only.
    """
    try:
        import gurobipy  # noqa: F401
        ver = ""
        try:
            import importlib.metadata
            ver = f"  v{importlib.metadata.version('gurobipy')}"
        except Exception:
            pass
        print(f"  [ok]      gurobipy{ver} — Gurobi exact solver available.")
    except ImportError:
        print("  [info]    gurobipy not found — Gurobi steps will be skipped.")
        print("            Gurobi exact-solver results will be absent from output.")
        print()
        print("            To enable Gurobi (optional):")
        print("              1. Get a free academic licence:")
        print("                 https://www.gurobi.com/academia/academic-program-and-licenses/")
        print("              2. Install the package:")
        print("                 pip install gurobipy")
        print("              3. Activate your licence:")
        print("                 grbgetkey <your-licence-key>")


def _insurance_baseline_exists() -> bool:
    return _BASELINE_CSV.exists()


def _insurance_echo_ready() -> bool:
    """ECHO runner needs baseline_full_results.csv to exist and be non-empty."""
    if not _BASELINE_CSV.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(_BASELINE_CSV)
        return len(df) > 0
    except Exception:
        return False


def _prompt_generate_insurance() -> bool:
    """
    Ask the user whether to generate insurance instances now.
    Returns True if they agreed and the run succeeded.
    """
    print("\n  Insurance baseline results not found.")
    print(f"  Expected: {_BASELINE_CSV.relative_to(ROOT)}")
    print("\n  This file is produced by running run_baseline_full.py")
    print("  (greedy heuristic + SA + optional Gurobi on all insurance instances).")
    ch = _ask("  Generate insurance baseline now? [y/n]:", ["y", "n"])
    if ch == "y":
        print()
        return _run(_BASELINE_SCRIPT)
    return False


# ---------------------------------------------------------------------------
# Option 2  —  Run ECHO SA Benchmark Pipeline  (insurance)
# ---------------------------------------------------------------------------

def _run_option2() -> bool:
    """
    Option 2: Run ECHO SA Benchmark Pipeline.

    Step 1  run_baseline_full.py   → results/baseline_full_results.csv
    Step 2  run_echo_full.py       → results/echo_full_results.csv

    If baseline results already exist, Step 1 is skipped and the user is
    informed.  Step 2 requires Step 1 output.
    """
    print()
    _hr()
    print("  Option 2 — Run Primary QUBO Pipeline")
    _hr()
    print()
    print("  Checking required files for Option 2...")
    if not _check_scripts_for_option("2"):
        print()
        print("  Cannot run Option 2. Ensure both runner scripts are present in:")
        print(f"  {_BASELINE_SCRIPT.parent.relative_to(ROOT)}/")
        return False
    _check_gurobi_optional()
    print()
    print("  Generate instances → Greedy → SA → Gurobi → ECHO-SA")
    print()
    print("  This pipeline runs on the insurance QUBO experiment corpus:")
    print("    Step 1  Generate instances + Greedy + SA + Gurobi")
    print("            → results/baseline_full_results.csv")
    print("    Step 2  ECHO-SA vs SA baseline")
    print("            → results/echo_full_results.csv")

    # --- Step 1 ---
    if _insurance_baseline_exists():
        print(f"\n  [found] {_BASELINE_CSV.relative_to(ROOT)}")
        ch = _ask("  Re-run baseline (Step 1)? Skipping is safe if data is current. [y/n]:", ["y", "n"])
        if ch == "y":
            print()
            ok = _run(_BASELINE_SCRIPT)
            if not ok:
                print("\n  [error] Baseline run failed. Check output above.")
                return False
    else:
        ok = _prompt_generate_insurance()
        if not ok:
            print("\n  Cannot proceed to Step 2 without baseline results.")
            return False

    # --- Step 2 ---
    if not _insurance_echo_ready():
        print("\n  [error] baseline_full_results.csv still missing or empty after Step 1.")
        print(f"          Expected: {_BASELINE_CSV.relative_to(ROOT)}")
        return False

    print("\n  Running Step 2: SA-ECHO on insurance instances...")
    print()
    ok = _run(_ECHO_SCRIPT)
    if not ok:
        print("\n  [error] SA-ECHO run failed. Check output above.")
    return ok


# ---------------------------------------------------------------------------
# Option 3  —  Run ECHO Max-Cut & Portfolio benchmark
# ---------------------------------------------------------------------------

def _run_option3() -> bool:
    """
    Option 3: Run ECHO Max-Cut & cardinality-constrained asset selection benchmark.

    Runs SA (solver="sa") and SA-ECHO (solver="sa_echo") on:
      - Max-Cut instances  (random Erdős-Rényi graphs)
      - Portfolio cardinality instances

    Results written to:
      results/maxcut/sa_results.csv
      results/maxcut/sa_echo_results.csv
      results/maxcut/results_master.csv
      results/portfolio_card/sa_results.csv
      results/portfolio_card/sa_echo_results.csv
      results/portfolio_card/results_master.csv
    """
    print()
    _hr()
    print("  Option 3 — Run Benchmark QUBO Families (Max-Cut, Portfolio) → SA → ECHO-SA")
    _hr()
    print()
    print("  Benchmark families:")
    print("    maxcut        — Max-Cut QUBO on random Erdős-Rényi graphs")
    print("    portfolio_card— Cardinality-constrained portfolio selection QUBO")
    print()
    print("  Solvers run on each instance:")
    print("    sa        — multistart simulated annealing (20 starts × 40k steps)")
    print("    sa_echo   — SA-ECHO homotopy (same 800k-step budget)")
    print()
    print("  Outputs per family:")
    print("    results/<family>/sa_results.csv")
    print("    results/<family>/sa_echo_results.csv")
    print("    results/<family>/results_master.csv  (paired SA vs SA-ECHO)")
    print()

    print("  Checking required files for Option 3...")
    if not _check_scripts_for_option("3"):
        print()
        print("  Cannot run Option 3. Ensure run_benchmarks.py is present in:")
        print(f"  {_BENCH_SCRIPT.parent.relative_to(ROOT)}/")
        return False
    print()

    ok = _run(_BENCH_SCRIPT, ["--family", "all"])
    if not ok:
        print("\n  [error] Benchmark run failed. Check output above.")
    return ok


# ---------------------------------------------------------------------------
# Option 1  —  Full Pipeline
# ---------------------------------------------------------------------------

def _run_option1() -> bool:
    """
    Option 1: Run Full Pipeline (Option 2 then Option 3).
    """
    print()
    _hr()
    print("  Option 1 — Run Full Pipeline")
    _hr()
    print()
    print("  Checking required files for Option 1...")
    if not _check_scripts_for_option("1"):
        print()
        print("  Cannot run Option 1. Ensure all three runner scripts are present.")
        return False
    _check_gurobi_optional()
    print()
    print("  Runs Option 2 (Primary QUBO Pipeline) then")
    print("  Option 3 (Benchmark QUBO Families) in sequence.")
    print()

    ok2 = _run_option2()
    if not ok2:
        print("\n  [warning] Option 2 did not complete cleanly.")
        ch = _ask("  Continue to Option 3 anyway? [y/n]:", ["y", "n"])
        if ch == "n":
            return False

    print()
    _hr("-")
    print("  Full Pipeline: starting Option 3...")
    _hr("-")

    ok3 = _run_option3()
    return ok2 and ok3


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _hr()
    print("  ECHO Experiment Launcher — checking requirements...")
    _hr()
    if not _check_requirements():
        sys.exit(1)

    # Script-file checks are done per-option (see _check_scripts_for_option).
    # No generic file warnings at startup — user sees only what's relevant.

    while True:
        print()
        _hr()
        print("  MAIN MENU")
        _hr()
        print("  1. Run Full Pipeline (option 2 & option 3)")
        print("  2. Run Primary QUBO Pipeline (Generate instances → Greedy → SA → Gurobi → ECHO-SA)")
        print("  3. Run Benchmark QUBO Families (Max-Cut, Portfolio, Spectral_dense) → SA → ECHO-SA")
        print("  Q. Quit")
        _hr()

        ch = _ask("  Select option [1 / 2 / 3 / Q]:", ["1", "2", "3", "q"])

        if ch == "1":
            _run_option1()
        elif ch == "2":
            _run_option2()
        elif ch == "3":
            _run_option3()
        else:
            print("\n  Goodbye.")
            break


if __name__ == "__main__":
    main()