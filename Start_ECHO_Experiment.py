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
_ECHO_SCRIPT     = ROOT / "src_code" / "runners" / "run_echo_full.py"

# New benchmark runner
_BENCH_SCRIPT    = ROOT / "src_code" / "runners" / "run_benchmarks.py"

# Key result files produced by Option 2
_BASELINE_CSV    = ROOT / "results" / "baseline_full_results.csv"
_ECHO_CSV        = ROOT / "results" / "echo_full_results.csv"


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

def _check_scripts_exist() -> bool:
    """Verify the three runner scripts are present."""
    ok = True
    for s in (_BASELINE_SCRIPT, _ECHO_SCRIPT, _BENCH_SCRIPT):
        if not s.exists():
            print(f"  [missing] {s.relative_to(ROOT)}")
            ok = False
    return ok


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

    if not _BENCH_SCRIPT.exists():
        print(f"  [missing] {_BENCH_SCRIPT.relative_to(ROOT)}")
        print("  Cannot run Option 3. Ensure run_benchmarks.py is present.")
        return False

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
    # Warn about missing scripts but do NOT exit — only the relevant option
    # will fail if its specific script is absent.
    for s in (_BASELINE_SCRIPT, _ECHO_SCRIPT):
        if not s.exists():
            print(f"  [warning] Missing: {s.relative_to(ROOT)}")

    while True:
        print()
        _hr()
        print("  MAIN MENU")
        _hr()
        print("  1. Run Full Pipeline (option 2 & option 3)")
        print("  2. Run Primary QUBO Pipeline (Generate instances → Greedy → SA → Gurobi → ECHO-SA)")
        print("  3. Run Benchmark QUBO Families (Max-Cut, Portfolio) → SA → ECHO-SA")
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