# src/run_experiments.py
"""
Run full experiment sweep (single-run, hybrid-design, EJOR-ready).

- Hybrid microdata -> QUBO:
  - generate_data.py creates members/claims/features and baseline artifacts per seed
  - extract_parameters.py derives c_vector and sigma_matrix once per seed at N_master
  - build_qubo.py truncates to requested N for each run and builds Q + index_map

- Solvers:
  - solve_classical.py: greedy + simulated annealing (SA)
  - solve_qubo_gurobi_exact.py: exact/MIP baseline and optional extended exactness block

Outputs:
  - results/results_master.csv
  - results/run_manifest.json
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

# -----------------------------
# Utilities
# -----------------------------
def run_cmd(cmd: list[str], cwd: Path, *, allow_fail: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        print("CMD FAILED:", " ".join(map(str, cmd)))
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)
        if not allow_fail:
            raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return p

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None

def safe_bool01(x) -> Optional[int]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, np.integer)):
        return 1 if int(x) != 0 else 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return 1
        if s in {"false", "f", "0", "no", "n"}:
            return 0
    return None

def safe_get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_git_hash(project_root: Path) -> Optional[str]:
    """
    Quiet git hash retrieval. Returns None if:
      - git isn't installed
      - folder isn't a git repo
      - any other error
    """
    try:
        p = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        return s if s else None
    except Exception:
        return None

# -----------------------------
# Seed directory checks
# -----------------------------
def seed_dir_is_sufficient(data_dir: Path, N_required: int) -> bool:
    """
    Verifies seed folder contains at least:
      - c_vector.csv with >= N_required rows in expected_cost
      - sigma_matrix.npz with sigma >= N_required x N_required
      - regulatory_set.csv present
    """
    try:
        c_path = data_dir / "c_vector.csv"
        s_path = data_dir / "sigma_matrix.npz"
        r_path = data_dir / "regulatory_set.csv"
        if not c_path.exists() or not s_path.exists() or not r_path.exists():
            return False

        c_df = pd.read_csv(c_path)
        if "expected_cost" not in c_df.columns:
            return False
        if len(c_df["expected_cost"]) < N_required:
            return False

        Sigma = np.load(s_path)["sigma"]
        if Sigma.shape[0] < N_required or Sigma.shape[1] < N_required:
            return False

        _ = pd.read_csv(r_path)
        return True
    except Exception:
        return False

def ensure_seed_data_dir(project_root: Path, seed: int, N_master: int, *, auto_generate: bool) -> Path:
    """
    Ensures data/seeds/seed_<seed>/ exists and has sufficient artifacts for N_master.

    IMPORTANT:
    - Generate ONCE per seed at N_master (max N used by that seed regime).
    - Never regenerate later for smaller N values.
    """
    data_dir = project_root / "data" / "seeds" / f"seed_{seed}"

    if data_dir.exists() and seed_dir_is_sufficient(data_dir, N_master):
        return data_dir

    if not auto_generate:
        msg = (
            f"Missing/insufficient data directory: {data_dir}\n"
            f"Need at least N={N_master} in c_vector.csv and sigma_matrix.npz, plus regulatory_set.csv.\n"
            f"Either generate it first, or re-run with auto generation enabled."
        )
        raise FileNotFoundError(msg)

    py = sys.executable
    gen_script = project_root / "src" / "generate_data.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"Missing generator script: {gen_script}")

    print(f"[seedgen] Generating master seed dir for seed={seed}, N_master={N_master} -> {data_dir}")
    cmd = [py, str(gen_script), "--seed", str(seed), "--N", str(N_master), "--overwrite"]
    run_cmd(cmd, project_root)

    if not data_dir.exists():
        raise RuntimeError(f"generate_data.py ran but did not create: {data_dir}")
    if not seed_dir_is_sufficient(data_dir, N_master):
        raise RuntimeError(f"Seed dir still insufficient after generation: {data_dir} need N={N_master}")

    return data_dir

def ensure_hybrid_extraction(project_root: Path, seed: int, data_dir: Path, N_master: int) -> None:
    """
    Runs extract_parameters.py ONCE per seed per sweep to derive QUBO parameters from microdata.
    """
    py = sys.executable
    extract_script = project_root / "src" / "extract_parameters.py"
    if not extract_script.exists():
        raise FileNotFoundError(f"Missing extractor script: {extract_script}")

    run_cmd([py, "src/extract_parameters.py", "--seed", str(seed), "--overwrite"], project_root)

    if not seed_dir_is_sufficient(data_dir, N_master):
        raise RuntimeError(f"Seed dir insufficient after extraction: {data_dir} need N={N_master}")

# -----------------------------
# Solution parsing helpers
# -----------------------------
def _solver_block(sol: dict, which: str) -> dict:
    blk = safe_get(sol, f"solvers.{which}", default=None)
    if isinstance(blk, dict):
        return blk
    raise KeyError(f"Missing solvers.{which} block. Re-run solve_classical.py after updating it.")

def _sfloat(blk: dict, path: str) -> Optional[float]:
    return safe_float(safe_get(blk, path, default=None))

def _sint(blk: dict, path: str) -> Optional[int]:
    return safe_int(safe_get(blk, path, default=None))

def _sbool01(blk: dict, path: str) -> Optional[int]:
    return safe_bool01(safe_get(blk, path, default=None))

def infer_D_affordability(idx: dict) -> Optional[float]:
    # 1) prefer explicit D from build_qubo (new schema)
    try:
        if idx.get("D_affordability", None) is not None:
            return float(abs(float(idx["D_affordability"])))
    except Exception:
        pass

    diag = idx.get("diagnostics", {}) or {}
    try:
        if diag.get("D_affordability", None) is not None:
            return float(abs(float(diag["D_affordability"])))
    except Exception:
        pass

    # 2) fallback: multiplier * A_base (decoupled)
    mult = idx.get("disallow_penalty_multiplier", None)

    # IMPORTANT: after decoupling, D scales with A_base, not A_onehot
    A_base = idx.get("A_base", None)
    A_onehot = idx.get("A_onehot", None)

    try:
        if mult is not None:
            if A_base is not None:
                return float(abs(float(mult)) * abs(float(A_base)))
            if A_onehot is not None:
                # last-resort for legacy index_map files
                return float(abs(float(mult)) * abs(float(A_onehot)))
    except Exception:
        pass

    return None

def _extract_gurobi(g: dict) -> dict:
    gb = safe_get(g, "solvers.gurobi", default=None)
    if not isinstance(gb, dict):
        gb = {}

    energy_total = safe_float(g.get("energy_total")) or safe_float(safe_get(gb, "energy_total"))
    raw_objective = safe_float(g.get("raw_objective")) or safe_float(safe_get(gb, "raw_objective"))

    penalty_energy = g.get("penalty_energy") if isinstance(g.get("penalty_energy"), dict) else (safe_get(gb, "penalty_energy", {}) or {})
    violations = g.get("violations") if isinstance(g.get("violations"), dict) else (safe_get(gb, "violations", {}) or {})
    checks = g.get("constraint_checks") if isinstance(g.get("constraint_checks"), dict) else (safe_get(gb, "constraint_checks", {}) or {})
    decoded = g.get("decoded") if isinstance(g.get("decoded"), dict) else (safe_get(gb, "decoded", {}) or {})
    status = g.get("status") if isinstance(g.get("status"), dict) else (safe_get(gb, "status", {}) or {})
    qterms = g.get("qubo_terms_signed") if isinstance(g.get("qubo_terms_signed"), dict) else (safe_get(gb, "qubo_terms_signed", {}) or {})

    return {
        "energy_total": energy_total,
        "raw_objective": raw_objective,

        "penE_onehot": safe_float(penalty_energy.get("penE_onehot")),
        "penE_reg": safe_float(penalty_energy.get("penE_reg")),
        "penE_afford": safe_float(penalty_energy.get("penE_afford")),
        "penE_total": safe_float(penalty_energy.get("penE_total")),

        "viol_onehot_y": safe_float(violations.get("onehot_y")),
        "viol_onehot_z": safe_float(violations.get("onehot_z")),
        "viol_reg_shortfall": safe_float(violations.get("reg_shortfall")),
        "viol_afford": safe_float(violations.get("affordability_indicator")),

        "chosen_premium_band": safe_int(decoded.get("chosen_premium_band")),
        "chosen_deductible_band": safe_int(decoded.get("chosen_deductible_band")),
        "selected_features_count": safe_int(decoded.get("selected_features_count")),
        "selected_reg_features_count": safe_int(decoded.get("selected_reg_features_count")),

        "onehot_ok": safe_bool01(checks.get("onehot_ok")),
        "reg_satisfied": safe_bool01(checks.get("reg_satisfied")),
        "affordability_satisfied": safe_bool01(checks.get("affordability_satisfied")),
        "is_feasible": safe_bool01(checks.get("is_feasible")),

        "status_code": safe_int(status.get("code")),
        "status_name": str(status.get("name") or ""),

        "mip_gap": safe_float(g.get("mip_gap") if g.get("mip_gap") is not None else safe_get(gb, "mip_gap")),
        "runtime_sec": safe_float(g.get("runtime_sec") if g.get("runtime_sec") is not None else safe_get(gb, "runtime_sec")),

        "qterm_total_signed": safe_float(qterms.get("qterm_total_signed")),
    }

# -----------------------------
# Sensitivity plan builder
# -----------------------------
def parse_int_list(csv_str: str) -> list[int]:
    out: list[int] = []
    for t in (csv_str or "").split(","):
        s = t.strip()
        if not s:
            continue
        out.append(int(s))
    return out

def parse_float_list(csv_str: str) -> list[float]:
    out: list[float] = []
    for t in (csv_str or "").split(","):
        s = t.strip()
        if not s:
            continue
        out.append(float(s))
    return out

def build_sensitivity_cases(
    include: bool,
    sens_factors: list[float],
    include_params: list[str],
) -> list[dict[str, Any]]:
    """
    Returns list of cases. Each case:
      - sweep_type: "baseline" or "sensitivity"
      - perturb_param: parameter name or ""
      - perturb_factor: float
      - overrides: dict applied to scenario cfg (lightweight)
    """
    cases: list[dict[str, Any]] = [
        {"sweep_type": "baseline", "perturb_param": "", "perturb_factor": 1.0, "overrides": {}}
    ]
    if not include:
        return cases

    allowed = {"A_onehot", "B_reg", "affordability_multiplier", "lambda_risk"}

    for p in include_params:
        if p not in allowed:
            raise ValueError(f"Invalid sensitivity param '{p}'. Allowed: {sorted(allowed)}")

    for param in include_params:
        for f in sens_factors:
            f = float(f)
            if abs(f - 1.0) < 1e-12:
                continue

            overrides: dict[str, Any] = {}
            if param == "A_onehot":
                overrides = {"penalties": {"onehot_multiplier": f}}
            elif param == "B_reg":
                overrides = {"penalties": {"reg_multiplier": f}}
            elif param == "affordability_multiplier":
                # EJOR DOCUMENTATION: This multiplies disallow_penalty_multiplier, which then
                # multiplies A_base (not A_onehot) in build_qubo.py. This ensures affordability
                # penalty perturbations are independent of onehot_multiplier sensitivity sweeps.
                overrides = {"affordability": {"disallow_penalty_multiplier_factor": f}}
            elif param == "lambda_risk":
                overrides = {"risk": {"lambda_risk_factor": f}}

            cases.append(
                {
                    "sweep_type": "sensitivity",
                    "perturb_param": param,
                    "perturb_factor": f,
                    "overrides": overrides,
                }
            )

    return cases

def apply_overrides_to_cfg(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Applies overrides for sensitivity in a minimally invasive way.
    We do NOT change schema; we only touch a few fields.

    Special handling:
      - affordability.disallow_penalty_multiplier_factor multiplies existing value
      - risk.lambda_risk_factor multiplies existing value
    """
    if not overrides:
        return cfg

    cfg = deep_copy(cfg)

    # penalties multipliers
    if "penalties" in overrides:
        cfg.setdefault("penalties", {})
        pblk = overrides["penalties"] or {}
        if "onehot_multiplier" in pblk:
            cfg["penalties"]["onehot_multiplier"] = float(pblk["onehot_multiplier"])
        if "reg_multiplier" in pblk:
            cfg["penalties"]["reg_multiplier"] = float(pblk["reg_multiplier"])

    # affordability multiplier factor
    if "affordability" in overrides:
        cfg.setdefault("affordability", {})
        ablk = overrides["affordability"] or {}
        if "disallow_penalty_multiplier_factor" in ablk:
            fac = float(ablk["disallow_penalty_multiplier_factor"])
            if "disallow_penalty_multiplier" in cfg["affordability"]:
                base = float(cfg["affordability"]["disallow_penalty_multiplier"])
                cfg["affordability"]["disallow_penalty_multiplier"] = float(base * fac)

    # risk lambda factor
    if "risk" in overrides:
        cfg.setdefault("risk", {})
        rblk = overrides["risk"] or {}
        if "lambda_risk_factor" in rblk:
            fac = float(rblk["lambda_risk_factor"])
            if "lambda_risk" in cfg["risk"]:
                base = float(cfg["risk"]["lambda_risk"])
                cfg["risk"]["lambda_risk"] = float(base * fac)

    return cfg

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run full experiment sweep (hybrid, single-run, EJOR-ready).")
    ap.add_argument("--fresh", action="store_true", help="Delete results/results_master.csv before running.")
    ap.add_argument("--gurobi_max_N", type=int, default=100, help="Run Gurobi only for N <= this value.")
    ap.add_argument("--gurobi_time_limit", type=int, default=3600, help="Gurobi time limit seconds.")
    ap.add_argument("--gurobi_mip_gap", type=float, default=0.0, help="Gurobi MIPGap (0.0 requests optimality).")
    ap.add_argument("--gurobi_quiet", action="store_true", help="Suppress Gurobi output.")
    ap.add_argument("--no_auto_generate_seed_data", action="store_true", help="Disable automatic seed data generation.")
    ap.add_argument("--no_extract_parameters", action="store_true", help="Debug only: skip extract_parameters.py")

    # EJOR Item 8 sensitivity sweep
    ap.add_argument("--include_sensitivity", action="store_true", help="Run 1-factor sensitivity sweeps in the SAME run.")
    ap.add_argument("--sens_N", type=str, default="50,100", help="Comma-separated N values for sensitivity (default 50,100).")
    ap.add_argument("--sens_seed_count", type=int, default=5, help="How many seeds per regime to use for sensitivity (default 5).")
    ap.add_argument("--sens_factors", type=str, default="0.9,1.1", help="Comma-separated perturbation factors (default 0.9,1.1).")
    ap.add_argument("--sens_params",   type=str,   default="A_onehot,B_reg,affordability_multiplier,lambda_risk",
                    help="Comma-separated sensitivity params from {A_onehot,B_reg,affordability_multiplier,lambda_risk}.")

    # Targeted extended exactness block (EJOR: show onset not artifact)
    ap.add_argument("--include_extended_exactness", action="store_true",
                    help="Run targeted Gurobi exact for selected scenarios/N/seeds (in addition to baseline gurobi_max_N).")
    ap.add_argument("--exact_scenarios", type=str, default="1,4",
                    help="Comma-separated scenario ids for extended exactness (default 1,4).")
    ap.add_argument("--exact_N", type=str, default="150,200,300",
                    help="Comma-separated N values for extended exactness (default 150,200,300).")
    ap.add_argument("--exact_seed_count", type=int, default=5,
                    help="How many seeds per regime to use for extended exactness (default 5).")
    ap.add_argument("--exact_time_limit", type=int, default=7200,
                    help="Gurobi time limit seconds for extended exactness (default 7200).")
    ap.add_argument("--exact_mip_gap", type=float, default=0.0,
                    help="Gurobi MIPGap for extended exactness (default 0.0).")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    plan_path = project_root / "config" / "experiment_plan.yaml"
    plan = load_yaml(plan_path)

    feature_sizes = [int(x) for x in plan["feature_sizes"]]
    scenarios = [int(x) for x in plan["scenarios"]]

    core_count = int(plan["seeds"]["core_count"])
    stress_count = int(plan["seeds"]["stress_count"])
    core_seed_start = int(plan["seeds"]["core_seed_start"])
    stress_seed_start = int(plan["seeds"]["stress_seed_start"])

    solver_cfg = plan.get("solver", {}) or {}

    auto_generate = not args.no_auto_generate_seed_data
    do_extract = not args.no_extract_parameters

    greedy_num_starts = int(solver_cfg.get("greedy_num_starts", solver_cfg.get("num_starts", 20)))
    sa_num_starts = int(solver_cfg.get("sa_num_starts", greedy_num_starts))
    sa_multistart = bool(solver_cfg.get("sa_multistart", True))

    # Determine per-regime master N to avoid mid-sweep regeneration.
    core_sizes = [n for n in feature_sizes if n <= 50]
    stress_sizes = [n for n in feature_sizes if n > 50]
    N_master_core = max(core_sizes) if core_sizes else 0
    N_master_stress = max(stress_sizes) if stress_sizes else 0

    # Sensitivity config
    sens_N = set(parse_int_list(args.sens_N))
    sens_factors = parse_float_list(args.sens_factors)
    sens_params = [s.strip() for s in (args.sens_params or "").split(",") if s.strip()]

    # Extended exactness config
    exact_scenarios = set(parse_int_list(args.exact_scenarios))
    exact_N = set(parse_int_list(args.exact_N))
    exact_seed_count = max(1, int(args.exact_seed_count))

    # Results folder + manifest
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    run_manifest = {
        "run_started_utc": utc_now_iso(),
        "project_root": str(project_root),
        "git_hash": get_git_hash(project_root),
        "python": sys.version,
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "plan_path": str(plan_path),
        "plan": plan,
        "cli_args": {
            "fresh": bool(args.fresh),
            "gurobi_max_N": int(args.gurobi_max_N),
            "gurobi_time_limit": int(args.gurobi_time_limit),
            "gurobi_mip_gap": float(args.gurobi_mip_gap),
            "gurobi_quiet": bool(args.gurobi_quiet),
            "auto_generate_seed_data": bool(auto_generate),
            "extract_parameters": bool(do_extract),

            "include_sensitivity": bool(args.include_sensitivity),
            "sens_N": sorted(list(sens_N)),
            "sens_seed_count": int(args.sens_seed_count),
            "sens_factors": sens_factors,
            "sens_params": sens_params,

            "include_extended_exactness": bool(args.include_extended_exactness),
            "exact_scenarios": sorted(list(exact_scenarios)),
            "exact_N": sorted(list(exact_N)),
            "exact_seed_count": int(args.exact_seed_count),
            "exact_time_limit": int(args.exact_time_limit),
            "exact_mip_gap": float(args.exact_mip_gap),
        },
        "design": {
            "mode": "hybrid_microdata_to_qubo",
            "seed_master_generation": {
                "N_master_core": int(N_master_core),
                "N_master_stress": int(N_master_stress),
                "note": "Each seed is generated once at its regime's N_master. Smaller N uses truncation at build_qubo stage."
            },
        },
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    out_csv = results_dir / "results_master.csv"
    if args.fresh and out_csv.exists():
        out_csv.unlink()

    # CRITICAL FIX: Define header as ordered list for dict-based CSV safety
    header = [
        "sweep_type", "perturb_param", "perturb_factor",

        "scenario_id", "scenario_name",
        "seed", "N", "M", "K", "L_reg_slack", "n_total",

        # EJOR H1: QUBO structural stability metrics
        "Q_condition_number", "Q_eigenvalue_min", "Q_eigenvalue_max", "Q_eigenvalue_count_negative",

        "energy_greedy",
        "objective_raw_greedy",
        "penE_onehot_greedy", "penE_reg_greedy", "penE_afford_greedy", "penE_total_greedy",
        "viol_onehot_y_greedy", "viol_onehot_z_greedy", "viol_reg_shortfall_greedy", "viol_afford_greedy",
        "chosen_premium_band_greedy", "chosen_deductible_band_greedy",
        "selected_features_count_greedy", "selected_reg_features_count_greedy",
        "onehot_ok_greedy", "reg_satisfied_greedy", "affordability_satisfied_greedy", "is_feasible_greedy",

        "energy_sa_best",
        "objective_raw_sa_best",
        "penE_onehot_sa", "penE_reg_sa", "penE_afford_sa", "penE_total_sa",
        "viol_onehot_y_sa", "viol_onehot_z_sa", "viol_reg_shortfall_sa", "viol_afford_sa",
        "chosen_premium_band_sa", "chosen_deductible_band_sa",
        "selected_features_count_sa", "selected_reg_features_count_sa",
        "onehot_ok_sa", "reg_satisfied_sa", "affordability_satisfied_sa", "is_feasible_sa",

        # Classical solver runtimes (EJOR H2 critical)
        "greedy_runtime_sec", "sa_runtime_sec", "sa_best_found_at_step", "sa_convergence_fraction", "total_classical_runtime_sec",

        "gap_greedy_to_sa_energy",
        "gap_greedy_to_sa_raw",

        "R_size", "Rmin",
        "lambda_risk",
        "affordability_enabled",
        "use_tight_regulation",
        "scale_factor",
        "A_onehot",
        "B_reg",
        "D_afford",

        "qterm_total_greedy_signed",
        "qterm_total_sa_signed",
        "qterm_total_gurobi_signed",

        "gurobi_ran",
        "energy_gurobi",
        "objective_raw_gurobi",
        "penE_onehot_gurobi", "penE_reg_gurobi", "penE_afford_gurobi", "penE_total_gurobi",
        "viol_onehot_y_gurobi", "viol_onehot_z_gurobi", "viol_reg_shortfall_gurobi", "viol_afford_gurobi",
        "chosen_premium_band_gurobi", "chosen_deductible_band_gurobi",
        "selected_features_count_gurobi", "selected_reg_features_count_gurobi",
        "onehot_ok_gurobi", "reg_satisfied_gurobi", "affordability_satisfied_gurobi", "is_feasible_gurobi",
        "gurobi_status_code", "gurobi_status_name",
        "gurobi_mip_gap", "gurobi_runtime_sec",

        "gap_sa_to_gurobi_energy",
        "gap_sa_to_gurobi_raw",

        "gurobi_mode",
        "gurobi_time_limit_used",
    ]

    if not out_csv.exists():
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    py = sys.executable
    prepared_seeds: set[int] = set()

    sens_cases = build_sensitivity_cases(
        include=bool(args.include_sensitivity),
        sens_factors=sens_factors if sens_factors else [0.9, 1.1],
        include_params=sens_params if sens_params else ["A_onehot", "B_reg", "affordability_multiplier", "lambda_risk"],
    )

    for scen_id in scenarios:
        cfg_path = project_root / "config" / f"config_qubo_S{scen_id}.yaml"
        cfg_original = load_yaml(cfg_path)

        try:
            for N in feature_sizes:
                # Seed regime selection
                seeds_full = (
                    [core_seed_start + i for i in range(core_count)]
                    if N <= 50
                    else [stress_seed_start + i for i in range(stress_count)]
                )

                # For sensitivity: use only first sens_seed_count seeds for that regime
                if N in sens_N:
                    seeds_sens = seeds_full[: max(1, int(args.sens_seed_count))]
                else:
                    seeds_sens = []

                # For extended exactness: use only first exact_seed_count seeds for that regime
                seeds_exact = seeds_full[: exact_seed_count]

                # Regime-specific master N
                N_master = N_master_core if N <= 50 else N_master_stress
                if N_master <= 0:
                    raise ValueError("Invalid N_master computed. Check experiment_plan.yaml feature_sizes.")

                for seed in seeds_full:
                    # Prepare seed ONCE per sweep at regime master N
                    if seed not in prepared_seeds:
                        data_dir = ensure_seed_data_dir(
                            project_root,
                            seed=int(seed),
                            N_master=int(N_master),
                            auto_generate=auto_generate,
                        )
                        if do_extract:
                            ensure_hybrid_extraction(project_root, seed=int(seed), data_dir=data_dir, N_master=int(N_master))
                        prepared_seeds.add(int(seed))
                    else:
                        data_dir = project_root / "data" / "seeds" / f"seed_{seed}"

                    # Decide which cases to run for this (seed,N)
                    cases_here = [sens_cases[0]]  # baseline always

                    if args.include_sensitivity and (N in sens_N) and (seed in seeds_sens):
                        cases_here.extend(sens_cases[1:])

                    for case in cases_here:
                        sweep_type = case["sweep_type"]
                        perturb_param = case["perturb_param"]
                        perturb_factor = float(case["perturb_factor"])
                        overrides = case["overrides"]

                        cfg = deep_copy(cfg_original)
                        cfg.setdefault("features", {})
                        cfg.setdefault("data", {})
                        cfg.setdefault("solver", {})
                        cfg.setdefault("penalties", {})
                        cfg.setdefault("risk", {})
                        cfg.setdefault("affordability", {})

                        cfg["features"]["N"] = int(N)
                        cfg["data"]["seed"] = int(seed)

                        cfg["solver"]["sa_steps"] = int(solver_cfg.get("sa_steps", 40000))
                        cfg["solver"]["sa_T0"] = float(solver_cfg.get("sa_T0", 5.0))
                        cfg["solver"]["sa_Tend"] = float(solver_cfg.get("sa_Tend", 0.01))
                        cfg["solver"]["random_seed"] = int(seed)
                        cfg["solver"]["sa_multistart"] = bool(sa_multistart)
                        cfg["solver"]["sa_num_starts"] = int(sa_num_starts)
                        cfg["solver"]["greedy_num_starts"] = int(greedy_num_starts)

                        cfg["penalties"].setdefault("onehot_multiplier", 1.0)
                        cfg["penalties"].setdefault("reg_multiplier", 1.0)

                        cfg = apply_overrides_to_cfg(cfg, overrides)

                        # Skip meaningless sweeps
                        if sweep_type == "sensitivity" and perturb_param == "affordability_multiplier":
                            if not bool(cfg.get("affordability", {}).get("enabled", False)):
                                continue

                        if sweep_type == "sensitivity" and perturb_param == "lambda_risk":
                            if abs(float(cfg.get("risk", {}).get("lambda_risk", 0.0))) < 1e-15:
                                continue

                        save_yaml(cfg_path, cfg)
                        scenario_name = str(cfg["scenario"]["name"])

                        # 1) Build QUBO
                        run_cmd([py, "src/build_qubo.py", "--scenario", str(scen_id), "--overwrite"], project_root)

                        idx_path = data_dir / f"index_map_{scenario_name}_N{N}.json"
                        q_path = data_dir / f"qubo_Q_{scenario_name}_N{N}.npz"
                        if not idx_path.exists():
                            raise FileNotFoundError(f"Missing index map after build: {idx_path}")
                        if not q_path.exists():
                            raise FileNotFoundError(f"Missing QUBO after build: {q_path}")
                        idx = read_json(idx_path)

                        # 2) Solve classical
                        run_cmd([py, "src/solve_classical.py", "--scenario", str(scen_id)], project_root)

                        sol_path = results_dir / f"solution_{scenario_name}_seed{seed}_N{N}.json"
                        if not sol_path.exists():
                            raise FileNotFoundError(f"Missing solution output: {sol_path}")
                        sol = read_json(sol_path)

                        gblk = _solver_block(sol, "greedy")
                        sblk = _solver_block(sol, "sa")

                        energy_greedy = safe_float(sol.get("energy_greedy") or safe_get(gblk, "energy_total"))
                        energy_sa = safe_float(sol.get("energy_sa_best") or safe_get(sblk, "energy_total"))

                        raw_greedy = safe_float(sol.get("raw_objective_greedy") or safe_get(gblk, "raw_objective"))
                        raw_sa = safe_float(sol.get("raw_objective_sa_best") or safe_get(sblk, "raw_objective"))

                        penE_onehot_g = _sfloat(gblk, "penalty_energy.penE_onehot")
                        penE_reg_g = _sfloat(gblk, "penalty_energy.penE_reg")
                        penE_aff_g = _sfloat(gblk, "penalty_energy.penE_afford")
                        penE_total_g = _sfloat(gblk, "penalty_energy.penE_total")

                        penE_onehot_s = _sfloat(sblk, "penalty_energy.penE_onehot")
                        penE_reg_s = _sfloat(sblk, "penalty_energy.penE_reg")
                        penE_aff_s = _sfloat(sblk, "penalty_energy.penE_afford")
                        penE_total_s = _sfloat(sblk, "penalty_energy.penE_total")

                        viol_onehot_y_g = _sfloat(gblk, "violations.onehot_y")
                        viol_onehot_z_g = _sfloat(gblk, "violations.onehot_z")
                        viol_reg_g = _sfloat(gblk, "violations.reg_shortfall")
                        viol_aff_g = _sfloat(gblk, "violations.affordability_indicator")

                        viol_onehot_y_s = _sfloat(sblk, "violations.onehot_y")
                        viol_onehot_z_s = _sfloat(sblk, "violations.onehot_z")
                        viol_reg_s = _sfloat(sblk, "violations.reg_shortfall")
                        viol_aff_s = _sfloat(sblk, "violations.affordability_indicator")

                        chosen_prem_g = _sint(gblk, "decoded.chosen_premium_band")
                        chosen_ded_g = _sint(gblk, "decoded.chosen_deductible_band")
                        feat_cnt_g = _sint(gblk, "decoded.selected_features_count")
                        reg_cnt_g = _sint(gblk, "decoded.selected_reg_features_count")

                        chosen_prem_s = _sint(sblk, "decoded.chosen_premium_band")
                        chosen_ded_s = _sint(sblk, "decoded.chosen_deductible_band")
                        feat_cnt_s = _sint(sblk, "decoded.selected_features_count")
                        reg_cnt_s = _sint(sblk, "decoded.selected_reg_features_count")

                        onehot_ok_g = _sbool01(gblk, "constraint_checks.onehot_ok")
                        reg_sat_g = _sbool01(gblk, "constraint_checks.reg_satisfied")
                        aff_sat_g = _sbool01(gblk, "constraint_checks.affordability_satisfied")
                        feas_g = _sbool01(gblk, "constraint_checks.is_feasible")

                        onehot_ok_s = _sbool01(sblk, "constraint_checks.onehot_ok")
                        reg_sat_s = _sbool01(sblk, "constraint_checks.reg_satisfied")
                        aff_sat_s = _sbool01(sblk, "constraint_checks.affordability_satisfied")
                        feas_s = _sbool01(sblk, "constraint_checks.is_feasible")

                        # Runtime tracking (EJOR H2 critical)
                        greedy_runtime_sec = safe_float(sol.get("greedy_runtime_sec"))
                        sa_runtime_sec = safe_float(sol.get("sa_runtime_sec"))
                        total_classical_runtime_sec = safe_float(sol.get("total_classical_runtime_sec"))
                        sa_best_found_at_step = safe_int(safe_get(sblk, "params.convergence.best_found_at_step"))
                        sa_convergence_fraction = safe_float(safe_get(sblk, "params.convergence.convergence_fraction"))

                        gap_greedy_to_sa_energy = (energy_greedy - energy_sa) if (energy_greedy is not None and energy_sa is not None) else None
                        gap_greedy_to_sa_raw = (raw_greedy - raw_sa) if (raw_greedy is not None and raw_sa is not None) else None

                        R_size = safe_int(idx.get("R_size"))
                        Rmin = safe_int(idx.get("Rmin"))
                        lambda_risk = safe_float(idx.get("lambda_risk"))
                        affordability_enabled = safe_bool01(idx.get("affordability_enabled"))
                        use_tight_regulation = safe_bool01(idx.get("use_tight_regulation"))
                        scale_factor = safe_float(idx.get("scale_factor"))
                        A_onehot = safe_float(idx.get("A_onehot"))
                        B_reg = safe_float(idx.get("B_reg"))

                        disallow_bands = idx.get("disallow_premium_bands") or idx.get("disallow_bands") or []
                        if affordability_enabled == 1 and len(disallow_bands) > 0:
                            D_afford = infer_D_affordability(idx)
                        else:
                            D_afford = 0.0

                        qterm_total_g_signed = safe_float(safe_get(gblk, "qubo_terms_signed.qterm_total_signed"))
                        qterm_total_s_signed = safe_float(safe_get(sblk, "qubo_terms_signed.qterm_total_signed"))

                        # 3) Optional: Gurobi exact benchmark (baseline and/or extended exactness)
                        gurobi_ran = 0
                        energy_gurobi = raw_objective_gurobi = None
                        penE_onehot_gurobi = penE_reg_gurobi = penE_aff_gurobi = penE_total_gurobi = None
                        viol_onehot_y_gurobi = viol_onehot_z_gurobi = viol_reg_gurobi = viol_aff_gurobi = None
                        chosen_premium_band_gurobi = chosen_deductible_band_gurobi = None
                        selected_features_count_gurobi = selected_reg_features_count_gurobi = None
                        onehot_ok_gurobi = reg_satisfied_gurobi = affordability_satisfied_gurobi = is_feasible_gurobi = None
                        gurobi_status_code = None
                        gurobi_status_name = ""
                        gurobi_mip_gap = None
                        gurobi_runtime_sec = None
                        gap_sa_to_gurobi_energy = None
                        gap_sa_to_gurobi_raw = None
                        qterm_total_gurobi_signed = None

                        gurobi_mode = ""
                        gurobi_time_limit_used = None

                        # CRITICAL FIX: Extended exactness REPLACES baseline to avoid double-runs
                        run_gurobi_extended = False
                        if args.include_extended_exactness:
                            # extended exactness only on baseline sweep rows (avoid exploding runs)
                            if sweep_type == "baseline":
                                if (int(scen_id) in exact_scenarios) and (int(N) in exact_N) and (int(seed) in seeds_exact):
                                    run_gurobi_extended = True

                        # Baseline only runs if NOT replaced by extended exactness
                        run_gurobi_baseline = (int(N) <= int(args.gurobi_max_N)) and not run_gurobi_extended

                        if run_gurobi_extended or run_gurobi_baseline:
                            gurobi_ran = 1

                            if run_gurobi_extended:
                                gurobi_mode = "extended"
                                time_limit = int(args.exact_time_limit)
                                mip_gap = float(args.exact_mip_gap)
                            else:
                                gurobi_mode = "baseline"
                                time_limit = int(args.gurobi_time_limit)
                                mip_gap = float(args.gurobi_mip_gap)

                            gurobi_time_limit_used = time_limit

                            cmd = [
                                py, "src/solve_qubo_gurobi_exact.py",
                                "--seed", str(seed),
                                "--scenario_name", scenario_name,
                                "--N", str(N),
                                "--time_limit", str(time_limit),
                                "--mip_gap", str(mip_gap),
                            ]
                            if args.gurobi_quiet:
                                cmd.append("--quiet")

                            run_cmd(cmd, project_root)

                            gjson = data_dir / f"gurobi_exact_{scenario_name}_N{N}.json"
                            if not gjson.exists():
                                raise FileNotFoundError(f"Gurobi JSON not found after run: {gjson}")
                            g = read_json(gjson)
                            gf = _extract_gurobi(g)

                            energy_gurobi = gf["energy_total"]
                            raw_objective_gurobi = gf["raw_objective"]

                            penE_onehot_gurobi = gf["penE_onehot"]
                            penE_reg_gurobi = gf["penE_reg"]
                            penE_aff_gurobi = gf["penE_afford"]
                            penE_total_gurobi = gf["penE_total"]

                            viol_onehot_y_gurobi = gf["viol_onehot_y"]
                            viol_onehot_z_gurobi = gf["viol_onehot_z"]
                            viol_reg_gurobi = gf["viol_reg_shortfall"]
                            viol_aff_gurobi = gf["viol_afford"]

                            chosen_premium_band_gurobi = gf["chosen_premium_band"]
                            chosen_deductible_band_gurobi = gf["chosen_deductible_band"]
                            selected_features_count_gurobi = gf["selected_features_count"]
                            selected_reg_features_count_gurobi = gf["selected_reg_features_count"]

                            onehot_ok_gurobi = gf["onehot_ok"]
                            reg_satisfied_gurobi = gf["reg_satisfied"]
                            affordability_satisfied_gurobi = gf["affordability_satisfied"]
                            is_feasible_gurobi = gf["is_feasible"]

                            gurobi_status_code = gf["status_code"]
                            gurobi_status_name = gf["status_name"]
                            gurobi_mip_gap = gf["mip_gap"]
                            gurobi_runtime_sec = gf["runtime_sec"]

                            if energy_sa is not None and energy_gurobi is not None:
                                gap_sa_to_gurobi_energy = energy_sa - energy_gurobi
                            if raw_sa is not None and raw_objective_gurobi is not None:
                                gap_sa_to_gurobi_raw = raw_sa - raw_objective_gurobi

                            qterm_total_gurobi_signed = gf["qterm_total_signed"]

                        # CRITICAL FIX: Dict-based row construction for CSV safety
                        row_dict = {
                            "sweep_type": sweep_type,
                            "perturb_param": perturb_param,
                            "perturb_factor": perturb_factor,

                            "scenario_id": int(scen_id),
                            "scenario_name": scenario_name,
                            "seed": int(seed),
                            "N": int(N),
                            "M": safe_int(idx.get("M")),
                            "K": safe_int(idx.get("K")),
                            "L_reg_slack": safe_int(idx.get("L_reg_slack", 0)),
                            "n_total": safe_int(idx.get("n_total")),

                            # EJOR H1: QUBO structural stability metrics
                            "Q_condition_number": safe_float(safe_get(idx, "diagnostics.Q_condition_number")),
                            "Q_eigenvalue_min": safe_float(safe_get(idx, "diagnostics.Q_eigenvalue_min")),
                            "Q_eigenvalue_max": safe_float(safe_get(idx, "diagnostics.Q_eigenvalue_max")),
                            "Q_eigenvalue_count_negative": safe_int(safe_get(idx, "diagnostics.Q_eigenvalue_count_negative")),

                            "energy_greedy": energy_greedy,
                            "objective_raw_greedy": raw_greedy,
                            "penE_onehot_greedy": penE_onehot_g,
                            "penE_reg_greedy": penE_reg_g,
                            "penE_afford_greedy": penE_aff_g,
                            "penE_total_greedy": penE_total_g,
                            "viol_onehot_y_greedy": viol_onehot_y_g,
                            "viol_onehot_z_greedy": viol_onehot_z_g,
                            "viol_reg_shortfall_greedy": viol_reg_g,
                            "viol_afford_greedy": viol_aff_g,
                            "chosen_premium_band_greedy": chosen_prem_g,
                            "chosen_deductible_band_greedy": chosen_ded_g,
                            "selected_features_count_greedy": feat_cnt_g,
                            "selected_reg_features_count_greedy": reg_cnt_g,
                            "onehot_ok_greedy": onehot_ok_g,
                            "reg_satisfied_greedy": reg_sat_g,
                            "affordability_satisfied_greedy": aff_sat_g,
                            "is_feasible_greedy": feas_g,

                            "energy_sa_best": energy_sa,
                            "objective_raw_sa_best": raw_sa,
                            "penE_onehot_sa": penE_onehot_s,
                            "penE_reg_sa": penE_reg_s,
                            "penE_afford_sa": penE_aff_s,
                            "penE_total_sa": penE_total_s,
                            "viol_onehot_y_sa": viol_onehot_y_s,
                            "viol_onehot_z_sa": viol_onehot_z_s,
                            "viol_reg_shortfall_sa": viol_reg_s,
                            "viol_afford_sa": viol_aff_s,
                            "chosen_premium_band_sa": chosen_prem_s,
                            "chosen_deductible_band_sa": chosen_ded_s,
                            "selected_features_count_sa": feat_cnt_s,
                            "selected_reg_features_count_sa": reg_cnt_s,
                            "onehot_ok_sa": onehot_ok_s,
                            "reg_satisfied_sa": reg_sat_s,
                            "affordability_satisfied_sa": aff_sat_s,
                            "is_feasible_sa": feas_s,

                            # Classical solver runtimes (EJOR H2 critical)
                            "greedy_runtime_sec": greedy_runtime_sec,
                            "sa_runtime_sec": sa_runtime_sec,
                            "sa_best_found_at_step": sa_best_found_at_step,
                            "sa_convergence_fraction": sa_convergence_fraction,
                            "total_classical_runtime_sec": total_classical_runtime_sec,

                            "gap_greedy_to_sa_energy": gap_greedy_to_sa_energy,
                            "gap_greedy_to_sa_raw": gap_greedy_to_sa_raw,

                            "R_size": R_size,
                            "Rmin": Rmin,
                            "lambda_risk": lambda_risk,
                            "affordability_enabled": affordability_enabled,
                            "use_tight_regulation": use_tight_regulation,
                            "scale_factor": scale_factor,
                            "A_onehot": A_onehot,
                            "B_reg": B_reg,
                            "D_afford": D_afford,

                            "qterm_total_greedy_signed": qterm_total_g_signed,
                            "qterm_total_sa_signed": qterm_total_s_signed,
                            "qterm_total_gurobi_signed": qterm_total_gurobi_signed,

                            "gurobi_ran": gurobi_ran,
                            "energy_gurobi": energy_gurobi,
                            "objective_raw_gurobi": raw_objective_gurobi,
                            "penE_onehot_gurobi": penE_onehot_gurobi,
                            "penE_reg_gurobi": penE_reg_gurobi,
                            "penE_afford_gurobi": penE_aff_gurobi,
                            "penE_total_gurobi": penE_total_gurobi,
                            "viol_onehot_y_gurobi": viol_onehot_y_gurobi,
                            "viol_onehot_z_gurobi": viol_onehot_z_gurobi,
                            "viol_reg_shortfall_gurobi": viol_reg_gurobi,
                            "viol_afford_gurobi": viol_aff_gurobi,
                            "chosen_premium_band_gurobi": chosen_premium_band_gurobi,
                            "chosen_deductible_band_gurobi": chosen_deductible_band_gurobi,
                            "selected_features_count_gurobi": selected_features_count_gurobi,
                            "selected_reg_features_count_gurobi": selected_reg_features_count_gurobi,
                            "onehot_ok_gurobi": onehot_ok_gurobi,
                            "reg_satisfied_gurobi": reg_satisfied_gurobi,
                            "affordability_satisfied_gurobi": affordability_satisfied_gurobi,
                            "is_feasible_gurobi": is_feasible_gurobi,
                            "gurobi_status_code": gurobi_status_code,
                            "gurobi_status_name": gurobi_status_name,
                            "gurobi_mip_gap": gurobi_mip_gap,
                            "gurobi_runtime_sec": gurobi_runtime_sec,

                            "gap_sa_to_gurobi_energy": gap_sa_to_gurobi_energy,
                            "gap_sa_to_gurobi_raw": gap_sa_to_gurobi_raw,

                            "gurobi_mode": gurobi_mode,
                            "gurobi_time_limit_used": gurobi_time_limit_used,
                        }

                        # Convert dict to row using header ordering (CSV safety)
                        row = [row_dict[col] for col in header]

                        with open(out_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)

                        tag = ""
                        if gurobi_ran:
                            tag = f" +Gurobi({gurobi_mode})"
                        extra = " +Extract" if do_extract else ""
                        sens_tag = f" +Sens({perturb_param}*{perturb_factor})" if sweep_type == "sensitivity" else ""
                        print(f"DONE scenario={scenario_name} N={N} seed={seed}{tag}{extra}{sens_tag}")

        finally:
            # Restore original config file (clean reproducibility)
            save_yaml(cfg_path, cfg_original)

    # Finalize manifest with end time
    try:
        mpath = results_dir / "run_manifest.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["run_finished_utc"] = utc_now_iso()
        mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"\nSaved consolidated results: {out_csv}")
    print(f"Saved run manifest: {results_dir / 'run_manifest.json'}")

if __name__ == "__main__":
    main()