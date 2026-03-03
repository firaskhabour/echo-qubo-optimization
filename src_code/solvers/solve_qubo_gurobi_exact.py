# src_code/solvers/solve_qubo_gurobi_exact.py
"""
Gurobi Exact Solver (MIQP Benchmark)
=====================================
Implements the Gurobi mixed-integer quadratic program (MIQP) benchmark
described in Section 4.2, item 4 of the accompanying paper.

Gurobi is used as a certified optimality reference where it successfully
proves optimality under configured time limits. Results are used
selectively: only instances with gurobi_status_name = OPTIMAL and
gurobi_mip_gap = 0.0 are treated as optimal references in analysis.

Recorded time limits (Section 4.2, item 4):
    3,600 s (1 hour)  — baseline instances
    7,200 s (2 hours) — larger instances (N ∈ {150, 200, 300})

Gurobi proves optimality for all instances up to N = 150. For N = 200
and N = 300, optimality is not certified in the baseline output.

Solver formulation:
    min  v^T Q v   subject to  v ∈ {0,1}^n

Hard one-hot constraints for the deductible block y and premium block z
are added as linear equality constraints, making the MIQP comparable to
the greedy and SA solvers which enforce one-hot via projection after every
move.

Energy decomposition validation (Section 4.2, item 5):
    E(v) = f(v) + Σ P̃_i(v), verified to relative tolerance 10⁻⁶.

Output per instance:
    data/seeds/seed_<S>/gurobi_exact_<scenario>_N<N>.json  — full record
    data/seeds/seed_<S>/gurobi_exact_results.csv           — appended row
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Solve a saved penalized QUBO with Gurobi (exact / gap-bounded)."
    )
    p.add_argument("--seed",          type=int, required=True,
                   help="Seed number (matches data/seeds/seed_<S>).")
    p.add_argument("--N",             type=int, default=None,
                   help="Feature size N. If omitted, auto-detect from newest index_map.")
    p.add_argument("--scenario_name", type=str, required=True,
                   help="Scenario name, e.g. S1_cost_only.")
    p.add_argument("--time_limit",    type=int, default=3600,
                   help="Gurobi time limit in seconds (default 3,600; use 7,200 for N ≥ 150).")
    p.add_argument("--mip_gap",       type=float, default=0.0,
                   help="Target MIP gap (0.0 requests proven optimality).")
    p.add_argument("--quiet",         action="store_true",
                   help="Suppress Gurobi solver output.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_index_map_and_qubo(
    data_dir: Path,
    scenario_name: str,
    N_cli: Optional[int],
) -> Tuple[Path, Path, dict, np.ndarray]:
    """
    Load the index map and QUBO matrix for the requested instance.

    If N_cli is provided, loads the exact file pair for that N.
    If N_cli is None, auto-detects the most recently modified index map
    matching the scenario name and infers N from its contents.
    """
    if N_cli is not None:
        idx_path  = data_dir / f"index_map_{scenario_name}_N{N_cli}.json"
        qubo_path = data_dir / f"qubo_Q_{scenario_name}_N{N_cli}.npz"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing index map: {idx_path}")
        if not qubo_path.exists():
            raise FileNotFoundError(f"Missing QUBO file: {qubo_path}")
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        Q   = np.load(qubo_path)["Q"]
        return idx_path, qubo_path, idx, Q

    candidates = sorted(
        data_dir.glob(f"index_map_{scenario_name}_N*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No index_map_{scenario_name}_N*.json found in {data_dir}. "
            f"Run build_qubo.py first."
        )

    idx_path  = candidates[0]
    idx       = json.loads(idx_path.read_text(encoding="utf-8"))
    N_detected = int(idx.get("N", 0))
    if N_detected <= 0:
        raise ValueError(f"index_map missing valid N field: {idx_path}")

    qubo_path = data_dir / f"qubo_Q_{scenario_name}_N{N_detected}.npz"
    if not qubo_path.exists():
        raise FileNotFoundError(
            f"Found index map but missing matching QUBO: {qubo_path}"
        )

    Q = np.load(qubo_path)["Q"]
    return idx_path, qubo_path, idx, Q


# ---------------------------------------------------------------------------
# Core QUBO primitives (mirror of solve_classical.py for cross-file consistency)
# ---------------------------------------------------------------------------

def energy(Q: np.ndarray, v01: np.ndarray) -> float:
    """Evaluate QUBO energy: E(v) = v^T Q v."""
    return float(v01.astype(float) @ Q @ v01.astype(float))


def compute_raw_objective(
    x01: np.ndarray,
    z01: np.ndarray,
    c: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float,
    premium_bands: np.ndarray,
) -> float:
    """
    Compute the raw economic objective f(v) excluding penalty energies
    (Section 4.3):

        f(v) = c^T x  +  lambda_risk · x^T Σ x  −  Σ_k P_k z_k
    """
    x = x01.astype(float)
    z = z01.astype(float)
    cost_term    = float(np.dot(c, x))
    risk_term    = float(lambda_risk * (x @ Sigma @ x)) if lambda_risk != 0.0 else 0.0
    premium_term = float(np.dot(premium_bands, z))
    return float(cost_term + risk_term - premium_term)


def _pos_float(idx: dict, key: str, default: float = 0.0) -> float:
    """Extract a non-negative float from the index map with a safe default."""
    v = idx.get(key, None)
    if v is None:
        return float(default)
    try:
        return float(abs(float(v)))
    except Exception:
        return float(default)


def _infer_D_affordability(idx: dict) -> float:
    """
    Recover the affordability penalty weight D from the index map.

    Resolution order (consistent across all solvers in the pipeline):
      1. Explicit D_affordability in root index map.
      2. D_affordability in the diagnostics sub-block (legacy location).
      3. disallow_penalty_multiplier × A_base (decoupled scaling).
      4. disallow_penalty_multiplier × A_onehot (legacy fallback only).
    """
    if idx.get("D_affordability", None) is not None:
        return float(abs(float(idx["D_affordability"])))
    diag = idx.get("diagnostics", {}) or {}
    if diag.get("D_affordability", None) is not None:
        return float(abs(float(diag["D_affordability"])))
    mult   = idx.get("disallow_penalty_multiplier", None)
    A_base = idx.get("A_base", None)
    if mult is not None and A_base is not None:
        return float(abs(float(mult)) * abs(float(A_base)))
    A = idx.get("A_onehot", None)
    if mult is not None and A is not None:
        return float(abs(float(mult)) * abs(float(A)))
    return 0.0


# ---------------------------------------------------------------------------
# Constraint metrics and energy decomposition
# ---------------------------------------------------------------------------

def _compute_constraint_metrics(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    R_zero: list[int],
    Rmin: int,
    affordability_enabled: bool,
    disallow_bands: list[int],
    A: float,
    B: float,
    D: float,
) -> Dict[str, Any]:
    """
    Compute violation amounts and penalty energy contributions.

    These are the violation-based quantities used for feasibility flags
    in Section 4.3. They differ from the constant-dropped QUBO terms
    used for energy decomposition validation.
    """
    sum_y = float(np.sum(y)) if y.size > 0 else 1.0
    sum_z = float(np.sum(z)) if z.size > 0 else 1.0

    viol_onehot_y = abs(sum_y - 1.0) if y.size > 0 else 0.0
    viol_onehot_z = abs(sum_z - 1.0) if z.size > 0 else 0.0

    S_reg = float(np.sum(x[np.array(R_zero, dtype=int)])) if R_zero else 0.0
    viol_reg_shortfall = max(0.0, float(Rmin) - S_reg) if (R_zero and Rmin > 0) else 0.0

    chosen_prem = int(np.argmax(z) + 1) if z.size > 0 else None
    viol_afford = 0.0
    if affordability_enabled and disallow_bands and chosen_prem is not None:
        viol_afford = 1.0 if (chosen_prem in disallow_bands) else 0.0

    penE_onehot = float(A) * (viol_onehot_y ** 2 + viol_onehot_z ** 2) if A != 0.0 else 0.0
    penE_reg    = float(B) * (viol_reg_shortfall ** 2)                  if B != 0.0 else 0.0
    penE_afford = float(D) * float(viol_afford)                          if D != 0.0 else 0.0

    onehot_ok             = bool((y.size == 0 or abs(sum_y - 1.0) < 1e-9) and
                                  (z.size == 0 or abs(sum_z - 1.0) < 1e-9))
    reg_satisfied         = bool(viol_reg_shortfall <= 1e-9)
    affordability_satisfied = bool(viol_afford < 0.5)
    is_feasible           = bool(onehot_ok and reg_satisfied and affordability_satisfied)

    return {
        "violations": {
            "onehot_y":                float(viol_onehot_y),
            "onehot_z":                float(viol_onehot_z),
            "reg_shortfall":           float(viol_reg_shortfall),
            "affordability_indicator": float(viol_afford),
        },
        "penalty_energy": {
            "penE_onehot": float(penE_onehot),
            "penE_reg":    float(penE_reg),
            "penE_afford": float(penE_afford),
            "penE_total":  float(penE_onehot + penE_reg + penE_afford),
        },
        "constraint_checks": {
            "onehot_ok":               bool(onehot_ok),
            "reg_satisfied":           bool(reg_satisfied),
            "reg_violation_amount":    int(round(float(viol_reg_shortfall))),
            "affordability_enabled":   bool(affordability_enabled),
            "affordability_disallow_bands": [int(b) for b in disallow_bands],
            "affordability_satisfied": bool(affordability_satisfied),
            "affordability_violation": bool(not affordability_satisfied) if (affordability_enabled and disallow_bands) else False,
            "is_feasible":             bool(is_feasible),
        },
        "feasibility_margin": {
            "onehot_worst_violation": float(max(viol_onehot_y, viol_onehot_z)),
            "reg_shortfall":          float(viol_reg_shortfall),
            "affordability_violated": bool(viol_afford > 0.5),
        },
    }


def _qubo_terms_signed_constant_dropped(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t_bits: np.ndarray,
    A: float,
    B: float,
    D: float,
    R_zero: list[int],
    Rmin: int,
    affordability_enabled: bool,
    disallow_bands: list[int],
) -> Dict[str, float]:
    """
    Compute the constant-dropped signed QUBO penalty contributions P̃_i(v)
    used in the energy decomposition validation (Section 4.2, item 5):

        E(v) = f(v) + Σ_i P̃_i(v)

    These terms can be negative at feasible solutions because the squared-
    form constants are dropped when building the QUBO in build_qubo.py.
    """
    sum_y = float(np.sum(y)) if y.size > 0 else 1.0
    sum_z = float(np.sum(z)) if z.size > 0 else 1.0

    qterm_onehot = 0.0
    if A != 0.0:
        if y.size > 0:
            qterm_onehot += A * ((1.0 - sum_y) ** 2 - 1.0)
        if z.size > 0:
            qterm_onehot += A * ((1.0 - sum_z) ** 2 - 1.0)

    qterm_reg = 0.0
    if B != 0.0 and Rmin > 0 and t_bits.size > 0 and len(R_zero) > 0:
        S = float(np.sum([x[i] for i in R_zero]))
        T = float(sum((2.0 ** l) * float(t_bits[l] > 0.5) for l in range(t_bits.size)))
        qterm_reg = B * ((S - T - float(Rmin)) ** 2 - float(Rmin ** 2))

    qterm_aff = 0.0
    if D != 0.0 and affordability_enabled and disallow_bands and z.size > 0:
        chosen_prem = int(np.argmax(z) + 1)
        if chosen_prem in disallow_bands:
            qterm_aff = float(D)

    return {
        "qterm_onehot_signed": float(qterm_onehot),
        "qterm_reg_signed":    float(qterm_reg),
        "qterm_afford_signed": float(qterm_aff),
        "qterm_total_signed":  float(qterm_onehot + qterm_reg + qterm_aff),
    }


# ---------------------------------------------------------------------------
# Gurobi MIQP solver
# ---------------------------------------------------------------------------

def solve_qubo_with_gurobi(
    *,
    Q: np.ndarray,
    N: int,
    M: int,
    K: int,
    time_limit_sec: int,
    mip_gap: float,
    verbose: bool,
    enforce_onehot: bool = True,
) -> Dict[str, Any]:
    """
    Solve min v^T Q v subject to v ∈ {0,1}^n using Gurobi MIQP.

    Hard one-hot equality constraints are added for the deductible block y
    (indices N..N+M) and premium block z (indices N+M..N+M+K) when
    enforce_onehot=True. This makes the MIQP formulation directly
    comparable to greedy and SA, which enforce one-hot via projection after
    every proposed move.

    The objective is constructed from the symmetrised upper-triangular
    entries of Q (Q[i,j] + Q[j,i] for i < j) to handle cases where Q
    is stored as a non-symmetric matrix.

    Returns a dict with:
        status_code      : Gurobi status integer
        status_name      : Human-readable status string (OPTIMAL, TIME_LIMIT, ...)
        obj              : Gurobi objective value (None if no incumbent)
        solution         : Continuous solution vector from Gurobi (None if no incumbent)
        mip_gap          : Final MIP gap (None if no incumbent)
        runtime_sec      : Wall-clock time for the solve call
        gurobi_runtime_sec : Gurobi's internal reported runtime
        enforce_onehot   : Whether hard one-hot constraints were added
    """
    if gp is None:
        raise RuntimeError(
            "gurobipy not available. Install Gurobi and gurobipy to run the exact baseline."
        )

    Q  = np.asarray(Q, dtype=float)
    n  = int(Q.shape[0])
    if Q.ndim != 2 or Q.shape[1] != n:
        raise ValueError("Q must be a square matrix.")
    if N + M + K > n:
        raise ValueError("Inconsistent N/M/K vs Q dimension.")

    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K

    m = gp.Model("qubo_exact")
    m.Params.OutputFlag = 0 if not verbose else 1
    m.Params.TimeLimit  = float(time_limit_sec)
    m.Params.MIPGap     = float(mip_gap)

    v = m.addVars(n, vtype=GRB.BINARY, name="v")

    # Build objective from symmetrised Q entries (handles non-symmetric storage)
    obj = gp.QuadExpr()
    for i in range(n):
        qii = float(Q[i, i])
        if abs(qii) > 1e-12:
            obj.add(qii * v[i])
    for i in range(n):
        for j in range(i + 1, n):
            qsym = float(Q[i, j] + Q[j, i])
            if abs(qsym) > 1e-12:
                obj.add(qsym * v[i] * v[j])
    m.setObjective(obj, GRB.MINIMIZE)

    # Hard one-hot constraints for deductible (y) and premium (z) bands
    if enforce_onehot and M > 0:
        m.addConstr(gp.quicksum(v[i] for i in range(y_start, y_end)) == 1, name="onehot_y")
    if enforce_onehot and K > 0:
        m.addConstr(gp.quicksum(v[i] for i in range(z_start, z_end)) == 1, name="onehot_z")

    t0 = time.perf_counter()
    m.optimize()
    wall_sec = float(time.perf_counter() - t0)

    status      = int(m.Status)
    status_name = {
        GRB.OPTIMAL:     "OPTIMAL",
        GRB.TIME_LIMIT:  "TIME_LIMIT",
        GRB.INFEASIBLE:  "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED:   "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL:  "SUBOPTIMAL",
        GRB.NUMERIC:     "NUMERIC",
    }.get(status, f"STATUS_{status}")

    sol = None
    if m.SolCount > 0:
        sol = np.array([float(v[i].X) for i in range(n)], dtype=float)

    return {
        "status_code":        status,
        "status_name":        status_name,
        "obj":                float(m.ObjVal) if m.SolCount > 0 else None,
        "solution":           sol,
        "mip_gap":            float(getattr(m, "MIPGap", 0.0)) if m.SolCount > 0 else None,
        "runtime_sec":        wall_sec,
        "gurobi_runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "enforce_onehot":     bool(enforce_onehot),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    args = parse_args()

    DATA_DIR = PROJECT_ROOT / "data" / "seeds" / f"seed_{args.seed}"
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    idx_path, qubo_path, idx, Q = _load_index_map_and_qubo(
        DATA_DIR, args.scenario_name, args.N
    )

    N = int(idx["N"])
    M = int(idx["M"])
    K = int(idx["K"])
    L = int(idx.get("L_reg_slack") or 0)

    affordability_enabled = bool(idx.get("affordability_enabled", False))
    disallow_bands        = [int(b) for b in (idx.get("disallow_premium_bands", []) or [])]
    Rmin                  = int(idx.get("Rmin", 0) or 0)

    regulatory = pd.read_csv(DATA_DIR / "regulatory_set.csv")
    R_set  = set(regulatory["feature_id"].astype(int).tolist())
    R_zero = [i for i in range(N) if (i + 1) in R_set]

    # Solve with hard one-hot constraints (Section 4.2, item 4)
    result = solve_qubo_with_gurobi(
        Q=Q, N=N, M=M, K=K,
        time_limit_sec=int(args.time_limit),
        mip_gap=float(args.mip_gap),
        verbose=not args.quiet,
        enforce_onehot=True,
    )

    # ------------------------------------------------------------------
    # Decode solution and validate energy decomposition
    # ------------------------------------------------------------------
    decoded:                  Dict[str, Any] = {}
    violations:               Dict[str, Any] = {}
    penalty_energy:           Dict[str, Any] = {}
    constraint_checks:        Dict[str, Any] = {}
    feasibility_margin:       Dict[str, Any] = {}
    qterms_signed:            Dict[str, Any] = {}
    weights:                  Dict[str, Any] = {}
    energy_decomposition_check: Dict[str, Any] = {}

    raw_objective = None
    energy_total  = None

    if result.get("solution") is not None:
        sol = np.asarray(result["solution"], dtype=float)
        v01 = (sol > 0.5).astype(float)

        energy_total = energy(Q, v01)

        x_sol  = v01[0:N]
        y_sol  = v01[N:N + M]
        z_sol  = v01[N + M:N + M + K]
        t_bits = v01[N + M + K:N + M + K + L] if L > 0 else np.array([], dtype=float)

        chosen_deductible = int(np.argmax(y_sol) + 1) if M > 0 else None
        chosen_premium    = int(np.argmax(z_sol) + 1) if K > 0 else None

        selected_features_count     = int(np.sum(x_sol > 0.5))
        selected_reg_features_count = int(np.sum(x_sol[np.array(R_zero, dtype=int)] > 0.5)) if R_zero else 0

        decoded = {
            "chosen_deductible_band":      chosen_deductible,
            "chosen_premium_band":         chosen_premium,
            "selected_features_count":     selected_features_count,
            "selected_reg_features_count": selected_reg_features_count,
        }

        c_df    = pd.read_csv(DATA_DIR / "c_vector.csv")
        c       = c_df["expected_cost"].to_numpy(dtype=float)[:N]
        Sigma   = np.load(DATA_DIR / "sigma_matrix.npz")["sigma"][:N, :N]

        premium_bands = np.array(idx.get("premium_bands", []), dtype=float)
        lambda_risk   = float(idx.get("lambda_risk", 0.0))

        raw_objective = compute_raw_objective(
            x01=x_sol, z01=z_sol,
            c=c, Sigma=Sigma,
            lambda_risk=lambda_risk, premium_bands=premium_bands,
        )

        A = _pos_float(idx, "A_onehot", default=0.0)
        B = _pos_float(idx, "B_reg",    default=0.0)
        D = float(_infer_D_affordability(idx))
        if (not affordability_enabled) or (not disallow_bands):
            D = 0.0
        weights = {"A_onehot": float(A), "B_reg": float(B), "D_afford": float(D)}

        cm = _compute_constraint_metrics(
            x=x_sol, y=y_sol, z=z_sol,
            R_zero=R_zero, Rmin=Rmin,
            affordability_enabled=affordability_enabled, disallow_bands=disallow_bands,
            A=A, B=B, D=D,
        )
        violations        = cm["violations"]
        penalty_energy    = cm["penalty_energy"]
        constraint_checks = cm["constraint_checks"]
        feasibility_margin = cm["feasibility_margin"]

        qterms_signed = _qubo_terms_signed_constant_dropped(
            x=x_sol, y=y_sol, z=z_sol, t_bits=t_bits,
            A=A, B=B, D=D,
            R_zero=R_zero, Rmin=Rmin,
            affordability_enabled=affordability_enabled, disallow_bands=disallow_bands,
        )

        # Energy decomposition validation (Section 4.2, item 5): 10⁻⁶ tolerance
        reconstructed_energy = raw_objective + qterms_signed["qterm_total_signed"]
        energy_error         = abs(energy_total - reconstructed_energy)
        relative_error       = energy_error / max(abs(energy_total), 1.0)

        if relative_error > 1e-6:
            raise ValueError(
                f"Energy decomposition validation failed:\n"
                f"  energy_total        = {energy_total:.12e}\n"
                f"  raw_objective       = {raw_objective:.12e}\n"
                f"  qterm_total_signed  = {qterms_signed['qterm_total_signed']:.12e}\n"
                f"  reconstructed       = {reconstructed_energy:.12e}\n"
                f"  absolute_error      = {energy_error:.12e}\n"
                f"  relative_error      = {relative_error:.12e}\n"
                f"This indicates a bug in penalty calculation or QUBO construction."
            )

        energy_decomposition_check = {
            "reconstructed_energy": float(reconstructed_energy),
            "absolute_error":       float(energy_error),
            "relative_error":       float(relative_error),
            "validation_passed":    True,
        }

    # ------------------------------------------------------------------
    # Assemble output record
    # ------------------------------------------------------------------
    out = {
        "timestamp_utc":     datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scenario":          args.scenario_name,
        "seed":              int(args.seed),
        "N": N, "M": M, "K": K, "L_reg_slack": L,
        "status":            {"code": result.get("status_code"), "name": result.get("status_name")},
        "objective_reported": result.get("obj"),
        "energy_total":      energy_total,
        "raw_objective":     raw_objective,
        "violations":        violations,
        "penalty_energy":    penalty_energy,
        "constraint_checks": constraint_checks,
        "feasibility_margin": feasibility_margin,
        "qubo_terms_signed": qterms_signed,
        "penalty_weights":   weights,
        "energy_decomposition_check": energy_decomposition_check,
        "mip_gap":           result.get("mip_gap"),
        "runtime_sec":       result.get("runtime_sec"),
        "gurobi_runtime_sec": result.get("gurobi_runtime_sec"),
        "decoded":           decoded,
        "files_used": {
            "index_map": str(idx_path),
            "qubo":      str(qubo_path),
            "seed_dir":  str(DATA_DIR),
        },
        "note": (
            "Gurobi solves the same penalized QUBO objective as greedy and SA, "
            "with hard one-hot constraints for comparability (Section 4.2, item 4)."
        ),
        "gurobi_modeling": {"enforce_onehot": bool(result.get("enforce_onehot", True))},
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_path = DATA_DIR / f"gurobi_exact_{args.scenario_name}_N{N}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = DATA_DIR / "gurobi_exact_results.csv"
    row = pd.DataFrame([{
        "scenario":              args.scenario_name,
        "seed":                  int(args.seed),
        "N":                     N,
        "status_code":           out["status"]["code"],
        "status_name":           out["status"]["name"],
        "energy_total":          out["energy_total"],
        "raw_objective":         out["raw_objective"],
        "penE_total":            (out.get("penalty_energy",  {}) or {}).get("penE_total"),
        "qterm_total_signed":    (out.get("qubo_terms_signed", {}) or {}).get("qterm_total_signed"),
        "mip_gap":               out["mip_gap"],
        "runtime_sec":           out["runtime_sec"],
        "gurobi_runtime_sec":    out.get("gurobi_runtime_sec"),
        "Rmin":                  int(Rmin),
        "affordability_enabled": bool(affordability_enabled),
        "is_feasible":           (out.get("constraint_checks", {}) or {}).get("is_feasible"),
        "energy_decomp_error":   (out.get("energy_decomposition_check", {}) or {}).get("relative_error"),
        "index_map_file":        idx_path.name,
        "qubo_file":             qubo_path.name,
        "enforce_onehot":        True,
    }])
    if csv_path.exists():
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("Gurobi exact (penalized QUBO) baseline completed.")
    print(f"Index map : {idx_path.name}")
    print(f"QUBO      : {qubo_path.name}")
    print(f"JSON      : {out_path.name}")
    print(f"CSV       : {csv_path.name}")

    if out["energy_total"] is not None:
        pe      = out.get("penalty_energy", {}) or {}
        raw_str = "None" if out["raw_objective"] is None else f"{out['raw_objective']:.6f}"
        print(
            f"energy_total={out['energy_total']:.6f}  raw={raw_str}  "
            f"penE_total={pe.get('penE_total')}  gap={out['mip_gap']}  "
            f"time={out['runtime_sec']:.2f}s  status={out['status']['name']}"
        )
        if decoded:
            print(
                f"premium={decoded.get('chosen_premium_band')}  "
                f"deductible={decoded.get('chosen_deductible_band')}  "
                f"features={decoded.get('selected_features_count')}  "
                f"reg_selected={decoded.get('selected_reg_features_count')} (Rmin={Rmin})"
            )
        cc = out.get("constraint_checks", {}) or {}
        print(f"is_feasible={cc.get('is_feasible')}")
        decomp = out.get("energy_decomposition_check", {}) or {}
        if decomp.get("validation_passed"):
            print(f"Energy decomposition validated: rel_err={decomp.get('relative_error', 0):.2e}")
    else:
        print(f"status={out['status']['name']} — no incumbent solution found.")


if __name__ == "__main__":
    main()