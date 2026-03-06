# src_code/solvers/solve_classical.py
"""
Classical Solvers: Greedy Heuristic and Simulated Annealing
============================================================
Implements the two classical baseline solvers described in Section 4.2 of
the accompanying paper:

  1. Greedy constructive heuristic (multi-start) — a deterministic local
     descent baseline with negligible runtime relative to metaheuristics.
     Provides a lower quality bound and a feasibility sanity check
     (Section 4.2, item 1).

  2. Multi-start Simulated Annealing (SA) — the primary metaheuristic
     baseline. Runs 20 independent starts of 40,000 steps each, for a
     fixed total evaluation budget of B = 800,000 per instance, identical
     across all scenarios and problem sizes (Section 4.2, item 2):

         SA steps per start : 40,000
         Number of starts   : 20
         Total steps        : 40,000 × 20 = 800,000

     SA uses no wall-clock stopping and no early termination. The reported
     value objective_raw_sa_best is the best raw objective observed across
     all 20 starts on the given instance.

Move set (Section 3.10, shared with ECHO's internal SA):
    Feature bit flip  : probability 0.85
    Deductible change : probability 0.075
    Premium change    : probability 0.075
    One-hot projection is enforced after every proposed move.

Energy decomposition validation (Section 4.2, item 5):
    For each solution the decomposition E(v) = f(v) + Σ P̃_i(v) is verified
    to relative tolerance 10⁻⁶. Solutions failing this check raise a
    ValueError and are excluded from output.

Output: results/solution_<scenario>_seed<SEED>_N<N>.json
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_scenario_from_args_or_prompt() -> int:
    """Parse --scenario from CLI, or prompt interactively if absent."""
    parser = argparse.ArgumentParser(description="Solve QUBO classically (greedy + SA).")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4],
                        help="Scenario number (1..4).")
    args, _ = parser.parse_known_args()

    if args.scenario is not None:
        return args.scenario

    while True:
        s = input("Select scenario to solve (1=S1, 2=S2, 3=S3, 4=S4): ").strip()
        if s in {"1", "2", "3", "4"}:
            return int(s)
        print("Invalid input. Please enter 1, 2, 3, or 4.")


# ---------------------------------------------------------------------------
# Core QUBO primitives
# ---------------------------------------------------------------------------

def energy(Q: np.ndarray, v: np.ndarray) -> float:
    """Evaluate QUBO energy: E(v) = v^T Q v."""
    return float(v @ Q @ v)


def project_onehot(v: np.ndarray, N: int, M: int, K: int,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Hard-project v to satisfy one-hot constraints on y and z.

    For the deductible block y (indices N..N+M) and premium block z
    (indices N+M..N+M+K), exactly one variable must equal 1. If all
    entries are zero, a random selection is made.
    """
    v2 = v.copy()

    y = v2[N:N + M]
    j = int(np.argmax(y)) if float(np.sum(y)) > 0.0 else int(rng.integers(0, M))
    y[:] = 0.0
    y[j] = 1.0

    z = v2[N + M:N + M + K]
    k = int(np.argmax(z)) if float(np.sum(z)) > 0.0 else int(rng.integers(0, K))
    z[:] = 0.0
    z[k] = 1.0

    return v2


# ---------------------------------------------------------------------------
# Greedy heuristic (Section 4.2, item 1)
# ---------------------------------------------------------------------------

def greedy_descent_single(Q: np.ndarray, N: int, M: int, K: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Single-start greedy local descent from a random binary initialization.

    Iterates over all non-band variables (x_1..x_N) and accepts any flip
    that strictly reduces QUBO energy. Repeats until no improving flip
    exists. One-hot projection is applied after every move.
    """
    n = Q.shape[0]
    v = rng.integers(0, 2, size=n).astype(float)
    v = project_onehot(v, N, M, K, rng)
    best_e = energy(Q, v)

    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K

    improved = True
    while improved:
        improved = False
        for i in range(n):
            if y_start <= i < y_end or z_start <= i < z_end:
                continue
            v_try = v.copy()
            v_try[i] = 1.0 - v_try[i]
            v_try = project_onehot(v_try, N, M, K, rng)
            e_try = energy(Q, v_try)
            if e_try < best_e:
                v = v_try
                best_e = e_try
                improved = True

    return v


def greedy_descent_multistart(
    Q: np.ndarray,
    N: int, M: int, K: int,
    rng: np.random.Generator,
    num_starts: int,
) -> Tuple[np.ndarray, float]:
    """
    Multi-start greedy: run greedy_descent_single num_starts times and
    return the best solution found.
    """
    best_v = None
    best_e = float("inf")
    for _ in range(int(num_starts)):
        v = greedy_descent_single(Q, N, M, K, rng)
        e = energy(Q, v)
        if e < best_e:
            best_e = e
            best_v = v
    assert best_v is not None
    return best_v, float(best_e)


# ---------------------------------------------------------------------------
# Simulated annealing (Section 4.2, item 2)
# ---------------------------------------------------------------------------

def simulated_annealing(
    Q: np.ndarray,
    N: int, M: int, K: int,
    rng: np.random.Generator,
    steps: int = 40000,
    T0: float = 5.0,
    Tend: float = 0.01,
    prem_move_prob: float = 0.075,
    ded_move_prob: float = 0.075,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Single-start SA for a fixed number of steps (Section 4.2, item 2).

    Move set (Section 3.10):
        Feature bit flip  : probability 1 − prem_move_prob − ded_move_prob = 0.85
        Deductible change : probability ded_move_prob  = 0.075
        Premium change    : probability prem_move_prob = 0.075

    Temperature follows a linear schedule from T0 to Tend. One-hot
    projection is enforced after every proposed move.

    Returns (best_v, best_e, diagnostics) where diagnostics records when
    the best solution was found — used in the Section 4.2 budget fairness
    analysis (sa_convergence_fraction in insurance_baseline_results.csv).
    """
    n = Q.shape[0]
    v = rng.integers(0, 2, size=n).astype(float)
    v = project_onehot(v, N, M, K, rng)
    e = energy(Q, v)

    best_v    = v.copy()
    best_e    = float(e)
    best_step = 0

    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K

    flip_prob = 1.0 - prem_move_prob - ded_move_prob
    if flip_prob <= 0:
        raise ValueError("Invalid SA move probabilities: flip_prob must be > 0.")

    for t in range(steps):
        frac = t / max(1, steps - 1)
        T    = T0 * (1 - frac) + Tend * frac

        v_new = v.copy()
        r = float(rng.random())

        if r < flip_prob:
            while True:
                i = int(rng.integers(0, n))
                if not (y_start <= i < y_end) and not (z_start <= i < z_end):
                    break
            v_new[i] = 1.0 - v_new[i]
        elif r < flip_prob + ded_move_prob:
            j = int(rng.integers(0, M))
            v_new[y_start:y_end] = 0.0
            v_new[y_start + j]   = 1.0
        else:
            k = int(rng.integers(0, K))
            v_new[z_start:z_end] = 0.0
            v_new[z_start + k]   = 1.0

        v_new = project_onehot(v_new, N, M, K, rng)
        e_new = energy(Q, v_new)

        if e_new < e:
            v, e = v_new, e_new
        else:
            p = math.exp(-(e_new - e) / max(1e-12, T))
            if float(rng.random()) < p:
                v, e = v_new, e_new

        if e < best_e:
            best_e    = float(e)
            best_v    = v.copy()
            best_step = t

    diagnostics = {
        "best_found_at_step":   int(best_step),
        "convergence_fraction": float(best_step) / max(1, steps),
        "converged_early":      bool(float(best_step) / max(1, steps) < 0.2),
    }

    return best_v, float(best_e), diagnostics


def _seed_for_start(base_seed: int, start_id: int) -> int:
    """
    Deterministic per-start seed using two large primes for good mixing:

        seed = (base_seed × 104729  +  start_id × 7919) mod 2³²

    Ensures independent, reproducible RNG streams across multi-start runs
    with no collision risk for the seed and start-id ranges used in the study.
    """
    return (int(base_seed) * 104729 + int(start_id) * 7919) & 0xFFFFFFFF


def simulated_annealing_multistart(
    Q: np.ndarray,
    N: int, M: int, K: int,
    base_seed: int,
    num_starts: int,
    steps: int,
    T0: float,
    Tend: float,
    prem_move_prob: float,
    ded_move_prob: float,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Multi-start SA: run SA num_starts times with independent per-start seeds
    and return the best solution found (Section 4.2, item 2).

    Per-start seeds are derived deterministically via _seed_for_start so
    that results are fully reproducible from base_seed alone.

    The diagnostics dict from the best-performing start is retained so that
    convergence_fraction in insurance_baseline_results.csv reflects the run
    that produced the reported solution.
    """
    best_v          = None
    best_e          = float("inf")
    best_diagnostics = None

    for s in range(int(num_starts)):
        run_seed = _seed_for_start(base_seed, s)
        rng = np.random.default_rng(run_seed)

        v, e, diag = simulated_annealing(
            Q, N, M, K, rng,
            steps=steps, T0=T0, Tend=Tend,
            prem_move_prob=prem_move_prob, ded_move_prob=ded_move_prob,
        )
        v = project_onehot(v, N, M, K, rng)
        e = energy(Q, v)

        if e < best_e:
            best_e          = float(e)
            best_v          = v.copy()
            best_diagnostics = diag

    assert best_v is not None
    params = {
        "multistart":          True,
        "num_starts":          int(num_starts),
        "start_seed_policy":   "seed_for_start = (base_seed*104729 + start_id*7919) mod 2^32",
        "random_seed_policy":  "solver_rng_tied_to_instance_seed",
        "random_seed":         int(base_seed),
        "convergence":         best_diagnostics,
    }
    return best_v, float(best_e), params


# ---------------------------------------------------------------------------
# File resolution helpers
# ---------------------------------------------------------------------------

def resolve_qubo_path(data_dir: Path, scenario_name: str, N: int) -> Path:
    """
    Resolve the QUBO file path, supporting both the current naming convention
    (qubo_Q_<scenario>_N<N>.npz) and the legacy convention (qubo_Q_<scenario>.npz).
    """
    q_new = data_dir / f"qubo_Q_{scenario_name}_N{N}.npz"
    if q_new.exists():
        return q_new
    q_old = data_dir / f"qubo_Q_{scenario_name}.npz"
    if q_old.exists():
        return q_old
    raise FileNotFoundError(
        f"Missing QUBO file: {q_new} (and no legacy file {q_old}). "
        f"Run build_qubo.py first."
    )


def load_index_map(data_dir: Path, scenario_name: str, N: int) -> dict:
    """Load the index map written by build_qubo.py for this instance."""
    idx_path = data_dir / f"index_map_{scenario_name}_N{N}.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Missing index map: {idx_path}. Run build_qubo.py first."
        )
    return json.loads(idx_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Objective and penalty computation
# ---------------------------------------------------------------------------

def compute_raw_objective(
    x01: np.ndarray,
    z01: np.ndarray,
    c: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float,
    premium_bands: np.ndarray,
) -> float:
    """
    Compute the raw economic objective f(v) excluding all penalty energies
    (Section 4.3):

        f(v) = c^T x  +  lambda_risk · x^T Σ x  −  Σ_k P_k z_k

    This is the business-meaningful insurance objective corresponding to
    expected cost, risk adjustment, and premium revenue (Section 2.3).
    It is the correct metric for cross-solver economic comparison because
    penalties are scaled and not directly comparable across scenarios or
    sizes (Section 4.3).
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

    Resolution order (same tiered logic as run_baseline_full.py and
    run_echo_full.py for consistency across the pipeline):
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
    Compute conceptual constraint violation amounts and the corresponding
    penalty energy contributions.

    These are the violation-based quantities used in the feasibility
    flags reported in Section 4.3 (viol_* and is_feasible_* columns).
    They differ from the constant-dropped QUBO terms computed in
    _qubo_signed_terms_constant_dropped, which are used for energy
    decomposition validation.
    """
    sum_y = float(np.sum(y)) if y.size > 0 else 1.0
    sum_z = float(np.sum(z)) if z.size > 0 else 1.0

    viol_onehot_y = abs(sum_y - 1.0) if y.size > 0 else 0.0
    viol_onehot_z = abs(sum_z - 1.0) if z.size > 0 else 0.0

    S_reg = float(np.sum([x[i] for i in R_zero])) if R_zero else 0.0
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
        "checks": {
            "onehot_ok":               bool(onehot_ok),
            "reg_satisfied":           bool(reg_satisfied),
            "reg_violation_amount":    int(round(float(viol_reg_shortfall))),
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


def _qubo_signed_terms_constant_dropped(
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
    Compute the constant-dropped signed QUBO penalty contributions P̃_i(v).

    These are the terms used in the energy decomposition validation
    (Section 4.2, item 5):

        E(v) = f(v) + Σ_i P̃_i(v)

    where f(v) is the raw economic objective. Unlike the violation-based
    penalty energies in _compute_constraint_metrics, these terms can be
    negative at feasible solutions because the squared-form constants are
    dropped when building the QUBO.
    """
    onehot_y_sum = float(np.sum(y)) if y.size > 0 else 1.0
    onehot_z_sum = float(np.sum(z)) if z.size > 0 else 1.0

    qterm_onehot = 0.0
    if A != 0.0:
        if y.size > 0:
            qterm_onehot += A * ((1.0 - onehot_y_sum) ** 2 - 1.0)
        if z.size > 0:
            qterm_onehot += A * ((1.0 - onehot_z_sum) ** 2 - 1.0)

    qterm_reg = 0.0
    if B != 0.0 and Rmin > 0 and t_bits.size > 0 and R_zero:
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


def _decode_and_score(
    v01: np.ndarray,
    *,
    Q: np.ndarray,
    N: int, M: int, K: int, L: int,
    idx_map: dict,
    c: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float,
    premium_bands: np.ndarray,
    deductible_bands: np.ndarray,
    R_zero: list[int],
    Rmin: int,
    affordability_enabled: bool,
    disallow_bands: list[int],
) -> Dict[str, Any]:
    """
    Decode a binary solution vector and compute all reported metrics.

    Performs the energy decomposition validation required by Section 4.2
    item 5: raises ValueError if relative error |E(v) − f(v) − Σ P̃_i(v)|
    / max(|E(v)|, 1) exceeds 10⁻⁶, which would indicate a bug in penalty
    calculation or QUBO construction.
    """
    v = v01.astype(float)

    x      = v[0:N]
    y      = v[N:N + M]
    z      = v[N + M:N + M + K]
    t_bits = v[N + M + K:] if L > 0 else np.array([], dtype=float)

    chosen_ded  = int(np.argmax(y) + 1) if M > 0 else None
    chosen_prem = int(np.argmax(z) + 1) if K > 0 else None

    selected_feature_ids        = [int(i + 1) for i in range(N) if x[i] > 0.5]
    selected_features_count     = int(len(selected_feature_ids))
    selected_reg_features_count = int(sum(1 for i in R_zero if x[i] > 0.5)) if R_zero else 0

    A = _pos_float(idx_map, "A_onehot", default=0.0)
    B = _pos_float(idx_map, "B_reg",    default=0.0)
    D = float(_infer_D_affordability(idx_map))
    if (not affordability_enabled) or (not disallow_bands):
        D = 0.0

    e_total = energy(Q, v)
    raw     = compute_raw_objective(
        x01=(x > 0.5).astype(float),
        z01=(z > 0.5).astype(float),
        c=c, Sigma=Sigma,
        lambda_risk=lambda_risk,
        premium_bands=premium_bands,
    )

    cm = _compute_constraint_metrics(
        x=(x > 0.5).astype(float), y=(y > 0.5).astype(float), z=(z > 0.5).astype(float),
        R_zero=R_zero, Rmin=Rmin,
        affordability_enabled=affordability_enabled, disallow_bands=disallow_bands,
        A=A, B=B, D=D,
    )

    qterms = _qubo_signed_terms_constant_dropped(
        x=x, y=y, z=z, t_bits=t_bits,
        A=A, B=B, D=D,
        R_zero=R_zero, Rmin=Rmin,
        affordability_enabled=affordability_enabled, disallow_bands=disallow_bands,
    )

    # Energy decomposition validation (Section 4.2, item 5): 10⁻⁶ tolerance
    reconstructed_energy = raw + qterms["qterm_total_signed"]
    energy_error         = abs(e_total - reconstructed_energy)
    relative_error       = energy_error / max(abs(e_total), 1.0)

    if relative_error > 1e-6:
        raise ValueError(
            f"Energy decomposition validation failed:\n"
            f"  energy_total        = {e_total:.12e}\n"
            f"  raw_objective       = {raw:.12e}\n"
            f"  qterm_total_signed  = {qterms['qterm_total_signed']:.12e}\n"
            f"  reconstructed       = {reconstructed_energy:.12e}\n"
            f"  absolute_error      = {energy_error:.12e}\n"
            f"  relative_error      = {relative_error:.12e}\n"
            f"This indicates a bug in penalty calculation or QUBO construction."
        )

    return {
        "energy_total":  float(e_total),
        "raw_objective": float(raw),
        "decoded": {
            "chosen_deductible_band":       chosen_ded,
            "chosen_premium_band":          chosen_prem,
            "selected_features_count":      selected_features_count,
            "selected_feature_ids":         selected_feature_ids,
            "selected_reg_features_count":  selected_reg_features_count,
        },
        "violations":          cm["violations"],
        "penalty_energy":      cm["penalty_energy"],
        "constraint_checks": {
            "onehot_ok":                    cm["checks"]["onehot_ok"],
            "reg_satisfied":                cm["checks"]["reg_satisfied"],
            "reg_violation_amount":         cm["checks"]["reg_violation_amount"],
            "affordability_enabled":        bool(affordability_enabled),
            "affordability_disallow_bands": [int(b) for b in disallow_bands],
            "affordability_satisfied":      cm["checks"]["affordability_satisfied"],
            "affordability_violation":      cm["checks"]["affordability_violation"],
            "is_feasible":                  cm["checks"]["is_feasible"],
        },
        "feasibility_margin": cm["feasibility_margin"],
        "qubo_terms_signed":  qterms,
        "penalty_weights":    {"A_onehot": float(A), "B_reg": float(B), "D_afford": float(D)},
        "energy_decomposition_check": {
            "reconstructed_energy": float(reconstructed_energy),
            "absolute_error":       float(energy_error),
            "relative_error":       float(relative_error),
            "validation_passed":    True,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    scen = get_scenario_from_args_or_prompt()

    cfg_path = PROJECT_ROOT / "config" / f"config_qubo_S{scen}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        qcfg = yaml.safe_load(f)

    SEED           = int(qcfg["data"]["seed"])
    N              = int(qcfg["features"]["N"])
    premium_bands  = np.array(qcfg["pricing"]["premium_bands"],    dtype=float)
    deductible_bands = np.array(qcfg["deductibles"]["deductible_bands"], dtype=float)
    lambda_risk    = float(qcfg["risk"]["lambda_risk"])

    M             = len(deductible_bands)
    K             = len(premium_bands)
    scenario_name = str(qcfg["scenario"]["name"])

    DATA_DIR = PROJECT_ROOT / "data" / "seeds" / f"seed_{SEED}"
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing seed folder: {DATA_DIR}. Run generate_data.py first.")

    c_all    = pd.read_csv(DATA_DIR / "c_vector.csv")["expected_cost"].to_numpy(dtype=float)
    c        = c_all[:N]
    Sigma    = np.load(DATA_DIR / "sigma_matrix.npz")["sigma"][:N, :N]

    Q_path = resolve_qubo_path(DATA_DIR, scenario_name, N)
    Q      = np.load(Q_path)["Q"]
    idx    = load_index_map(DATA_DIR, scenario_name, N)

    n_total = int(Q.shape[0])
    L       = n_total - (N + M + K)
    if L < 0:
        raise ValueError("Q has fewer variables than N+M+K; inconsistent build or inputs.")

    regulatory = pd.read_csv(DATA_DIR / "regulatory_set.csv")
    R_set  = set(regulatory["feature_id"].astype(int).tolist())
    R_zero = [i for i in range(N) if (i + 1) in R_set]

    Rmin                 = int(idx.get("Rmin", 0))
    affordability_enabled = bool(idx.get("affordability_enabled", False))
    disallow_bands       = [int(b) for b in (idx.get("disallow_premium_bands", []) or [])]

    # ------------------------------------------------------------------
    # Solver parameters from YAML
    # ------------------------------------------------------------------
    solver_cfg = qcfg.get("solver", {}) or {}

    # Greedy multi-start count: use config if present, else size-based default
    greedy_num_starts_cfg = solver_cfg.get("greedy_num_starts", None)
    if greedy_num_starts_cfg is not None:
        num_starts_greedy = int(greedy_num_starts_cfg)
    else:
        if N <= 30:
            num_starts_greedy = 20
        elif N <= 50:
            num_starts_greedy = 50
        else:
            num_starts_greedy = 100

    # SA parameters (Section 4.2, item 2)
    steps  = int(solver_cfg.get("sa_steps", 40000))
    T0     = float(solver_cfg.get("sa_T0",   5.0))
    Tend   = float(solver_cfg.get("sa_Tend", 0.01))

    sa_multistart  = bool(solver_cfg.get("sa_multistart", True))
    sa_num_starts_cfg = solver_cfg.get("sa_num_starts", None)
    sa_num_starts  = int(sa_num_starts_cfg) if sa_num_starts_cfg is not None else int(num_starts_greedy)

    # Move probabilities: 0.85 / 0.075 / 0.075 across all scenarios (Section 3.10).
    # Uniform across all four scenarios — no scenario-specific parameter tuning
    # (abstract and Section 4.2).
    prem_prob = 0.075
    ded_prob  = 0.075

    # ------------------------------------------------------------------
    # Independent RNG streams for greedy and SA.
    # Offset scheme: base = SEED * 104729 + offset (same prime as _seed_for_start).
    # ------------------------------------------------------------------
    RNG_OFFSET_GREEDY = 1
    RNG_OFFSET_SA     = 2

    rng_greedy = np.random.default_rng(int(SEED) * 104729 + RNG_OFFSET_GREEDY)

    t0_greedy = time.perf_counter()
    v_greedy, _ = greedy_descent_multistart(Q, N, M, K, rng_greedy, num_starts=num_starts_greedy)
    greedy_runtime_sec = time.perf_counter() - t0_greedy

    v_greedy = project_onehot(v_greedy, N, M, K, rng_greedy)
    e_greedy = energy(Q, v_greedy)

    t0_sa = time.perf_counter()

    if sa_multistart and sa_num_starts > 1:
        sa_base_seed = int(SEED) * 104729 + RNG_OFFSET_SA
        v_sa, e_sa, sa_params = simulated_annealing_multistart(
            Q, N, M, K,
            base_seed=sa_base_seed,
            num_starts=int(sa_num_starts),
            steps=int(steps), T0=float(T0), Tend=float(Tend),
            prem_move_prob=float(prem_prob), ded_move_prob=float(ded_prob),
        )
    else:
        rng_sa = np.random.default_rng(int(SEED) * 104729 + RNG_OFFSET_SA)
        v_sa, e_sa, sa_diag = simulated_annealing(
            Q, N, M, K, rng_sa,
            steps=steps, T0=T0, Tend=Tend,
            prem_move_prob=prem_prob, ded_move_prob=ded_prob,
        )
        v_sa  = project_onehot(v_sa, N, M, K, rng_sa)
        e_sa  = energy(Q, v_sa)
        sa_params = {
            "multistart":         False,
            "num_starts":         1,
            "random_seed_policy": "solver_rng_tied_to_instance_seed",
            "random_seed":        int(SEED) * 104729 + RNG_OFFSET_SA,
            "convergence":        sa_diag,
        }

    sa_runtime_sec = time.perf_counter() - t0_sa

    # ------------------------------------------------------------------
    # Decode, score, and validate both solutions
    # ------------------------------------------------------------------
    decode_kwargs = dict(
        Q=Q, N=N, M=M, K=K, L=L, idx_map=idx,
        c=c, Sigma=Sigma, lambda_risk=lambda_risk,
        premium_bands=premium_bands, deductible_bands=deductible_bands,
        R_zero=R_zero, Rmin=Rmin,
        affordability_enabled=affordability_enabled, disallow_bands=disallow_bands,
    )
    greedy_pack = _decode_and_score(v_greedy, **decode_kwargs)
    sa_pack     = _decode_and_score(v_sa,     **decode_kwargs)

    # ------------------------------------------------------------------
    # Assemble output JSON
    # ------------------------------------------------------------------
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    out: Dict[str, Any] = {
        "scenario":       scenario_name,
        "scenario_file":  cfg_path.name,
        "seed":           SEED,
        "N": N, "M": M, "K": K, "L_reg_slack": L, "n_total": n_total,
        "qubo_file_used": Q_path.name,

        "energy_greedy":     float(e_greedy),
        "energy_sa_best":    float(e_sa),
        "objective_raw_greedy":   float(greedy_pack["raw_objective"]),
        "objective_raw_sa_best":  float(sa_pack["raw_objective"]),

        # Runtime (recorded for Section 4.2 overhead analysis)
        "greedy_runtime_sec":          float(greedy_runtime_sec),
        "sa_runtime_sec":              float(sa_runtime_sec),
        "total_classical_runtime_sec": float(greedy_runtime_sec + sa_runtime_sec),

        "solvers": {
            "greedy": greedy_pack | {
                "params": {
                    "multistart":  True,
                    "num_starts":  int(num_starts_greedy),
                    "random_seed": int(SEED) * 104729 + RNG_OFFSET_GREEDY,
                }
            },
            "sa": sa_pack | {
                "params": {
                    "sa_steps": int(steps),
                    "sa_T0":    float(T0),
                    "sa_Tend":  float(Tend),
                    "sa_move_probs": {
                        "premium":    float(prem_prob),
                        "deductible": float(ded_prob),
                        "flip_other": float(1.0 - prem_prob - ded_prob),
                    },
                    **sa_params,
                    "post_sa_hard_onehot_repair": True,
                }
            },
        },
    }

    out_path = RESULTS_DIR / f"solution_{scenario_name}_seed{SEED}_N{N}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"Loaded {scenario_name} | seed={SEED} | N={N} M={M} K={K} | n={n_total} | L={L}")
    print(f"QUBO: {Q_path.name}")

    print(f"Greedy (best of {num_starts_greedy}):  "
          f"energy={out['energy_greedy']:.6f}  "
          f"raw={out['objective_raw_greedy']:.6f}  "
          f"time={greedy_runtime_sec:.3f}s")

    conv = out["solvers"]["sa"]["params"]["convergence"]
    label = f"SA (best of {sa_num_starts})" if (sa_multistart and sa_num_starts > 1) else "SA (single run)"
    print(f"{label}:  "
          f"energy={out['energy_sa_best']:.6f}  "
          f"raw={out['objective_raw_sa_best']:.6f}  "
          f"time={sa_runtime_sec:.3f}s")
    print(f"  Convergence: best found at step {conv['best_found_at_step']}/{steps} "
          f"({conv['convergence_fraction']:.1%})")

    gp = out["solvers"]["greedy"]["penalty_energy"]
    sp = out["solvers"]["sa"]["penalty_energy"]
    print(f"Greedy penE_total={gp['penE_total']:.6f}  "
          f"feasible={out['solvers']['greedy']['constraint_checks']['is_feasible']}")
    print(f"SA     penE_total={sp['penE_total']:.6f}  "
          f"feasible={out['solvers']['sa']['constraint_checks']['is_feasible']}")

    g_d = out["solvers"]["greedy"]["energy_decomposition_check"]
    s_d = out["solvers"]["sa"]["energy_decomposition_check"]
    print(f"Energy decomposition check:  "
          f"greedy rel_err={g_d['relative_error']:.2e}  "
          f"SA rel_err={s_d['relative_error']:.2e}")

    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()