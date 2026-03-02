# src/build_qubo.py
import argparse
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

def parse_args():
    p = argparse.ArgumentParser(description="Build QUBO matrix for selected scenario.")
    p.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], required=False, help="Scenario number (1..4)")
    p.add_argument("--seed", type=int, default=None, help="Override seed (uses data/seeds/seed_<seed>).")
    p.add_argument("--N", type=int, default=None, help="Override feature size N.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing QUBO/index_map files if present.")
    return p.parse_args()

def config_path_for_scenario(project_root: Path, scenario_id: int) -> Path:
    return project_root / "config" / f"config_qubo_S{scenario_id}.yaml"

def _symmetrize_copy_upper(Q: np.ndarray) -> np.ndarray:
    """
    Symmetrize by copying the upper triangle into the lower triangle (NO averaging).
    This preserves intended off-diagonal coefficients when we only write Q[i,j] for i<j.
    """
    iu = np.triu_indices_from(Q, k=1)
    Q[iu[1], iu[0]] = Q[iu]
    return Q

def add_onehot(Q: np.ndarray, indices: list[int], weight: float) -> None:
    """
    Adds weight * (1 - sum v)^2 to Q (dropping constant).
    For symmetric Q in v^T Q v:
      - diagonal gets -weight
      - off-diagonal Q_ij (i<j) gets +weight
        because energy contributes 2*Q_ij*v_i*v_j.
    """
    w = float(weight)
    for i in indices:
        Q[i, i] += -w
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            Q[indices[a], indices[b]] += w

def add_square_linear_form(Q: np.ndarray, idx: list[int], coef: list[float], const: float, weight: float) -> None:
    """
    Adds weight * ( const + sum_t coef[t] * v[idx[t]] )^2 to Q, dropping weight*const^2 only.
    For symmetric Q in v^T Q v:
      - diagonal: w*(2*const*a_i + a_i^2)
      - off-diagonal (i<j): w*(a_i*a_j)
    """
    if len(idx) != len(coef):
        raise ValueError("idx and coef must have same length")

    w = float(weight)

    # diagonal
    for i, a in zip(idx, coef):
        a = float(a)
        Q[i, i] += w * (2.0 * float(const) * a + a * a)

    # off-diagonal (upper triangle)
    m = len(idx)
    for p in range(m):
        for q in range(p + 1, m):
            Q[idx[p], idx[q]] += w * (float(coef[p]) * float(coef[q]))

def _stable_hash_dict(d: dict) -> str:
    """Stable hash for reproducibility: JSON-dump with sorted keys, then SHA256."""
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _compute_qubo_condition_number(Q: np.ndarray) -> dict:
    """
    Compute QUBO condition number and eigenvalue spectrum for EJOR structural analysis.
    
    For hypothesis H1 (structural stability), we expect:
    - Bounded condition number as N grows
    - Eigenvalue spectrum that doesn't explode
    
    Returns dict with condition number, eigenvalue range, and spectrum percentiles.
    """
    try:
        # Use eigvalsh for symmetric matrices (faster and more stable than eig)
        eigenvalues = np.linalg.eigvalsh(Q)
        
        eig_min = float(eigenvalues.min())
        eig_max = float(eigenvalues.max())
        
        # Condition number: ratio of largest to smallest absolute eigenvalue
        # Handle near-zero eigenvalues carefully
        abs_eigs = np.abs(eigenvalues)
        abs_eig_max = float(abs_eigs.max())
        abs_eig_min_nonzero = float(abs_eigs[abs_eigs > 1e-12].min()) if np.any(abs_eigs > 1e-12) else 1e-12
        
        cond_number = abs_eig_max / abs_eig_min_nonzero
        
        # Spectrum percentiles for distributional analysis
        percentiles = [0, 25, 50, 75, 100]
        eig_percentiles = {
            f"p{p}": float(np.percentile(eigenvalues, p)) 
            for p in percentiles
        }
        
        return {
            "Q_condition_number": float(cond_number),
            "Q_eigenvalue_min": eig_min,
            "Q_eigenvalue_max": eig_max,
            "Q_eigenvalue_range": [eig_min, eig_max],
            "Q_eigenvalue_percentiles": eig_percentiles,
            "Q_eigenvalue_count_negative": int(np.sum(eigenvalues < -1e-12)),
            "Q_eigenvalue_count_positive": int(np.sum(eigenvalues > 1e-12)),
            "Q_eigenvalue_count_near_zero": int(np.sum(np.abs(eigenvalues) <= 1e-12)),
            "note": "Condition number measures numerical stability; bounded growth with N suggests stable QUBO structure.",
        }
    except Exception as e:
        return {
            "Q_condition_number": None,
            "error": f"Eigenvalue computation failed: {str(e)}",
        }

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    args = parse_args()

    if args.scenario is None:
        while True:
            s = input("Select scenario to build (1=S1, 2=S2, 3=S3, 4=S4): ").strip()
            if s in {"1", "2", "3", "4"}:
                scen = int(s)
                break
            print("Invalid input. Please enter 1, 2, 3, or 4.")
    else:
        scen = args.scenario

    cfg_path = config_path_for_scenario(PROJECT_ROOT, scen)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing scenario config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        qcfg = yaml.safe_load(f)

    # Capture hash BEFORE CLI overrides (useful to detect drift between runs)
    cfg_hash = _stable_hash_dict(qcfg)

    SEED = int(qcfg["data"]["seed"])
    N = int(qcfg["features"]["N"])
    if args.seed is not None:
        SEED = int(args.seed)
    if args.N is not None:
        N = int(args.N)

    premium_bands = np.array(qcfg["pricing"]["premium_bands"], dtype=float)
    deductible_bands = np.array(qcfg["deductibles"]["deductible_bands"], dtype=float)
    K = len(premium_bands)
    M = len(deductible_bands)

    lambda_risk = float(qcfg["risk"]["lambda_risk"])

    Rmin_share_baseline = float(qcfg["regulation"]["Rmin_share_baseline"])
    Rmin_share_tight = float(qcfg["regulation"]["Rmin_share_tight"])
    use_tight = bool(qcfg["regulation"]["use_tight"])

    # Affordability block exists for schema stability across scenarios
    affordability_enabled = bool(qcfg["affordability"]["enabled"])
    disallow_premium_bands = list(qcfg["affordability"]["disallow_premium_bands"])
    disallow_penalty_multiplier = float(qcfg["affordability"]["disallow_penalty_multiplier"])

    # Penalties: independent 1-factor sensitivity via multipliers
    scale_factor = float(qcfg["penalties"]["scale_factor"])
    onehot_mult = float(qcfg["penalties"].get("onehot_multiplier", 1.0))
    reg_mult = float(qcfg["penalties"].get("reg_multiplier", 1.0))

    scenario_name = str(qcfg["scenario"]["name"])

    # Load data for this SEED
    DATA_DIR = PROJECT_ROOT / "data" / "seeds" / f"seed_{SEED}"
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    c_df = pd.read_csv(DATA_DIR / "c_vector.csv")
    if "expected_cost" not in c_df.columns:
        raise ValueError("c_vector.csv must contain column 'expected_cost'")
    c_all = c_df["expected_cost"].to_numpy(dtype=float)
    if len(c_all) < N:
        raise ValueError(f"c_vector has only {len(c_all)} entries but N={N}")

    Sigma_all = np.load(DATA_DIR / "sigma_matrix.npz")["sigma"]
    if Sigma_all.shape[0] < N or Sigma_all.shape[1] < N:
        raise ValueError(f"sigma_matrix is {Sigma_all.shape} but needs at least ({N},{N})")

    regulatory = pd.read_csv(DATA_DIR / "regulatory_set.csv")
    if "feature_id" not in regulatory.columns:
        raise ValueError("regulatory_set.csv must contain column 'feature_id'")
    R_feature_ids_1based = regulatory["feature_id"].astype(int).tolist()
    R_set = set(R_feature_ids_1based)

    c = c_all[:N]
    Sigma = Sigma_all[:N, :N]

    # Regulatory indices and Rmin
    R_indices_zero_based = [i for i in range(N) if (i + 1) in R_set]
    R_size = len(R_indices_zero_based)

    Rmin_share = Rmin_share_tight if use_tight else Rmin_share_baseline
    Rmin = int(math.ceil(Rmin_share * R_size)) if R_size > 0 else 0

    # Penalty scaling baseline
    C_scale = float(np.sum(c))
    if C_scale <= 0:
        raise ValueError("C_scale must be positive. Check c_vector generation.")

    # --- CLEAN SENSITIVITY DECOUPLING ---
    # A_base = scale_factor * C_scale is the base feasibility scale.
    # Multipliers (onehot_mult / reg_mult) move A and B independently.
    # 
    # AFFORDABILITY PENALTY DESIGN (EJOR documentation):
    # D is intentionally tied to A_base (NOT A_onehot) to ensure that when we perform
    # 1-factor sensitivity on onehot_multiplier (varying A), the affordability penalty D
    # remains constant relative to the baseline problem scale. This design choice enables
    # clean interpretation of sensitivity results:
    #   - Varying onehot_mult affects ONLY the one-hot constraint penalties (A)
    #   - D remains proportional to the baseline objective magnitude (C_scale)
    #   - This prevents confounding when analyzing the effect of A on feasibility
    #
    # Mathematically: D = disallow_penalty_multiplier * A_base
    #                 where A_base = scale_factor * C_scale
    #                 and A = A_base * onehot_mult (can vary independently)
    A_base = float(scale_factor * C_scale)
    A = float(A_base * onehot_mult)   # one-hot penalty
    B = float(A_base * reg_mult)      # regulation penalty

    # Slack bits for surplus T >= 0 in (S - T - Rmin)^2
    if Rmin > 0:
        L = int(math.ceil(math.log2(Rmin + 1)))
    else:
        L = 0

    if Rmin > 0 and R_size == 0:
        raise ValueError("Rmin > 0 but R_size == 0. Check regulatory_set.csv and Rmin_share settings.")

    # Variable ordering v = [x_1..x_N, y_1..y_M, z_1..z_K, t_0..t_{L-1}]
    def idx_x(i0): return i0
    def idx_y(j0): return N + j0
    def idx_z(k0): return N + M + k0
    def idx_t(l0): return N + M + K + l0

    n = N + M + K + L
    Q = np.zeros((n, n), dtype=float)

    # Objective: sum c_i x_i + lambda * x^T Sigma x - sum premium_k z_k
    for i in range(N):
        Q[idx_x(i), idx_x(i)] += float(c[i])

    if lambda_risk != 0.0:
        for i in range(N):
            Q[idx_x(i), idx_x(i)] += float(lambda_risk * Sigma[i, i])
        for i in range(N):
            for j in range(i + 1, N):
                Q[idx_x(i), idx_x(j)] += float(lambda_risk * Sigma[i, j])

    for k in range(K):
        Q[idx_z(k), idx_z(k)] += -float(premium_bands[k])

    # One-hot constraints (soft)
    add_onehot(Q, [idx_z(k) for k in range(K)], A)
    add_onehot(Q, [idx_y(j) for j in range(M)], A)

    # Regulation: (S - T - Rmin)^2  (skipped if Rmin == 0)
    if R_size > 0 and Rmin > 0 and L > 0:
        idx_list = []
        coef_list = []
        for i0 in R_indices_zero_based:
            idx_list.append(idx_x(i0))
            coef_list.append(1.0)
        for l0 in range(L):
            idx_list.append(idx_t(l0))
            coef_list.append(-float(2 ** l0))
        add_square_linear_form(Q, idx_list, coef_list, const=-float(Rmin), weight=B)

    # Affordability: soft disallow bands by adding +D to that z_k diagonal
    # (See comment above about D being tied to A_base for sensitivity decoupling)
    D_affordability = None
    if affordability_enabled:
        D_affordability = float(disallow_penalty_multiplier * A_base)  # decoupled from A sensitivity
        for band in disallow_premium_bands:
            k0 = int(band) - 1
            if k0 < 0 or k0 >= K:
                raise ValueError(f"Invalid disallowed premium band {band}; must be in 1..{K}")
            Q[idx_z(k0), idx_z(k0)] += float(D_affordability)

    # Symmetrize by copying (NO averaging)
    Q = _symmetrize_copy_upper(Q)

    # Clean tiny entries
    Q[np.abs(Q) < 1e-12] = 0.0

    # EJOR ADDITION: Compute condition number and eigenvalue spectrum
    qubo_spectrum = _compute_qubo_condition_number(Q)

    qubo_out = DATA_DIR / f"qubo_Q_{scenario_name}_N{N}.npz"
    idx_out = DATA_DIR / f"index_map_{scenario_name}_N{N}.json"

    if (not args.overwrite) and (qubo_out.exists() or idx_out.exists()):
        print(f"QUBO or index already exists for scenario={scenario_name} seed={SEED} N={N}")
        print(f"  {qubo_out.name}")
        print(f"  {idx_out.name}")
        print("Use --overwrite to regenerate.")
        return

    np.savez_compressed(qubo_out, Q=Q)

    # Diagnostics (dominance sanity check)
    premium_max_abs = float(np.max(np.abs(premium_bands))) if K > 0 else 0.0
    sigma_abs_sum = float(np.sum(np.abs(Sigma)))
    obj_mag_bound = float(np.sum(np.abs(c)) + abs(lambda_risk) * sigma_abs_sum + premium_max_abs)
    if obj_mag_bound <= 0:
        obj_mag_bound = 1.0

    diag = {
        "cfg_hash_sha256": cfg_hash,
        "C_scale": float(C_scale),
        "scale_factor": float(scale_factor),
        "A_base": float(A_base),
        "A_onehot": float(A),
        "B_reg": float(B),
        "onehot_multiplier": float(onehot_mult),
        "reg_multiplier": float(reg_mult),
        "objective_magnitude_bound": float(obj_mag_bound),
        "A_over_obj_bound": float(A / obj_mag_bound),
        "B_over_obj_bound": float(B / obj_mag_bound),
        "premium_max_abs": float(premium_max_abs),
        "sigma_abs_sum": float(sigma_abs_sum),
        "R_size": int(R_size),
        "Rmin": int(Rmin),
        "affordability_enabled": bool(affordability_enabled),
        "disallow_penalty_multiplier": float(disallow_penalty_multiplier),
        "D_is_scaled_to_A_base_not_A": True,
        # EJOR: Structural analysis for H1
        **qubo_spectrum,
    }
    if affordability_enabled and D_affordability is not None:
        diag["D_affordability"] = float(D_affordability)
        diag["D_over_premium_max_abs"] = float(D_affordability / max(1e-12, premium_max_abs))
        diag["D_over_obj_bound"] = float(D_affordability / max(1e-12, obj_mag_bound))

    index_map = {
        "scenario": scenario_name,
        "scenario_file": cfg_path.name,
        "seed": int(SEED),
        "ordering": "v = [x_1..x_N, y_1..y_M, z_1..z_K, t_0..t_{L-1}]",
        "N": int(N), "M": int(M), "K": int(K), "L_reg_slack": int(L),
        "n_total": int(n),
        "x_range": [0, N - 1],
        "y_range": [N, N + M - 1],
        "z_range": [N + M, N + M + K - 1],
        "t_range": [N + M + K, N + M + K + L - 1] if L > 0 else None,

        "C_scale": float(C_scale),
        "scale_factor": float(scale_factor),
        "onehot_multiplier": float(onehot_mult),
        "reg_multiplier": float(reg_mult),
        "A_base": float(A_base),
        "A_onehot": float(A),
        "B_reg": float(B),
        "lambda_risk": float(lambda_risk),

        "R_size": int(R_size),
        "Rmin": int(Rmin),
        "R_indices_zero_based": [int(i) for i in R_indices_zero_based],
        "R_feature_ids_1based": [int(i) for i in R_feature_ids_1based],

        "premium_bands": premium_bands.tolist(),
        "deductible_bands": deductible_bands.tolist(),

        "affordability_enabled": bool(affordability_enabled),
        "disallow_premium_bands": [int(b) for b in disallow_premium_bands] if affordability_enabled else [],
        "disallow_penalty_multiplier": float(disallow_penalty_multiplier),
        "D_affordability": float(D_affordability) if D_affordability is not None else None,

        "use_tight_regulation": bool(use_tight),
        "qubo_file": qubo_out.name,
        "regulation_encoding": "surplus T>=0 with (sum_{i in R} x_i) - (sum_l 2^l t_l) = Rmin",
        "diagnostics": diag,
    }

    with open(idx_out, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)

    print("QUBO construction completed.")
    print(
        f"scenario={scenario_name} seed={SEED} N={N} n_total={n} "
        f"(M={M},K={K},L={L}) lambda={lambda_risk} A={A:.2f} B={B:.2f} C_scale={C_scale:.2f} "
        f"(onehot_mult={onehot_mult:.3f}, reg_mult={reg_mult:.3f})"
    )
    print(
        f"Penalty diagnostics: A/obj_bound={diag['A_over_obj_bound']:.3f} "
        f"| B/obj_bound={diag['B_over_obj_bound']:.3f} | obj_bound={obj_mag_bound:.2f}"
    )
    if affordability_enabled and D_affordability is not None:
        print(
            f"Affordability enabled: bands {index_map['disallow_premium_bands']} "
            f"| D={D_affordability:.2f} | D/premium_max={diag.get('D_over_premium_max_abs'):.3f}"
        )
    
    # EJOR: Report structural analysis
    if qubo_spectrum.get("Q_condition_number") is not None:
        print(
            f"QUBO structural analysis: condition_number={qubo_spectrum['Q_condition_number']:.2e} "
            f"| eig_range=[{qubo_spectrum['Q_eigenvalue_min']:.2e}, {qubo_spectrum['Q_eigenvalue_max']:.2e}]"
        )
    
    print(f"Saved QUBO:  {qubo_out.name}")
    print(f"Saved index: {idx_out.name}")

if __name__ == "__main__":
    main()