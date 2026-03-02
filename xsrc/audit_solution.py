import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(project_root: Path, scenario_id: int) -> tuple[Path, dict]:
    cfg_path = project_root / "config" / f"config_qubo_S{scenario_id}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing scenario config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is not a YAML mapping: {cfg_path}")
    return cfg_path, cfg


def main():
    parser = argparse.ArgumentParser(description="Audit QUBO solution feasibility and term breakdown.")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], required=True, help="Scenario number (1..4)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default from config)")
    parser.add_argument("--N", type=int, default=None, help="Override N (default from config)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path, cfg = load_config(project_root, args.scenario)

    scenario_name = str(_get(cfg, ["scenario", "name"], f"S{args.scenario}"))
    seed = int(args.seed) if args.seed is not None else int(_get(cfg, ["data", "seed"], 1000))
    N = int(args.N) if args.N is not None else int(_get(cfg, ["features", "N"]))
    premium_bands = np.array(_get(cfg, ["pricing", "premium_bands"]), dtype=float)
    deductible_bands = np.array(_get(cfg, ["deductibles", "deductible_bands"]), dtype=float)
    K = len(premium_bands)
    M = len(deductible_bands)
    lambda_risk = float(_get(cfg, ["risk", "lambda_risk"], 0.0))

    data_dir = project_root / "data" / "seeds" / f"seed_{seed}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    # Load QUBO (scenario-specific)
    q_path = data_dir / f"qubo_Q_{scenario_name}.npz"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing QUBO file: {q_path} (run build_qubo for this scenario)")
    Q = np.load(q_path)["Q"]

    n_total = int(Q.shape[0])
    L = n_total - (N + M + K)
    if L < 0:
        raise ValueError("Q has fewer variables than N+M+K; inconsistent build/inputs.")

    # Load extracted parameters for decomposition
    c_df = pd.read_csv(data_dir / "c_vector.csv")
    c_all = c_df["expected_cost"].to_numpy(dtype=float)
    c = c_all[:N]

    Sigma_all = np.load(data_dir / "sigma_matrix.npz")["sigma"]
    Sigma = Sigma_all[:N, :N]

    # Regulatory set
    regulatory = pd.read_csv(data_dir / "regulatory_set.csv")
    R_set = set(regulatory["feature_id"].astype(int).tolist())
    R_zero = [i for i in range(N) if (i + 1) in R_set]
    R_size = len(R_zero)

    # penalties
    scale_factor = float(_get(cfg, ["penalties", "scale_factor"], 10.0))
    C_scale = float(np.sum(c))
    A = scale_factor * C_scale
    B = scale_factor * C_scale

    use_tight = bool(_get(cfg, ["regulation", "use_tight"], False))
    Rmin_share_baseline = float(_get(cfg, ["regulation", "Rmin_share_baseline"], 0.60))
    Rmin_share_tight = float(_get(cfg, ["regulation", "Rmin_share_tight"], 0.85))
    Rmin_share = Rmin_share_tight if use_tight else Rmin_share_baseline
    Rmin = int(math.ceil(Rmin_share * R_size)) if R_size > 0 else 0

    affordability_enabled = bool(_get(cfg, ["affordability", "enabled"], False))
    disallow_premium_bands = list(_get(cfg, ["affordability", "disallow_premium_bands"], []))
    disallow_mult = float(_get(cfg, ["affordability", "disallow_penalty_multiplier"], 50.0))
    D = disallow_mult * A

    # Load solution JSON
    sol_path = project_root / "results" / f"solution_{scenario_name}_seed{seed}_N{N}.json"
    if not sol_path.exists():
        raise FileNotFoundError(f"Missing solution file: {sol_path}")

    with open(sol_path, "r", encoding="utf-8") as f:
        sol = json.load(f)

    selected_ids = sol.get("selected_feature_ids", [])
    chosen_ded = int(sol.get("chosen_deductible_band", 1))
    chosen_prem = int(sol.get("chosen_premium_band", 1))
    e_best_reported = sol.get("energy_best", None)

    # Build v from solution including surplus bits if present
    v = np.zeros(n_total, dtype=float)

    for fid in selected_ids:
        fid = int(fid)
        if 1 <= fid <= N:
            v[fid - 1] = 1.0

    if not (1 <= chosen_ded <= M):
        raise ValueError(f"chosen_deductible_band={chosen_ded} out of range 1..{M}")
    v[N + (chosen_ded - 1)] = 1.0

    if not (1 <= chosen_prem <= K):
        raise ValueError(f"chosen_premium_band={chosen_prem} out of range 1..{K}")
    v[N + M + (chosen_prem - 1)] = 1.0

    # load surplus bits
    bits = sol.get("reg_surplus_bits", None)
    if bits is not None and isinstance(bits, list) and L > 0:
        for l in range(min(L, len(bits))):
            v[N + M + K + l] = 1.0 if int(bits[l]) == 1 else 0.0

    # QUBO energy
    e_qubo = float(v @ Q @ v)

    # blocks
    x = v[:N]
    y = v[N:N + M]
    z = v[N + M:N + M + K]
    t_bits = v[N + M + K:] if L > 0 else np.array([], dtype=float)

    # compute surplus value
    T_surplus = float(sum((2 ** l) * t_bits[l] for l in range(L))) if L > 0 else 0.0

    # Terms (intuitive, model-level)
    cost_term = float(np.dot(c, x))
    risk_term = float(lambda_risk * (x @ Sigma @ x))
    premium_term = float(-np.dot(premium_bands[:K], z))

    # Onehot penalties (paper-level)
    y_sum = float(y.sum())
    z_sum = float(z.sum())
    onehot_y_pen = float(A * (1.0 - y_sum) ** 2)
    onehot_z_pen = float(A * (1.0 - z_sum) ** 2)

    # Regulation
    S_reg = float(x[R_zero].sum()) if R_size > 0 else 0.0
    reg_feasible = (S_reg + 1e-9) >= float(Rmin)

    # Paper-aligned hinge penalty: B * max(0, Rmin - S)^2
    violation = max(0.0, float(Rmin) - S_reg)
    reg_hinge_pen = float(B * (violation ** 2))

    # Engineering/QUBO encoding residual (surplus model): S - T - Rmin
    reg_eq_val = float(S_reg - T_surplus - float(Rmin))

    # Affordability
    disallowed_hit = []
    afford_pen = 0.0
    if affordability_enabled and disallow_premium_bands:
        for band in disallow_premium_bands:
            k = int(band)
            if 1 <= k <= K and z[k - 1] > 0.5:
                disallowed_hit.append(k)
        afford_pen = float(D * len(disallowed_hit))

    # Print
    print("\n==================== AUDIT ====================")
    print(f"Scenario:         {scenario_name}  (config: {cfg_path.name})")
    print(f"Seed:             {seed}")
    print(f"N/M/K/L:          {N}/{M}/{K}/{L}   (n={n_total})")
    print("----------------------------------------------")
    print("Feasibility checks")
    print(f"  Onehot y sum:   {y_sum:.0f}   (chosen deductible band={chosen_ded})")
    print(f"  Onehot z sum:   {z_sum:.0f}   (chosen premium band={chosen_prem})")
    if affordability_enabled:
        print(f"  Affordability:  enabled, disallowed={disallow_premium_bands}, hit={disallowed_hit}")
    else:
        print("  Affordability:  disabled")

    print("----------------------------------------------")
    print("Regulation checks (paper-aligned inequality)")
    print(f"  |R|:            {R_size}")
    print(f"  Rmin share:     {Rmin_share:.2f}  (use_tight={use_tight})")
    print(f"  Rmin:           {Rmin}")
    print(f"  Selected in R:  {S_reg:.0f}   (feasible S>=Rmin: {reg_feasible})")
    print(f"  Hinge violation max(0,Rmin-S): {violation:.3f}")
    print(f"  Hinge penalty (paper):         {reg_hinge_pen:.6f}")
    print(f"  Surplus T (encoding):          {T_surplus:.0f}   (bits={ [int(b>0.5) for b in t_bits] })")
    print(f"  Encoding residual S-T-Rmin:    {reg_eq_val:.3f}")

    print("----------------------------------------------")
    print("Objective/penalty parameters")
    print(f"  lambda_risk:    {lambda_risk}")
    print(f"  C_scale:        {C_scale:.4f}")
    print(f"  A (onehot):     {A:.4f}")
    print(f"  B (reg):        {B:.4f}")
    if affordability_enabled:
        print(f"  D (afford):     {D:.4f}  (mult={disallow_mult})")

    print("----------------------------------------------")
    print("Term breakdown (model-level, intuitive)")
    print(f"  Cost term:      {cost_term:.6f}")
    print(f"  Risk term:      {risk_term:.6f}")
    print(f"  Premium term:   {premium_term:.6f}")
    print(f"  Onehot y pen:   {onehot_y_pen:.6f}")
    print(f"  Onehot z pen:   {onehot_z_pen:.6f}")
    print(f"  Reg hinge pen:  {reg_hinge_pen:.6f}")
    print(f"  Afford pen:     {afford_pen:.6f}")

    print("----------------------------------------------")
    print("QUBO energy consistency")
    print(f"  Energy v^TQv:   {e_qubo:.12f}")
    if e_best_reported is not None:
        print(f"  Reported best:  {float(e_best_reported):.12f}")
        print(f"  Delta:          {e_qubo - float(e_best_reported):.12f}")
    else:
        print("  Reported best:  (missing in solution json)")

    print("==============================================\n")


if __name__ == "__main__":
    main()
