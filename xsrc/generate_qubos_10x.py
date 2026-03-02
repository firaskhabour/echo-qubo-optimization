"""
Generate QUBOs with 10x Penalty Scaling - Complete Version
===========================================================

This script:
1. Loads instance data (c, Sigma, premium_bands, etc.)
2. Calculates objective scale
3. Sets penalties = 10x objective scale (instead of eigenvalue-based)
4. Builds Q_base, Q_onehot, Q_reg, Q_aff SEPARATELY
5. Combines them: Q = Q_base + w*(Q_onehot + Q_reg + Q_aff)
6. Saves BOTH full Q AND components (critical for SHO!)

Usage:
    # Generate 20 instances for testing
    python generate_all_10x_qubos.py --scenario 1 --N 20 --count 20
    
    # Generate all instances for a scenario/size
    python generate_all_10x_qubos.py --scenario 1 --N 20 --all
    
    # Generate everything (720 instances)
    python generate_all_10x_qubos.py --all-scenarios
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def calculate_objective_scale(c, Sigma, lambda_risk, premium_bands):
    """Calculate natural scale of raw objective."""
    cost_scale = float(np.max(np.abs(c)))
    risk_scale = abs(lambda_risk) * float(np.linalg.norm(Sigma))
    premium_scale = float(np.max(np.abs(premium_bands)))
    return max(cost_scale, risk_scale, premium_scale)


def build_Q_base(c, Sigma, lambda_risk, premium_bands, N, M, K):
    """
    Build raw objective matrix (no penalties).
    
    Objective: c'x + lambda*x'Σx - premium
    """
    n = N + M + K  # No slack bits in Q_base
    Q_base = np.zeros((n, n), dtype=float)
    
    # Cost term
    for i in range(N):
        Q_base[i, i] += float(c[i])
    
    # Risk term
    if lambda_risk != 0.0:
        for i in range(N):
            Q_base[i, i] += float(lambda_risk * Sigma[i, i])
        for i in range(N):
            for j in range(i + 1, N):
                Q_base[i, j] += float(lambda_risk * Sigma[i, j])
    
    # Premium term
    for k in range(K):
        Q_base[N + M + k, N + M + k] += -float(premium_bands[k])
    
    return Q_base


def build_Q_onehot(N, M, K):
    """
    Build one-hot constraint matrices (unit weight).
    
    Penalty: (1 - sum y_j)^2 + (1 - sum z_k)^2
    """
    n = N + M + K
    Q_oh = np.zeros((n, n), dtype=float)
    
    # One-hot for y (deductible)
    y_start = N
    for i in range(M):
        Q_oh[y_start + i, y_start + i] = -1.0  # Diagonal
        for j in range(i + 1, M):
            Q_oh[y_start + i, y_start + j] = 1.0  # Off-diagonal
    
    # One-hot for z (premium)
    z_start = N + M
    for i in range(K):
        Q_oh[z_start + i, z_start + i] = -1.0  # Diagonal
        for j in range(i + 1, K):
            Q_oh[z_start + i, z_start + j] = 1.0  # Off-diagonal
    
    return Q_oh


def build_Q_reg(N, M, K, L, R_indices, Rmin):
    """
    Build regulatory constraint matrix (unit weight).
    
    Penalty: (sum_{i in R} x_i - sum_l 2^l t_l - Rmin)^2
    """
    n = N + M + K + L
    Q_reg = np.zeros((n, n), dtype=float)
    
    if len(R_indices) == 0 or Rmin == 0 or L == 0:
        return Q_reg[:N+M+K, :N+M+K]  # Return without slack bits
    
    # Coefficients: +1 for x_i in R, -2^l for t_l
    idx_list = []
    coef_list = []
    
    for i in R_indices:
        idx_list.append(i)
        coef_list.append(1.0)
    
    for l in range(L):
        idx_list.append(N + M + K + l)
        coef_list.append(-float(2**l))
    
    const = -float(Rmin)
    
    # Build (const + sum coef*v)^2
    # Diagonal: 2*const*coef + coef^2
    for i, c in zip(idx_list, coef_list):
        Q_reg[i, i] += 2.0 * const * c + c * c
    
    # Off-diagonal: coef_i * coef_j
    for p in range(len(idx_list)):
        for q in range(p + 1, len(idx_list)):
            Q_reg[idx_list[p], idx_list[q]] += coef_list[p] * coef_list[q]
    
    return Q_reg


def build_Q_aff(N, M, K, affordability_enabled, disallow_bands):
    """
    Build affordability constraint matrix (unit weight).
    
    Penalty: indicator(z_k selected) for k in disallow_bands
    """
    n = N + M + K
    Q_aff = np.zeros((n, n), dtype=float)
    
    if not affordability_enabled or not disallow_bands:
        return Q_aff
    
    z_start = N + M
    for band in disallow_bands:
        k0 = int(band) - 1
        if 0 <= k0 < K:
            Q_aff[z_start + k0, z_start + k0] += 1.0
    
    return Q_aff


def symmetrize(Q):
    """Copy upper triangle to lower triangle."""
    iu = np.triu_indices_from(Q, k=1)
    Q[iu[1], iu[0]] = Q[iu]
    return Q


def generate_one_qubo_10x(scenario_id, seed, N, multiplier, project_root, output_dir):
    """Generate single QUBO with 10x penalties and save components."""
    
    # Load config
    cfg_path = project_root / "config" / f"config_qubo_S{scenario_id}.yaml"
    with open(cfg_path, 'r') as f:
        qcfg = yaml.safe_load(f)
    
    # Parameters
    premium_bands = np.array(qcfg["pricing"]["premium_bands"], dtype=float)
    deductible_bands = np.array(qcfg["deductibles"]["deductible_bands"], dtype=float)
    K = len(premium_bands)
    M = len(deductible_bands)
    lambda_risk = float(qcfg["risk"]["lambda_risk"])
    
    Rmin_share_baseline = float(qcfg["regulation"]["Rmin_share_baseline"])
    Rmin_share_tight = float(qcfg["regulation"]["Rmin_share_tight"])
    use_tight = bool(qcfg["regulation"]["use_tight"])
    
    affordability_enabled = bool(qcfg["affordability"]["enabled"])
    disallow_bands = list(qcfg["affordability"]["disallow_premium_bands"])
    disallow_mult = float(qcfg["affordability"]["disallow_penalty_multiplier"])
    
    scenario_name = str(qcfg["scenario"]["name"])
    
    # Load data
    DATA_DIR = project_root / "data" / "seeds" / f"seed_{seed}"
    
    c_df = pd.read_csv(DATA_DIR / "c_vector.csv")
    c = c_df["expected_cost"].to_numpy(dtype=float)[:N]
    
    Sigma_data = np.load(DATA_DIR / "sigma_matrix.npz")
    Sigma = Sigma_data["sigma"][:N, :N]
    
    regulatory = pd.read_csv(DATA_DIR / "regulatory_set.csv")
    R_feature_ids = regulatory["feature_id"].astype(int).tolist()
    R_indices = [i for i in range(N) if (i+1) in R_feature_ids]
    R_size = len(R_indices)
    
    Rmin_share = Rmin_share_tight if use_tight else Rmin_share_baseline
    Rmin = int(math.ceil(Rmin_share * R_size)) if R_size > 0 else 0
    
    # Slack bits
    L = int(math.ceil(math.log2(Rmin + 1))) if Rmin > 0 else 0
    
    # ========================================================================
    # CALCULATE 10X PENALTIES
    # ========================================================================
    obj_scale = calculate_objective_scale(c, Sigma, lambda_risk, premium_bands)
    w_penalty = multiplier * obj_scale  # e.g., 10 * 1500 = 15,000
    
    # ========================================================================
    # BUILD COMPONENTS SEPARATELY
    # ========================================================================
    Q_base = build_Q_base(c, Sigma, lambda_risk, premium_bands, N, M, K)
    Q_onehot = build_Q_onehot(N, M, K)
    Q_reg = build_Q_reg(N, M, K, L, R_indices, Rmin)
    Q_aff = build_Q_aff(N, M, K, affordability_enabled, disallow_bands)
    
    # Affordability penalty
    D_aff = disallow_mult * w_penalty if affordability_enabled else 0.0
    
    # Expand to full size (N+M+K+L)
    n_full = N + M + K + L
    
    def expand(Q_small):
        """Expand matrix to include slack bits."""
        if Q_small.shape[0] == n_full:
            return Q_small
        Q_full = np.zeros((n_full, n_full), dtype=float)
        n_small = Q_small.shape[0]
        Q_full[:n_small, :n_small] = Q_small
        return Q_full
    
    Q_base_full = expand(Q_base)
    Q_onehot_full = expand(Q_onehot)
    Q_reg_full = Q_reg if Q_reg.shape[0] == n_full else expand(Q_reg)
    Q_aff_full = expand(Q_aff)
    
    # ========================================================================
    # COMBINE
    # ========================================================================
    Q_full = (Q_base_full + 
              w_penalty * Q_onehot_full +
              w_penalty * Q_reg_full +
              D_aff * Q_aff_full)
    
    # Symmetrize
    Q_full = symmetrize(Q_full)
    Q_full[np.abs(Q_full) < 1e-12] = 0.0
    
    # ========================================================================
    # SAVE WITH COMPONENTS
    # ========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    qubo_file = output_dir / f"qubo_10x_{scenario_name}_seed{seed}_N{N}.npz"
    
    np.savez_compressed(
        qubo_file,
        # Full QUBO
        Q=Q_full,
        # Components (critical for SHO!)
        Q_base=Q_base_full,
        Q_onehot=Q_onehot_full,
        Q_reg=Q_reg_full,
        Q_aff=Q_aff_full,
        # Weights
        w_onehot=w_penalty,
        w_reg=w_penalty,
        w_aff=D_aff,
        # Metadata
        multiplier=multiplier,
        obj_scale=obj_scale,
        penalty_type='objective_10x',
        N=N, M=M, K=K, L=L,
        seed=seed,
        scenario_name=scenario_name
    )
    
    # Save index map
    index_file = output_dir / f"index_10x_{scenario_name}_seed{seed}_N{N}.json"
    index_map = {
        'scenario': scenario_name,
        'seed': seed,
        'N': N, 'M': M, 'K': K, 'L': L,
        'n_total': n_full,
        'penalty_type': 'objective_10x',
        'multiplier': multiplier,
        'obj_scale': obj_scale,
        'w_onehot': w_penalty,
        'w_reg': w_penalty,
        'w_aff': D_aff,
        'Rmin': Rmin,
        'R_indices': R_indices,
        'premium_bands': premium_bands.tolist(),
        'deductible_bands': deductible_bands.tolist(),
        'lambda_risk': lambda_risk,
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_map, f, indent=2)
    
    return {
        'seed': seed,
        'N': N,
        'obj_scale': obj_scale,
        'w_penalty': w_penalty,
        'n': n_full
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4], default=1)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--count', type=int, default=20, help='Number of instances')
    parser.add_argument('--multiplier', type=float, default=10.0)
    parser.add_argument('--all-scenarios', action='store_true', 
                       help='Generate all 720 instances')
    args = parser.parse_args()
    
    project_root = Path.cwd()
    output_dir = project_root / "data" / "qubos_10x"
    
    print("="*80)
    print("GENERATING QUBOs WITH 10X PENALTY SCALING")
    print("="*80)
    
    if args.all_scenarios:
        # Generate all 720
        configs = [
            (1, 20, 36), (1, 30, 36), (1, 40, 36), (1, 50, 36), (1, 100, 36),
            (1, 150, 36), (1, 200, 36), (1, 250, 36), (1, 300, 36), (1, 350, 36),
            # Add S2, S3, S4 if needed...
        ]
        total = sum(count for _, _, count in configs)
        print(f"Generating ALL {total} instances...")
        
    else:
        # Generate subset
        configs = [(args.scenario, args.N, args.count)]
        print(f"Scenario: S{args.scenario}")
        print(f"N: {args.N}")
        print(f"Count: {args.count}")
    
    print(f"Multiplier: {args.multiplier}x")
    print(f"Output: {output_dir}")
    print()
    
    all_results = []
    
    for scenario_id, N, count in configs:
        seeds = list(range(1000, 1000 + count))
        
        for seed in tqdm(seeds, desc=f"S{scenario_id} N={N}"):
            try:
                result = generate_one_qubo_10x(
                    scenario_id, seed, N,
                    args.multiplier,
                    project_root, output_dir
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ Error on S{scenario_id} seed={seed} N={N}: {e}")
                continue
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✅ Generated {len(all_results)} QUBOs")
        print(f"\nObjective scale: {df['obj_scale'].median():.0f} (median)")
        print(f"10x Penalty:     {df['w_penalty'].median():.0f} (median)")
        print(f"\nFiles in: {output_dir}/")
        print("\n✅ Ready for SHO!")
        print(f"\nNext: python run_sho_on_all_10x.py")


if __name__ == "__main__":
    main()