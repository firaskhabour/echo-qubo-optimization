"""
Run Spectral Landscape Navigation (SLN) Experiments
===================================================

This script runs SLN on your existing high-penalty QUBOs and compares
to existing SA/Greedy/Gurobi results.

Usage:
    # Test on 5 instances
    python run_sln_experiments.py --scenario 1 --N 20 --test 5
    
    # Run on all instances for one size
    python run_sln_experiments.py --scenario 1 --N 20
    
    # Run everything (720 instances)
    python run_sln_experiments.py --all
    
    # Test multi-start
    python run_sln_experiments.py --scenario 1 --N 20 --test 5 --multistart 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, 'src')

# Import SLN
from spectral_landscape_navigation import (
    spectral_landscape_navigation,
    spectral_landscape_navigation_multistart,
    get_default_SLN_parameters
)


def load_existing_qubo(seed: int, scenario_name: str, N: int, 
                       data_dir: Path) -> dict:
    """
    Load existing high-penalty QUBO and index map.
    
    These are your original QUBOs with eigenvalue-based penalty scaling.
    """
    seed_dir = data_dir / f"seed_{seed}"
    
    # Load QUBO
    qubo_file = seed_dir / f"qubo_Q_{scenario_name}_N{N}.npz"
    if not qubo_file.exists():
        raise FileNotFoundError(f"QUBO not found: {qubo_file}")
    
    Q_data = np.load(qubo_file)
    Q_full = Q_data['Q']
    
    # Load index map
    index_file = seed_dir / f"index_map_{scenario_name}_N{N}.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index map not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index_map = json.load(f)
    
    return {
        'Q_full': Q_full,
        'index_map': index_map,
        'N': index_map['N'],
        'M': index_map['M'],
        'K': index_map['K'],
        'L': index_map.get('L_reg_slack', 0),
        'seed': seed,
        'scenario': scenario_name,
    }


def decompose_qubo(Q_full: np.ndarray, index_map: dict, 
                   c: np.ndarray, Sigma: np.ndarray, lambda_risk: float,
                   premium_bands: np.ndarray, deductible_bands: np.ndarray,
                   N: int, M: int, K: int, L: int) -> dict:
    """
    Decompose existing QUBO into Q_base and penalty matrices.
    
    Strategy:
    - Rebuild Q_base from instance data
    - Rebuild penalty matrices from structure
    - Extract weights by comparing to Q_full
    
    This allows us to use SLN on existing high-penalty QUBOs.
    """
    
    n = N + M + K + L
    
    # ========================================================================
    # BUILD Q_BASE (raw objective, no penalties)
    # ========================================================================
    
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
                Q_base[j, i] = Q_base[i, j]  # Symmetrize
    
    # Premium term
    for k in range(K):
        Q_base[N + M + k, N + M + k] += -float(premium_bands[k])
    
    # ========================================================================
    # BUILD PENALTY MATRICES (unit weight)
    # ========================================================================
    
    Q_onehot = np.zeros((n, n), dtype=float)
    
    # One-hot for y (deductible)
    y_start = N
    for i in range(M):
        Q_onehot[y_start + i, y_start + i] = -1.0
        for j in range(i + 1, M):
            Q_onehot[y_start + i, y_start + j] = 1.0
            Q_onehot[y_start + j, y_start + i] = 1.0
    
    # One-hot for z (premium)
    z_start = N + M
    for i in range(K):
        Q_onehot[z_start + i, z_start + i] = -1.0
        for j in range(i + 1, K):
            Q_onehot[z_start + i, z_start + j] = 1.0
            Q_onehot[z_start + j, z_start + i] = 1.0
    
    # Regulatory constraint matrix
    Q_reg = np.zeros((n, n), dtype=float)
    R_indices = index_map.get('R_indices_zero_based', [])
    Rmin = index_map.get('Rmin', 0)
    
    if R_indices and Rmin > 0 and L > 0:
        # Build (sum x_i - sum 2^l t_l - Rmin)^2
        idx_list = []
        coef_list = []
        
        for i in R_indices:
            idx_list.append(i)
            coef_list.append(1.0)
        
        for l in range(L):
            idx_list.append(N + M + K + l)
            coef_list.append(-float(2**l))
        
        const = -float(Rmin)
        
        # Diagonal
        for i, a in zip(idx_list, coef_list):
            Q_reg[i, i] += 2.0 * const * a + a * a
        
        # Off-diagonal
        for p in range(len(idx_list)):
            for q in range(p + 1, len(idx_list)):
                val = coef_list[p] * coef_list[q]
                Q_reg[idx_list[p], idx_list[q]] += val
                Q_reg[idx_list[q], idx_list[p]] += val
    
    # Affordability matrix (placeholder)
    Q_aff = np.zeros((n, n), dtype=float)
    
    # ========================================================================
    # EXTRACT WEIGHTS from index_map
    # ========================================================================
    
    w_onehot = float(index_map.get('A_onehot', 0.0))
    w_reg = float(index_map.get('B_reg', 0.0))
    D_aff_raw = index_map.get('D_affordability', 0.0)
    w_aff = float(D_aff_raw) if D_aff_raw is not None else 0.0
    
    return {
        'Q_base': Q_base,
        'Q_onehot': Q_onehot,
        'Q_reg': Q_reg,
        'Q_aff': Q_aff,
        'w_onehot': w_onehot,
        'w_reg': w_reg,
        'w_aff': w_aff,
    }


def load_baseline_results(baseline_csv: Path, scenario: str, seed: int, N: int) -> dict:
    """Load existing SA/Greedy/Gurobi results for comparison."""
    
    df = pd.read_csv(baseline_csv)
    mask = (
        (df['scenario_name'] == scenario) &
        (df['seed'] == seed) &
        (df['N'] == N)
    )
    
    row = df[mask]
    if len(row) == 0:
        return None
    
    row = row.iloc[0]
    
    return {
        'sa_objective': float(row['objective_raw_sa_best']),
        'sa_energy': float(row['energy_sa_best']),
        'sa_feasible': bool(row['is_feasible_sa']),
        'greedy_objective': float(row['objective_raw_greedy']),
        'greedy_energy': float(row['energy_greedy']),
        'gurobi_objective': float(row.get('objective_raw_gurobi', np.nan)),
        'gurobi_energy': float(row.get('energy_gurobi', np.nan)),
    }


def evaluate_raw_objective(solution: np.ndarray, c: np.ndarray, Sigma: np.ndarray,
                           lambda_risk: float, premium_bands: np.ndarray,
                           N: int, M: int, K: int) -> float:
    """Evaluate raw business objective."""
    
    x = solution[:N]
    z = solution[N+M:N+M+K]
    
    cost = np.dot(c, x)
    risk = lambda_risk * x.T @ Sigma @ x if lambda_risk != 0 else 0.0
    premium_idx = np.argmax(z)
    premium = premium_bands[premium_idx]
    
    return float(cost + risk - premium)


def check_feasibility(solution: np.ndarray, N: int, M: int, K: int,
                     R_indices: list, Rmin: int) -> bool:
    """Check solution feasibility."""
    
    y = solution[N:N+M]
    z = solution[N+M:N+M+K]
    x = solution[:N]
    
    # One-hot y
    y_ok = abs(np.sum(y) - 1.0) < 1e-6
    
    # One-hot z
    z_ok = abs(np.sum(z) - 1.0) < 1e-6
    
    # Regulatory
    if R_indices and Rmin > 0:
        R_sum = sum(x[i] for i in R_indices)
        reg_ok = R_sum >= (Rmin - 1e-6)
    else:
        reg_ok = True
    
    return y_ok and z_ok and reg_ok


def run_one_instance(seed: int, scenario_name: str, N: int,
                    data_dir: Path, baseline_results: dict,
                    use_multistart: bool = False, num_starts: int = 1,
                    verbose: bool = False) -> dict:
    """Run SLN on one instance."""
    
    # Load QUBO
    qubo_data = load_existing_qubo(seed, scenario_name, N, data_dir)
    Q_full = qubo_data['Q_full']
    index_map = qubo_data['index_map']
    M = qubo_data['M']
    K = qubo_data['K']
    L = qubo_data['L']
    
    # Load instance data
    seed_dir = data_dir / f"seed_{seed}"
    
    c_df = pd.read_csv(seed_dir / "c_vector.csv")
    c = c_df["expected_cost"].to_numpy(dtype=float)[:N]
    
    Sigma_data = np.load(seed_dir / "sigma_matrix.npz")
    Sigma = Sigma_data["sigma"][:N, :N]
    
    regulatory = pd.read_csv(seed_dir / "regulatory_set.csv")
    R_ids = regulatory["feature_id"].astype(int).tolist()
    R_indices = [i for i in range(N) if (i+1) in R_ids]
    
    # Get premium bands and lambda_risk from index map
    premium_bands = np.array(index_map['premium_bands'])
    deductible_bands = np.array(index_map['deductible_bands'])
    lambda_risk = float(index_map['lambda_risk'])
    Rmin = int(index_map.get('Rmin', 0))
    
    # Decompose QUBO
    decomp = decompose_qubo(
        Q_full, index_map, c, Sigma, lambda_risk,
        premium_bands, deductible_bands, N, M, K, L
    )
    
    # Run SLN
    if use_multistart:
        result = spectral_landscape_navigation_multistart(
            decomp['Q_base'], decomp['Q_onehot'], decomp['Q_reg'], decomp['Q_aff'],
            N, M, K, L,
            decomp['w_onehot'], decomp['w_reg'], decomp['w_aff'],
            seed=seed,
            num_starts=num_starts,
            params=None,
            verbose=verbose
        )
    else:
        result = spectral_landscape_navigation(
            decomp['Q_base'], decomp['Q_onehot'], decomp['Q_reg'], decomp['Q_aff'],
            N, M, K, L,
            decomp['w_onehot'], decomp['w_reg'], decomp['w_aff'],
            seed=seed,
            params=None,
            verbose=verbose
        )
    
    # Evaluate raw objective
    sln_raw_obj = evaluate_raw_objective(
        result['solution'], c, Sigma, lambda_risk, premium_bands, N, M, K
    )
    
    # Check feasibility
    sln_feasible = check_feasibility(
        result['solution'], N, M, K, R_indices, Rmin
    )
    
    # Compare to baselines
    gap_to_sa = sln_raw_obj - baseline_results['sa_objective']
    gap_to_greedy = sln_raw_obj - baseline_results['greedy_objective']
    gap_to_gurobi = sln_raw_obj - baseline_results['gurobi_objective']
    
    return {
        'scenario': scenario_name,
        'seed': seed,
        'N': N, 'M': M, 'K': K, 'L': L,
        # SLN results
        'sln_raw_objective': sln_raw_obj,
        'sln_energy': result['energy'],
        'sln_feasible': int(sln_feasible),
        'sln_runtime': result['runtime'],
        'sln_lambda_0': result['lambda_0'],
        'sln_best_stage': result['best_stage'],
        'sln_num_stages': len(result['stages']),
        # Baseline comparisons
        'sa_raw_objective': baseline_results['sa_objective'],
        'sa_energy': baseline_results['sa_energy'],
        'greedy_raw_objective': baseline_results['greedy_objective'],
        'gurobi_raw_objective': baseline_results['gurobi_objective'],
        # Gaps
        'gap_to_sa': gap_to_sa,
        'gap_to_greedy': gap_to_greedy,
        'gap_to_gurobi': gap_to_gurobi,
        # Wins
        'sln_beats_sa': int(gap_to_sa < -1e-6),
        'sln_ties_sa': int(abs(gap_to_sa) <= 1e-6),
        'sln_beats_greedy': int(gap_to_greedy < -1e-6),
        'sln_matches_gurobi': int(abs(gap_to_gurobi) <= 1e-6),
        # Multistart metadata
        'multistart': use_multistart,
        'num_starts': num_starts if use_multistart else 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4], default=1)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--test', type=int, default=None, help='Test on first N instances')
    parser.add_argument('--all', action='store_true', help='Run all 720 instances')
    parser.add_argument('--multistart', type=int, default=None, 
                       help='Use multi-start with N runs')
    parser.add_argument('--baseline', type=str, default='results/results_master.csv')
    parser.add_argument('--output', type=str, default='results/sln_results.csv')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    data_dir = project_root / "data" / "seeds"
    baseline_csv = Path(args.baseline)
    
    print("="*80)
    print("RUNNING SPECTRAL LANDSCAPE NAVIGATION (SLN)")
    print("="*80)
    
    if not baseline_csv.exists():
        print(f"❌ Baseline CSV not found: {baseline_csv}")
        return
    
    # Determine instances to run
    if args.all:
        # All 720 instances
        configs = [
            (1, 20, 36), (1, 30, 36), (1, 40, 36), (1, 50, 36),
            (1, 100, 36), (1, 150, 36), (1, 200, 36), (1, 250, 36),
            (1, 300, 36), (1, 350, 36),
        ]
    else:
        count = args.test if args.test else 36
        configs = [(args.scenario, args.N, count)]
    
    use_multistart = args.multistart is not None
    num_starts = args.multistart if use_multistart else 1
    
    scenario_name = f"S{args.scenario}_cost_only"
    
    print(f"Scenario: {scenario_name}")
    print(f"Instances: {sum(c[2] for c in configs)}")
    if use_multistart:
        print(f"Multi-start: {num_starts} runs per instance")
    print(f"Baseline: {baseline_csv}")
    print(f"Output: {args.output}")
    print()
    
    all_results = []
    failed = []
    
    for scenario_id, N, count in configs:
        scenario_name = f"S{scenario_id}_cost_only"
        seeds = list(range(1000, 1000 + count))
        
        for seed in tqdm(seeds, desc=f"S{scenario_id} N={N}"):
            # Load baseline
            baseline = load_baseline_results(baseline_csv, scenario_name, seed, N)
            if baseline is None:
                print(f"\n⚠️  No baseline for seed={seed} N={N}")
                failed.append((scenario_id, seed, N))
                continue
            
            try:
                result = run_one_instance(
                    seed, scenario_name, N, data_dir, baseline,
                    use_multistart=use_multistart,
                    num_starts=num_starts,
                    verbose=args.verbose
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"\n❌ Error on seed={seed}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                failed.append((scenario_id, seed, N))
                continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"✅ Completed: {len(all_results)}")
        if failed:
            print(f"⚠️  Failed: {len(failed)}")
        
        # Performance vs SA
        wins_sa = df['sln_beats_sa'].sum()
        ties_sa = df['sln_ties_sa'].sum()
        losses_sa = len(df) - wins_sa - ties_sa
        
        print(f"\nSLN vs SA:")
        print(f"  Wins:   {wins_sa:3d} / {len(df)} ({wins_sa/len(df)*100:5.1f}%)")
        print(f"  Ties:   {ties_sa:3d} / {len(df)} ({ties_sa/len(df)*100:5.1f}%)")
        print(f"  Losses: {losses_sa:3d} / {len(df)} ({losses_sa/len(df)*100:5.1f}%)")
        
        # Performance vs Greedy
        wins_greedy = df['sln_beats_greedy'].sum()
        print(f"\nSLN vs Greedy:")
        print(f"  Wins:   {wins_greedy:3d} / {len(df)} ({wins_greedy/len(df)*100:5.1f}%)")
        
        # Performance vs Gurobi
        matches_gurobi = df['sln_matches_gurobi'].sum()
        print(f"\nSLN vs Gurobi:")
        print(f"  Matches: {matches_gurobi:3d} / {len(df)} ({matches_gurobi/len(df)*100:5.1f}%)")
        
        print(f"\nGap statistics (vs SA):")
        print(f"  Median: {df['gap_to_sa'].median():>10,.0f}")
        print(f"  Mean:   {df['gap_to_sa'].mean():>10,.0f}")
        
        print(f"\nSaved: {output_path}")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)
        
        win_rate = wins_sa / len(df) * 100
        
        if win_rate >= 40:
            print(f"\n✅✅✅ EXCELLENT! {win_rate:.1f}% win rate vs SA")
            print("   SLN is competitive!")
        elif win_rate >= 25:
            print(f"\n✅✅ GOOD! {win_rate:.1f}% win rate vs SA")
            print("   SLN shows promise")
        elif win_rate >= 15:
            print(f"\n✅ IMPROVEMENT! {win_rate:.1f}% win rate vs SA")
            print("   Better than baseline SHO (14%)")
        else:
            print(f"\n⚠️  {win_rate:.1f}% win rate vs SA")
            print("   Similar to baseline SHO")
        
    else:
        print("\n❌ No results generated")


if __name__ == "__main__":
    main()