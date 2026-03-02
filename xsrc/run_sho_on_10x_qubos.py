"""
Run Improved SHO on 10x Penalty QUBOs - Complete Version
=========================================================

This script:
1. Loads 10x QUBOs WITH components (Q_base, Q_onehot, Q_reg, Q_aff)
2. Runs improved SHO properly
3. Compares SHO raw objective to existing SA raw objective
4. Saves results

Usage:
    # Test on 20 instances
    python run_sho_on_all_10x.py --scenario 1 --N 20 --count 20
    
    # Run all for one size
    python run_sho_on_all_10x.py --scenario 1 --N 20 --all
    
    # Run everything (720 instances)
    python run_sho_on_all_10x.py --all-scenarios
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Import SHO
sys.path.insert(0, 'src')
from spectral_homotopy_optimization import (
    spectral_homotopy_QUBO,
    get_default_SHO_parameters
)
from solve_classical import (
    simulated_annealing,
    energy,
    project_onehot
)


def load_10x_qubo_with_components(qubo_file, index_file):
    """Load 10x QUBO with all components."""
    data = np.load(qubo_file)
    
    with open(index_file, 'r') as f:
        index_map = json.load(f)
    
    return {
        'Q_full': data['Q'],
        'Q_base': data['Q_base'],
        'Q_onehot': data['Q_onehot'],
        'Q_reg': data['Q_reg'],
        'Q_aff': data['Q_aff'],
        'w_onehot': float(data['w_onehot']),
        'w_reg': float(data['w_reg']),
        'w_aff': float(data['w_aff']),
        'N': index_map['N'],
        'M': index_map['M'],
        'K': index_map['K'],
        'L': index_map['L'],
        'seed': index_map['seed'],
        'scenario': index_map['scenario'],
        'obj_scale': index_map['obj_scale'],
    }


def load_sa_baseline(baseline_csv, scenario, seed, N):
    """Load SA result from existing results."""
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
        'objective_raw': float(row['objective_raw_sa_best']),
        'energy': float(row['energy_sa_best']),
        'is_feasible': bool(row['is_feasible_sa']),
    }


def evaluate_raw_objective(x, y, z, c, Sigma, lambda_risk, premium_bands):
    """
    Evaluate raw business objective.
    
    f(x,y,z) = c'x + lambda*x'Σx - premium
    """
    cost = np.dot(c, x)
    risk = lambda_risk * x.T @ Sigma @ x if lambda_risk != 0 else 0.0
    premium_idx = np.argmax(z)
    premium = premium_bands[premium_idx]
    
    return float(cost + risk - premium)


def check_feasibility(x, y, z, R_indices, Rmin):
    """Check if solution is feasible."""
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


def run_sho_one_instance(qubo_file, index_file, data_dir, baseline_result, verbose=False):
    """Run improved SHO on one instance."""
    
    # Load QUBO with components
    qubo = load_10x_qubo_with_components(qubo_file, index_file)
    
    N = qubo['N']
    M = qubo['M']
    K = qubo['K']
    L = qubo['L']
    seed = qubo['seed']
    scenario = qubo['scenario']
    
    # Load instance data for objective evaluation
    seed_dir = data_dir / f"seed_{seed}"
    
    c_df = pd.read_csv(seed_dir / "c_vector.csv")
    c = c_df["expected_cost"].to_numpy(dtype=float)[:N]
    
    Sigma_data = np.load(seed_dir / "sigma_matrix.npz")
    Sigma = Sigma_data["sigma"][:N, :N]
    
    regulatory = pd.read_csv(seed_dir / "regulatory_set.csv")
    R_ids = regulatory["feature_id"].astype(int).tolist()
    R_indices = [i for i in range(N) if (i+1) in R_ids]
    
    # Load parameters from index
    with open(index_file, 'r') as f:
        idx = json.load(f)
    
    premium_bands = np.array(idx['premium_bands'])
    lambda_risk = float(idx['lambda_risk'])
    Rmin = int(idx['Rmin'])
    
    # Get improved SHO parameters
    sho_params = get_default_SHO_parameters()
    
    # SA solver wrapper
    def sa_solver_wrapper(Q_local, N_local, M_local, K_local, L_local, rng,
                         steps, T0, Tend, initial_solution=None):
        solution, e, diagnostics = simulated_annealing(
            Q=Q_local, N=N_local, M=M_local, K=K_local,
            rng=rng, steps=steps, T0=T0, Tend=Tend,
            prem_move_prob=0.075, ded_move_prob=0.075
        )
        solution = project_onehot(solution, N_local, M_local, K_local, rng)
        e = energy(Q_local, solution)
        return solution, e
    
    # Run SHO with proper components!
    if verbose:
        print(f"  Running SHO on {scenario}, seed={seed}, N={N}...")
    
    sho_result = spectral_homotopy_QUBO(
        Q_base=qubo['Q_base'],
        Q_oh=qubo['Q_onehot'],
        Q_reg=qubo['Q_reg'],
        Q_aff=qubo['Q_aff'],
        N=N, M=M, K=K, L=L,
        w_oh=qubo['w_onehot'],
        w_reg=qubo['w_reg'],
        w_aff=qubo['w_aff'],
        seed=seed,
        params=sho_params,
        SA_solver=sa_solver_wrapper,
        verbose=verbose
    )
    
    # Extract solution
    solution = sho_result['solution']
    x = solution[:N]
    y = solution[N:N+M]
    z = solution[N+M:N+M+K]
    
    # Evaluate raw objective
    sho_raw_obj = evaluate_raw_objective(
        x, y, z, c, Sigma, lambda_risk, premium_bands
    )
    
    # Check feasibility
    sho_feasible = check_feasibility(x, y, z, R_indices, Rmin)
    
    # Compare to SA
    sa_raw_obj = baseline_result['objective_raw']
    gap = sho_raw_obj - sa_raw_obj
    
    sho_wins = gap < -1e-6
    sho_ties = abs(gap) <= 1e-6
    
    return {
        'scenario': scenario,
        'seed': seed,
        'N': N,
        'M': M,
        'K': K,
        'L': L,
        'obj_scale': qubo['obj_scale'],
        'w_penalty': qubo['w_onehot'],
        # SHO results
        'sho_raw_objective': sho_raw_obj,
        'sho_energy': sho_result['energy'],
        'sho_feasible': int(sho_feasible),
        'sho_runtime': sho_result['runtime'],
        'sho_lambda_0': sho_result['lambda_0'],
        'sho_best_stage': sho_result['best_stage'],
        # SA baseline
        'sa_raw_objective': sa_raw_obj,
        'sa_energy': baseline_result['energy'],
        'sa_feasible': int(baseline_result['is_feasible']),
        # Comparison
        'gap': gap,
        'gap_pct': (gap / abs(sa_raw_obj) * 100) if sa_raw_obj != 0 else 0,
        'sho_wins': int(sho_wins),
        'sho_ties': int(sho_ties),
        'sho_loses': int(not sho_wins and not sho_ties),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4], default=1)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--baseline', type=str, default='results/results_master.csv')
    parser.add_argument('--output', type=str, default='results/sho_10x_results.csv')
    parser.add_argument('--all-scenarios', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    project_root = Path.cwd()
    qubos_dir = project_root / "data" / "qubos_10x"
    data_dir = project_root / "data" / "seeds"
    baseline_csv = Path(args.baseline)
    
    print("="*80)
    print("RUNNING IMPROVED SHO ON 10X QUBOS")
    print("="*80)
    
    if not qubos_dir.exists():
        print(f"❌ {qubos_dir} not found!")
        print(f"Run: python generate_all_10x_qubos.py first")
        return
    
    if not baseline_csv.exists():
        print(f"❌ Baseline CSV not found: {baseline_csv}")
        return
    
    # Determine instances to run
    if args.all_scenarios:
        configs = [
            (1, 20, 36), (1, 30, 36), # ... etc
        ]
    else:
        configs = [(args.scenario, args.N, args.count)]
    
    print(f"Baseline: {baseline_csv}")
    print(f"Output:   {args.output}")
    print()
    
    all_results = []
    failed = []
    
    for scenario_id, N, count in configs:
        scenario_name = f"S{scenario_id}_cost_only"
        seeds = list(range(1000, 1000 + count))
        
        for seed in tqdm(seeds, desc=f"S{scenario_id} N={N}"):
            qubo_file = qubos_dir / f"qubo_10x_{scenario_name}_seed{seed}_N{N}.npz"
            index_file = qubos_dir / f"index_10x_{scenario_name}_seed{seed}_N{N}.json"
            
            if not qubo_file.exists():
                print(f"\n⚠️  QUBO not found: {qubo_file.name}")
                failed.append((scenario_id, seed, N))
                continue
            
            # Load SA baseline
            baseline = load_sa_baseline(baseline_csv, scenario_name, seed, N)
            if baseline is None:
                print(f"\n⚠️  No SA baseline for seed={seed} N={N}")
                failed.append((scenario_id, seed, N))
                continue
            
            try:
                result = run_sho_one_instance(
                    qubo_file, index_file, data_dir, baseline, args.verbose
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
        
        wins = df['sho_wins'].sum()
        ties = df['sho_ties'].sum()
        losses = df['sho_loses'].sum()
        
        print(f"\nSHO (10x) vs SA (high penalties):")
        print(f"  Wins:   {wins:3d} / {len(df)} ({wins/len(df)*100:5.1f}%)")
        print(f"  Ties:   {ties:3d} / {len(df)} ({ties/len(df)*100:5.1f}%)")
        print(f"  Losses: {losses:3d} / {len(df)} ({losses/len(df)*100:5.1f}%)")
        
        print(f"\nGap statistics:")
        print(f"  Median: {df['gap'].median():>10,.0f}")
        print(f"  Mean:   {df['gap'].mean():>10,.0f}")
        
        print(f"\nSaved: {output_path}")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)
        win_rate = wins / len(df) * 100
        
        if win_rate >= 40:
            print(f"\n✅✅✅ EXCELLENT! {win_rate:.1f}% win rate")
            print("   10x penalty scaling WORKS!")
            print("   Strong paper contribution")
        elif win_rate >= 25:
            print(f"\n✅✅ GOOD! {win_rate:.1f}% win rate")
            print("   10x helps significantly")
            print("   Moderate paper contribution")
        elif win_rate >= 15:
            print(f"\n✅ IMPROVEMENT. {win_rate:.1f}% win rate")
            print("   Better than baseline (14%)")
            print("   But needs more improvement")
        else:
            print(f"\n❌ INSUFFICIENT. {win_rate:.1f}% win rate")
            print("   Try 5x or different approach")
    else:
        print("\n❌ No results generated")


if __name__ == "__main__":
    main()
    