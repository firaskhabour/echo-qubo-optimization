"""
Full ECHO Experiment - compairs baseline_full_results.csv
==================================================

Runs ECHO on ALL 520 baseline instances to compairs baseline results.

Usage:
    # Run everything (all 520 baseline instances)
    python run_echo_full.py
    
    # Run specific scenario
    python run_echo_full.py --scenario 4
    
    # Run specific scenario and APPEND to existing
    python run_echo_full.py --scenario 4 --append
    
    # Run specific size
    python run_echo_full.py --N 50
    
    # Test mode (first 5 instances)
    python run_echo_full.py --test 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add src_code to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import ECHO
from solvers.echo_optimizer import spectral_landscape_navigation  # noqa: F401


# Configuration matching BASELINE instances in Results_Run_V3.xlsx
# Total: 520 baseline instances (excludes 200 sensitivity instances)
ALL_CONFIGS = {
    # Format: (scenario_id, N): [list of seeds]
    # All scenarios use seeds 1000-1019 for N=20,30,40,50
    # All scenarios use seeds 2000-2009 for N=60,100,150,200,300
    
    # S1_cost_only (130 instances)
    (1, 20): list(range(1000, 1020)),   # 20
    (1, 30): list(range(1000, 1020)),   # 20
    (1, 40): list(range(1000, 1020)),   # 20
    (1, 50): list(range(1000, 1020)),   # 20
    (1, 60): list(range(2000, 2010)),   # 10
    (1, 100): list(range(2000, 2010)),  # 10
    (1, 150): list(range(2000, 2010)),  # 10
    (1, 200): list(range(2000, 2010)),  # 10
    (1, 300): list(range(2000, 2010)),  # 10
    
    # S2_risk_adjusted (130 instances)
    (2, 20): list(range(1000, 1020)),
    (2, 30): list(range(1000, 1020)),
    (2, 40): list(range(1000, 1020)),
    (2, 50): list(range(1000, 1020)),
    (2, 60): list(range(2000, 2010)),
    (2, 100): list(range(2000, 2010)),
    (2, 150): list(range(2000, 2010)),
    (2, 200): list(range(2000, 2010)),
    (2, 300): list(range(2000, 2010)),
    
    # S3_tight_regulation (130 instances)
    (3, 20): list(range(1000, 1020)),
    (3, 30): list(range(1000, 1020)),
    (3, 40): list(range(1000, 1020)),
    (3, 50): list(range(1000, 1020)),
    (3, 60): list(range(2000, 2010)),
    (3, 100): list(range(2000, 2010)),
    (3, 150): list(range(2000, 2010)),
    (3, 200): list(range(2000, 2010)),
    (3, 300): list(range(2000, 2010)),
    
    # S4_affordability (130 instances)
    (4, 20): list(range(1000, 1020)),
    (4, 30): list(range(1000, 1020)),
    (4, 40): list(range(1000, 1020)),
    (4, 50): list(range(1000, 1020)),
    (4, 60): list(range(2000, 2010)),
    (4, 100): list(range(2000, 2010)),
    (4, 150): list(range(2000, 2010)),
    (4, 200): list(range(2000, 2010)),
    (4, 300): list(range(2000, 2010)),
}

SCENARIO_NAMES = {
    1: 'S1_cost_only',
    2: 'S2_risk_adjusted',
    3: 'S3_tight_regulation',
    4: 'S4_affordability',
}


def load_qubo_and_decompose(seed: int, scenario_id: int, N: int, data_dir: Path):
    """Load QUBO and decompose into components for ECHO."""
    
    scenario_name = SCENARIO_NAMES[scenario_id]
    seed_dir = data_dir / f"seed_{seed}"
    
    # Load QUBO
    qubo_file = seed_dir / f"qubo_Q_{scenario_name}_N{N}.npz"
    Q_data = np.load(qubo_file)
    Q_full = Q_data['Q']
    
    # Load index map
    index_file = seed_dir / f"index_map_{scenario_name}_N{N}.json"
    with open(index_file, 'r') as f:
        index_map = json.load(f)
    
    M = index_map['M']
    K = index_map['K']
    L = index_map.get('L_reg_slack', 0)
    n = N + M + K + L
    
    # Load instance data
    c_df = pd.read_csv(seed_dir / "c_vector.csv")
    c = c_df["expected_cost"].to_numpy(dtype=float)[:N]
    
    Sigma_data = np.load(seed_dir / "sigma_matrix.npz")
    Sigma = Sigma_data["sigma"][:N, :N]
    
    regulatory = pd.read_csv(seed_dir / "regulatory_set.csv")
    R_ids = regulatory["feature_id"].astype(int).tolist()
    R_indices = [i for i in range(N) if (i+1) in R_ids]
    
    premium_bands = np.array(index_map['premium_bands'])
    lambda_risk = float(index_map['lambda_risk'])
    Rmin = int(index_map.get('Rmin', 0))
    
    # Build Q_base (objective only, no penalties)
    Q_base = np.zeros((n, n), dtype=float)
    for i in range(N):
        Q_base[i, i] += float(c[i])
    if lambda_risk != 0.0:
        for i in range(N):
            Q_base[i, i] += float(lambda_risk * Sigma[i, i])
        for i in range(N):
            for j in range(i + 1, N):
                val = float(lambda_risk * Sigma[i, j])
                Q_base[i, j] += val
                Q_base[j, i] += val
    for k in range(K):
        Q_base[N + M + k, N + M + k] += -float(premium_bands[k])
    
    # Build penalty matrices (unit weight)
    Q_onehot = np.zeros((n, n), dtype=float)
    y_start = N
    for i in range(M):
        Q_onehot[y_start + i, y_start + i] = -1.0
        for j in range(i + 1, M):
            Q_onehot[y_start + i, y_start + j] = 1.0
            Q_onehot[y_start + j, y_start + i] = 1.0
    z_start = N + M
    for i in range(K):
        Q_onehot[z_start + i, z_start + i] = -1.0
        for j in range(i + 1, K):
            Q_onehot[z_start + i, z_start + j] = 1.0
            Q_onehot[z_start + j, z_start + i] = 1.0
    
    Q_reg = np.zeros((n, n), dtype=float)
    if R_indices and Rmin > 0 and L > 0:
        idx_list = R_indices + [N + M + K + l for l in range(L)]
        coef_list = [1.0] * len(R_indices) + [-float(2**l) for l in range(L)]
        const = -float(Rmin)
        for i, a in zip(idx_list, coef_list):
            Q_reg[i, i] += 2.0 * const * a + a * a
        for p in range(len(idx_list)):
            for q in range(p + 1, len(idx_list)):
                val = coef_list[p] * coef_list[q]
                Q_reg[idx_list[p], idx_list[q]] += val
                Q_reg[idx_list[q], idx_list[p]] += val
    
    # Build affordability constraint (S4 only) - CRITICAL FIX!
    Q_aff = np.zeros((n, n), dtype=float)
    
    # S4: Forbid premium bands 4 and 5
    affordability_enabled = index_map.get('affordability_enabled', False)
    disallow_premium_bands = index_map.get('disallow_premium_bands', [])
    
    if affordability_enabled and disallow_premium_bands:
        # Add unit penalty to diagonal of forbidden premium band variables
        for band_1based in disallow_premium_bands:
            k0 = int(band_1based) - 1  # Convert to 0-indexed
            if 0 <= k0 < K:
                z_idx = N + M + k0  # z variables start at N+M
                Q_aff[z_idx, z_idx] = 1.0  # Unit weight (multiplied by w_aff later)
    
    # Extract penalty weights
    w_oh = float(index_map.get('A_onehot', 0.0))
    w_reg = float(index_map.get('B_reg', 0.0))
    
    # D_affordability might be stored under different keys
    D_raw = index_map.get('D_affordability')
    if D_raw is None:
        D_raw = index_map.get('D_afford', 0.0)
    w_aff = float(D_raw) if D_raw is not None else 0.0
    
    return {
        'Q_base': Q_base, 'Q_onehot': Q_onehot, 'Q_reg': Q_reg, 'Q_aff': Q_aff,
        'w_oh': w_oh, 'w_reg': w_reg, 'w_aff': w_aff,
        'N': N, 'M': M, 'K': K, 'L': L,
        'c': c, 'Sigma': Sigma, 'lambda_risk': lambda_risk,
        'premium_bands': premium_bands, 'R_indices': R_indices, 'Rmin': Rmin,
    }


def evaluate_solution(solution, data):
    """Evaluate solution and return all metrics."""
    N, M, K = data['N'], data['M'], data['K']
    x = solution[:N]
    y = solution[N:N+M]
    z = solution[N+M:N+M+K]
    
    # Raw objective
    cost = np.dot(data['c'], x)
    risk = data['lambda_risk'] * x.T @ data['Sigma'] @ x if data['lambda_risk'] != 0 else 0.0
    prem_idx = int(np.argmax(z))
    premium = data['premium_bands'][prem_idx]
    raw_obj = float(cost + risk - premium)
    
    # Feasibility
    y_ok = abs(np.sum(y) - 1.0) < 1e-6
    z_ok = abs(np.sum(z) - 1.0) < 1e-6
    if data['R_indices'] and data['Rmin'] > 0:
        R_sum = sum(x[i] for i in data['R_indices'])
        reg_ok = R_sum >= (data['Rmin'] - 1e-6)
    else:
        reg_ok = True
    feasible = y_ok and z_ok and reg_ok
    
    # Choices
    ded_idx = int(np.argmax(y)) + 1
    prem_idx += 1
    num_features = int(np.sum(x > 0.5))
    num_reg_features = int(sum(1 for i in data['R_indices'] if x[i] > 0.5)) if data['R_indices'] else 0
    
    return {
        'raw_objective': raw_obj,
        'feasible': feasible,
        'chosen_premium_band': prem_idx,
        'chosen_deductible_band': ded_idx,
        'selected_features_count': num_features,
        'selected_reg_features_count': num_reg_features,
    }


def run_echo_on_instance(seed, scenario_id, N, data_dir, baseline_row, verbose=False):
    """Run ECHO on one instance."""
    
    # Load and decompose QUBO
    data = load_qubo_and_decompose(seed, scenario_id, N, data_dir)
    
    # Run ECHO
    result = spectral_landscape_navigation(
        data['Q_base'], data['Q_onehot'], data['Q_reg'], data['Q_aff'],
        data['N'], data['M'], data['K'], data['L'],
        data['w_oh'], data['w_reg'], data['w_aff'],
        seed=seed,
        params=None,
        verbose=verbose
    )
    
    # Evaluate solution
    echo_eval = evaluate_solution(result['solution'], data)
    
    # Extract baseline results
    sa_raw = float(baseline_row['objective_raw_sa_best'])
    sa_energy = float(baseline_row['energy_sa_best'])
    sa_feasible = bool(baseline_row['is_feasible_sa'])
    sa_runtime = float(baseline_row['sa_runtime_sec'])
    
    greedy_raw = float(baseline_row['objective_raw_greedy'])
    greedy_energy = float(baseline_row['energy_greedy'])
    greedy_feasible = bool(baseline_row['is_feasible_greedy'])
    
    # Gurobi
    gurobi_ran = bool(baseline_row.get('gurobi_ran', False))
    if gurobi_ran:
        gurobi_raw = float(baseline_row.get('objective_raw_gurobi', np.nan))
        gurobi_energy = float(baseline_row.get('energy_gurobi', np.nan))
        gurobi_feasible = bool(baseline_row.get('is_feasible_gurobi', False))
        gurobi_status = str(baseline_row.get('gurobi_status_name', 'unknown'))
    else:
        gurobi_raw = np.nan
        gurobi_energy = np.nan
        gurobi_feasible = False
        gurobi_status = 'not_run'
    
    # Compute gaps
    gap_sa = echo_eval['raw_objective'] - sa_raw
    gap_greedy = echo_eval['raw_objective'] - greedy_raw
    gap_gurobi = echo_eval['raw_objective'] - gurobi_raw if not np.isnan(gurobi_raw) else np.nan
    
    # Win/tie/loss
    echo_beats_sa = int(gap_sa < -1e-6)
    echo_ties_sa = int(abs(gap_sa) <= 1e-6)
    echo_loses_sa = int(gap_sa > 1e-6)
    
    echo_beats_greedy = int(gap_greedy < -1e-6)
    echo_ties_greedy = int(abs(gap_greedy) <= 1e-6)
    
    if not np.isnan(gurobi_raw):
        echo_beats_gurobi = int(gap_gurobi < -1e-6)
        echo_ties_gurobi = int(abs(gap_gurobi) <= 1e-6)
        echo_loses_gurobi = int(gap_gurobi > 1e-6)
    else:
        echo_beats_gurobi = 0
        echo_ties_gurobi = 0
        echo_loses_gurobi = 0
    
    # Spectral info
    spectra = result.get('spectra', [])
    final_spec = spectra[-1] if spectra else {}
    
    return {
        # Instance ID
        'scenario_id': scenario_id,
        'scenario_name': SCENARIO_NAMES[scenario_id],
        'seed': seed,
        'N': N,
        'M': data['M'],
        'K': data['K'],
        'L': data['L'],
        
        # ECHO results
        'echo_raw_objective': echo_eval['raw_objective'],
        'echo_energy': result['energy'],
        'echo_feasible': int(echo_eval['feasible']),
        'echo_runtime_sec': result['runtime'],
        'echo_chosen_premium_band': echo_eval['chosen_premium_band'],
        'echo_chosen_deductible_band': echo_eval['chosen_deductible_band'],
        'echo_selected_features_count': echo_eval['selected_features_count'],
        'echo_selected_reg_features_count': echo_eval['selected_reg_features_count'],
        
        # ECHO algorithm details
        'echo_lambda_0': result['lambda_0'],
        'echo_num_stages': len(result['stages']),
        'echo_best_stage': result['best_stage'],
        'echo_steps_used': result['steps_used'],
        'echo_final_condition_number': final_spec.get('condition_number', np.nan),
        'echo_final_negative_eigs': final_spec.get('negative_count', 0),
        'echo_final_roughness': final_spec.get('roughness', np.nan),
        
        # Baseline results
        'sa_raw_objective': sa_raw,
        'sa_energy': sa_energy,
        'sa_feasible': int(sa_feasible),
        'sa_runtime_sec': sa_runtime,
        
        'greedy_raw_objective': greedy_raw,
        'greedy_energy': greedy_energy,
        'greedy_feasible': int(greedy_feasible),
        
        'gurobi_ran': int(gurobi_ran),
        'gurobi_raw_objective': gurobi_raw,
        'gurobi_energy': gurobi_energy,
        'gurobi_feasible': int(gurobi_feasible),
        'gurobi_status': gurobi_status,
        
        # Gaps (negative = ECHO better)
        'gap_echo_to_sa': gap_sa,
        'gap_echo_to_greedy': gap_greedy,
        'gap_echo_to_gurobi': gap_gurobi,
        
        # Performance indicators
        'echo_beats_sa': echo_beats_sa,
        'echo_ties_sa': echo_ties_sa,
        'echo_loses_sa': echo_loses_sa,
        'echo_beats_greedy': echo_beats_greedy,
        'echo_ties_greedy': echo_ties_greedy,
        'echo_beats_gurobi': echo_beats_gurobi,
        'echo_ties_gurobi': echo_ties_gurobi,
        'echo_loses_gurobi': echo_loses_gurobi,
        
        # Penalty weights
        'w_onehot': data['w_oh'],
        'w_reg': data['w_reg'],
        'w_aff': data['w_aff'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4], default=None)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--test', type=int, default=None, help='Run first N instances only')
    parser.add_argument('--baseline', type=str, default='results/baseline_full_results.csv')
    parser.add_argument('--output', type=str, default='results/echo_full_results.csv')
    parser.add_argument('--append', action='store_true', help='Append to existing output file')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    data_dir = project_root / "data" / "seeds"
    baseline_file = Path(args.baseline)
    
    print("="*80)
    print("SPECTRAL LANDSCAPE NAVIGATION - FULL EXPERIMENT")
    print("="*80)
    
    # Load baseline results (CSV only, no Excel)
    baseline_df = pd.read_csv(baseline_file)
    
    # Filter to BASELINE instances only (exclude sensitivity)
    if 'sweep_type' in baseline_df.columns:
        baseline_df = baseline_df[baseline_df['sweep_type'] == 'baseline']
    
    print(f"Loaded baseline: {baseline_file} ({len(baseline_df)} baseline instances)")

    
    # Determine which instances to run
    instances_to_run = []
    
    for (scen_id, N), seeds in ALL_CONFIGS.items():
        # Filter by command line args
        if args.scenario is not None and scen_id != args.scenario:
            continue
        if args.N is not None and N != args.N:
            continue
        
        for seed in seeds:
            instances_to_run.append((scen_id, N, seed))
    
    # Apply test limit
    if args.test:
        instances_to_run = instances_to_run[:args.test]
    
    print(f"Will run: {len(instances_to_run)} instances")
    print(f"Output: {args.output}")
    if args.append:
        print("Mode: APPEND to existing file")
    print()
    
    # Run experiments
    results = []
    failed = []
    
    # Prepare output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If append mode and file exists, load existing results
    if args.append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        print(f"Loaded existing results: {len(existing_df)} rows")
        print()
    else:
        existing_df = None
    
    for idx, (scen_id, N, seed) in enumerate(tqdm(instances_to_run, desc="Running ECHO"), 1):
        scenario_name = SCENARIO_NAMES[scen_id]
        
        # Find baseline row
        mask = (
            (baseline_df['scenario_name'] == scenario_name) &
            (baseline_df['seed'] == seed) &
            (baseline_df['N'] == N)
        )
        baseline_rows = baseline_df[mask]
        
        if len(baseline_rows) == 0:
            print(f"\n⚠️  No baseline for {scenario_name} seed={seed} N={N}")
            failed.append((scen_id, N, seed))
            continue
        
        baseline_row = baseline_rows.iloc[0]
        
        try:
            result = run_echo_on_instance(
                seed, scen_id, N, data_dir, baseline_row, args.verbose
            )
            results.append(result)
            
            # INCREMENTAL SAVE: Save after each instance
            df_new = pd.DataFrame(results)
            
            if existing_df is not None:
                # Append mode: combine with existing
                df_combined = pd.concat([existing_df, df_new], ignore_index=True)
                df_combined = df_combined.sort_values(['scenario_id', 'N', 'seed']).reset_index(drop=True)
                df_combined.to_csv(output_path, index=False)
            else:
                # Normal mode: just save new results
                df_new.to_csv(output_path, index=False)
            
            # Print progress every 10 instances
            if idx % 10 == 0:
                print(f"\n  💾 Saved checkpoint: {len(results)} instances completed")
                
        except Exception as e:
            print(f"\n❌ Error on {scenario_name} seed={seed} N={N}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed.append((scen_id, N, seed))
    
    # Final summary
    if results:
        df_new = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"✅ Completed: {len(df_new)}")
        if failed:
            print(f"⚠️  Failed: {len(failed)}")
        
        # Overall stats
        print(f"\nECHO vs SA:")
        wins = df_new['echo_beats_sa'].sum()
        ties = df_new['echo_ties_sa'].sum()
        losses = df_new['echo_loses_sa'].sum()
        print(f"  Wins:   {wins:4d} / {len(df_new)} ({wins/len(df_new)*100:5.1f}%)")
        print(f"  Ties:   {ties:4d} / {len(df_new)} ({ties/len(df_new)*100:5.1f}%)")
        print(f"  Losses: {losses:4d} / {len(df_new)} ({losses/len(df_new)*100:5.1f}%)")
        
        print(f"\nECHO vs Greedy:")
        wins_g = df_new['echo_beats_greedy'].sum()
        print(f"  Wins:   {wins_g:4d} / {len(df_new)} ({wins_g/len(df_new)*100:5.1f}%)")
        
        if df_new['gurobi_ran'].sum() > 0:
            gurobi_df = df_new[df_new['gurobi_ran'] == 1]
            matches = gurobi_df['echo_ties_gurobi'].sum()
            print(f"\nECHO vs Gurobi (where Gurobi ran):")
            print(f"  Matches: {matches:4d} / {len(gurobi_df)} ({matches/len(gurobi_df)*100:5.1f}%)")
        
        print(f"\nGap statistics (vs SA):")
        print(f"  Median: {df_new['gap_echo_to_sa'].median():>12,.1f}")
        print(f"  Mean:   {df_new['gap_echo_to_sa'].mean():>12,.1f}")
        
        # By size (if multiple sizes)
        if len(df_new['N'].unique()) > 1:
            print(f"\nPerformance by N:")
            print("-"*80)
            print(f"{'N':<6} {'Count':<8} {'Wins':<8} {'Ties':<8} {'Win%':<8} {'Median Gap':<12}")
            print("-"*80)
            for N_val in sorted(df_new['N'].unique()):
                df_n = df_new[df_new['N'] == N_val]
                w = df_n['echo_beats_sa'].sum()
                t = df_n['echo_ties_sa'].sum()
                gap = df_n['gap_echo_to_sa'].median()
                print(f"{N_val:<6} {len(df_n):<8} {w:<8} {t:<8} {w/len(df_n)*100:<8.1f} {gap:<12,.1f}")
        
        print("\n" + "="*80)
        print(f"Results saved incrementally to: {output_path}")
        if existing_df is not None:
            print(f"Total rows in file: {len(existing_df) + len(df_new)}")
        else:
            print(f"Total rows in file: {len(df_new)}")
        print("="*80)
    else:
        print("\n❌ No results generated")


if __name__ == "__main__":
    main()