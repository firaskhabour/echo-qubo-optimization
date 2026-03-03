# src_code/runners/run_echo_full.py
"""
ECHO Full Experiment Runner
============================
Runs ECHO on all 520 baseline instances and compares results against
the baseline_full_results.csv produced by run_baseline_full.py.

Each instance is identified by the triplet (scenario_id, N, seed), matching
the experimental corpus defined in Section 4.1 of the accompanying paper.
Baseline SA and greedy results are read directly from baseline_full_results.csv;
no re-running of classical solvers is performed.

Output: results/echo_full_results.csv — one row per instance, containing
ECHO results alongside the corresponding baseline metrics for direct
paired comparison.

Usage:
    # Run all 520 baseline instances
    python run_echo_full.py

    # Run a single scenario
    python run_echo_full.py --scenario 4

    # Run a single scenario and append to an existing output file
    python run_echo_full.py --scenario 4 --append

    # Run a specific problem size only
    python run_echo_full.py --N 50

    # Test mode: run only the first N instances
    python run_echo_full.py --test 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add src_code to path so solvers can be imported as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solvers.echo_optimizer import spectral_landscape_navigation  # noqa: F401


# ---------------------------------------------------------------------------
# Instance corpus definition
#
# Matches the 520 baseline instances in Section 4.1:
#   Core regime   (N ∈ {20,30,40,50}): seeds 1000–1019 (20 seeds per cell)
#   Stress regime (N ∈ {60,...,300}):  seeds 2000–2009 (10 seeds per cell)
# ---------------------------------------------------------------------------

ALL_CONFIGS = {
    # Format: (scenario_id, N): [list of seeds]

    # S1_cost_only (130 instances)
    (1, 20): list(range(1000, 1020)),
    (1, 30): list(range(1000, 1020)),
    (1, 40): list(range(1000, 1020)),
    (1, 50): list(range(1000, 1020)),
    (1, 60): list(range(2000, 2010)),
    (1, 100): list(range(2000, 2010)),
    (1, 150): list(range(2000, 2010)),
    (1, 200): list(range(2000, 2010)),
    (1, 300): list(range(2000, 2010)),

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


# ---------------------------------------------------------------------------
# QUBO loading and decomposition
# ---------------------------------------------------------------------------

def load_qubo_and_decompose(seed: int, scenario_id: int, N: int, data_dir: Path):
    """
    Load the pre-built QUBO for a given instance and decompose it into the
    four component matrices required by spectral_landscape_navigation.

    ECHO expects the full QUBO decomposed as:
        Q = Q_base + w_oh * Q_oh + w_reg * Q_reg + w_aff * Q_aff

    where Q_base captures the economic objective and Q_oh, Q_reg, Q_aff are
    unit-weight penalty matrices. This decomposition matches the formulation
    in Sections 2.7–2.10 of the paper.

    Variable ordering:  v = [x_1..x_N | y_1..y_M | z_1..z_K | t_0..t_{L-1}]
    (consistent with build_qubo.py)
    """
    scenario_name = SCENARIO_NAMES[scenario_id]
    seed_dir = data_dir / f"seed_{seed}"

    # Load QUBO matrix and index map
    qubo_file = seed_dir / f"qubo_Q_{scenario_name}_N{N}.npz"
    Q_data = np.load(qubo_file)
    Q_full = Q_data['Q']

    index_file = seed_dir / f"index_map_{scenario_name}_N{N}.json"
    with open(index_file, 'r') as f:
        index_map = json.load(f)

    M = index_map['M']
    K = index_map['K']
    L = index_map.get('L_reg_slack', 0)
    n = N + M + K + L

    # Load instance data (prefix-sliced to N)
    c_df = pd.read_csv(seed_dir / "c_vector.csv")
    c = c_df["expected_cost"].to_numpy(dtype=float)[:N]

    Sigma_data = np.load(seed_dir / "sigma_matrix.npz")
    Sigma = Sigma_data["sigma"][:N, :N]

    regulatory = pd.read_csv(seed_dir / "regulatory_set.csv")
    R_ids = regulatory["feature_id"].astype(int).tolist()
    R_indices = [i for i in range(N) if (i + 1) in R_ids]

    premium_bands = np.array(index_map['premium_bands'])
    lambda_risk = float(index_map['lambda_risk'])
    Rmin = int(index_map.get('Rmin', 0))

    # ------------------------------------------------------------------
    # Q_base: economic objective (cost + risk curvature + premium revenue)
    # min  c^T x  +  lambda_risk * x^T Sigma x  -  sum_k P_k z_k
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Q_onehot: unit-weight one-hot penalty for deductible and premium bands
    # Penalty form: (1 - sum v_i)^2, expanded for the quadratic form v^T Q v.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Q_reg: unit-weight regulatory minimum coverage penalty
    # Squared penalty: (sum_{i in R} x_i - sum_l 2^l t_l - Rmin)^2
    # Skipped when Rmin == 0 (no minimum coverage requirement active).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Q_aff: unit-weight affordability penalty (S4 only)
    # Adds a unit diagonal penalty to each disallowed premium band variable.
    # Active only when affordability_enabled = true in the index map.
    # ------------------------------------------------------------------
    Q_aff = np.zeros((n, n), dtype=float)
    affordability_enabled = index_map.get('affordability_enabled', False)
    disallow_premium_bands = index_map.get('disallow_premium_bands', [])

    if affordability_enabled and disallow_premium_bands:
        for band_1based in disallow_premium_bands:
            k0 = int(band_1based) - 1   # convert to zero-based
            if 0 <= k0 < K:
                z_idx = N + M + k0
                Q_aff[z_idx, z_idx] = 1.0  # unit weight; multiplied by w_aff below

    # ------------------------------------------------------------------
    # Penalty weights from index map
    # w_aff resolution follows the same tiered logic as infer_D_affordability
    # in run_baseline_full.py to handle both current and legacy index maps.
    # ------------------------------------------------------------------
    w_oh  = float(index_map.get('A_onehot', 0.0))
    w_reg = float(index_map.get('B_reg', 0.0))

    D_raw = index_map.get('D_affordability')
    if D_raw is None:
        diag = index_map.get('diagnostics', {}) or {}
        D_raw = diag.get('D_affordability')
    if D_raw is None:
        mult   = index_map.get('disallow_penalty_multiplier')
        A_base = index_map.get('A_base')
        A_oh   = index_map.get('A_onehot')
        if mult is not None and A_base is not None:
            D_raw = float(mult) * float(A_base)
        elif mult is not None and A_oh is not None:
            D_raw = float(mult) * float(A_oh)   # legacy fallback
    w_aff = float(D_raw) if D_raw is not None else 0.0

    return {
        'Q_base': Q_base, 'Q_onehot': Q_onehot, 'Q_reg': Q_reg, 'Q_aff': Q_aff,
        'w_oh': w_oh, 'w_reg': w_reg, 'w_aff': w_aff,
        'N': N, 'M': M, 'K': K, 'L': L,
        'c': c, 'Sigma': Sigma, 'lambda_risk': lambda_risk,
        'premium_bands': premium_bands, 'R_indices': R_indices, 'Rmin': Rmin,
    }


# ---------------------------------------------------------------------------
# Solution evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(solution, data):
    """
    Evaluate a binary solution vector and return the economic objective and
    feasibility metrics defined in Section 4.3.

    Returns the raw economic objective f(x) (excluding penalty energies),
    feasibility flags, and decoded product structure (selected bands, feature
    counts). This matches the objective_raw_* reporting convention used in
    baseline_full_results.csv.
    """
    N, M, K = data['N'], data['M'], data['K']
    x = solution[:N]
    y = solution[N:N + M]
    z = solution[N + M:N + M + K]

    # Raw economic objective: cost + risk - premium (Section 2.3)
    cost  = np.dot(data['c'], x)
    risk  = data['lambda_risk'] * x.T @ data['Sigma'] @ x if data['lambda_risk'] != 0 else 0.0
    prem_idx = int(np.argmax(z))
    premium  = data['premium_bands'][prem_idx]
    raw_obj  = float(cost + risk - premium)

    # Feasibility (Section 4.3): one-hot validity + regulatory minimum coverage
    y_ok  = abs(np.sum(y) - 1.0) < 1e-6
    z_ok  = abs(np.sum(z) - 1.0) < 1e-6
    if data['R_indices'] and data['Rmin'] > 0:
        R_sum  = sum(x[i] for i in data['R_indices'])
        reg_ok = R_sum >= (data['Rmin'] - 1e-6)
    else:
        reg_ok = True
    feasible = y_ok and z_ok and reg_ok

    # Decoded product structure
    ded_idx  = int(np.argmax(y)) + 1   # 1-based band index
    prem_idx = prem_idx + 1            # 1-based band index
    num_features     = int(np.sum(x > 0.5))
    num_reg_features = int(sum(1 for i in data['R_indices'] if x[i] > 0.5)) if data['R_indices'] else 0

    return {
        'raw_objective': raw_obj,
        'feasible': feasible,
        'chosen_premium_band': prem_idx,
        'chosen_deductible_band': ded_idx,
        'selected_features_count': num_features,
        'selected_reg_features_count': num_reg_features,
    }


# ---------------------------------------------------------------------------
# Per-instance ECHO execution
# ---------------------------------------------------------------------------

def run_echo_on_instance(seed, scenario_id, N, data_dir, baseline_row, verbose=False):
    """
    Run ECHO on a single instance and assemble the full result record.

    Loads the pre-built QUBO, decomposes it into components, runs
    spectral_landscape_navigation, evaluates the resulting solution, and
    joins with the corresponding baseline row from baseline_full_results.csv.

    The paired gap Δ = echo_raw_objective − sa_raw_objective follows the
    sign convention in Section 4.3: negative values indicate ECHO improvement.
    Win/tie/loss classification uses tolerance ε = 10⁻⁶ (Section 4.3).
    """
    data = load_qubo_and_decompose(seed, scenario_id, N, data_dir)

    result = spectral_landscape_navigation(
        data['Q_base'], data['Q_onehot'], data['Q_reg'], data['Q_aff'],
        data['N'], data['M'], data['K'], data['L'],
        data['w_oh'], data['w_reg'], data['w_aff'],
        seed=seed,
        params=None,
        verbose=verbose,
    )

    echo_eval = evaluate_solution(result['solution'], data)

    # Baseline metrics from run_baseline_full.py output
    sa_raw      = float(baseline_row['objective_raw_sa_best'])
    sa_energy   = float(baseline_row['energy_sa_best'])
    sa_feasible = bool(baseline_row['is_feasible_sa'])
    sa_runtime  = float(baseline_row['sa_runtime_sec'])

    greedy_raw      = float(baseline_row['objective_raw_greedy'])
    greedy_energy   = float(baseline_row['energy_greedy'])
    greedy_feasible = bool(baseline_row['is_feasible_greedy'])

    gurobi_ran = bool(baseline_row.get('gurobi_ran', False))
    if gurobi_ran:
        gurobi_raw      = float(baseline_row.get('objective_raw_gurobi', np.nan))
        gurobi_energy   = float(baseline_row.get('energy_gurobi', np.nan))
        gurobi_feasible = bool(baseline_row.get('is_feasible_gurobi', False))
        gurobi_status   = str(baseline_row.get('gurobi_status_name', 'unknown'))
    else:
        gurobi_raw      = np.nan
        gurobi_energy   = np.nan
        gurobi_feasible = False
        gurobi_status   = 'not_run'

    # Paired gaps (Section 4.3): negative = ECHO improvement
    gap_sa     = echo_eval['raw_objective'] - sa_raw
    gap_greedy = echo_eval['raw_objective'] - greedy_raw
    gap_gurobi = echo_eval['raw_objective'] - gurobi_raw if not np.isnan(gurobi_raw) else np.nan

    # Win/tie/loss with ε = 10⁻⁶ tolerance (Section 4.3)
    echo_beats_sa  = int(gap_sa  < -1e-6)
    echo_ties_sa   = int(abs(gap_sa)  <= 1e-6)
    echo_loses_sa  = int(gap_sa  >  1e-6)

    echo_beats_greedy = int(gap_greedy < -1e-6)
    echo_ties_greedy  = int(abs(gap_greedy) <= 1e-6)

    if not np.isnan(gurobi_raw):
        echo_beats_gurobi = int(gap_gurobi < -1e-6)
        echo_ties_gurobi  = int(abs(gap_gurobi) <= 1e-6)
        echo_loses_gurobi = int(gap_gurobi >  1e-6)
    else:
        echo_beats_gurobi = echo_ties_gurobi = echo_loses_gurobi = 0

    # Spectral diagnostics from the final homotopy stage
    spectra    = result.get('spectra', [])
    final_spec = spectra[-1] if spectra else {}

    return {
        # Instance identity
        'scenario_id': scenario_id,
        'scenario_name': SCENARIO_NAMES[scenario_id],
        'seed': seed,
        'N': N,
        'M': data['M'],
        'K': data['K'],
        'L': data['L'],

        # ECHO solution metrics
        'echo_raw_objective': echo_eval['raw_objective'],
        'echo_energy': result['energy'],
        'echo_feasible': int(echo_eval['feasible']),
        'echo_runtime_sec': result['runtime'],
        'echo_chosen_premium_band': echo_eval['chosen_premium_band'],
        'echo_chosen_deductible_band': echo_eval['chosen_deductible_band'],
        'echo_selected_features_count': echo_eval['selected_features_count'],
        'echo_selected_reg_features_count': echo_eval['selected_reg_features_count'],

        # ECHO algorithm diagnostics
        'echo_tau0': result['tau0'],
        'echo_num_stages': len(result['stages']),
        'echo_best_stage': result['best_stage'],
        'echo_steps_used': result['steps_used'],
        'echo_final_condition_number': final_spec.get('condition_number', np.nan),
        'echo_final_negative_eigs': final_spec.get('negative_count', 0),
        'echo_final_roughness': final_spec.get('roughness', np.nan),

        # Baseline: SA
        'sa_raw_objective': sa_raw,
        'sa_energy': sa_energy,
        'sa_feasible': int(sa_feasible),
        'sa_runtime_sec': sa_runtime,

        # Baseline: greedy
        'greedy_raw_objective': greedy_raw,
        'greedy_energy': greedy_energy,
        'greedy_feasible': int(greedy_feasible),

        # Baseline: Gurobi (where available)
        'gurobi_ran': int(gurobi_ran),
        'gurobi_raw_objective': gurobi_raw,
        'gurobi_energy': gurobi_energy,
        'gurobi_feasible': int(gurobi_feasible),
        'gurobi_status': gurobi_status,

        # Paired gaps (negative = ECHO improvement)
        'gap_echo_to_sa': gap_sa,
        'gap_echo_to_greedy': gap_greedy,
        'gap_echo_to_gurobi': gap_gurobi,

        # Win/tie/loss indicators
        'echo_beats_sa': echo_beats_sa,
        'echo_ties_sa': echo_ties_sa,
        'echo_loses_sa': echo_loses_sa,
        'echo_beats_greedy': echo_beats_greedy,
        'echo_ties_greedy': echo_ties_greedy,
        'echo_beats_gurobi': echo_beats_gurobi,
        'echo_ties_gurobi': echo_ties_gurobi,
        'echo_loses_gurobi': echo_loses_gurobi,

        # Penalty weights (for cross-referencing with index map)
        'w_onehot': data['w_oh'],
        'w_reg': data['w_reg'],
        'w_aff': data['w_aff'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ECHO on all 520 baseline instances (Section 4 of the paper)."
    )
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run a single scenario only.")
    parser.add_argument('--N', type=int, default=None,
                        help="Run a single problem size only.")
    parser.add_argument('--test', type=int, default=None,
                        help="Test mode: run only the first N instances.")
    parser.add_argument('--baseline', type=str, default='results/baseline_full_results.csv',
                        help="Path to baseline_full_results.csv.")
    parser.add_argument('--output', type=str, default='results/echo_full_results.csv',
                        help="Path for output CSV.")
    parser.add_argument('--append', action='store_true',
                        help="Append to an existing output file rather than overwriting.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print ECHO solver progress for each instance.")

    args = parser.parse_args()

    project_root = Path.cwd()
    data_dir     = project_root / "data" / "seeds"
    baseline_file = Path(args.baseline)

    print("=" * 80)
    print("ECHO FULL EXPERIMENT — SPECTRAL LANDSCAPE NAVIGATION")
    print("=" * 80)

    # Load baseline results; filter to baseline sweep rows only
    baseline_df = pd.read_csv(baseline_file)
    if 'sweep_type' in baseline_df.columns:
        baseline_df = baseline_df[baseline_df['sweep_type'] == 'baseline']

    print(f"Loaded baseline: {baseline_file} ({len(baseline_df)} baseline instances)")

    # ------------------------------------------------------------------
    # Build instance list, applying any CLI filters
    # ------------------------------------------------------------------
    instances_to_run = []
    for (scen_id, N), seeds in ALL_CONFIGS.items():
        if args.scenario is not None and scen_id != args.scenario:
            continue
        if args.N is not None and N != args.N:
            continue
        for seed in seeds:
            instances_to_run.append((scen_id, N, seed))

    if args.test:
        instances_to_run = instances_to_run[:args.test]

    print(f"Instances to run: {len(instances_to_run)}")
    print(f"Output: {args.output}")
    if args.append:
        print("Mode: APPEND to existing file")
    print()

    # ------------------------------------------------------------------
    # Prepare output file and load existing results if appending
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if args.append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        print(f"Loaded existing results: {len(existing_df)} rows")
        print()

    # ------------------------------------------------------------------
    # Main execution loop — saves incrementally after each instance
    # ------------------------------------------------------------------
    results = []
    failed  = []

    for idx, (scen_id, N, seed) in enumerate(tqdm(instances_to_run, desc="Running ECHO"), 1):
        scenario_name = SCENARIO_NAMES[scen_id]

        mask = (
            (baseline_df['scenario_name'] == scenario_name) &
            (baseline_df['seed'] == seed) &
            (baseline_df['N'] == N)
        )
        baseline_rows = baseline_df[mask]

        if len(baseline_rows) == 0:
            print(f"\n  No baseline row found for {scenario_name} seed={seed} N={N} — skipping.")
            failed.append((scen_id, N, seed))
            continue

        baseline_row = baseline_rows.iloc[0]

        try:
            result = run_echo_on_instance(
                seed, scen_id, N, data_dir, baseline_row, args.verbose
            )
            results.append(result)

            # Incremental save after every instance for fault tolerance
            df_new = pd.DataFrame(results)
            if existing_df is not None:
                df_combined = pd.concat([existing_df, df_new], ignore_index=True)
                df_combined = df_combined.sort_values(['scenario_id', 'N', 'seed']).reset_index(drop=True)
                df_combined.to_csv(output_path, index=False)
            else:
                df_new.to_csv(output_path, index=False)

            if idx % 10 == 0:
                print(f"\n  Checkpoint: {len(results)} instances saved.")

        except Exception as e:
            print(f"\n  Error — {scenario_name} seed={seed} N={N}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed.append((scen_id, N, seed))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if results:
        df_new = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Completed: {len(df_new)}")
        if failed:
            print(f"Failed:    {len(failed)}")

        wins   = df_new['echo_beats_sa'].sum()
        ties   = df_new['echo_ties_sa'].sum()
        losses = df_new['echo_loses_sa'].sum()
        n_tot  = len(df_new)
        print(f"\nECHO vs SA:")
        print(f"  Wins:   {wins:4d} / {n_tot} ({wins   / n_tot * 100:5.1f}%)")
        print(f"  Ties:   {ties:4d} / {n_tot} ({ties   / n_tot * 100:5.1f}%)")
        print(f"  Losses: {losses:4d} / {n_tot} ({losses / n_tot * 100:5.1f}%)")

        wins_g = df_new['echo_beats_greedy'].sum()
        print(f"\nECHO vs Greedy:")
        print(f"  Wins: {wins_g:4d} / {n_tot} ({wins_g / n_tot * 100:5.1f}%)")

        gurobi_mask = df_new['gurobi_ran'] == 1
        if gurobi_mask.sum() > 0:
            gdf     = df_new[gurobi_mask]
            matches = gdf['echo_ties_gurobi'].sum()
            print(f"\nECHO vs Gurobi (where Gurobi ran, n={len(gdf)}):")
            print(f"  Matches: {matches:4d} / {len(gdf)} ({matches / len(gdf) * 100:5.1f}%)")

        print(f"\nRaw objective gap vs SA (negative = ECHO improvement):")
        print(f"  Median: {df_new['gap_echo_to_sa'].median():>12,.1f}")
        print(f"  Mean:   {df_new['gap_echo_to_sa'].mean():>12,.1f}")

        if len(df_new['N'].unique()) > 1:
            print(f"\nPerformance by N:")
            print("-" * 72)
            print(f"{'N':<6} {'Count':<8} {'Wins':<8} {'Ties':<8} {'Win%':<8} {'Median gap':<12}")
            print("-" * 72)
            for N_val in sorted(df_new['N'].unique()):
                df_n = df_new[df_new['N'] == N_val]
                w    = df_n['echo_beats_sa'].sum()
                t    = df_n['echo_ties_sa'].sum()
                gap  = df_n['gap_echo_to_sa'].median()
                print(f"{N_val:<6} {len(df_n):<8} {w:<8} {t:<8} {w / len(df_n) * 100:<8.1f} {gap:<12,.1f}")

        print("\n" + "=" * 80)
        total_rows = (len(existing_df) + len(df_new)) if existing_df is not None else len(df_new)
        print(f"Output: {output_path}  ({total_rows} total rows)")
        print("=" * 80)

    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()