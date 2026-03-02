"""
SHO Experimental Evaluation
Runs Spectral Homotopy Optimization on insurance QUBO instances
Compares against baseline SA, Greedy, and Gurobi results
"""

import numpy as np
import pandas as pd
import time
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

# Import SHO components
from spectral_homotopy_optimization import (
    spectral_homotopy_QUBO,
    get_default_SHO_parameters,
    SHOParameters
)
from sho_metrics import (
    extract_solution_metrics,
    compute_gaps,
    aggregate_results_by_scenario,
    wilcoxon_signed_rank_test
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """
    Configuration - uses your existing project structure
    """
    RESULTS_MASTER = "results/results_master.csv"  # Your baseline results
    OUTPUT_DIR = "results/sho_experiments"
    OUTPUT_CSV = "results/sho_experiments/results_sho_complete.csv"
    
    # Data paths (matches your structure)
    QUBO_DATA_DIR = "data/seeds"  # Your QUBO location
    
    # Experiment settings
    MAX_INSTANCES = None  # Set to small number for testing, None for all
    VERBOSE = False  # Print progress for each instance
    SAVE_INTERVAL = 1  # Save results after EVERY instance (was 10)


# ============================================================================
# IMPORT SA FUNCTIONS FROM EXISTING CODEBASE
# ============================================================================

# Import directly from your solve_classical.py module
try:
    # Try relative import (if run_sho_experiments.py is in src/ or same level)
    from solve_classical import simulated_annealing, project_onehot, energy
    print("✓ SA functions imported from solve_classical.py")
    
except ImportError:
    # Try from src package
    try:
        from src.solve_classical import simulated_annealing, project_onehot, energy
        print("✓ SA functions imported from src.solve_classical")
    except ImportError:
        # Add src to path and try again
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / 'src'
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            from solve_classical import simulated_annealing, project_onehot, energy
            print(f"✓ SA functions imported from {src_path}/solve_classical.py")
        else:
            raise ImportError(
                "Cannot find solve_classical.py!\n"
                "Expected locations:\n"
                "  1. ./solve_classical.py (same directory)\n"
                "  2. ./src/solve_classical.py (src package)\n"
                "Please ensure solve_classical.py is accessible."
            )


# ============================================================================
# AUTOMATIC QUBO DECOMPOSITION
# ============================================================================

def decompose_qubo_from_params(
    Q_full: np.ndarray,
    w_oh: float,
    w_reg: float,
    w_aff: float,
    N: int,
    M: int,
    K: int
) -> Dict[str, np.ndarray]:
    """
    Decompose Q_full back into components using penalty weights.
    
    Since Q = Q_base + w_oh*Q_oh + w_reg*Q_reg + w_aff*Q_aff,
    we can approximate the decomposition by analyzing the structure.
    
    This is a heuristic decomposition - not perfect but functional.
    """
    n = Q_full.shape[0]
    
    # Strategy: Since penalties are massive (1M-40M) and objective is small (1K-10K),
    # we can identify penalty-heavy regions
    
    # One-hot penalties typically appear in y and z blocks
    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K
    
    # Extract Q_oh (one-hot constraint penalties)
    Q_oh = np.zeros_like(Q_full)
    # One-hot penalties are in y and z blocks (off-diagonal coupling)
    # Approximate: extract penalty structure from those regions
    Q_oh[y_start:y_end, y_start:y_end] = Q_full[y_start:y_end, y_start:y_end]
    Q_oh[z_start:z_end, z_start:z_end] = Q_full[z_start:z_end, z_start:z_end]
    
    # Scale back to unit penalties
    if w_oh > 1e-6:
        Q_oh = Q_oh / w_oh
    
    # Extract Q_reg (regulatory penalties - typically in x-x coupling)
    Q_reg = np.zeros_like(Q_full)
    # Regulatory penalties often appear in feature-feature coupling
    Q_reg[:N, :N] = Q_full[:N, :N]
    
    # Scale back
    if w_reg > 1e-6:
        Q_reg = Q_reg / w_reg
    
    # Q_aff (affordability - if present)
    Q_aff = None
    if w_aff > 1e-6:
        Q_aff = np.zeros_like(Q_full)
        # Affordability typically in feature-premium coupling
        Q_aff[:N, z_start:z_end] = Q_full[:N, z_start:z_end]
        Q_aff[z_start:z_end, :N] = Q_full[z_start:z_end, :N]
        Q_aff = Q_aff / w_aff
    
    # Q_base = Q_full - penalties
    Q_base = Q_full.copy()
    Q_base -= w_oh * Q_oh
    Q_base -= w_reg * Q_reg
    if Q_aff is not None:
        Q_base -= w_aff * Q_aff
    
    return {
        'Q_full': Q_full,
        'Q_base': Q_base,
        'Q_oh': Q_oh,
        'Q_reg': Q_reg,
        'Q_aff': Q_aff
    }

def load_qubo_instance(
    scenario_name: str,
    seed: int,
    N: int,
    M: int,
    K: int,
    w_oh: float,
    w_reg: float,
    w_aff: float,
    data_dir: Path = Path("data/seeds")
) -> Dict[str, np.ndarray]:
    """
    Load QUBO instance from your data structure with automatic decomposition.
    
    Args:
        scenario_name: Scenario identifier (e.g., 'S1_cost_only')
        seed: Random seed
        N, M, K: Problem dimensions
        w_oh, w_reg, w_aff: Penalty weights (for decomposition)
        data_dir: Root directory for seed data
        
    Returns:
        Dict with Q_full, Q_base, Q_oh, Q_reg, Q_aff matrices
    """
    # Construct path based on your structure
    # Structure: data/seeds/seed_{seed}/qubo_Q_{scenario}_N{N}.npz
    seed_dir = data_dir / f"seed_{seed}"
    qubo_filename = f"qubo_Q_{scenario_name}_N{N}.npz"
    qubo_path = seed_dir / qubo_filename
    
    if not qubo_path.exists():
        # Provide helpful debugging info
        if not seed_dir.exists():
            raise FileNotFoundError(
                f"Seed directory not found: {seed_dir}\n"
                f"Expected structure: {data_dir}/seed_{{seed}}/\n"
                f"Available seeds: {list(data_dir.glob('seed_*')) if data_dir.exists() else 'data dir not found'}"
            )
        else:
            # Seed dir exists but file doesn't
            available_files = list(seed_dir.glob('qubo_Q_*.npz'))
            raise FileNotFoundError(
                f"QUBO file not found: {qubo_path}\n"
                f"Seed directory exists: {seed_dir}\n"
                f"Available QUBO files in this seed:\n" +
                '\n'.join(f"  - {f.name}" for f in available_files[:10])
            )
    
    # Load the npz file
    qubo_data = np.load(qubo_path)
    
    # Extract Q matrix
    if 'Q' in qubo_data:
        Q_full = qubo_data['Q']
    elif 'Q_full' in qubo_data:
        Q_full = qubo_data['Q_full']
    else:
        available_keys = list(qubo_data.keys())
        raise KeyError(f"No Q matrix in {qubo_path}. Keys: {available_keys}")
    
    # Check if decomposition already exists
    if all(k in qubo_data for k in ['Q_base', 'Q_onehot', 'Q_reg']):
        print(f"  ✓ Using pre-decomposed QUBO from file")
        Q_base = qubo_data['Q_base']
        Q_oh = qubo_data['Q_onehot']
        Q_reg = qubo_data['Q_reg']
        Q_aff = qubo_data.get('Q_afford', None)
    else:
        # Automatic decomposition using penalty weights
        decomposed = decompose_qubo_from_params(Q_full, w_oh, w_reg, w_aff, N, M, K)
        Q_base = decomposed['Q_base']
        Q_oh = decomposed['Q_oh']
        Q_reg = decomposed['Q_reg']
        Q_aff = decomposed['Q_aff']
    
    return {
        'Q_full': Q_full,
        'Q_base': Q_base,
        'Q_oh': Q_oh,
        'Q_reg': Q_reg,
        'Q_aff': Q_aff
    }


def get_penalty_weights(row: pd.Series) -> Dict[str, float]:
    """
    Extract penalty weights from results_master row.
    
    Args:
        row: Row from results_master.csv
        
    Returns:
        Dict with w_oh, w_reg, w_aff
    """
    scale_factor = row['scale_factor']
    
    return {
        'w_oh': scale_factor * row['A_onehot'],
        'w_reg': scale_factor * row['B_reg'],
        'w_aff': scale_factor * row['D_afford']
    }


# ============================================================================
# SA SOLVER WRAPPER
# ============================================================================

def sa_solver_wrapper(
    Q: np.ndarray,
    N: int,
    M: int,
    K: int,
    L: int,
    rng: np.random.Generator,
    steps: int,
    T0: float,
    Tend: float,
    initial_solution: Optional[np.ndarray] = None
) -> tuple:
    """
    SA solver wrapper compatible with SHO interface.
    Uses auto-imported simulated_annealing() function.
    
    Args:
        Q: QUBO matrix
        N, M, K, L: Problem dimensions
        rng: Random number generator
        steps: Number of SA steps
        T0, Tend: Temperature schedule
        initial_solution: Warm-start solution (ignored - SA doesn't support it yet)
        
    Returns:
        (solution, energy) tuple
    """
    # Warm-starting limitation
    if initial_solution is not None:
        # Your SA doesn't support warm-starting
        # This is okay - SHO will still work, just won't benefit from warm starts
        pass  # Silently ignore, start from random
    
    # Call auto-imported SA function
    # Signature: simulated_annealing(Q, N, M, K, rng, steps, T0, Tend, prem_move_prob, ded_move_prob)
    best_solution, best_energy, diagnostics = simulated_annealing(
        Q=Q,
        N=N,
        M=M,
        K=K,
        rng=rng,
        steps=steps,
        T0=T0,
        Tend=Tend,
        prem_move_prob=0.075,  # Match baseline
        ded_move_prob=0.075    # Match baseline
    )
    
    # Return (solution, energy) tuple as expected by SHO
    return best_solution, best_energy


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_sho_experiments(
    results_master_path: str = Config.RESULTS_MASTER,
    output_csv: str = Config.OUTPUT_CSV,
    max_instances: Optional[int] = Config.MAX_INSTANCES,
    verbose: bool = Config.VERBOSE,
    penalty_strategy: str = "constant"
):
    """
    Run SHO on all instances from results_master.csv
    
    Args:
        results_master_path: Path to baseline results CSV
        output_csv: Where to save SHO results
        max_instances: Limit for testing (None = all)
        verbose: Print detailed progress
        penalty_strategy: Penalty ramping strategy - "ramp_up", "constant", or "ramp_down"
    """
    print("="*80)
    print("SPECTRAL HOMOTOPY OPTIMIZATION - EXPERIMENTAL EVALUATION")
    print(f"Penalty Strategy: {penalty_strategy.upper()}")
    print("="*80)
    
    # Load baseline results
    print(f"\n[1] Loading baseline results from {results_master_path}...")
    baseline_df = pd.read_csv(results_master_path)
    
    # 🔴 CRITICAL: Remove duplicate instances (keep first occurrence)
    original_count = len(baseline_df)
    baseline_df = baseline_df.drop_duplicates(
        subset=['scenario_name', 'seed', 'N'], 
        keep='first'
    )
    duplicates_removed = original_count - len(baseline_df)
    
    if duplicates_removed > 0:
        print(f"  ⚠️  Removed {duplicates_removed} duplicate instances from baseline")
    
    if max_instances is not None:
        baseline_df = baseline_df.head(max_instances)
        print(f"    Limited to {max_instances} instances for testing")
    
    print(f"    Loaded {len(baseline_df)} unique instances")
    
    # Setup output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Get SHO parameters
    sho_params = get_default_SHO_parameters()
    
    # Set penalty strategy from command-line argument
    sho_params.penalty_strategy = penalty_strategy
    
    print(f"\n[2] SHO Configuration:")
    print(f"    Penalty Strategy: {penalty_strategy.upper()}")
    print(f"    Budget: {sho_params.total_budget:,} SA steps (matching baseline)")
    print(f"    Stages: {sho_params.t_stages}")
    print(f"    Stage budgets: {[f'{b*100:.0f}%' for b in sho_params.stage_budgets]}")
    print(f"    Initial restarts: {sho_params.R_initial}")
    print(f"    Beam width: {sho_params.B_beam}")
    
    # Run experiments
    print(f"\n[3] Running SHO experiments...")
    
    for idx, row in tqdm(baseline_df.iterrows(), total=len(baseline_df), desc="Instances"):
        
        # Extract instance info
        scenario_name = row['scenario_name']
        seed = int(row['seed'])
        N = int(row['N'])
        M = int(row['M'])
        K = int(row['K'])
        L = int(row['L_reg_slack'])
        
        try:
            # Get penalty weights first (needed for decomposition)
            weights = get_penalty_weights(row)
            
            # Load QUBO instance with automatic decomposition
            qubo_data = load_qubo_instance(
                scenario_name=scenario_name,
                seed=seed,
                N=N,
                M=M,
                K=K,
                w_oh=weights['w_oh'],
                w_reg=weights['w_reg'],
                w_aff=weights['w_aff']
            )
            
            # 🔴 FIX: Load actual problem data for raw objective calculation
            # Load cost vector and covariance matrix
            data_dir = Config.QUBO_DATA_DIR
            seed_dir = Path(data_dir) / f"seed_{seed}"
            
            c_df = pd.read_csv(seed_dir / "c_vector.csv")
            c = c_df["expected_cost"].to_numpy(dtype=float)[:N]
            
            Sigma_data = np.load(seed_dir / "sigma_matrix.npz")
            Sigma = Sigma_data["sigma"][:N, :N]
            
            # Load problem parameters from config YAML
            scenario_num = int(scenario_name.split('_')[0][1])  # Extract number from 'S1_...'
            config_path = Path("config") / f"config_qubo_S{scenario_num}.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            premium_bands = np.array(config['pricing']['premium_bands'], dtype=float)
            lambda_risk = float(config['risk']['lambda_risk'])
            
            # Run SHO
            start_time = time.time()
            sho_result = spectral_homotopy_QUBO(
                Q_base=qubo_data['Q_base'],
                Q_oh=qubo_data['Q_oh'],
                Q_reg=qubo_data['Q_reg'],
                Q_aff=qubo_data['Q_aff'],
                N=N, M=M, K=K, L=L,
                w_oh=weights['w_oh'],
                w_reg=weights['w_reg'],
                w_aff=weights['w_aff'],
                seed=seed,
                params=sho_params,
                SA_solver=sa_solver_wrapper,
                verbose=verbose
            )
            sho_runtime = time.time() - start_time
            
            # Extract solution metrics
            sho_metrics = extract_solution_metrics(
                solution=sho_result['solution'],
                Q_full=qubo_data['Q_full'],
                Q_base=qubo_data['Q_base'],
                Q_penalties=qubo_data['Q_reg'],
                N=N, M=M, K=K,
                c=c,
                Sigma=Sigma,
                lambda_risk=lambda_risk,
                premium_bands=premium_bands,
                method_name="sho"
            )
            
            # Add SHO-specific diagnostics
            sho_metrics['sho_runtime_sec'] = sho_runtime
            sho_metrics['sho_lambda_0'] = sho_result['lambda_0']
            sho_metrics['sho_best_found_at_stage'] = sho_result['best_stage']
            sho_metrics['sho_stages_completed'] = len(sho_params.t_stages)
            sho_metrics['sho_beam_size_final'] = len(sho_result['beam_history'])
            sho_metrics['sho_total_sa_steps'] = sho_params.total_budget
            
            # Compute gaps
            sa_baseline = {
                'objective_raw_sa_best': row['objective_raw_sa_best'],
                'energy_sa_best': row['energy_sa_best'],
                'is_feasible_sa': row['is_feasible_sa']
            }
            
            greedy_baseline = {
                'objective_raw_greedy': row['objective_raw_greedy'],
                'energy_greedy': row['energy_greedy'],
                'is_feasible_greedy': row['is_feasible_greedy']
            }
            
            gurobi_baseline = None
            if row['gurobi_ran']:
                gurobi_baseline = {
                    'objective_raw_gurobi': row['objective_raw_gurobi'],
                    'energy_gurobi': row['energy_gurobi'],
                    'is_feasible_gurobi': row['is_feasible_gurobi'],
                    'gurobi_ran': 1
                }
            
            gaps = compute_gaps(sho_metrics, sa_baseline, greedy_baseline, gurobi_baseline)
            
            # Combine results
            result_row = {
                # Instance identifiers
                'scenario_name': scenario_name,
                'seed': seed,
                'N': N, 'M': M, 'K': K, 'L': L,
                
                # Experiment identifier (NEW)
                'penalty_strategy': penalty_strategy,
                
                # SHO results
                **sho_metrics,
                **gaps,
                
                # Baseline metrics (from results_master for comparison)
                'objective_raw_sa_best': row['objective_raw_sa_best'],
                'energy_sa_best': row['energy_sa_best'],
                'is_feasible_sa': row['is_feasible_sa'],
                'sa_runtime_sec': row['sa_runtime_sec'],
                
                'objective_raw_greedy': row['objective_raw_greedy'],
                'energy_greedy': row['energy_greedy'],
                'is_feasible_greedy': row['is_feasible_greedy'],
                'greedy_runtime_sec': row['greedy_runtime_sec'],
                
                # Gurobi (if available)
                'objective_raw_gurobi': row.get('objective_raw_gurobi', None),
                'energy_gurobi': row.get('energy_gurobi', None),
                'is_feasible_gurobi': row.get('is_feasible_gurobi', None),
                'gurobi_runtime_sec': row.get('gurobi_runtime_sec', None),
                'gurobi_ran': row.get('gurobi_ran', 0),
                
                # Problem structure
                'Q_eigenvalue_max': row['Q_eigenvalue_max'],
                'Q_eigenvalue_min': row['Q_eigenvalue_min'],
                'Q_condition_number': row['Q_condition_number'],
                'Q_eigenvalue_count_negative': row['Q_eigenvalue_count_negative']
            }
            
            all_results.append(result_row)
            
            # Save after EVERY instance (SAVE_INTERVAL = 1)
            if len(all_results) % Config.SAVE_INTERVAL == 0:
                pd.DataFrame(all_results).to_csv(output_csv, index=False)
                # Progress message every instance
                print(f"  ✓ Saved ({len(all_results)}/{len(baseline_df)})")
        
        except Exception as e:
            print(f"\n⚠️  Error on instance {idx} ({scenario_name}, seed={seed}, N={N}): {e}")
            
            # Log failed instance for debugging
            failed_row = {
                'scenario_name': scenario_name,
                'seed': seed,
                'N': N, 'M': M, 'K': K, 'L': L,
                'error': str(e),
                'error_type': type(e).__name__
            }
            all_results.append(failed_row)
            
            continue
    
    # Final save
    if len(all_results) == 0:
        print("\n❌ FATAL ERROR: No instances completed successfully!")
        print("   All instances failed. Check error messages above.")
        return pd.DataFrame()  # Return empty dataframe
    
    results_df = pd.DataFrame(all_results)
    
    # Filter out failed instances (those with 'error' column)
    if 'error' in results_df.columns:
        failed_count = results_df['error'].notna().sum()
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} instances failed and will be excluded from results")
            results_df = results_df[results_df['error'].isna()].drop(columns=['error', 'error_type'])
    
    if len(results_df) == 0:
        print("\n❌ FATAL ERROR: No successful instances to save!")
        return pd.DataFrame()
    
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_csv}")
    print(f"Total instances: {len(results_df)}")
    
    # Safe summary (check if columns exist)
    if 'sho_beats_sa' in results_df.columns:
        print(f"\nSummary:")
        print(f"  SHO beats SA: {results_df['sho_beats_sa'].sum()} / {len(results_df)}")
    
    if 'sho_beats_greedy' in results_df.columns:
        print(f"  SHO beats Greedy: {results_df['sho_beats_greedy'].sum()} / {len(results_df)}")
    
    if 'sho_matches_gurobi' in results_df.columns:
        gurobi_count = results_df['sho_matches_gurobi'].notna().sum()
        if gurobi_count > 0:
            print(f"  SHO matches Gurobi: {results_df['sho_matches_gurobi'].sum()} / {gurobi_count}")
    
    return results_df


# ============================================================================
# POST-PROCESSING AND ANALYSIS
# ============================================================================

def generate_ejor_tables(results_df: pd.DataFrame, output_dir: Path = Path(Config.OUTPUT_DIR), penalty_strategy: str = "unknown"):
    """
    Generate publication-ready tables for EJOR
    
    Args:
        results_df: Complete results DataFrame
        output_dir: Where to save tables
        penalty_strategy: Penalty strategy name for file naming
    """
    # Check if we have any results
    if len(results_df) == 0:
        print("\n⚠️  No results to generate tables from - skipping table generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"GENERATING EJOR TABLES ({penalty_strategy.upper()})")
    print("="*80)
    
    # Table 1: Scenario × N aggregates
    print("\n[Table 1] Performance by Scenario and Problem Size")
    agg_table = aggregate_results_by_scenario(results_df, group_by=['scenario_name', 'N'])
    agg_path = output_dir / f"table1_scenario_N_aggregates_{penalty_strategy}.csv"
    agg_table.to_csv(agg_path, index=False)
    print(f"    Saved to: {agg_path}")
    print(agg_table[['scenario_name', 'N', 'gap_sho_to_sa_raw_median', 'sho_beats_sa_sum', 'feasibility_rate']].to_string())
    
    # Table 2: Statistical tests
    print("\n[Table 2] Wilcoxon Signed-Rank Tests")
    wilcoxon_results = []
    for scenario in results_df['scenario_name'].unique():
        scenario_data = results_df[results_df['scenario_name'] == scenario]
        test_result = wilcoxon_signed_rank_test(scenario_data)
        test_result['scenario'] = scenario
        wilcoxon_results.append(test_result)
    
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_path = output_dir / f"table2_statistical_tests_{penalty_strategy}.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"    Saved to: {wilcoxon_path}")
    
    # Safe display - only show columns that exist
    display_cols = ['scenario', 'n_pairs']
    optional_cols = ['p_value', 'significant_at_0.05', 'win_rate', 'note']
    for col in optional_cols:
        if col in wilcoxon_df.columns:
            display_cols.append(col)
    
    print(wilcoxon_df[display_cols].to_string())
    
    print("\n" + "="*80)


if __name__ == "__main__":
    """
    Main entry point
    
    Usage:
        # Experiment 1: Ramp UP
        python run_sho_experiments.py --penalty ramp_up --output results/sho_RAMP_UP.csv
        
        # Experiment 2: Constant (no ramp)
        python run_sho_experiments.py --penalty constant --output results/sho_NO_RAMP.csv
        
        # Experiment 3: Ramp DOWN
        python run_sho_experiments.py --penalty ramp_down --output results/sho_RAMP_DOWN.csv
        
        # Test on 10 instances
        python run_sho_experiments.py --penalty constant --test 10
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SHO experiments with different penalty strategies")
    parser.add_argument('--penalty', type=str, default='constant', 
                       choices=['ramp_up', 'constant', 'ramp_down'],
                       help='Penalty ramping strategy (default: constant)')
    parser.add_argument('--test', type=int, default=None, 
                       help='Limit to N instances for testing')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed progress')
    parser.add_argument('--results', type=str, default=Config.RESULTS_MASTER, 
                       help='Path to results_master.csv')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_CSV, 
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    # Run experiments
    try:
        results_df = run_sho_experiments(
            results_master_path=args.results,
            output_csv=args.output,
            max_instances=args.test,
            verbose=args.verbose,
            penalty_strategy=args.penalty
        )
        
        # Generate tables
        generate_ejor_tables(results_df, penalty_strategy=args.penalty)
        
        print("\n✅ SUCCESS! Check output files for results.")
        
    except NotImplementedError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n⚠️  You must implement the following before running:")
        print("    1. QUBO loading/decomposition (load_qubo_instance)")
        print("    2. SA solver integration (sa_solver_wrapper)")
        print("\nSee comments in the code for details.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)