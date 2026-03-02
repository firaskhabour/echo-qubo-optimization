"""
Clean Up RAMP_UP Experiment Results
====================================

Removes duplicate instances from sensitivity analysis and regenerates tables.
This ensures fair comparison across all three experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from sho_metrics import (
    aggregate_results_by_scenario,
    wilcoxon_signed_rank_test
)

def cleanup_ramp_up_results():
    """
    Clean up RAMP_UP results by removing duplicates and regenerating tables.
    """
    
    print("="*80)
    print("CLEANING UP RAMP_UP EXPERIMENT RESULTS")
    print("="*80)
    
    # Load raw results
    input_path = "results/sho_experiments/results_sho_RAMP_UP.csv"
    
    print(f"\n[1] Loading results from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"    Original rows: {len(df)}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['scenario_name', 'seed', 'N'], keep=False)
    print(f"    Duplicate instances: {duplicates.sum()}")
    
    # Remove duplicates (keep first occurrence)
    df_clean = df.drop_duplicates(subset=['scenario_name', 'seed', 'N'], keep='first')
    
    print(f"    After deduplication: {len(df_clean)}")
    print(f"    Removed: {len(df) - len(df_clean)} duplicate rows")
    
    # Add penalty_strategy column if missing
    if 'penalty_strategy' not in df_clean.columns:
        df_clean['penalty_strategy'] = 'ramp_up'
        print(f"\n[2] Added 'penalty_strategy' column: ramp_up")
    
    # Save cleaned results
    output_path = "results/sho_experiments/results_sho_RAMP_UP.csv"
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n[3] Saved cleaned results to: {output_path}")
    
    # Regenerate tables
    print("\n" + "="*80)
    print("REGENERATING TABLES WITH CLEAN DATA")
    print("="*80)
    
    output_dir = Path("results/sho_experiments")
    
    # Table 1: Scenario × N aggregates
    print("\n[Table 1] Performance by Scenario and Problem Size")
    agg_table = aggregate_results_by_scenario(df_clean, group_by=['scenario_name', 'N'])
    agg_path = output_dir / "table1_scenario_N_aggregates_ramp_up.csv"
    agg_table.to_csv(agg_path, index=False)
    print(f"    Saved to: {agg_path}")
    
    # Display key columns
    display_cols = ['scenario_name', 'N', 'gap_sho_to_sa_raw_median', 'sho_beats_sa_sum', 'feasibility_rate']
    if all(col in agg_table.columns for col in display_cols):
        print(agg_table[display_cols].to_string(index=False))
    
    # Table 2: Statistical tests
    print("\n[Table 2] Wilcoxon Signed-Rank Tests")
    wilcoxon_results = []
    for scenario in df_clean['scenario_name'].unique():
        scenario_data = df_clean[df_clean['scenario_name'] == scenario]
        test_result = wilcoxon_signed_rank_test(scenario_data)
        test_result['scenario'] = scenario
        wilcoxon_results.append(test_result)
    
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_path = output_dir / "table2_statistical_tests_ramp_up.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"    Saved to: {wilcoxon_path}")
    
    # Display
    display_cols = ['scenario', 'n_pairs']
    optional_cols = ['p_value', 'significant_at_0.05', 'win_rate', 'note']
    for col in optional_cols:
        if col in wilcoxon_df.columns:
            display_cols.append(col)
    
    print(wilcoxon_df[display_cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("UPDATED SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal instances: {len(df_clean)}")
    print(f"  SHO beats SA: {df_clean['sho_beats_sa'].sum()} / {len(df_clean)} ({df_clean['sho_beats_sa'].mean()*100:.1f}%)")
    print(f"  SHO beats Greedy: {df_clean['sho_beats_greedy'].sum()} / {len(df_clean)} ({df_clean['sho_beats_greedy'].mean()*100:.1f}%)")
    
    if 'sho_matches_gurobi' in df_clean.columns:
        gurobi_count = df_clean['sho_matches_gurobi'].notna().sum()
        if gurobi_count > 0:
            matches = df_clean['sho_matches_gurobi'].sum()
            print(f"  SHO matches Gurobi: {matches} / {gurobi_count} ({matches/gurobi_count*100:.1f}%)")
    
    print(f"\nFeasibility by scenario:")
    for scenario in sorted(df_clean['scenario_name'].unique()):
        scenario_data = df_clean[df_clean['scenario_name'] == scenario]
        feas_rate = scenario_data['is_feasible_sho'].mean() * 100
        print(f"  {scenario:30s}: {feas_rate:5.1f}%")
    
    print("\n" + "="*80)
    print("CLEANUP COMPLETE!")
    print("="*80)
    
    print("\nFiles updated:")
    print(f"  ✓ {output_path}")
    print(f"  ✓ {agg_path}")
    print(f"  ✓ {wilcoxon_path}")
    
    print("\nThese files are now ready for comparison with CONSTANT and RAMP_DOWN experiments.")
    
    return df_clean

if __name__ == "__main__":
    df_clean = cleanup_ramp_up_results()