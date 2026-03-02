"""
SHO Metrics Calculator
Computes gaps and comparisons following EJOR standards and matching results_master structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def extract_solution_metrics(
    solution: np.ndarray,
    Q_full: np.ndarray,
    Q_base: np.ndarray,
    Q_penalties: np.ndarray,
    N: int,
    M: int,
    K: int,
    c: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float,
    premium_bands: np.ndarray,
    method_name: str = "SHO"
) -> Dict[str, Any]:
    """
    Extract comprehensive metrics from a solution.
    Mirrors the structure used in results_master for SA/Greedy/Gurobi.
    
    🔴 FIX: Computes raw objective using ACTUAL cost/risk/premium data,
    not Q_base (which is QUBO-transformed and unreliable for business metrics).
    
    Args:
        solution: Binary solution vector
        Q_full: Full QUBO matrix (with penalties)
        Q_base: Base objective QUBO (NOT USED - kept for compatibility)
        Q_penalties: Penalty terms only
        N, M, K: Problem dimensions
        c: Cost vector (N-dimensional)
        Sigma: Risk covariance matrix (N×N)
        lambda_risk: Risk aversion parameter
        premium_bands: Premium values for each band (K-dimensional)
        method_name: Identifier (e.g., "SHO", "SA", "Greedy")
        
    Returns:
        Dict with metrics matching results_master column structure
    """
    # Energy (penalized objective)
    energy = float(solution @ Q_full @ solution)
    
    # 🔴 Safety check
    if np.isnan(energy) or np.isinf(energy):
        raise ValueError(f"Invalid energy computed: {energy}")
    
    # Extract decision variables
    x = solution[:N]  # Features
    y = solution[N:N+M]  # Deductible
    z = solution[N+M:N+M+K]  # Premium
    
    # Convert to binary (threshold at 0.5)
    x_binary = (x > 0.5).astype(float)
    z_binary = (z > 0.5).astype(float)
    
    # 🔴 FIX: Compute raw objective using ACTUAL problem data
    # Raw = Cost + Risk - Premium
    cost_term = float(c @ x_binary)
    risk_term = float(lambda_risk * (x_binary @ Sigma @ x_binary)) if lambda_risk != 0.0 else 0.0
    premium_term = float(premium_bands @ z_binary)
    
    objective_raw = cost_term + risk_term - premium_term
    
    # Penalty components
    penalty_total = float(solution @ Q_penalties @ solution)
    
    # Feature selection
    selected_features_count = int(np.sum(x_binary))
    
    # One-hot selections
    chosen_deductible_band = int(np.argmax(y) + 1) if np.sum(y) > 0 else 0
    chosen_premium_band = int(np.argmax(z) + 1) if np.sum(z) > 0 else 0
    
    # Feasibility checks (one-hot constraints)
    y_sum = np.sum(y)
    z_sum = np.sum(z)
    onehot_ok = bool(abs(y_sum - 1.0) < 1e-6 and abs(z_sum - 1.0) < 1e-6)
    
    # 🔴 CRITICAL: Feasibility requires both one-hot AND low penalties
    is_feasible = bool(penalty_total < 1e-6 and onehot_ok)
    
    # Return in results_master format
    prefix = method_name.lower()
    return {
        f'energy_{prefix}': energy,
        f'objective_raw_{prefix}': objective_raw,
        f'penE_total_{prefix}': penalty_total,
        f'selected_features_count_{prefix}': selected_features_count,
        f'chosen_deductible_band_{prefix}': chosen_deductible_band,
        f'chosen_premium_band_{prefix}': chosen_premium_band,
        f'is_feasible_{prefix}': int(is_feasible),
        f'onehot_ok_{prefix}': int(onehot_ok)
    }


def compute_gaps(
    sho_metrics: Dict[str, Any],
    sa_metrics: Dict[str, Any],
    greedy_metrics: Optional[Dict[str, Any]] = None,
    gurobi_metrics: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-9
) -> Dict[str, Any]:
    """
    Compute gaps following ChatGPT's EJOR-defensible methodology.
    
    Primary comparison: raw objective (business metric) on feasible solutions only.
    Secondary: energy (penalized) for landscape analysis.
    
    Args:
        sho_metrics: SHO solution metrics
        sa_metrics: SA baseline metrics
        greedy_metrics: Greedy baseline metrics (optional)
        gurobi_metrics: Gurobi reference metrics (optional)
        tolerance: For near-zero handling in relative gaps
        
    Returns:
        Dict with gap metrics following results_master naming
    """
    gaps = {}
    
    # Extract feasibility
    sho_feasible = bool(sho_metrics.get('is_feasible_sho', 0))
    sa_feasible = bool(sa_metrics.get('is_feasible_sa', 0))
    
    # SHO vs SA (primary comparison)
    if sho_feasible and sa_feasible:
        # Raw objective gap (business metric)
        delta_raw = sho_metrics['objective_raw_sho'] - sa_metrics['objective_raw_sa_best']
        gaps['gap_sho_to_sa_raw'] = delta_raw
        
        # Relative gap (for varying scales)
        denominator = max(abs(sa_metrics['objective_raw_sa_best']), tolerance)
        gaps['gap_sho_to_sa_raw_pct'] = (delta_raw / denominator) * 100
        
        # Win/tie/loss
        gaps['sho_beats_sa'] = int(delta_raw < -tolerance)  # SHO better
        gaps['sho_ties_sa'] = int(abs(delta_raw) <= tolerance)
        gaps['sho_loses_sa'] = int(delta_raw > tolerance)
    else:
        # Infeasible comparison
        gaps['gap_sho_to_sa_raw'] = None
        gaps['gap_sho_to_sa_raw_pct'] = None
        gaps['sho_beats_sa'] = 0
        gaps['sho_ties_sa'] = 0
        gaps['sho_loses_sa'] = 0
    
    # Energy gap (secondary, for landscape analysis)
    delta_energy = sho_metrics['energy_sho'] - sa_metrics['energy_sa_best']
    gaps['gap_sho_to_sa_energy'] = delta_energy
    
    # SHO vs Greedy
    if greedy_metrics is not None:
        greedy_feasible = bool(greedy_metrics.get('is_feasible_greedy', 0))
        if sho_feasible and greedy_feasible:
            delta_greedy = sho_metrics['objective_raw_sho'] - greedy_metrics['objective_raw_greedy']
            gaps['gap_sho_to_greedy_raw'] = delta_greedy
            gaps['sho_beats_greedy'] = int(delta_greedy < -tolerance)
        else:
            gaps['gap_sho_to_greedy_raw'] = None
            gaps['sho_beats_greedy'] = 0
    
    # SHO vs Gurobi (reference)
    if gurobi_metrics is not None and gurobi_metrics.get('gurobi_ran', 0):
        gurobi_feasible = bool(gurobi_metrics.get('is_feasible_gurobi', 0))
        if sho_feasible and gurobi_feasible:
            delta_gurobi = sho_metrics['objective_raw_sho'] - gurobi_metrics['objective_raw_gurobi']
            gaps['gap_sho_to_gurobi_raw'] = delta_gurobi
            gaps['gap_sho_to_gurobi_energy'] = sho_metrics['energy_sho'] - gurobi_metrics['energy_gurobi']
            
            # Match Gurobi optimality (within tolerance)
            gaps['sho_matches_gurobi'] = int(abs(delta_gurobi) <= tolerance)
        else:
            gaps['gap_sho_to_gurobi_raw'] = None
            gaps['gap_sho_to_gurobi_energy'] = None
            gaps['sho_matches_gurobi'] = 0
    
    return gaps


def aggregate_results_by_scenario(
    results_df: pd.DataFrame,
    group_by: list = ['scenario_name', 'N']
) -> pd.DataFrame:
    """
    Aggregate results for EJOR tables.
    
    Reports median, IQR, win/tie/loss counts per scenario×N.
    Handles feasible-only and infeasible separately.
    
    Args:
        results_df: DataFrame with individual instance results
        group_by: Grouping columns (default: scenario, N)
        
    Returns:
        DataFrame with aggregated statistics
    """
    # Filter to feasible instances only for raw gaps
    feasible = results_df[
        (results_df['is_feasible_sho'] == 1) & 
        (results_df['is_feasible_sa'] == 1)
    ].copy()
    
    # Aggregate statistics
    agg_funcs = {
        'gap_sho_to_sa_raw': ['median', 'mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'gap_sho_to_sa_energy': ['median'],
        'sho_beats_sa': 'sum',
        'sho_ties_sa': 'sum',
        'sho_loses_sa': 'sum',
        'sho_runtime_sec': ['median', 'mean'],
        'sa_runtime_sec': ['median', 'mean']
    }
    
    # Add optional columns if present
    if 'gap_sho_to_gurobi_raw' in feasible.columns:
        agg_funcs['gap_sho_to_gurobi_raw'] = ['median', 'mean']
        agg_funcs['sho_matches_gurobi'] = 'sum'
    
    if 'sho_beats_greedy' in feasible.columns:
        agg_funcs['sho_beats_greedy'] = 'sum'
    
    aggregated = feasible.groupby(group_by).agg(agg_funcs)
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    # Add instance counts
    counts = results_df.groupby(group_by).size()
    feasible_counts = feasible.groupby(group_by).size()
    
    aggregated['total_instances'] = counts
    aggregated['feasible_instances'] = feasible_counts
    aggregated['feasibility_rate'] = (feasible_counts / counts * 100).round(1)
    
    return aggregated.reset_index()


def wilcoxon_signed_rank_test(
    results_df: pd.DataFrame,
    method1_col: str = 'objective_raw_sho',
    method2_col: str = 'objective_raw_sa_best'
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test for EJOR statistical reporting.
    
    Tests null hypothesis: method1 and method2 have same distribution.
    
    Args:
        results_df: Results with paired instances
        method1_col: Column for method 1 (e.g., SHO)
        method2_col: Column for method 2 (e.g., SA)
        
    Returns:
        Dict with test statistics
    """
    from scipy.stats import wilcoxon
    
    # Filter to feasible pairs
    feasible = results_df[
        (results_df['is_feasible_sho'] == 1) & 
        (results_df['is_feasible_sa'] == 1)
    ].copy()
    
    if len(feasible) < 10:
        return {
            'n_pairs': len(feasible),
            'statistic': None,
            'p_value': None,
            'note': 'Insufficient pairs for test (n<10)'
        }
    
    # Wilcoxon test
    stat, p_value = wilcoxon(
        feasible[method1_col],
        feasible[method2_col],
        alternative='two-sided',
        zero_method='wilcox'
    )
    
    # Effect size (median difference)
    median_diff = (feasible[method1_col] - feasible[method2_col]).median()
    
    # Win rate
    wins = (feasible[method1_col] < feasible[method2_col]).sum()
    ties = (feasible[method1_col] == feasible[method2_col]).sum()
    losses = (feasible[method1_col] > feasible[method2_col]).sum()
    
    return {
        'n_pairs': len(feasible),
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'significant_at_0.01': bool(p_value < 0.01),
        'median_difference': float(median_diff),
        'wins': int(wins),
        'ties': int(ties),
        'losses': int(losses),
        'win_rate': float(wins / len(feasible) * 100)
    }