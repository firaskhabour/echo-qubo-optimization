"""
Spectral Homotopy Optimization (SHO) for QUBO Problems
Structured Penalty-Ramp Continuation with Beam Search

This implements the penalty-ramp homotopy approach where:
1. One-hot constraints are enforced early (structural validity)
2. Regulatory constraints are ramped gradually (feasibility learning)
3. Affordability constraints are introduced last (economic preferences)
4. Spectral smoothing (diagonal shift) decays throughout

Reference: [Your EJOR Paper Title]
Authors: [Your Names]
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class SHOParameters:
    """Parameters for Spectral Homotopy Optimization"""
    # Homotopy schedule
    t_stages: List[float]  # e.g., [0, 0.25, 0.5, 0.75, 1.0]
    
    # Penalty strategy (NEW: controls how penalties evolve)
    penalty_strategy: str = "constant"  # Options: "ramp_up", "constant", "ramp_down"
    
    # Constraint ramp parameters (used by ramp_up and ramp_down strategies)
    tau_oh: float = 0.0  # One-hot ramp completion point (deprecated, kept for compatibility)
    p_reg: float = 2.0   # Regulatory ramp exponent (used for ramp_up)
    q_aff: float = 2.0   # Affordability ramp exponent (used for ramp_up)
    r_spectral: float = 2.0  # Spectral decay exponent (always used)
    
    # Beam search parameters
    R_initial: int = 20  # Initial restarts (e.g., 20)
    B_beam: int = 6     # Beam width (e.g., 6)
    R_fresh: int = 0    # Fresh restarts per stage (e.g., 2)
    
    # Budget allocation (must sum to 1.0)
    stage_budgets: List[float] = None  # Fraction per stage, e.g., [0.4, 0.2, 0.15, 0.25]
    total_budget: int = 800000  # Total SA steps
    
    # SA parameters
    T0: float = 5.0
    Tend: float = 0.01
    prem_move_prob: float = 0.075
    ded_move_prob: float = 0.075


def compute_lambda_0_gershgorin(Q: np.ndarray, epsilon_percentile: float = 0.05) -> float:
    """
    Compute initial spectral shift using Gershgorin lower bound.
    
    This avoids O(n^3) eigendecomposition by using the cheap Gershgorin bound:
    λ_min(Q) ≥ min_i (Q_ii - Σ_j≠i |Q_ij|)
    
    Args:
        Q: QUBO matrix (or Q_base for initial computation)
        epsilon_percentile: Safety margin as percentile of diagonal magnitudes
        
    Returns:
        λ_0: Initial spectral shift (0 if Q is already well-conditioned)
    """
    n = Q.shape[0]
    
    # Gershgorin lower bound for each row
    gershgorin_bounds = np.zeros(n)
    for i in range(n):
        row_sum = np.sum(np.abs(Q[i, :])) - np.abs(Q[i, i])
        gershgorin_bounds[i] = Q[i, i] - row_sum
    
    lambda_min_lower = np.min(gershgorin_bounds)
    
    # Safety margin
    epsilon = epsilon_percentile * np.median(np.abs(np.diag(Q)))
    
    # Return shift needed to make spectrum less negative
    lambda_0 = max(0.0, -lambda_min_lower + epsilon)
    
    return lambda_0


def build_Q_homotopy(
    t: float,
    Q_base: np.ndarray,
    Q_oh: np.ndarray,
    Q_reg: np.ndarray,
    Q_aff: Optional[np.ndarray],
    w_oh: float,
    w_reg: float,
    w_aff: float,
    lambda_0: float,
    params: SHOParameters
) -> np.ndarray:
    """
    Build Q(t) at homotopy parameter t ∈ [0, 1].
    
    Q(t) = Q_base + α(t)·w_oh·Q_oh + β(t)·w_reg·Q_reg 
           + δ(t)·w_aff·Q_aff + λ(t)·I
    
    Where:
    - α(t): One-hot ramp (fast, enforces structural validity)
    - β(t): Regulatory ramp (slower, learns feasibility)
    - δ(t): Affordability ramp (slowest, refines economics)
    - λ(t): Spectral shift (decays to 0)
    
    Args:
        t: Homotopy parameter in [0, 1]
        Q_base: Base objective QUBO
        Q_oh: One-hot constraint penalties
        Q_reg: Regulatory constraint penalties
        Q_aff: Affordability constraint penalties (or None)
        w_oh, w_reg, w_aff: Penalty weights
        lambda_0: Initial spectral shift
        params: SHO parameters
        
    Returns:
        Q_t: QUBO matrix at parameter t
    """
    n = Q_base.shape[0]
    
    # CRITICAL: One-hot enforced by projection, NOT by penalty
    # With w_onehot up to 20.6M, including penalty swamps objective and worsens ruggedness
    # Set α(t) = 0 (or tiny constant 0.01 for tie-breaking)
    alpha_t = 0.01 if params.tau_oh > 0 else 0.0  # Tiny tie-breaker if tau_oh specified
    
    # ============================================================================
    # PENALTY STRATEGY SELECTION (Controlled by params.penalty_strategy)
    # ============================================================================
    
    if params.penalty_strategy == "ramp_up":
        # 🔵 EXPERIMENT 1: Penalty Ramp UP (0% → 100%)
        # Penalties start at 0% and gradually increase to 100%
        # Hypothesis: Allows exploration of infeasible regions early
        beta_t = t ** params.p_reg  # e.g., t² → 0.00 at t=0, 1.00 at t=1
        delta_t = t ** params.q_aff if (Q_aff is not None and w_aff > 0) else 0.0
        
    elif params.penalty_strategy == "constant":
        # 🟢 EXPERIMENT 2: No Penalty Ramp (Always 100%)
        # Penalties remain constant at 100% throughout
        # Hypothesis: Maintains landscape consistency like SA/Gurobi
        beta_t = 1.0
        delta_t = 1.0 if (Q_aff is not None and w_aff > 0) else 0.0
        
    elif params.penalty_strategy == "ramp_down":
        # 🟣 EXPERIMENT 3: Penalty Ramp DOWN (200% → 100%)
        # Penalties start at 200% (over-constrained) and relax to 100%
        # Hypothesis: Ensures feasibility early, allows optimization later
        beta_t = 1.0 + (1.0 - t) ** 2  # 2.00 at t=0, 1.00 at t=1
        delta_t = (1.0 + (1.0 - t) ** 2) if (Q_aff is not None and w_aff > 0) else 0.0
        
    else:
        raise ValueError(f"Unknown penalty_strategy: {params.penalty_strategy}. "
                        f"Must be one of: 'ramp_up', 'constant', 'ramp_down'")
    
    # Spectral shift: Quadratic decay λ(t) = λ₀(1-t)²
    # This is ALWAYS applied regardless of penalty strategy
    # Smooths early landscape, vanishes at t=1
    lambda_t = lambda_0 * ((1.0 - t) ** params.r_spectral)
    
    # Build Q(t)
    Q_t = Q_base.copy()  # 🔴 CRITICAL: Create fresh copy, do not mutate
    Q_t += alpha_t * w_oh * Q_oh
    Q_t += beta_t * w_reg * Q_reg
    
    if Q_aff is not None and w_aff > 0:
        Q_t += delta_t * w_aff * Q_aff
    
    # Add spectral shift to diagonal
    if lambda_t > 1e-10:
        np.fill_diagonal(Q_t, np.diag(Q_t) + lambda_t)
    
    # 🔴 Safety check: Verify energy scales are reasonable
    max_q = np.max(np.abs(Q_t))
    if max_q > 1e10:  # Sanity threshold
        import warnings
        warnings.warn(
            f"Q(t={t:.2f}) has very large values: max={max_q:.2e}. "
            f"This may indicate penalty scaling issue. "
            f"w_oh={w_oh:.2e}, w_reg={w_reg:.2e}, β(t)={beta_t:.4f}"
        )
    
    return Q_t


def spectral_homotopy_QUBO(
    Q_base: np.ndarray,
    Q_oh: np.ndarray,
    Q_reg: np.ndarray,
    Q_aff: Optional[np.ndarray],
    N: int,
    M: int,
    K: int,
    L: int,
    w_oh: float,
    w_reg: float,
    w_aff: float,
    seed: int,
    params: SHOParameters,
    SA_solver,  # Function: SA_solver(Q, N, M, K, L, rng, steps, T0, Tend) -> (solution, energy)
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Spectral Homotopy Optimization with Beam Continuation.
    
    Algorithm:
    1. Compute λ_0 from spectral analysis of Q_base
    2. For each stage t in homotopy schedule:
       a. Build Q(t) with ramped penalties
       b. Run beam of SA searches (warm-started from previous stage)
       c. Add fresh restarts for diversity
       d. Keep top B candidates
    3. Return best solution found
    
    Args:
        Q_base, Q_oh, Q_reg, Q_aff: QUBO decomposition
        N, M, K, L: Problem dimensions
        w_oh, w_reg, w_aff: Penalty weights
        seed: Random seed
        params: SHO parameters
        SA_solver: Simulated annealing function
        verbose: Print progress
        
    Returns:
        Dict with solution, energy, diagnostics
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Compute initial spectral shift
    lambda_0 = compute_lambda_0_gershgorin(Q_base)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SPECTRAL HOMOTOPY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"  λ_0 (spectral shift): {lambda_0:.2f}")
        print(f"  Stages: {params.t_stages}")
        print(f"  Initial restarts: {params.R_initial}")
        print(f"  Beam width: {params.B_beam}")
    
    # Beam: list of (solution, energy) tuples
    beam = []
    
    # Stage 0: Initial diversification on smoothest landscape
    t_0 = params.t_stages[0]
    Q_0 = build_Q_homotopy(t_0, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
    
    # 🔴 FIX: Distribute stage budget properly to avoid truncation losses
    stage_0_total = int(params.stage_budgets[0] * params.total_budget)
    base_steps_per_restart = stage_0_total // params.R_initial
    remainder = stage_0_total % params.R_initial
    
    # 🔴 CRITICAL: Track exact steps used for budget verification
    total_steps_used = 0
    
    if verbose:
        print(f"\n[Stage 0] t={t_0:.2f}, Q smoothed, {params.R_initial} restarts")
        print(f"  Budget: {stage_0_total:,} total, {base_steps_per_restart:,}-{base_steps_per_restart+1:,} per restart")
    
    for r in range(params.R_initial):
        # Distribute remainder to first 'remainder' restarts
        steps_for_this_restart = base_steps_per_restart + (1 if r < remainder else 0)
        
        solution, energy = SA_solver(
            Q_0, N, M, K, L, rng, 
            steps=steps_for_this_restart,
            T0=params.T0,
            Tend=params.Tend
        )
        total_steps_used += steps_for_this_restart  # Count every SA run with actual steps
        beam.append((solution.copy(), energy, 0))  # (solution, energy, stage_found)
    
    # Verify Stage 0 budget
    assert total_steps_used == stage_0_total, \
        f"Stage 0 budget error: {total_steps_used} != {stage_0_total}"
    
    # Keep top B
    beam.sort(key=lambda x: x[1])
    beam = beam[:params.B_beam]
    
    if verbose:
        print(f"  Best energy: {beam[0][1]:.4f}")
        print(f"  Steps used so far: {total_steps_used:,} / {params.total_budget:,}")
    
    # Subsequent stages: Beam continuation with tightening
    # 🔴 CRITICAL: Track trajectory for paper plots
    stage_trajectory = [{
        'stage': 0,
        't': t_0,
        'best_energy': beam[0][1],
        'beam_size': len(beam),
        'steps_used': total_steps_used
    }]
    
    for stage_idx in range(1, len(params.t_stages)):
        t_k = params.t_stages[stage_idx]
        
        # 🔴 CRITICAL: Rebuild Q(t) cleanly (do not mutate in place)
        Q_k = build_Q_homotopy(t_k, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
        
        # Verify Q(1.0) matches original at final stage
        if abs(t_k - 1.0) < 1e-10:
            Q_original = Q_base + w_reg * Q_reg
            if Q_aff is not None and w_aff > 0:
                Q_original += w_aff * Q_aff
            if not np.allclose(Q_k, Q_original, rtol=1e-8):
                raise ValueError(f"Q(t=1) does not match original QUBO! Max diff: {np.max(np.abs(Q_k - Q_original))}")
        
        # Beam tightening: Keep B=6 until stage 3, then tighten to B=4
        beam_size = 4 if stage_idx >= 3 else params.B_beam
        
        # 🔴 FIX: Final stage (t=1.0) only refines beam, no fresh restarts
        is_final_stage = (stage_idx == len(params.t_stages) - 1)
        n_fresh = 0 if is_final_stage else params.R_fresh
        n_candidates = len(beam) + n_fresh
        
        # 🔴 FIX: Compute budget for this stage with proper remainder distribution
        stage_total_steps = int(params.stage_budgets[stage_idx] * params.total_budget)
        
        # Base steps per candidate
        base_steps_per_candidate = stage_total_steps // n_candidates
        remainder = stage_total_steps % n_candidates
        
        # Distribute remainder to first 'remainder' candidates
        steps_per_candidate = []
        for c in range(n_candidates):
            if c < remainder:
                steps_per_candidate.append(base_steps_per_candidate + 1)
            else:
                steps_per_candidate.append(base_steps_per_candidate)
        
        # Verify
        assert sum(steps_per_candidate) == stage_total_steps, \
            f"Stage {stage_idx} budget error: {sum(steps_per_candidate)} != {stage_total_steps}"
        
        if verbose:
            fresh_msg = f"{n_fresh} fresh" if n_fresh > 0 else "no fresh"
            print(f"\n[Stage {stage_idx}] t={t_k:.2f}, {n_candidates} candidates ({len(beam)} beam + {fresh_msg}), target beam={beam_size}")
            print(f"  Budget: {stage_total_steps:,} steps total, {base_steps_per_candidate:,}-{base_steps_per_candidate+1:,} per candidate")
        
        new_beam = []
        candidate_idx = 0
        
        # Propagate beam (warm-started)
        for solution, _, _ in beam:
            refined_solution, refined_energy = SA_solver(
                Q_k, N, M, K, L, rng,
                steps=steps_per_candidate[candidate_idx],  # Use distributed budget
                T0=params.T0,  # 🔴 CRITICAL: Same T0 as baseline
                Tend=params.Tend,  # 🔴 CRITICAL: Same Tend as baseline
                initial_solution=solution  # Warm-start
            )
            total_steps_used += steps_per_candidate[candidate_idx]  # 🔴 Count actual steps used
            candidate_idx += 1
            new_beam.append((refined_solution.copy(), refined_energy, stage_idx))
        
        # Add fresh restarts for diversity (except final stage)
        for _ in range(n_fresh):
            fresh_solution, fresh_energy = SA_solver(
                Q_k, N, M, K, L, rng,
                steps=steps_per_candidate[candidate_idx],  # Use distributed budget
                T0=params.T0,
                Tend=params.Tend
            )
            total_steps_used += steps_per_candidate[candidate_idx]  # 🔴 Count actual steps used
            candidate_idx += 1
            new_beam.append((fresh_solution.copy(), fresh_energy, stage_idx))
        
        # Keep top B (with tightening at stage 3)
        new_beam.sort(key=lambda x: x[1])
        beam = new_beam[:beam_size]
        
        # Record trajectory
        stage_trajectory.append({
            'stage': stage_idx,
            't': t_k,
            'best_energy': beam[0][1],
            'beam_size': len(beam),
            'steps_used': total_steps_used
        })
        
        if verbose:
            print(f"  Best energy: {beam[0][1]:.4f}, Beam size: {len(beam)}")
            print(f"  Steps used so far: {total_steps_used:,} / {params.total_budget:,}")
    
    # Final solution
    best_solution, best_energy, best_stage = beam[0]
    
    runtime = time.time() - start_time
    
    # 🔴 CRITICAL: Verify exact budget was used
    if total_steps_used != params.total_budget:
        # Diagnostic: Show where steps went
        print(f"\n❌ BUDGET VIOLATION DETECTED:")
        print(f"   Expected: {params.total_budget:,} steps")
        print(f"   Used: {total_steps_used:,} steps")
        print(f"   Difference: {total_steps_used - params.total_budget:,}")
        print(f"\n   Budget breakdown:")
        for i, (t, frac) in enumerate(zip(params.t_stages, params.stage_budgets)):
            expected = int(frac * params.total_budget)
            print(f"     Stage {i} (t={t:.2f}): {frac*100:.1f}% = {expected:,} steps")
        print(f"   Total from fractions: {sum(int(f * params.total_budget) for f in params.stage_budgets):,}")
        
        raise ValueError(
            f"Budget violation! Used {total_steps_used} steps, expected {params.total_budget}. "
            f"Difference: {total_steps_used - params.total_budget}"
        )
    
    # 🔴 Safety check: Detect NaN or overflow
    if np.isnan(best_energy) or np.isinf(best_energy):
        raise ValueError(f"Invalid energy detected: {best_energy}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SHO COMPLETE: Energy={best_energy:.4f}, Runtime={runtime:.2f}s")
        print(f"Best found at stage: {best_stage}")
        print(f"✓ Budget verified: {total_steps_used:,} steps used")
        print(f"{'='*70}\n")
    
    return {
        'solution': best_solution,
        'energy': best_energy,
        'best_stage': best_stage,
        'lambda_0': lambda_0,
        'runtime': runtime,
        'beam_history': beam,  # Top B solutions
        'stage_trajectory': stage_trajectory,  # 🔴 For paper plots
        'total_steps_used': total_steps_used,  # 🔴 For verification
        'method': 'SHO'
    }


# ============================================================================
# DEFAULT PARAMETERS (will be updated with ChatGPT's exact specifications)
# ============================================================================

def get_default_SHO_parameters() -> SHOParameters:
    """
    Default SHO parameters based on ChatGPT specifications.
    
    Key design principles:
    - α(t)=0: One-hot enforced by projection, not penalty (w_onehot up to 20.6M too large)
    - β(t)=t²: Regulatory quadratic ramp (smooth introduction of 10-40M penalties)
    - λ(t)=λ₀(1-t)²: Spectral smoothing decays quadratically
    - Budget: 800K total, front-loaded exploration (30%) + final refinement (20%)
    - Beam: Diversity early (B=6), efficiency late (B=4)
    
    🔴 CRITICAL: Stage budgets adjusted to avoid integer rounding errors
    """
    # Budget allocation percentages (must sum to 1.0)
    # ADJUSTED to ensure integer division gives exactly 800K
    stage_fractions = [0.30, 0.20, 0.15, 0.15, 0.20]
    total_budget = 800_000
    
    # Compute actual integer steps per stage
    stage_steps = [int(frac * total_budget) for frac in stage_fractions]
    
    # Fix rounding error by adjusting last stage
    actual_total = sum(stage_steps)
    if actual_total != total_budget:
        stage_steps[-1] += (total_budget - actual_total)
    
    # Verify (critical assertion)
    assert sum(stage_steps) == total_budget, f"Budget allocation error: {sum(stage_steps)} != {total_budget}"
    
    # Convert back to fractions for storage
    verified_fractions = [s / total_budget for s in stage_steps]
    
    return SHOParameters(
        # Homotopy schedule (5 stages)
        t_stages=[0.0, 0.25, 0.5, 0.75, 1.0],
        
        # Constraint ramp parameters (ChatGPT final spec)
        tau_oh=0.0,      # One-hot via projection only (α=0)
        p_reg=2.0,       # Quadratic regulatory ramp: β(t)=t²
        q_aff=2.0,       # Quadratic affordability ramp: δ(t)=t²
        r_spectral=2.0,  # Quadratic spectral decay: λ(t)=λ₀(1-t)²
        
        # Beam parameters (multi-start essential, SHO improves success probability)
        R_initial=20,    # Match baseline
        B_beam=6,        # Keep top 6 (stages 0-2)
        R_fresh=2,       # Add 2 fresh per stage (diversity)
        
        # Budget allocation (verified to sum exactly to 800K)
        stage_budgets=verified_fractions,
        total_budget=total_budget,
        
        # SA parameters (match baseline)
        T0=5.0,
        Tend=0.01,
        prem_move_prob=0.075,
        ded_move_prob=0.075
    )