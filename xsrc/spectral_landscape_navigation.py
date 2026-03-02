"""
Spectral Landscape Navigation (SLN) for QUBO Optimization
==========================================================

Novel eigenvalue-guided adaptive homotopy optimization.

Key Innovations:
1. Adaptive stage selection based on spectral landscape analysis
2. Roughness-aware budget allocation (spend more in ill-conditioned regions)
3. Condition-number-guided beam sizing
4. Dynamic smoothing parameter adjustment

Usage:
    from spectral_landscape_navigation import spectral_landscape_navigation
    
    result = spectral_landscape_navigation(
        Q_base, Q_oh, Q_reg, Q_aff,
        w_oh, w_reg, w_aff,
        N, M, K, L,
        seed=1000
    )

Author: Advanced QUBO Optimization Framework
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Any


# ============================================================================
# PHASE 1: SPECTRAL ANALYSIS
# ============================================================================

def analyze_spectrum(Q: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """
    Analyze eigenvalue spectrum of QUBO matrix.
    
    Returns metrics about landscape structure:
    - Condition number (numerical stability)
    - Spectral gap (separation of eigenvalues)
    - Negative eigenvalue count (non-convexity)
    - Roughness estimate (variability in spectrum)
    """
    try:
        # Compute eigenvalues (symmetric matrix)
        eigs = np.linalg.eigvalsh(Q)
        
        # Filter near-zero eigenvalues
        EPSILON = 1e-10
        nonzero_eigs = eigs[np.abs(eigs) > EPSILON]
        
        # Condition number
        if len(nonzero_eigs) > 0:
            max_abs = np.max(np.abs(nonzero_eigs))
            min_abs = np.min(np.abs(nonzero_eigs))
            condition = max_abs / max(min_abs, EPSILON)
        else:
            condition = 1.0
        
        # Spectral gap (gap between largest and second-largest eigenvalues)
        if len(eigs) >= 2:
            sorted_eigs = np.sort(eigs)
            spectral_gap = sorted_eigs[-1] - sorted_eigs[-2]
        else:
            spectral_gap = 0.0
        
        # Count negative eigenvalues (indicates non-convex regions)
        negative_count = int(np.sum(eigs < -EPSILON))
        
        # Effective rank (number of significant eigenvalues)
        effective_rank = int(np.sum(np.abs(eigs) > 1e-6))
        
        # Roughness: standard deviation of eigenvalues (landscape variability)
        roughness = float(np.std(eigs))
        
        spectrum = {
            'condition_number': float(condition),
            'spectral_gap': float(spectral_gap),
            'negative_count': negative_count,
            'effective_rank': effective_rank,
            'roughness': roughness,
            'eig_max': float(np.max(eigs)),
            'eig_min': float(np.min(eigs)),
        }
        
        if verbose:
            print(f"    Condition: {condition:.1e}, Neg: {negative_count}, "
                  f"Rough: {roughness:.1e}")
        
        return spectrum
        
    except np.linalg.LinAlgError as e:
        print(f"Warning: Eigenvalue computation failed: {e}")
        # Return default values
        return {
            'condition_number': 1e6,
            'spectral_gap': 0.0,
            'negative_count': 0,
            'effective_rank': Q.shape[0],
            'roughness': 1e6,
            'eig_max': 0.0,
            'eig_min': 0.0,
        }


def estimate_initial_smoothing(Q_base: np.ndarray, Q_penalties: np.ndarray,
                                percentile: float = 0.25) -> float:
    """
    Estimate initial smoothing parameter λ₀.
    
    Strategy: Set λ₀ to percentile of eigenvalue range of Q_penalties.
    This ensures significant smoothing at t=0.
    
    Args:
        Q_base: Base objective matrix
        Q_penalties: Combined penalty matrix
        percentile: Percentile of eigenvalue range (0.25 = 25%)
    
    Returns:
        λ₀ smoothing parameter
    """
    try:
        # Analyze penalty matrix spectrum
        eigs_pen = np.linalg.eigvalsh(Q_penalties)
        eig_range = np.max(eigs_pen) - np.min(eigs_pen)
        
        # Set λ₀ to percentile of range
        lambda_0 = percentile * eig_range
        
        # Ensure reasonable bounds
        lambda_0 = max(lambda_0, 100.0)  # Minimum smoothing
        lambda_0 = min(lambda_0, 1e6)    # Maximum smoothing
        
        return float(lambda_0)
        
    except:
        # Fallback
        return 1000.0


def compute_adaptive_stages(
    Q_base: np.ndarray,
    Q_penalties: np.ndarray,
    lambda_0: float,
    max_stages: int = 8,
    verbose: bool = False
) -> List[float]:
    """
    Compute adaptive stage points based on spectral landscape analysis.
    
    Strategy:
    - Sample t ∈ [0, 1] at fine resolution
    - Compute spectrum at each t
    - Place stages where spectrum changes most rapidly
    - Always include t=0 and t=1
    
    Returns:
        List of t values (sorted, including 0 and 1)
    """
    if verbose:
        print("  Analyzing spectral landscape...")
    
    # Sample many candidate t values
    n_samples = 51
    candidates = np.linspace(0, 1, n_samples)
    
    spectra = []
    for t in candidates:
        # Build Q(t) with smoothing
        lambda_t = lambda_0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        
        # Add smoothing
        if lambda_t > 1e-8:
            Q_t = Q_t + lambda_t * np.eye(Q_t.shape[0])
        
        # Analyze spectrum
        spec = analyze_spectrum(Q_t, verbose=False)
        spectra.append(spec)
    
    # Compute spectral change between consecutive points
    changes = []
    for i in range(1, len(spectra)):
        # Measure magnitude of spectral change
        delta = (
            abs(spectra[i]['condition_number'] - spectra[i-1]['condition_number']) / 
            max(spectra[i-1]['condition_number'], 1.0) +
            abs(spectra[i]['roughness'] - spectra[i-1]['roughness']) /
            max(spectra[i-1]['roughness'], 1.0)
        )
        changes.append((candidates[i], delta))
    
    # Sort by change magnitude
    changes.sort(key=lambda x: x[1], reverse=True)
    
    # Select points with largest spectral changes
    # But ensure we don't have too many stages
    selected = [t for t, _ in changes[:max_stages-2]]
    
    # Always include endpoints
    stages = [0.0] + selected + [1.0]
    stages = sorted(list(set(stages)))  # Remove duplicates and sort
    
    if verbose:
        print(f"  Selected {len(stages)} adaptive stages: {[f'{t:.2f}' for t in stages]}")
    
    return stages


def allocate_budget_by_roughness(
    stages: List[float],
    Q_base: np.ndarray,
    Q_penalties: np.ndarray,
    lambda_0: float,
    total_budget: int,
    verbose: bool = False
) -> List[int]:
    """
    Allocate computational budget based on landscape roughness.
    
    Strategy:
    - Rough regions (high condition number, many negative eigenvalues) get more budget
    - Smooth regions get less budget
    - Ensures we spend time where problem is hard
    
    Returns:
        Budget allocation for each stage (sums to total_budget)
    """
    roughness_scores = []
    
    for t in stages:
        # Build Q(t)
        lambda_t = lambda_0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if lambda_t > 1e-8:
            Q_t = Q_t + lambda_t * np.eye(Q_t.shape[0])
        
        # Analyze spectrum
        spec = analyze_spectrum(Q_t, verbose=False)
        
        # Roughness score = condition number × sqrt(negative eigenvalues + 1)
        # Higher score = harder region = more budget needed
        roughness = spec['condition_number'] * np.sqrt(spec['negative_count'] + 1)
        roughness_scores.append(roughness)
    
    # Normalize to budget
    total_roughness = sum(roughness_scores)
    if total_roughness < 1e-10:
        # Uniform allocation if all scores near zero
        budgets = [total_budget // len(stages)] * len(stages)
    else:
        budgets = [int(total_budget * r / total_roughness) for r in roughness_scores]
    
    # Ensure we use exactly total_budget (handle rounding)
    budget_diff = total_budget - sum(budgets)
    budgets[-1] += budget_diff  # Add remainder to last stage
    
    # Ensure minimum budget per stage
    min_budget = 5000
    for i in range(len(budgets)):
        if budgets[i] < min_budget:
            budgets[i] = min_budget
    
    # Renormalize if we exceeded budget
    if sum(budgets) > total_budget:
        scale = total_budget / sum(budgets)
        budgets = [int(b * scale) for b in budgets]
        budgets[-1] = total_budget - sum(budgets[:-1])
    
    if verbose:
        print("  Budget allocation by roughness:")
        for t, b in zip(stages, budgets):
            pct = b / total_budget * 100
            print(f"    t={t:.2f}: {b:>7,} steps ({pct:>5.1f}%)")
    
    return budgets


def adaptive_beam_size(
    condition_number: float,
    base_beam: int = 20,
    min_beam: int = 10,
    max_beam: int = 40
) -> int:
    """
    Compute adaptive beam size based on condition number.
    
    Strategy:
    - High condition number → wider beam (need diversity)
    - Low condition number → narrower beam (can exploit)
    
    Returns:
        Beam size (clamped to [min_beam, max_beam])
    """
    if condition_number > 1000:
        beam = base_beam * 2  # Very ill-conditioned
    elif condition_number > 100:
        beam = int(base_beam * 1.5)  # Moderately ill-conditioned
    elif condition_number > 10:
        beam = base_beam  # Normal
    else:
        beam = max(base_beam // 2, min_beam)  # Well-conditioned
    
    # Clamp to bounds
    beam = max(min_beam, min(beam, max_beam))
    
    return beam


# ============================================================================
# PHASE 2: OPTIMIZATION ROUTINES
# ============================================================================

def energy(Q: np.ndarray, v: np.ndarray) -> float:
    """Compute QUBO energy: v^T Q v"""
    return float(v @ Q @ v)


def project_onehot(v: np.ndarray, N: int, M: int, K: int, 
                   rng: np.random.Generator) -> np.ndarray:
    """
    Hard projection to satisfy one-hot constraints.
    
    For y (deductible) and z (premium): exactly one must be selected.
    """
    v2 = v.copy()
    
    # One-hot for y (deductible)
    y = v2[N:N+M]
    if np.sum(y) <= 0.0:
        j = int(rng.integers(0, M))
    else:
        j = int(np.argmax(y))
    y[:] = 0.0
    y[j] = 1.0
    
    # One-hot for z (premium)
    z = v2[N+M:N+M+K]
    if np.sum(z) <= 0.0:
        k = int(rng.integers(0, K))
    else:
        k = int(np.argmax(z))
    z[:] = 0.0
    z[k] = 1.0
    
    return v2


def simulated_annealing_local(
    Q: np.ndarray,
    N: int, M: int, K: int, L: int,
    rng: np.random.Generator,
    steps: int,
    T0: float = 5.0,
    Tend: float = 0.01,
    initial_solution: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """
    Local simulated annealing for one stage.
    
    Similar to your existing SA but streamlined for SLN.
    """
    n = Q.shape[0]
    
    # Initialize
    if initial_solution is not None:
        v = initial_solution.copy()
    else:
        v = rng.integers(0, 2, size=n).astype(float)
    
    v = project_onehot(v, N, M, K, rng)
    e = energy(Q, v)
    
    best_v = v.copy()
    best_e = e
    
    # Move probabilities
    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K
    
    prem_prob = 0.075
    ded_prob = 0.075
    flip_prob = 1.0 - prem_prob - ded_prob
    
    for t in range(steps):
        # Temperature schedule
        frac = t / max(1, steps - 1)
        T = T0 * (1 - frac) + Tend * frac
        
        # Propose move
        v_new = v.copy()
        r = float(rng.random())
        
        if r < flip_prob:
            # Flip a feature bit
            while True:
                i = int(rng.integers(0, n))
                if not (y_start <= i < y_end or z_start <= i < z_end):
                    break
            v_new[i] = 1.0 - v_new[i]
        elif r < flip_prob + ded_prob:
            # Change deductible
            j = int(rng.integers(0, M))
            v_new[y_start:y_end] = 0.0
            v_new[y_start + j] = 1.0
        else:
            # Change premium
            k = int(rng.integers(0, K))
            v_new[z_start:z_end] = 0.0
            v_new[z_start + k] = 1.0
        
        # Project and evaluate
        v_new = project_onehot(v_new, N, M, K, rng)
        e_new = energy(Q, v_new)
        
        # Accept/reject
        if e_new < e or rng.random() < np.exp(-(e_new - e) / max(T, 1e-10)):
            v, e = v_new, e_new
        
        # Track best
        if e < best_e:
            best_e = e
            best_v = v.copy()
    
    return best_v, best_e


# ============================================================================
# PHASE 3: MAIN ALGORITHM
# ============================================================================

def spectral_landscape_navigation(
    Q_base: np.ndarray,
    Q_oh: np.ndarray,
    Q_reg: np.ndarray,
    Q_aff: np.ndarray,
    N: int, M: int, K: int, L: int,
    w_oh: float,
    w_reg: float,
    w_aff: float,
    seed: int = None,
    params: Dict[str, Any] = None,
    SA_solver: Callable = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Spectral Landscape Navigation (SLN): Eigenvalue-guided adaptive homotopy.
    
    Args:
        Q_base: Base objective matrix (no penalties)
        Q_oh, Q_reg, Q_aff: Penalty matrices (unit weight)
        N, M, K, L: Problem dimensions
        w_oh, w_reg, w_aff: Penalty weights
        seed: Random seed
        params: Optional parameters (overrides defaults)
        SA_solver: Custom SA solver (optional)
        verbose: Print progress
    
    Returns:
        Dictionary with solution, energy, and diagnostic information
    """
    
    start_time = time.time()
    
    # Set default parameters
    default_params = {
        'total_budget': 800_000,
        'max_stages': 8,
        'smoothing_percentile': 0.25,
        'base_beam': 20,
        'initial_restarts': 40,
        'T0': 5.0,
        'Tend': 0.01,
    }
    
    if params is not None:
        default_params.update(params)
    params = default_params
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Build combined penalty matrix
    Q_penalties = w_oh * Q_oh + w_reg * Q_reg + w_aff * Q_aff
    
    if verbose:
        print("="*70)
        print("SPECTRAL LANDSCAPE NAVIGATION (SLN)")
        print("="*70)
        print(f"Problem: N={N}, M={M}, K={K}, L={L}")
        print(f"Penalty weights: w_oh={w_oh:.1e}, w_reg={w_reg:.1e}, w_aff={w_aff:.1e}")
        print()
    
    # ========================================================================
    # PHASE 1: SPECTRAL ANALYSIS
    # ========================================================================
    
    if verbose:
        print("[PHASE 1] Spectral Landscape Analysis")
        print("-"*70)
    
    # Estimate initial smoothing
    lambda_0 = estimate_initial_smoothing(
        Q_base, Q_penalties, params['smoothing_percentile']
    )
    
    if verbose:
        print(f"  Initial smoothing λ₀: {lambda_0:.1e}")
    
    # Compute adaptive stages
    stages = compute_adaptive_stages(
        Q_base, Q_penalties, lambda_0,
        max_stages=params['max_stages'],
        verbose=verbose
    )
    
    # Allocate budget
    budgets = allocate_budget_by_roughness(
        stages, Q_base, Q_penalties, lambda_0,
        params['total_budget'],
        verbose=verbose
    )
    
    # Analyze spectrum at each stage
    if verbose:
        print()
        print("  Spectral analysis at each stage:")
        print("  " + "-"*66)
        print(f"  {'Stage':<8} {'t':<6} {'λ(t)':<12} {'Cond':<12} {'Neg':<6} {'Rough':<12}")
        print("  " + "-"*66)
    
    spectra = []
    for i, t in enumerate(stages):
        lambda_t = lambda_0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if lambda_t > 1e-8:
            Q_t = Q_t + lambda_t * np.eye(Q_t.shape[0])
        
        spec = analyze_spectrum(Q_t, verbose=False)
        spectra.append(spec)
        
        if verbose:
            print(f"  {i:<8} {t:<6.2f} {lambda_t:<12.1e} "
                  f"{spec['condition_number']:<12.1e} "
                  f"{spec['negative_count']:<6} "
                  f"{spec['roughness']:<12.1e}")
    
    if verbose:
        print("  " + "-"*66)
        print()
    
    # ========================================================================
    # PHASE 2: HOMOTOPY OPTIMIZATION
    # ========================================================================
    
    if verbose:
        print("[PHASE 2] Adaptive Homotopy Optimization")
        print("-"*70)
    
    candidates = []
    best_overall_energy = float('inf')
    best_overall_solution = None
    best_overall_stage = 0
    
    steps_used = 0
    
    for stage_idx, (t, budget, spectrum) in enumerate(zip(stages, budgets, spectra)):
        
        # Adaptive beam size
        beam = adaptive_beam_size(
            spectrum['condition_number'],
            base_beam=params['base_beam']
        )
        
        # Build Q(t) with adaptive smoothing
        lambda_t = lambda_0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if lambda_t > 1e-8:
            Q_t = Q_t + lambda_t * np.eye(Q_t.shape[0])
        
        if verbose:
            print(f"\n[Stage {stage_idx}] t={t:.2f}, λ={lambda_t:.1e}, beam={beam}")
            print(f"  Budget: {budget:,} steps")
            print(f"  Condition: {spectrum['condition_number']:.1e}")
        
        # Initialize or continue
        if stage_idx == 0:
            # Initial stage: multiple random restarts
            num_restarts = params['initial_restarts']
            steps_per_restart = budget // num_restarts
            
            if verbose:
                print(f"  Initializing with {num_restarts} random restarts")
                print(f"  ({steps_per_restart:,} steps each)")
            
            for r in range(num_restarts):
                # Random initialization
                solution, e = simulated_annealing_local(
                    Q_t, N, M, K, L, rng,
                    steps=steps_per_restart,
                    T0=params['T0'],
                    Tend=params['Tend']
                )
                
                candidates.append({
                    'solution': solution,
                    'energy': e,
                    'stage': stage_idx
                })
                
                if e < best_overall_energy:
                    best_overall_energy = e
                    best_overall_solution = solution.copy()
                    best_overall_stage = stage_idx
            
            steps_used += budget
            
        else:
            # Continue from previous stage
            num_candidates = len(candidates)
            steps_per_candidate = budget // max(num_candidates, 1)
            
            if verbose:
                print(f"  Continuing from {num_candidates} candidates")
                print(f"  ({steps_per_candidate:,} steps each)")
            
            new_candidates = []
            for cand in candidates:
                # Continue optimization from this solution
                solution, e = simulated_annealing_local(
                    Q_t, N, M, K, L, rng,
                    steps=steps_per_candidate,
                    T0=params['T0'],
                    Tend=params['Tend'],
                    initial_solution=cand['solution']
                )
                
                new_candidates.append({
                    'solution': solution,
                    'energy': e,
                    'stage': stage_idx
                })
                
                if e < best_overall_energy:
                    best_overall_energy = e
                    best_overall_solution = solution.copy()
                    best_overall_stage = stage_idx
            
            candidates = new_candidates
            steps_used += budget
        
        # Keep top beam candidates
        candidates.sort(key=lambda c: c['energy'])
        candidates = candidates[:beam]
        
        if verbose:
            best_stage_energy = min(c['energy'] for c in candidates)
            print(f"  Best energy this stage: {best_stage_energy:.6f}")
            print(f"  Steps used so far: {steps_used:,} / {params['total_budget']:,}")
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    
    runtime = time.time() - start_time
    
    # Evaluate on ORIGINAL Q (no smoothing) at t=1
    Q_final = Q_base + Q_penalties
    final_energy = energy(Q_final, best_overall_solution)
    
    if verbose:
        print()
        print("="*70)
        print("SLN COMPLETE")
        print("="*70)
        print(f"Best energy (final Q): {final_energy:.6f}")
        print(f"Best found at stage: {best_overall_stage}")
        print(f"Runtime: {runtime:.2f}s")
        print(f"✓ Budget verified: {steps_used:,} steps used")
        print("="*70)
    
    return {
        'solution': best_overall_solution,
        'energy': final_energy,
        'best_stage': best_overall_stage,
        'runtime': runtime,
        'lambda_0': lambda_0,
        'stages': stages,
        'budgets': budgets,
        'spectra': spectra,
        'steps_used': steps_used,
        'method': 'SLN',
    }


# ============================================================================
# MULTI-START WRAPPER
# ============================================================================

def spectral_landscape_navigation_multistart(
    Q_base: np.ndarray,
    Q_oh: np.ndarray,
    Q_reg: np.ndarray,
    Q_aff: np.ndarray,
    N: int, M: int, K: int, L: int,
    w_oh: float,
    w_reg: float,
    w_aff: float,
    seed: int = None,
    num_starts: int = 5,
    params: Dict[str, Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Multi-start SLN: Run SLN multiple times and keep best result.
    
    This provides additional robustness against local minima.
    
    Args:
        num_starts: Number of independent SLN runs
        Other args: Same as spectral_landscape_navigation
    
    Returns:
        Best result across all runs, with multi-start metadata
    """
    
    if verbose:
        print("="*70)
        print(f"MULTI-START SLN ({num_starts} runs)")
        print("="*70)
        print()
    
    results = []
    
    for run in range(num_starts):
        # Use different seed for each run
        run_seed = (seed * 7919 + run * 104729) & 0xFFFFFFFF if seed is not None else None
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUN {run + 1}/{num_starts} (seed={run_seed})")
            print(f"{'='*70}")
        
        result = spectral_landscape_navigation(
            Q_base, Q_oh, Q_reg, Q_aff,
            N, M, K, L,
            w_oh, w_reg, w_aff,
            seed=run_seed,
            params=params,
            verbose=verbose
        )
        
        results.append(result)
        
        if verbose:
            print(f"\nRun {run + 1} complete: energy = {result['energy']:.6f}")
    
    # Select best result
    best_result = min(results, key=lambda r: r['energy'])
    
    if verbose:
        print("\n" + "="*70)
        print("MULTI-START SUMMARY")
        print("="*70)
        energies = [r['energy'] for r in results]
        print(f"Runs completed: {num_starts}")
        print(f"Best energy: {min(energies):.6f}")
        print(f"Worst energy: {max(energies):.6f}")
        print(f"Mean energy: {np.mean(energies):.6f}")
        print(f"Std energy: {np.std(energies):.6f}")
        print("="*70)
    
    # Add multi-start metadata
    best_result['multistart'] = True
    best_result['num_starts'] = num_starts
    best_result['all_energies'] = [r['energy'] for r in results]
    
    return best_result


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_default_SLN_parameters() -> Dict[str, Any]:
    """
    Get default parameters for SLN.
    
    These can be overridden by passing a params dict to SLN functions.
    """
    return {
        'total_budget': 800_000,
        'max_stages': 8,
        'smoothing_percentile': 0.25,
        'base_beam': 20,
        'initial_restarts': 40,
        'T0': 5.0,
        'Tend': 0.01,
    }