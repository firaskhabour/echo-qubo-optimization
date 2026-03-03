# src_code/solvers/echo_optimizer.py
"""
ECHO: Eigenvalue-guided Constrained Homotopy Optimization
==========================================================
Implements the ECHO algorithm described in Section 3 of the accompanying paper.

ECHO improves solution quality over multistart SA under an identical fixed
evaluation budget (B_total = 800,000) by introducing three spectral adaptations
derived solely from the QUBO matrix (Section 3.11):

  1. Spectral initialization of τ₀ from the eigenvalue range of Q^(pen)
     (Section 3.5).
  2. Eigenvalue-driven stage selection based on spectral change across
     t ∈ [0, 1] (Section 3.6).
  3. Condition number guided beam sizing to preserve diversity in
     ill-conditioned regions (Section 3.9).

ECHO uses no gradients, no surrogate models, and no learning components.
All adaptation is computed from spectral properties of staged matrices Q(t).

Homotopy family (Section 3.4):

    Q(t) = Q^(obj) + t·Q^(pen) + τ(t)·I,    τ(t) = τ₀(1−t)²

where Q^(obj) = Q_base and Q^(pen) = w_oh·Q_oh + w_reg·Q_reg + w_aff·Q_aff.
At t = 1: τ(1) = 0 and Q(1) = Q, the original target QUBO.

Variable ordering (inherited from build_qubo.py, Section 2):
    v = [x_1 ... x_N | y_1 ... y_M | z_1 ... z_K | t_0 ... t_{L-1}]

Entry point:
    spectral_landscape_navigation(Q_base, Q_oh, Q_reg, Q_aff,
                                   N, M, K, L,
                                   w_oh, w_reg, w_aff,
                                   seed=1000)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Any


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def analyze_spectrum(Q: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """
    Compute spectral diagnostics for a symmetric QUBO matrix.

    Used at each candidate stage point t to characterize the landscape of
    Q(t). The diagnostics drive stage selection (Section 3.6), budget
    allocation (Section 3.7), and beam sizing (Section 3.9).

    Returns a dict with:
        condition_number  -- κ(Q): max|λ| / min|λ| over nonzero eigenvalues
        spectral_gap      -- λ_max − λ_{n-1} (gap at the top of the spectrum)
        negative_count    -- n_neg: number of eigenvalues below −ε
        effective_rank    -- number of eigenvalues with |λ| > 10⁻⁶
        roughness         -- std(eigenvalues): landscape variability proxy
        eig_max, eig_min  -- largest and smallest eigenvalues
    """
    try:
        eigs = np.linalg.eigvalsh(Q)

        EPSILON = 1e-10
        nonzero_eigs = eigs[np.abs(eigs) > EPSILON]

        if len(nonzero_eigs) > 0:
            max_abs = np.max(np.abs(nonzero_eigs))
            min_abs = np.min(np.abs(nonzero_eigs))
            condition = max_abs / max(min_abs, EPSILON)
        else:
            condition = 1.0

        spectral_gap = float(eigs[-1] - eigs[-2]) if len(eigs) >= 2 else 0.0

        # Negative eigenvalue count: indicates non-convex landscape regions
        negative_count = int(np.sum(eigs < -EPSILON))

        effective_rank = int(np.sum(np.abs(eigs) > 1e-6))

        # Roughness proxy: eigenvalue standard deviation (Section 3.6)
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
        print(f"Warning: eigenvalue computation failed: {e}")
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
    Compute the initial smoothing parameter τ₀ from the penalty spectrum.

    Implements Section 3.5:

        τ₀ = percentile × (μ_max − μ_min)

    where μ_max and μ_min are the maximum and minimum eigenvalues of Q^(pen),
    and percentile = 0.25 by default. τ₀ is clamped to [100, 10⁶]:

        τ₀ ← max(τ₀, 100),   τ₀ ← min(τ₀, 10⁶)

    This anchors smoothing intensity to the penalty spectrum and prevents
    degenerate values at both small- and large-scale instances.
    """
    try:
        eigs_pen = np.linalg.eigvalsh(Q_penalties)
        eig_range = float(np.max(eigs_pen) - np.min(eigs_pen))
        tau0 = percentile * eig_range
        tau0 = max(tau0, 100.0)
        tau0 = min(tau0, 1e6)
        return float(tau0)
    except Exception:
        return 1000.0


def compute_adaptive_stages(
    Q_base: np.ndarray,
    Q_penalties: np.ndarray,
    tau0: float,
    max_stages: int = 8,
    verbose: bool = False
) -> List[float]:
    """
    Select homotopy stage points by spectral change (Section 3.6).

    Samples 51 candidate values uniformly on [0, 1]. For each candidate t,
    forms Q(t) = Q_base + t·Q_penalties + τ₀(1−t)²·I and computes spectral
    diagnostics. Consecutive-point spectral change scores are then computed as:

        Δ(t_i) = |κ(t_i) − κ(t_{i-1})| / κ(t_{i-1})
                + |ρ(t_i) − ρ(t_{i-1})| / ρ(t_{i-1})

    The (max_stages − 2) candidate points with the largest Δ are selected as
    intermediate stages. Endpoints t = 0 and t = 1 are always included.
    Duplicate values are removed.

    Returns a sorted list of stage t-values with |S| ≤ max_stages.
    """
    if verbose:
        print("  Analyzing spectral landscape...")

    n_samples = 51
    candidates = np.linspace(0, 1, n_samples)

    spectra = []
    for t in candidates:
        tau_t = tau0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if tau_t > 1e-8:
            Q_t = Q_t + tau_t * np.eye(Q_t.shape[0])
        spectra.append(analyze_spectrum(Q_t, verbose=False))

    # Spectral change score between consecutive candidate points
    changes = []
    for i in range(1, len(spectra)):
        delta = (
            abs(spectra[i]['condition_number'] - spectra[i-1]['condition_number']) /
            max(spectra[i-1]['condition_number'], 1.0) +
            abs(spectra[i]['roughness'] - spectra[i-1]['roughness']) /
            max(spectra[i-1]['roughness'], 1.0)
        )
        changes.append((candidates[i], delta))

    changes.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in changes[:max_stages - 2]]

    # Always include endpoints; remove duplicates
    stages = sorted(set([0.0] + selected + [1.0]))

    if verbose:
        print(f"  Selected {len(stages)} adaptive stages: {[f'{t:.2f}' for t in stages]}")

    return stages


def allocate_budget_by_roughness(
    stages: List[float],
    Q_base: np.ndarray,
    Q_penalties: np.ndarray,
    tau0: float,
    total_budget: int,
    verbose: bool = False
) -> List[int]:
    """
    Allocate the total evaluation budget across stages by roughness score
    (Section 3.7).

    For each stage t_s, the roughness score is:

        r_s = κ(Q(t_s)) × √(n_neg(Q(t_s)) + 1)

    Stage budgets are assigned proportionally:

        B_s = ⌊(r_s / Σ r_j) × B_total⌋

    with a minimum of B_s ≥ 5,000. If the minimum constraint causes the sum
    to exceed B_total, budgets are rescaled. The remainder from integer
    division is added to the final stage so that Σ B_s = B_total.

    Note: within each stage, SA assigns per-restart and per-candidate steps
    via integer division, so the executed iteration count can be marginally
    below the allocated stage budget. This is conservative and cannot
    artificially inflate ECHO's evaluation count.
    """
    roughness_scores = []
    for t in stages:
        tau_t = tau0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if tau_t > 1e-8:
            Q_t = Q_t + tau_t * np.eye(Q_t.shape[0])
        spec = analyze_spectrum(Q_t, verbose=False)
        roughness_scores.append(spec['condition_number'] * np.sqrt(spec['negative_count'] + 1))

    total_roughness = sum(roughness_scores)
    if total_roughness < 1e-10:
        budgets = [total_budget // len(stages)] * len(stages)
    else:
        budgets = [int(total_budget * r / total_roughness) for r in roughness_scores]

    # Assign integer-division remainder to the last stage
    budgets[-1] += total_budget - sum(budgets)

    # Enforce minimum budget per stage (B_s ≥ 5,000)
    min_budget = 5000
    for i in range(len(budgets)):
        if budgets[i] < min_budget:
            budgets[i] = min_budget

    # Rescale if minimum enforcement caused overshoot
    if sum(budgets) > total_budget:
        scale = total_budget / sum(budgets)
        budgets = [int(b * scale) for b in budgets]
        budgets[-1] = total_budget - sum(budgets[:-1])

    if verbose:
        print("  Budget allocation by roughness score:")
        for t, b in zip(stages, budgets):
            print(f"    t={t:.2f}: {b:>7,} steps ({b / total_budget * 100:>5.1f}%)")

    return budgets


def adaptive_beam_size(
    condition_number: float,
    base_beam: int = 20,
    min_beam: int = 10,
    max_beam: int = 40
) -> int:
    """
    Compute the beam size for candidate population control (Section 3.9).

    With base_beam = 20 the rule is:

        κ > 1000  → beam = 40   (very ill-conditioned: maximise diversity)
        κ > 100   → beam = 30   (moderately ill-conditioned)
        κ > 10    → beam = 20   (normal)
        κ ≤ 10    → beam = 10   (well-conditioned: exploit more)

    Beam is clamped to [min_beam, max_beam] = [10, 40].
    """
    if condition_number > 1000:
        beam = base_beam * 2
    elif condition_number > 100:
        beam = int(base_beam * 1.5)
    elif condition_number > 10:
        beam = base_beam
    else:
        beam = max(base_beam // 2, min_beam)

    return max(min_beam, min(beam, max_beam))


# ---------------------------------------------------------------------------
# SA core routines
# ---------------------------------------------------------------------------

def energy(Q: np.ndarray, v: np.ndarray) -> float:
    """Evaluate the QUBO energy: E(v) = v^T Q v (Section 3.3)."""
    return float(v @ Q @ v)


def project_onehot(v: np.ndarray, N: int, M: int, K: int,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Hard-project v to satisfy the one-hot constraints on y and z.

    For the deductible block y (indices N..N+M) and premium block z
    (indices N+M..N+M+K), exactly one variable must equal 1. If all
    entries are zero, a random selection is made.
    """
    v2 = v.copy()

    y = v2[N:N + M]
    j = int(np.argmax(y)) if np.sum(y) > 0.0 else int(rng.integers(0, M))
    y[:] = 0.0
    y[j] = 1.0

    z = v2[N + M:N + M + K]
    k = int(np.argmax(z)) if np.sum(z) > 0.0 else int(rng.integers(0, K))
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
    Single-run SA on Q for a fixed number of steps (Section 3.10).

    Move set (Section 3.10):
        Feature bit flip  : probability 0.85
        Deductible change : probability 0.075
        Premium change    : probability 0.075

    One-hot projection is enforced after each proposed move.
    Temperature follows a linear schedule from T0 to Tend.

    When initial_solution is None, a random binary vector is used and
    one-hot projection is applied before the first energy evaluation
    (Stage 0 initialization, Section 3.8).
    """
    n = Q.shape[0]

    if initial_solution is not None:
        v = initial_solution.copy()
    else:
        v = rng.integers(0, 2, size=n).astype(float)

    v = project_onehot(v, N, M, K, rng)
    e = energy(Q, v)

    best_v = v.copy()
    best_e = e

    y_start, y_end = N, N + M
    z_start, z_end = N + M, N + M + K

    prem_prob = 0.075
    ded_prob  = 0.075
    flip_prob = 1.0 - prem_prob - ded_prob   # 0.85

    for t in range(steps):
        frac = t / max(1, steps - 1)
        T = T0 * (1 - frac) + Tend * frac

        v_new = v.copy()
        r = float(rng.random())

        if r < flip_prob:
            # Feature bit flip: choose any variable outside y and z blocks
            while True:
                i = int(rng.integers(0, n))
                if not (y_start <= i < y_end or z_start <= i < z_end):
                    break
            v_new[i] = 1.0 - v_new[i]
        elif r < flip_prob + ded_prob:
            j = int(rng.integers(0, M))
            v_new[y_start:y_end] = 0.0
            v_new[y_start + j] = 1.0
        else:
            k = int(rng.integers(0, K))
            v_new[z_start:z_end] = 0.0
            v_new[z_start + k] = 1.0

        v_new = project_onehot(v_new, N, M, K, rng)
        e_new = energy(Q, v_new)

        if e_new < e or rng.random() < np.exp(-(e_new - e) / max(T, 1e-10)):
            v, e = v_new, e_new

        if e < best_e:
            best_e = e
            best_v = v.copy()

    return best_v, best_e


# ---------------------------------------------------------------------------
# Main ECHO algorithm
# ---------------------------------------------------------------------------

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
    ECHO: Eigenvalue-guided Constrained Homotopy Optimization (Algorithm 1).

    Implements the full algorithm described in Sections 3.4–3.10. Executes
    exactly B_total = 800,000 SA steps (modulo integer-division rounding)
    across adaptive homotopy stages, matching the baseline SA evaluation
    budget (Section 3.7).

    Args:
        Q_base         : Economic objective matrix Q^(obj) (Section 3.3).
        Q_oh, Q_reg, Q_aff : Unit-weight penalty matrices (Section 3.3).
        N, M, K, L     : Problem dimensions (features, deductible bands,
                         premium bands, regulatory slack variables).
        w_oh, w_reg, w_aff : Penalty weights applied to the unit matrices.
        seed           : Random seed for full reproducibility.
        params         : Optional dict overriding default parameters.
        SA_solver      : Unused; reserved for future custom SA injection.
        verbose        : Print stage-level diagnostic output.

    Returns a dict with:
        solution       : Best binary vector v* found on the original Q.
        energy         : E(v*) = v*^T Q v* on the unsmoothed Q at t=1.
        best_stage     : Index of the stage at which v* was found.
        runtime        : Wall-clock time in seconds (recorded, not used
                         for termination; Section 3.7).
        tau0           : Initial smoothing parameter τ₀ (Section 3.5).
        stages         : List of homotopy stage t-values.
        budgets        : SA step budget allocated to each stage.
        spectra        : Spectral diagnostics at each stage.
        steps_used     : Total SA steps executed.
        method         : 'ECHO'.
    """
    start_time = time.time()

    # Default parameters (Algorithm 1 header)
    default_params = {
        'total_budget':        800_000,
        'max_stages':          8,
        'smoothing_percentile': 0.25,
        'base_beam':           20,
        'initial_restarts':    40,
        'T0':                  5.0,
        'Tend':                0.01,
    }
    if params is not None:
        default_params.update(params)
    params = default_params

    rng = np.random.default_rng(seed)

    # Step 1: Construct Q^(pen) (Algorithm 1, step 1)
    Q_penalties = w_oh * Q_oh + w_reg * Q_reg + w_aff * Q_aff

    if verbose:
        print("=" * 70)
        print("ECHO — Eigenvalue-guided Constrained Homotopy Optimization")
        print("=" * 70)
        print(f"Problem: N={N}, M={M}, K={K}, L={L}")
        print(f"Penalty weights: w_oh={w_oh:.1e}, w_reg={w_reg:.1e}, w_aff={w_aff:.1e}")
        print()

    # ------------------------------------------------------------------
    # Phase 1: Spectral analysis
    # ------------------------------------------------------------------
    if verbose:
        print("[Phase 1] Spectral Landscape Analysis")
        print("-" * 70)

    # Step 2: Compute τ₀ (Section 3.5)
    tau0 = estimate_initial_smoothing(
        Q_base, Q_penalties, params['smoothing_percentile']
    )
    if verbose:
        print(f"  Initial smoothing τ₀: {tau0:.1e}")

    # Step 3–4: Sample candidates, select stages (Section 3.6)
    stages = compute_adaptive_stages(
        Q_base, Q_penalties, tau0,
        max_stages=params['max_stages'],
        verbose=verbose
    )

    # Step 5: Allocate budget by roughness score (Section 3.7)
    budgets = allocate_budget_by_roughness(
        stages, Q_base, Q_penalties, tau0,
        params['total_budget'],
        verbose=verbose
    )

    # Spectral diagnostics at each stage (used in Phase 2 and for output)
    if verbose:
        print()
        print("  Spectral diagnostics by stage:")
        print("  " + "-" * 66)
        print(f"  {'Stage':<8} {'t':<6} {'τ(t)':<12} {'Cond':<12} {'Neg':<6} {'Rough':<12}")
        print("  " + "-" * 66)

    spectra = []
    for i, t in enumerate(stages):
        tau_t = tau0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if tau_t > 1e-8:
            Q_t = Q_t + tau_t * np.eye(Q_t.shape[0])
        spec = analyze_spectrum(Q_t, verbose=False)
        spectra.append(spec)
        if verbose:
            print(f"  {i:<8} {t:<6.2f} {tau_t:<12.1e} "
                  f"{spec['condition_number']:<12.1e} "
                  f"{spec['negative_count']:<6} "
                  f"{spec['roughness']:<12.1e}")

    if verbose:
        print("  " + "-" * 66)
        print()

    # ------------------------------------------------------------------
    # Phase 2: Homotopy optimization
    # ------------------------------------------------------------------
    if verbose:
        print("[Phase 2] Adaptive Homotopy Optimization")
        print("-" * 70)

    candidates = []
    best_overall_energy   = float('inf')
    best_overall_solution = None
    best_overall_stage    = 0
    steps_used = 0

    for stage_idx, (t, budget, spectrum) in enumerate(zip(stages, budgets, spectra)):

        # Step 7a: Beam size from condition number (Section 3.9)
        beam = adaptive_beam_size(
            spectrum['condition_number'],
            base_beam=params['base_beam']
        )

        # Build Q(t) with diagonal smoothing (Section 3.4)
        tau_t = tau0 * (1 - t) ** 2
        Q_t = Q_base + t * Q_penalties
        if tau_t > 1e-8:
            Q_t = Q_t + tau_t * np.eye(Q_t.shape[0])

        if verbose:
            print(f"\n[Stage {stage_idx}]  t={t:.2f},  τ(t)={tau_t:.1e},  beam={beam}")
            print(f"  Budget: {budget:,} steps")
            print(f"  Condition: {spectrum['condition_number']:.1e}")

        if stage_idx == 0:
            # Step 6: Stage 0 — multistart exploration (Section 3.8)
            num_restarts       = params['initial_restarts']
            steps_per_restart  = budget // num_restarts

            if verbose:
                print(f"  Stage 0: {num_restarts} random restarts, "
                      f"{steps_per_restart:,} steps each")

            for r in range(num_restarts):
                solution, e = simulated_annealing_local(
                    Q_t, N, M, K, L, rng,
                    steps=steps_per_restart,
                    T0=params['T0'],
                    Tend=params['Tend'],
                )
                candidates.append({'solution': solution, 'energy': e, 'stage': stage_idx})
                if e < best_overall_energy:
                    best_overall_energy   = e
                    best_overall_solution = solution.copy()
                    best_overall_stage    = stage_idx

        else:
            # Step 7b: Subsequent stages — refine each surviving candidate
            # (Section 3.10): steps_per_candidate = ⌊B_s / |candidates|⌋
            num_candidates       = len(candidates)
            steps_per_candidate  = budget // max(num_candidates, 1)

            if verbose:
                print(f"  Refining {num_candidates} candidates, "
                      f"{steps_per_candidate:,} steps each")

            new_candidates = []
            for cand in candidates:
                solution, e = simulated_annealing_local(
                    Q_t, N, M, K, L, rng,
                    steps=steps_per_candidate,
                    T0=params['T0'],
                    Tend=params['Tend'],
                    initial_solution=cand['solution'],
                )
                new_candidates.append({'solution': solution, 'energy': e, 'stage': stage_idx})
                if e < best_overall_energy:
                    best_overall_energy   = e
                    best_overall_solution = solution.copy()
                    best_overall_stage    = stage_idx

            candidates = new_candidates

        steps_used += budget

        # Step 7c: Beam pruning — keep top beam by staged energy
        candidates.sort(key=lambda c: c['energy'])
        candidates = candidates[:beam]

        if verbose:
            print(f"  Best staged energy: {min(c['energy'] for c in candidates):.6f}")
            print(f"  Steps used: {steps_used:,} / {params['total_budget']:,}")

    # ------------------------------------------------------------------
    # Step 8: Evaluate best solution on original unsmoothed Q (Section 3.4)
    # ------------------------------------------------------------------
    runtime = time.time() - start_time
    Q_final = Q_base + Q_penalties
    final_energy = energy(Q_final, best_overall_solution)

    if verbose:
        print()
        print("=" * 70)
        print("ECHO COMPLETE")
        print("=" * 70)
        print(f"Best energy on Q: {final_energy:.6f}")
        print(f"Best found at stage: {best_overall_stage}")
        print(f"Runtime: {runtime:.2f} s  (recorded; not used for termination)")
        print(f"Steps used: {steps_used:,} / {params['total_budget']:,}")
        print("=" * 70)

    return {
        'solution':   best_overall_solution,
        'energy':     final_energy,
        'best_stage': best_overall_stage,
        'runtime':    runtime,
        'tau0':       tau0,
        'stages':     stages,
        'budgets':    budgets,
        'spectra':    spectra,
        'steps_used': steps_used,
        'method':     'ECHO',
    }


# ---------------------------------------------------------------------------
# Multi-start wrapper (optional robustness layer)
# ---------------------------------------------------------------------------

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
    Run ECHO multiple times with independent seeds and return the best result.

    This wrapper is not used in the paper's main experiments; the single-run
    ECHO already incorporates multi-start exploration at Stage 0
    (initial_restarts = 40, Section 3.8). It is provided for users who want
    additional robustness at higher computational cost.

    Each run uses a deterministic seed derived from the base seed so that
    multi-start results are fully reproducible.
    """
    if verbose:
        print("=" * 70)
        print(f"ECHO MULTI-START ({num_starts} runs)")
        print("=" * 70)

    results = []
    for run in range(num_starts):
        run_seed = (seed * 7919 + run * 104729) & 0xFFFFFFFF if seed is not None else None
        if verbose:
            print(f"\nRun {run + 1}/{num_starts}  (seed={run_seed})")
        result = spectral_landscape_navigation(
            Q_base, Q_oh, Q_reg, Q_aff,
            N, M, K, L,
            w_oh, w_reg, w_aff,
            seed=run_seed,
            params=params,
            verbose=verbose,
        )
        results.append(result)
        if verbose:
            print(f"Run {run + 1} energy: {result['energy']:.6f}")

    best_result = min(results, key=lambda r: r['energy'])

    if verbose:
        energies = [r['energy'] for r in results]
        print("\n" + "=" * 70)
        print("MULTI-START SUMMARY")
        print("=" * 70)
        print(f"Best energy:  {min(energies):.6f}")
        print(f"Worst energy: {max(energies):.6f}")
        print(f"Mean energy:  {np.mean(energies):.6f}")
        print(f"Std energy:   {np.std(energies):.6f}")
        print("=" * 70)

    best_result['multistart']   = True
    best_result['num_starts']   = num_starts
    best_result['all_energies'] = [r['energy'] for r in results]
    return best_result


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_default_SLN_parameters() -> Dict[str, Any]:
    """
    Return the default ECHO parameter dict (Algorithm 1 header values).

    These match the values used in all paper experiments and can be passed
    as the params argument to spectral_landscape_navigation to override
    individual entries.
    """
    return {
        'total_budget':         800_000,
        'max_stages':           8,
        'smoothing_percentile': 0.25,
        'base_beam':            20,
        'initial_restarts':     40,
        'T0':                   5.0,
        'Tend':                 0.01,
    }


def echo_optimize(*args, **kwargs):
    """Alias for spectral_landscape_navigation."""
    return spectral_landscape_navigation(*args, **kwargs)