# src_code/generators/benchmark_families.py
"""
Benchmark QUBO Generators
==========================
Produces QUBO matrices for two benchmark families used in menu Option 3.

Families
--------
maxcut
    Weighted Max-Cut on a random Erdős-Rényi graph G(N, edge_prob).
    Edge weights ~ Uniform(0, 1).  Unconstrained binary QUBO.
    No penalty terms (M=K=L=0, w_pen=0).

portfolio_card
    Cardinality-constrained portfolio selection.
    Select exactly K = round(k_frac * N) assets.
    Constraint encoded as quadratic penalty in Q_pen.
    M=K_bands=L=0.

spectral_dense
    Dense QUBO with controlled eigenvalue dispersion.
    Eigenvalues are log-spaced over kappa_target; neg_frac are negative.
    Uses smoothing continuation: Q_pen = diag(row_abs_sums), w_pen = 1.0.
    Designed to maximise ECHO gain by activating all spectral mechanisms.

All return a dict consumed by run_benchmarks.py:
    Q_base  : objective-only QUBO (n x n, symmetric)
    Q_pen   : smoothing / penalty matrix
    w_pen   : scalar penalty weight
    N       : number of binary decision variables
    M, K, L : always 0 for all families
    meta    : dict of instance parameters written into result rows
"""

from __future__ import annotations
import math
from typing import Any, Dict
import numpy as np


def generate_maxcut(N: int, seed: int, edge_prob: float = 0.5) -> Dict[str, Any]:
    """
    Weighted Max-Cut QUBO on G(N, edge_prob) with Normal(0,1) edge weights.

    QUBO formulation (signed Laplacian, minimisation):
        Q_full_ij  =  w_ij            (i ≠ j, edge weight ~ Normal(0,1))
        Q_full_ii  = -sum_j w_ij      (negative signed row sum on diagonal)
    Minimising v^T Q_full v  ≡  maximising the signed cut weight.

    ECHO decomposition  (reviewer-corrected, Step B):
    -------------------------------------------------------
    Previous decomposition split Q_full = W + (-1)*diag(deg), which made
    the homotopy PATH go from W (no diagonal) → W-D (target) — i.e., it
    was ADDING ruggedness as t increased, the opposite of ECHO's intention.
    This caused ECHO to consistently lose vs plain SA at N≥100.

    Correct decomposition:
        Q_base = Q_full          (the complete target QUBO, fixed)
        Q_pen  = diag(row_abs)   (row absolute-sum diagonal — always PSD)
        w_pen  = 1.0             (positive: smoothing is ADDED, not removed)

    Homotopy path:  Q(t) = Q_base + tau*(1-t)^2 * Q_pen
        t=0, tau large:  Q(0) = Q_full + tau*diag(row_abs)
                         All eigenvalues lifted → smooth, nearly-convex landscape.
                         SA at stage 0 finds good global structure easily.
        t=1, tau=0:      Q(1) = Q_full
                         Exact original landscape recovered at the final stage.

    This aligns with ECHO's core assumption:
        "Q_pen adds positive curvature that is gradually removed,
         guiding the solver from a smooth surrogate to the true landscape."

    References: Boros & Hammer (2002); Dunning et al. (2018).
    """
    rng = np.random.default_rng(seed)

    # Build edge weight matrix W (off-diagonal, symmetric, Normal(0,1) weights)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < edge_prob:
                w = float(rng.standard_normal())
                W[i, j] = w
                W[j, i] = w

    # Full signed-Laplacian QUBO: Q_full = W - diag(deg)
    deg_signed = W.sum(axis=1)
    Q_full = W - np.diag(deg_signed)

    # Smoothing matrix: positive diagonal = row absolute sums of Q_full
    # PSD by construction; entries scale with local edge-weight magnitude
    row_abs = np.abs(Q_full).sum(axis=1)
    Q_pen   = np.diag(row_abs)

    # w_pen = 1.0: positive so homotopy ADDS smoothing at stage 0
    w_pen = 1.0

    num_edges = int((W != 0).sum() // 2)
    meta = {
        "benchmark_family": "maxcut",
        "N": N, "seed": seed,
        "edge_prob":   edge_prob,
        "num_edges":   num_edges,
        "weight_dist": "normal01",
        "k_frac":      None,
        "K_card":      None,
        "penalty_weight": w_pen,
    }
    return {"Q_base": Q_full, "Q_pen": Q_pen,
            "w_pen": w_pen, "N": N, "M": 0, "K": 0, "L": 0, "meta": meta}


def generate_portfolio_card(
    N: int, seed: int,
    k_frac: float = 0.15,
    lambda_risk: float = 1.0,
    scale_factor: float = 50.0,
) -> Dict[str, Any]:
    """
    Cardinality-constrained portfolio QUBO.

    Objective (minimise):
        -mu^T x  +  lambda_risk * x^T Sigma x

    Constraint:
        sum_i x_i = K_card  (encoded as w_pen*(sum x_i - K)^2 in Q_pen)

    mu_i ~ Uniform(0,1), Sigma = A A^T + 0.01*I (rank-min(5,N) factor model).
    Penalty weight w_pen = scale_factor * max|Q_base|.

    scale_factor=50 ensures the constraint eigenvalues dominate the objective
    eigenvalues at all N, giving ECHO a strong spectral signal to navigate.
    At scale_factor=10 the dominance weakens at large N and ECHO loses the
    conditioning signal it relies on.

    Note on condition number: κ(Q_full) ≈ 5 at all N. This is structural —
    the large uniform penalty diagonal flattens the spectrum. As a result,
    adaptive_beam_size always returns the minimum beam (10). This does not
    break ECHO: roughness-based budget allocation and stage selection remain
    fully active via the neg_eigs * kappa roughness score. The beam sizing
    mechanism simply adds no extra diversity for this family.
    """
    rng    = np.random.default_rng(seed)
    K_card = max(1, round(k_frac * N))

    mu  = rng.uniform(0.0, 1.0, size=N)
    A_f = rng.standard_normal((N, min(5, N))) / math.sqrt(N)
    Sigma = A_f @ A_f.T + 0.01 * np.eye(N)

    Q_base = -np.diag(mu) + lambda_risk * Sigma

    # Penalty: (sum x_i - K)^2 expanded for binary x_i in QUBO form x^T Q_pen x
    # (sum x_i - K)^2 = (1-2K)*sum x_i + 2*sum_{i<j} x_i x_j + K^2
    # In QUBO: x^T Q x = sum_i Q_ii x_i + 2*sum_{i<j} Q_ij x_i x_j
    # Therefore: Q_pen_ii = (1-2K),  Q_pen_ij = 1  (NOT 2 — the factor of 2 is
    # already present in the x^T Q x expansion for off-diagonal terms)
    # Verification: x^T Q_pen x = (1-2K)*s + s*(s-1) = s^2-2Ks = (s-K)^2-K^2 ✓
    Q_pen = np.full((N, N), 1.0, dtype=float)
    np.fill_diagonal(Q_pen, 1.0 - 2.0 * K_card)

    obj_scale = max(float(np.max(np.abs(Q_base))), 1e-6)
    w_pen = float(scale_factor * obj_scale)

    meta = {
        "benchmark_family": "portfolio_card",
        "N": N, "seed": seed,
        "edge_prob": None,
        "k_frac": k_frac, "K_card": K_card,
        "lambda_risk": lambda_risk,
        "scale_factor": scale_factor,
        "penalty_weight": w_pen,
    }
    return {"Q_base": Q_base, "Q_pen": Q_pen,
            "w_pen": w_pen, "N": N, "M": 0, "K": 0, "L": 0, "meta": meta}


def generate_spectral_dense(
    N: int,
    seed: int,
    kappa_target: float = 1e4,
    neg_frac: float = 0.3,
) -> Dict[str, Any]:
    """
    Dense QUBO with controlled eigenvalue dispersion.

    Construction
    ------------
    1. Draw a random orthogonal matrix U via QR of an N×N Gaussian matrix.
    2. Build log-spaced eigenvalue magnitudes over [1, kappa_target].
    3. Assign signs: the first ceil(neg_frac * N) magnitudes are negated
       (making the matrix indefinite), then the sign vector is randomly
       permuted so negative eigenvalues are spread throughout the spectrum.
    4. Q = U @ diag(λ) @ U.T, forced exactly symmetric with (Q + Q.T) / 2.
    5. Scale Q so max(|Q|) = 1 — preserves κ exactly while keeping energy
       magnitudes comparable across N (otherwise larger N → larger energies).

    ECHO decomposition (smoothing continuation)
    --------------------------------------------
    For unconstrained QUBOs without a natural penalty term, ECHO uses a
    positive-diagonal smoothing matrix:

        Q_base = Q          (the full target QUBO, fixed)
        Q_pen  = diag(row_abs_sums)   where row_abs_sums[i] = Σ_j |Q[i,j]|
        w_pen  = 1.0        (positive; smoothing ADDED at stage 0, removed at final)

    Homotopy path: Q(τ) = Q + τ * Q_pen
        τ large:  all eigenvalues lifted → smooth, near-convex landscape
        τ = 0:    exact original Q recovered

    Condition number
    ----------------
    κ(Q) = |λ_max| / |λ_min| where both extremes are the absolute values of
    the outermost eigenvalues.  After scaling, κ ≈ kappa_target (exact up to
    floating-point and the fact that |λ| are log-spaced across N values).

    Parameters
    ----------
    N            : problem dimension
    seed         : RNG seed for reproducibility
    kappa_target : target condition number (default 1e4)
    neg_frac     : fraction of eigenvalues that are negative (default 0.3)
    """
    rng   = np.random.default_rng(seed)
    n_neg = max(1, int(math.ceil(neg_frac * N)))
    n_pos = N - n_neg
    kt    = max(kappa_target, 1.0 + 1e-9)

    # ── Orthogonal basis via QR ──────────────────────────────────────────────
    G    = rng.standard_normal((N, N))
    U, _ = np.linalg.qr(G)

    # ── Eigenvalues: κ_actual = kappa_target exactly ─────────────────────────
    # Positive branch: logspace(1, kappa_target) → [1 .. kappa_target]
    # Negative branch: -logspace(1, 1/kappa_target) → [-1 .. -1/kappa_target]
    # After scaling by kappa_target: λ_max = +1, |λ_min| = 1/kappa_target
    # → κ = λ_max / |λ_min| = kappa_target  ✓
    pos_eigs = np.logspace(0,  math.log10(kt), n_pos)   # [1 .. kappa_target]
    neg_eigs = -np.logspace(0, -math.log10(kt), n_neg)  # [-1 .. -1/kappa_target]

    lambdas  = np.concatenate([neg_eigs, pos_eigs])
    lambdas  = rng.permutation(lambdas)    # randomise eigenvector associations
    lambdas  = lambdas / kt               # scale: λ_max=1, |λ_min|=1/kappa_target

    # ── Assemble Q and force exact symmetry ──────────────────────────────────
    Q = U @ np.diag(lambdas) @ U.T
    Q = (Q + Q.T) * 0.5

    # ── Smoothing decomposition ──────────────────────────────────────────────
    # Q_pen = diag(row_abs_sums): always PSD, scales with Q's energy magnitude.
    # w_pen = +1: homotopy ADDS smoothing at stage 0 and removes it at final.
    row_abs = np.abs(Q).sum(axis=1)
    Q_pen   = np.diag(row_abs)
    w_pen   = 1.0

    # ── Filename-safe kappa tag ───────────────────────────────────────────────
    exp       = round(math.log10(kappa_target))
    kappa_tag = f"1e{exp}"
    instance_id = f"spectral_dense_N{N}_k{kappa_tag}_seed{seed}"

    meta = {
        "benchmark_family": "spectral_dense",
        "N":            N,
        "seed":         seed,
        "kappa_target": kappa_target,
        "neg_frac":     neg_frac,
        "kappa_tag":    kappa_tag,
        "instance_id":  instance_id,
        # _BASE_COLS fields not applicable:
        "edge_prob":      None,
        "num_edges":      None,
        "k_frac":         None,
        "K_card":         None,
        "penalty_weight": w_pen,
    }
    return {
        "Q_base": Q,
        "Q_pen":  Q_pen,
        "w_pen":  w_pen,
        "N":      N,
        "M":      0,
        "K":      0,
        "L":      0,
        "meta":   meta,
    }


def generate_instance(family: str, N: int, seed: int, **kwargs) -> Dict[str, Any]:
    """Dispatch to the correct generator. Unknown kwargs are silently ignored."""
    if family == "maxcut":
        return generate_maxcut(N, seed,
            edge_prob=float(kwargs.get("edge_prob", 0.5)))
    elif family == "portfolio_card":
        return generate_portfolio_card(N, seed,
            k_frac=float(kwargs.get("k_frac", 0.15)),
            lambda_risk=float(kwargs.get("lambda_risk", 1.0)),
            scale_factor=float(kwargs.get("scale_factor", 10.0)))
    elif family == "spectral_dense":
        return generate_spectral_dense(N, seed,
            kappa_target=float(kwargs.get("kappa_target", 1e4)),
            neg_frac=float(kwargs.get("neg_frac", 0.3)))
    else:
        raise ValueError(
            f"Unknown benchmark family '{family}'. "
            "Expected 'maxcut', 'portfolio_card', or 'spectral_dense'.")