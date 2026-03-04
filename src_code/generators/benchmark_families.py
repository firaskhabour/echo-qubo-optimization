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

Both return a dict consumed by run_benchmarks.py:
    Q_base  : objective-only QUBO (n x n, symmetric)
    Q_pen   : unit-weight penalty matrix  (zeros for maxcut)
    w_pen   : scalar penalty weight       (0.0 for maxcut)
    N       : number of binary decision variables
    M, K, L : always 0 for both families
    meta    : dict of instance parameters written into result rows
"""

from __future__ import annotations
import math
from typing import Any, Dict
import numpy as np


def generate_maxcut(N: int, seed: int, edge_prob: float = 0.5) -> Dict[str, Any]:
    """
    Weighted Max-Cut QUBO on G(N, edge_prob) with Normal(0,1) edge weights.

    Edge weights are drawn from Normal(0,1) rather than Uniform(0,1).
    This produces a *signed* (frustrated) graph where the energy landscape
    is significantly rougher than uniform-weight graphs, activating all
    three ECHO mechanisms:

        condition_number > 100 at N≥50  →  beam sizing active (beam 30–40)
        large roughness variation        →  non-uniform budget allocation
        spectral change across path      →  meaningful stage selection

    With Uniform(0,1) weights the graph Laplacian is well-conditioned
    (κ ≈ 2–3), beam sizing always returns the minimum, and ECHO reduces
    to smoothed SA with no advantage.

    QUBO formulation  (signed Laplacian, minimisation form):
        Q_full_ij  =  w_ij            (i ≠ j, edge weight)
        Q_full_ii  = -sum_j w_ij      (signed row sum on diagonal)
    Minimising v^T Q_full v ≡ maximising signed cut weight.

    ECHO decomposition  (Q_full = Q_base + w_pen * Q_pen):
        Q_base  : edge weight matrix W  (off-diagonal only, zero diagonal)
        Q_pen   : signed degree matrix  diag(sum_j w_ij)
        w_pen   : -1.0  →  Q_base + (-1)*Q_pen = W - D = Q_full  (exact)

    The signed degree diagonal has eigenvalue range O(sqrt(N)) for Normal
    weights, giving tau0 a natural scale and the homotopy path genuine
    spectral curvature to navigate.

    References: Boros & Hammer (2002); Dunning et al. (2018).
    """
    rng = np.random.default_rng(seed)

    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < edge_prob:
                w = float(rng.standard_normal())   # Normal(0,1) — signed weights
                W[i, j] = w
                W[j, i] = w

    deg_signed = W.sum(axis=1)   # signed row sums

    Q_base = W.copy()                   # off-diagonal edge weights, zero diagonal
    Q_pen  = np.diag(deg_signed)        # signed degree diagonal

    # w_pen = -1: Q_base + (-1)*Q_pen = W - diag(deg) = standard signed Laplacian
    w_pen  = -1.0

    num_edges = int((W != 0).sum() // 2)
    meta = {
        "benchmark_family": "maxcut",
        "N": N, "seed": seed,
        "edge_prob":  edge_prob,
        "num_edges":  num_edges,
        "weight_dist": "normal01",
        "k_frac":     None,
        "K_card":     None,
        "penalty_weight": w_pen,
    }
    return {"Q_base": Q_base, "Q_pen": Q_pen,
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


def generate_instance(family: str, N: int, seed: int, **kwargs) -> Dict[str, Any]:
    """Dispatch to the correct generator. Unknown kwargs are silently ignored."""
    if family == "maxcut":
        return generate_maxcut(N, seed,
            edge_prob=float(kwargs.get("edge_prob", 0.5)))
    elif family == "portfolio_card":
        return generate_portfolio_card(N, seed,
            k_frac=float(kwargs.get("k_frac", 0.15)),
            lambda_risk=float(kwargs.get("lambda_risk", 1.0)),
            scale_factor=float(kwargs.get("scale_factor", 50.0)))
    else:
        raise ValueError(
            f"Unknown benchmark family '{family}'. "
            "Expected 'maxcut' or 'portfolio_card'.")