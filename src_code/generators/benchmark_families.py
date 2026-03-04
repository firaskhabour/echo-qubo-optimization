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
    Max-Cut QUBO instance on G(N, edge_prob) with Uniform(0,1) edge weights.

    Standard QUBO reduction (Lucas 2014, Sec 2.3):
        Q_ii = -sum_j w_ij     (diagonal)
        Q_ij =  w_ij           (i != j)
    Minimising v^T Q v is equivalent to maximising the cut weight.
    """
    rng = np.random.default_rng(seed)

    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < edge_prob:
                w = float(rng.uniform(0.0, 1.0))
                W[i, j] = w
                W[j, i] = w

    Q_base = W.copy()
    np.fill_diagonal(Q_base, -W.sum(axis=1))

    meta = {
        "benchmark_family": "maxcut",
        "N": N, "seed": seed,
        "edge_prob": edge_prob,
        "num_edges": int((W > 0).sum() // 2),
        "k_frac": None, "K_card": None, "penalty_weight": 0.0,
    }
    return {"Q_base": Q_base, "Q_pen": np.zeros_like(Q_base),
            "w_pen": 0.0, "N": N, "M": 0, "K": 0, "L": 0, "meta": meta}


def generate_portfolio_card(
    N: int, seed: int,
    k_frac: float = 0.3,
    lambda_risk: float = 1.0,
    scale_factor: float = 10.0,
) -> Dict[str, Any]:
    """
    Cardinality-constrained portfolio QUBO.

    Objective (minimise):
        -mu^T x  +  lambda_risk * x^T Sigma x

    Constraint:
        sum_i x_i = K_card  (encoded as A*(sum x_i - K)^2 in Q_pen)

    mu_i ~ Uniform(0,1), Sigma = A A^T + 0.01*I (rank-min(5,N) factor model).
    Penalty weight A_pen = scale_factor * max|Q_base| for constraint dominance.
    """
    rng    = np.random.default_rng(seed)
    K_card = max(1, round(k_frac * N))

    mu  = rng.uniform(0.0, 1.0, size=N)
    A_f = rng.standard_normal((N, min(5, N))) / math.sqrt(N)
    Sigma = A_f @ A_f.T + 0.01 * np.eye(N)

    Q_base = -np.diag(mu) + lambda_risk * Sigma

    # Penalty: (sum x_i - K)^2 expanded for binary x_i
    # diagonal: (1 - 2K), off-diagonal: 2
    Q_pen = np.full((N, N), 2.0, dtype=float)
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
            k_frac=float(kwargs.get("k_frac", 0.3)),
            lambda_risk=float(kwargs.get("lambda_risk", 1.0)),
            scale_factor=float(kwargs.get("scale_factor", 10.0)))
    else:
        raise ValueError(
            f"Unknown benchmark family '{family}'. "
            "Expected 'maxcut' or 'portfolio_card'.")