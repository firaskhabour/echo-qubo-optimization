# src_code/utils/extract_parameters.py
"""
Hybrid Parameter Extraction: c_vector and sigma_matrix
========================================================
Derives the two QUBO input parameters from the synthetic seed data
produced by generate_data.py:

  c ∈ ℝ^N   — expected cost vector for coverage features (Section 2.3)
  Σ ∈ ℝ^{N×N} — feature-level claims covariance matrix (Section 2.3)

Both parameters are computed from the members/claims/feature_catalog
artefacts in data/seeds/seed_<SEED>/ and written back to the same
directory as c_vector.csv and sigma_matrix.npz.

Derivation:

  1. Build a member × feature cost matrix F ∈ ℝ^{U×N} where
         F[u, i] = impact_factor[i] × (S[u, cat1[i]] + S[u, cat2[i]])
     and S[u, c] is member u's total claim spend in category c.

  2. c[i] = mean_u F[u, i]  (expected cost per feature).

  3. Σ = cov(F) + ε·I, where ε is a small diagonal regularisation
     that ensures strict positive definiteness consistent with the
     Σ = AA^T + εI formulation described in Section 2.3 (ε = 0.001
     in the master generation; the --epsilon argument here controls
     the regularisation applied at extraction time).

  4. Off-diagonal entries of Σ are sparsified by retaining only the
     top-d absolute off-diagonal entries per row (symmetric union),
     which controls the density of the risk interaction matrix passed
     to build_qubo.py.

This script is the intermediate pipeline step between generate_data.py
and build_qubo.py. It is idempotent when --overwrite is not set.

Usage:
    python extract_parameters.py --seed 1000
    python extract_parameters.py --seed 1000 --overwrite --d 6 --epsilon 1e-6

Output:
    data/seeds/seed_<SEED>/c_vector.csv       — one column: expected_cost
    data/seeds/seed_<SEED>/sigma_matrix.npz   — key: sigma
    data/seeds/seed_<SEED>/instance_meta.json — updated with extraction metadata
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract c_vector and sigma_matrix from claims and feature catalog."
    )
    p.add_argument("--seed",      type=int,   required=True,
                   help="Seed id (matches data/seeds/seed_<SEED>).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite c_vector.csv and sigma_matrix.npz if already present.")
    p.add_argument("--d",         type=int,   default=6,
                   help="Top-d off-diagonal interactions per row to retain (default 6).")
    p.add_argument("--epsilon",   type=float, default=1e-6,
                   help="Diagonal regularisation ε added after sparsification (default 1e-6).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, raising FileNotFoundError with a clear message if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _write_meta(meta_path: Path, updates: Dict[str, Any]) -> None:
    """
    Merge updates into instance_meta.json, creating the file if absent.
    Existing keys outside the updated block are preserved.
    """
    base: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            base = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    base.update(updates)
    meta_path.write_text(json.dumps(base, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _build_member_category_spend(
    claims: pd.DataFrame,
    members: pd.DataFrame,
    C: int,
) -> np.ndarray:
    """
    Aggregate total claim spend per member per category.

    Returns S of shape (U, C) where S[u, c] is the total claim amount
    for member u (0-based index) in category c (0-based index).
    Member IDs in claims.csv are 1-based and are converted internally.

    If claims is empty or contains no valid rows, returns a zero matrix.
    """
    U = int(len(members))

    if claims.empty:
        return np.zeros((U, C), dtype=float)

    req_cols = {"member_id", "category_id", "amount"}
    missing  = req_cols - set(claims.columns)
    if missing:
        raise ValueError(f"claims.csv missing columns: {sorted(missing)}")

    claims = claims.copy()
    claims["member_id"]   = claims["member_id"].astype(int)
    claims["category_id"] = claims["category_id"].astype(int)
    claims["amount"]      = claims["amount"].astype(float)

    # Discard rows with out-of-range category ids
    claims = claims[(claims["category_id"] >= 1) & (claims["category_id"] <= C)]
    if claims.empty:
        return np.zeros((U, C), dtype=float)

    piv = claims.pivot_table(
        index="member_id",
        columns="category_id",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
    )

    S = np.zeros((U, C), dtype=float)
    for cat in range(1, C + 1):
        if cat in piv.columns:
            idx  = piv.index.to_numpy(dtype=int) - 1   # convert 1-based → 0-based
            vals = piv[cat].to_numpy(dtype=float)
            ok   = (idx >= 0) & (idx < U)
            S[idx[ok], cat - 1] = vals[ok]

    return S


def _sparsify_topd_sigma(sigma_dense: np.ndarray, d: int, epsilon: float) -> np.ndarray:
    """
    Sparsify a dense covariance matrix by retaining only the top-d
    largest-magnitude off-diagonal entries per row, then symmetrising
    by taking the element-wise maximum of the result and its transpose,
    and finally adding ε·I for diagonal regularisation.

    The symmetric-union step ensures that if entry (i, j) is retained
    from row i's top-d, it is also present at (j, i). Setting d = 0
    returns a diagonal matrix. Setting d ≥ N−1 retains all entries.
    """
    N      = int(sigma_dense.shape[0])
    sigma  = np.zeros_like(sigma_dense, dtype=float)
    np.fill_diagonal(sigma, np.diag(sigma_dense))

    if d > 0 and N > 1:
        for i in range(N):
            row    = sigma_dense[i, :].copy()
            row[i] = 0.0
            k      = min(int(d), N - 1)
            if k <= 0:
                continue
            top_k = np.argpartition(np.abs(row), -k)[-k:]
            sigma[i, top_k] = sigma_dense[i, top_k]

    # Symmetric union
    sigma = np.maximum(sigma, sigma.T)

    # Diagonal regularisation (ε·I)
    if epsilon is not None and float(epsilon) > 0.0:
        sigma = sigma + float(epsilon) * np.eye(N)

    return sigma.astype(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args         = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_dir     = project_root / "data" / "seeds" / f"seed_{int(args.seed)}"

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing seed directory: {data_dir}. Run generate_data.py first.")

    members  = _safe_read_csv(data_dir / "members.csv")
    claims   = _safe_read_csv(data_dir / "claims.csv")
    features = _safe_read_csv(data_dir / "feature_catalog.csv")

    needed_feat = {"feature_id", "cat1_id", "cat2_id", "impact_factor"}
    missing     = needed_feat - set(features.columns)
    if missing:
        raise ValueError(
            f"feature_catalog.csv missing columns: {sorted(missing)}. "
            f"Hybrid design requires cat1_id, cat2_id, and impact_factor."
        )

    U = int(len(members))
    N = int(len(features))
    if U <= 0:
        raise ValueError("members.csv is empty; cannot extract parameters.")
    if N <= 0:
        raise ValueError("feature_catalog.csv is empty; cannot extract parameters.")

    # Infer number of categories C: take the maximum across claims, feature ids, and 12
    C_claims = int(claims["category_id"].max()) if (not claims.empty and "category_id" in claims.columns) else 0
    C_feats  = int(max(features["cat1_id"].max(), features["cat2_id"].max())) if N > 0 else 0
    C        = int(max(C_claims, C_feats, 12))

    # ------------------------------------------------------------------
    # Step 1: Member × category spend matrix S ∈ ℝ^{U×C}
    # ------------------------------------------------------------------
    S = _build_member_category_spend(claims, members, C=C)

    # ------------------------------------------------------------------
    # Step 2: Member × feature cost matrix F ∈ ℝ^{U×N}
    # F[u, i] = impact_factor[i] × (S[u, cat1[i]] + S[u, cat2[i]])
    # ------------------------------------------------------------------
    cat1   = features["cat1_id"].astype(int).to_numpy() - 1   # 0-based
    cat2   = features["cat2_id"].astype(int).to_numpy() - 1
    impact = features["impact_factor"].astype(float).to_numpy()

    if np.any(cat1 < 0) or np.any(cat2 < 0) or np.any(cat1 >= C) or np.any(cat2 >= C):
        raise ValueError(
            "Feature category ids out of bounds relative to inferred C. "
            "Check generate_data.py output."
        )

    feature_costs = (S[:, cat1] + S[:, cat2]) * impact.reshape(1, -1)

    # ------------------------------------------------------------------
    # Step 3: c[i] = mean_u F[u, i]  (expected cost per feature)
    # ------------------------------------------------------------------
    c_vector = feature_costs.mean(axis=0).astype(float)

    # ------------------------------------------------------------------
    # Step 4: Σ = sparsify(cov(F)) + ε·I
    # Consistent with the Σ = AA^T + εI positive-definite structure
    # described in Section 2.3.
    # ------------------------------------------------------------------
    sigma_dense = np.cov(feature_costs, rowvar=False)
    sigma       = _sparsify_topd_sigma(sigma_dense, d=int(args.d), epsilon=float(args.epsilon))

    # ------------------------------------------------------------------
    # Write outputs (guard against silent overwrites)
    # ------------------------------------------------------------------
    c_out = data_dir / "c_vector.csv"
    s_out = data_dir / "sigma_matrix.npz"

    if (not args.overwrite) and (c_out.exists() or s_out.exists()):
        raise FileExistsError(
            f"{c_out.name} or {s_out.name} already exists in {data_dir}. "
            f"Pass --overwrite to regenerate."
        )

    pd.DataFrame({"expected_cost": c_vector}).to_csv(c_out, index=False)
    np.savez_compressed(s_out, sigma=sigma)

    _write_meta(
        data_dir / "instance_meta.json",
        updates={
            "extraction": {
                "method":        "claims_derived",
                "seed":          int(args.seed),
                "num_members":   U,
                "num_features":  N,
                "num_categories": C,
                "sparsity_top_d": int(args.d),
                "epsilon":       float(args.epsilon),
            }
        },
    )

    print(f"Parameter extraction completed.")
    print(f"Seed: {args.seed}  U={U}  N={N}  C={C}")
    print(f"Saved: {c_out.name}, {s_out.name}")


if __name__ == "__main__":
    main()