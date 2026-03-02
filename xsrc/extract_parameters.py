# src/extract_parameters.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Extract QUBO parameters (c_vector and sigma_matrix) from claims + feature catalog.")
    p.add_argument("--seed", type=int, required=True, help="Seed id (matches data/seeds/seed_<seed>).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite c_vector.csv and sigma_matrix.npz if present.")
    p.add_argument("--d", type=int, default=6, help="Top-d off-diagonal interactions per feature row to keep (default 6).")
    p.add_argument("--epsilon", type=float, default=1e-6, help="Diagonal regularization epsilon (default 1e-6).")
    return p.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _write_meta(meta_path: Path, updates: Dict[str, Any]) -> None:
    base: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            base = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    base.update(updates)
    meta_path.write_text(json.dumps(base, indent=2), encoding="utf-8")


def _build_member_category_spend(claims: pd.DataFrame, members: pd.DataFrame, C: int) -> np.ndarray:
    """
    Returns S of shape (U, C) where S[u, c] is total claim amount for member u in category (1..C).
    Uses member_id as 1-based identifiers.
    """
    if claims.empty:
        U = int(len(members))
        return np.zeros((U, C), dtype=float)

    req_cols = {"member_id", "category_id", "amount"}
    missing = req_cols - set(claims.columns)
    if missing:
        raise ValueError(f"claims.csv missing columns: {sorted(missing)}")

    # Ensure numeric
    claims = claims.copy()
    claims["member_id"] = claims["member_id"].astype(int)
    claims["category_id"] = claims["category_id"].astype(int)
    claims["amount"] = claims["amount"].astype(float)

    # Filter valid category ids
    claims = claims[(claims["category_id"] >= 1) & (claims["category_id"] <= C)]
    if claims.empty:
        U = int(len(members))
        return np.zeros((U, C), dtype=float)

    piv = claims.pivot_table(
        index="member_id",
        columns="category_id",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
    )

    U = int(len(members))
    S = np.zeros((U, C), dtype=float)

    # members are assumed 1..U in generate_data
    # Align pivot rows into S
    for cat in range(1, C + 1):
        if cat in piv.columns:
            # member_id is 1-based; convert to 0-based row index
            idx = piv.index.to_numpy(dtype=int) - 1
            vals = piv[cat].to_numpy(dtype=float)
            ok = (idx >= 0) & (idx < U)
            S[idx[ok], cat - 1] = vals[ok]

    return S


def _sparsify_topd_sigma(sigma_dense: np.ndarray, d: int, epsilon: float) -> np.ndarray:
    """
    Keep diagonal + top-d absolute off-diagonal entries per row, symmetric union, then +epsilon*I.
    """
    N = int(sigma_dense.shape[0])
    sigma = np.zeros_like(sigma_dense, dtype=float)

    # Diagonal
    np.fill_diagonal(sigma, np.diag(sigma_dense))

    if d > 0 and N > 1:
        for i in range(N):
            row = sigma_dense[i, :].copy()
            row[i] = 0.0
            # If d >= N-1 keep all
            k = min(int(d), N - 1)
            if k <= 0:
                continue
            # argpartition for speed
            nn = np.argpartition(np.abs(row), -k)[-k:]
            sigma[i, nn] = sigma_dense[i, nn]

    # Symmetric union
    sigma = np.maximum(sigma, sigma.T)

    # Regularisation
    if epsilon is not None and float(epsilon) > 0.0:
        sigma = sigma + float(epsilon) * np.eye(N)

    return sigma.astype(float)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "seeds" / f"seed_{int(args.seed)}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing seed folder: {data_dir}")

    members_path = data_dir / "members.csv"
    claims_path = data_dir / "claims.csv"
    feats_path = data_dir / "feature_catalog.csv"

    members = _safe_read_csv(members_path)
    claims = _safe_read_csv(claims_path)
    features = _safe_read_csv(feats_path)

    # Validate feature catalog required columns
    needed_feat = {"feature_id", "cat1_id", "cat2_id", "impact_factor"}
    missing = needed_feat - set(features.columns)
    if missing:
        raise ValueError(
            f"feature_catalog.csv missing columns: {sorted(missing)}. "
            f"Hybrid design requires cat1_id, cat2_id, impact_factor."
        )

    U = int(len(members))
    if U <= 0:
        raise ValueError("members.csv is empty; cannot extract parameters.")
    N = int(len(features))
    if N <= 0:
        raise ValueError("feature_catalog.csv is empty; cannot extract parameters.")

    # Infer number of categories C
    # Prefer max in claims; fallback to max(cat ids in features); else 12.
    C_claims = int(claims["category_id"].max()) if (not claims.empty and "category_id" in claims.columns) else 0
    C_feats = int(max(features["cat1_id"].max(), features["cat2_id"].max())) if N > 0 else 0
    C = int(max(C_claims, C_feats, 12))

    # Build spend matrix S (U x C)
    S = _build_member_category_spend(claims, members, C=C)

    # Vectorised feature cost matrix (U x N)
    cat1 = features["cat1_id"].astype(int).to_numpy() - 1
    cat2 = features["cat2_id"].astype(int).to_numpy() - 1
    impact = features["impact_factor"].astype(float).to_numpy()

    if np.any(cat1 < 0) or np.any(cat2 < 0) or np.any(cat1 >= C) or np.any(cat2 >= C):
        raise ValueError("Feature category ids out of bounds relative to inferred C. Check generation.")

    # feature_costs[u, i] = impact[i] * (S[u, cat1[i]] + S[u, cat2[i]])
    feature_costs = (S[:, cat1] + S[:, cat2]) * impact.reshape(1, -1)

    # c_vector
    c_vector = feature_costs.mean(axis=0).astype(float)

    # Sigma dense then sparsify
    # np.cov expects rows as observations when rowvar=False
    sigma_dense = np.cov(feature_costs, rowvar=False)
    sigma = _sparsify_topd_sigma(sigma_dense, d=int(args.d), epsilon=float(args.epsilon))

    # Outputs
    c_out = data_dir / "c_vector.csv"
    s_out = data_dir / "sigma_matrix.npz"

    if (not args.overwrite) and (c_out.exists() or s_out.exists()):
        raise FileExistsError(
            f"{c_out.name} or {s_out.name} already exists. Use --overwrite to regenerate."
        )

    pd.DataFrame({"expected_cost": c_vector}).to_csv(c_out, index=False)
    np.savez_compressed(s_out, sigma=sigma)

    # Update meta
    meta_path = data_dir / "instance_meta.json"
    _write_meta(
        meta_path,
        updates={
            "extraction": {
                "method": "claims_derived",
                "seed": int(args.seed),
                "num_members": U,
                "num_features": N,
                "num_categories": C,
                "sparsity_top_d": int(args.d),
                "epsilon": float(args.epsilon),
            }
        },
    )

    print("Parameter extraction completed.")
    print(f"Seed: {args.seed} | U={U} | N={N} | C={C}")
    print(f"Saved: {c_out.name}, {s_out.name}")


if __name__ == "__main__":
    main()
