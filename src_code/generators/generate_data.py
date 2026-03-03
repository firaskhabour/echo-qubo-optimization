# src_code/generators/generate_data.py
"""
Synthetic Seed Data Generator
==============================
Generates all seed data required by the QUBO construction pipeline for a
single seed identifier. Outputs are written to data/seeds/seed_<seed>/ and
constitute the reproducible instance corpus described in Section 4.1 of the
accompanying paper.

Design principle — hybrid generation:
    Demographic and claims artefacts (members, claims, feature catalogue) are
    generated for auditability and narrative coherence with the insurance
    application. The quantities that enter the QUBO directly — the expected
    cost vector c and the covariance matrix Σ — are generated as controlled
    synthetic quantities whose statistical properties are governed by
    config.yaml (see generate_feature_costs_master and generate_sigma_master).
    This hybrid approach keeps the pipeline defensible while providing full
    control over the spectral properties studied in the paper.

Master-instance design:
    All artefacts are generated at a master size N_max (the largest N in the
    experiment plan). Smaller instances are obtained by taking the leading
    prefix: c[:N], Sigma[:N, :N], and regulatory IDs with id <= N. This
    guarantees strict prefix consistency — any instance at size N is a
    structural sub-problem of the master — and ensures that changes in N
    reflect only dimensional growth, not changes in the underlying population.

Output files (written to data/seeds/seed_<SEED>/):
    c_vector.csv          -- expected cost vector (N_max entries)
    sigma_matrix.npz      -- covariance matrix (N_max × N_max)
    regulatory_set.csv    -- regulatory feature IDs (1-based, subset of 1..N_max)
    members.csv           -- synthetic member population (audit support)
    feature_catalog.csv   -- feature-to-category mapping (audit support)
    claims.csv            -- simulated claims (audit support)
    instance_meta.json    -- generation parameters and provenance metadata

Usage:
    python generate_data.py --seed 1000
    python generate_data.py --seed 2005 --overwrite
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_cfg(project_root: Path) -> dict:
    """Load the main config.yaml from the project config directory."""
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_load_experiment_plan(project_root: Path) -> Optional[dict]:
    """
    Load experiment_plan.yaml if present; return None if absent.

    The experiment plan is the authoritative source for the N grid used in
    the computational study (Section 4.1). When present, its maximum N
    determines the master generation size.
    """
    plan_path = project_root / "config" / "experiment_plan.yaml"
    if not plan_path.exists():
        return None
    with open(plan_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_N_max(cfg: dict, plan: Optional[dict], N_cli: Optional[int]) -> int:
    """
    Determine the master generation size N_max.

    All artefacts are generated at N_max to guarantee prefix consistency
    across the N grid. Resolution priority:
      1. experiment_plan.yaml  max(feature_sizes)  -- canonical source
      2. config.yaml           features.N_default   -- fallback
      3. --N CLI argument                            -- last resort
    """
    if plan and "feature_sizes" in plan and len(plan["feature_sizes"]) > 0:
        return int(max(int(x) for x in plan["feature_sizes"]))

    n_default = int(cfg.get("features", {}).get("N_default", 100))
    if n_default > 0:
        return n_default

    if N_cli is None:
        raise ValueError(
            "Cannot infer N_max: no experiment_plan.yaml, no config features.N_default, "
            "and no --N provided."
        )
    return int(N_cli)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

def _category_names_12() -> Tuple[list[str], dict[int, str]]:
    """
    Return the fixed 12-category set used across all seeds.

    Categories are stable across seeds and generations to ensure that
    feature-to-category mappings in the feature catalogue are interpretable
    in the insurance domain context described in Section 2.1.
    """
    names = [
        "inpatient",
        "outpatient",
        "pharmacy",
        "oncology",
        "maternity",
        "diagnostics",
        "emergency",
        "chronic_care",
        "mental_health",
        "physiotherapy",
        "dental",
        "international",
    ]
    id_to_name = {i + 1: names[i] for i in range(len(names))}
    return names, id_to_name


# ---------------------------------------------------------------------------
# Synthetic population and claims (audit artefacts)
# ---------------------------------------------------------------------------

def generate_members(cfg: dict, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic member population.

    Member demographics (age, region, risk score) are generated for narrative
    coherence with the insurance application. They do not directly enter the
    QUBO; the cost vector c and covariance matrix Σ are generated independently
    via generate_feature_costs_master and generate_sigma_master.
    """
    rng = np.random.default_rng(seed)
    U = int(cfg["population"]["num_members"])

    # Age groups: construct by share, then fix rounding drift
    age_groups = []
    age_keys = list(cfg["age_distribution"].keys())
    for k in age_keys:
        share = float(cfg["age_distribution"][k]["share"])
        age_groups += [k] * int(round(share * U))
    if len(age_groups) < U:
        age_groups += [age_keys[0]] * (U - len(age_groups))
    elif len(age_groups) > U:
        age_groups = age_groups[:U]
    rng.shuffle(age_groups)

    ages = []
    for g in age_groups:
        dist = cfg["age_distribution"][g]
        ages.append(int(rng.integers(int(dist["min_age"]), int(dist["max_age"]) + 1)))

    regions = rng.choice(cfg["regions"], size=U)

    # Risk score: Lognormal(mean, sigma) on the log scale (NumPy convention)
    risk_cfg = cfg.get("risk_score", {}) or {}
    mean = float(risk_cfg.get("mean", 0.0))
    sigma = float(risk_cfg.get("sigma", 0.8))
    risk = rng.lognormal(mean=mean, sigma=sigma, size=U)

    return pd.DataFrame({
        "member_id": np.arange(1, U + 1),
        "age": ages,
        "age_group": age_groups,
        "region": regions,
        "risk_score": risk,
        "seed": seed,
    })


def generate_claims(cfg: dict, seed: int, members: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a lightweight synthetic claims dataset.

    Claims are generated to support the 'claims → feature costs' narrative and
    to provide an optional validation path. They do not determine c or Σ
    directly; those quantities are generated via their own controlled processes.

    Generation model:
        Frequency : Poisson(base_lambda × risk_score) per member
        Categories: sampled from 12 fixed categories with a fixed probability
                    vector reflecting plausible utilisation patterns
        Severity  : Lognormal(mu, sigma) where mu is chosen to target a
                    configurable mean claim amount (config: claims_severity.mean_amount)
    """
    rng = np.random.default_rng(seed + 55_000)

    U = int(len(members))
    months = int(cfg.get("population", {}).get("coverage_months", 12))

    # Claim frequency: Poisson scaled by individual risk score
    lam0 = float(cfg.get("claims_frequency", {}).get("base_lambda", 1.5))
    risk = members["risk_score"].to_numpy(dtype=float)
    lam = np.clip(lam0 * risk, 0.0, None)
    n_claims = rng.poisson(lam=lam, size=U)

    # Category selection
    _, id_to_name = _category_names_12()
    cat_ids = np.array(list(id_to_name.keys()), dtype=int)

    # Fixed probability vector — reflects plausible utilisation shares across
    # the 12 categories. Stable across seeds for reproducibility.
    p = np.array(
        [0.08,   # inpatient       (low frequency, high severity)
         0.25,   # outpatient      (high frequency, low severity)
         0.30,   # pharmacy        (very high frequency)
         0.02,   # oncology        (rare)
         0.01,   # maternity       (rare in general population)
         0.12,   # diagnostics     (common)
         0.06,   # emergency
         0.08,   # chronic_care
         0.03,   # mental_health
         0.03,   # physiotherapy
         0.01,   # dental
         0.01],  # international
        dtype=float,
    )
    p = p / np.sum(p)

    # Claim severity: Lognormal with mu chosen to target a configurable mean amount.
    # For Lognormal: E[X] = exp(mu + 0.5 * sigma^2), so mu = log(target_mean) - 0.5 * sigma^2.
    sev_cfg = cfg.get("claims_severity", {}) or {}
    sigma = float(sev_cfg.get("sigma", 1.2))
    target_mean = float(sev_cfg.get("mean_amount", 250.0))
    mu = float(np.log(max(1e-9, target_mean)) - 0.5 * sigma * sigma)

    rows = []
    claim_id = 1
    for u in range(U):
        k = int(n_claims[u])
        if k <= 0:
            continue
        cats = rng.choice(cat_ids, size=k, replace=True, p=p)
        amounts = rng.lognormal(mean=mu, sigma=sigma, size=k)
        months_draw = rng.integers(1, months + 1, size=k)
        member_id = int(members.iloc[u]["member_id"])
        for j in range(k):
            rows.append({
                "claim_id": claim_id,
                "member_id": member_id,
                "category_id": int(cats[j]),
                "amount": float(amounts[j]),
                "month": int(months_draw[j]),
            })
            claim_id += 1

    return pd.DataFrame(rows)


def generate_feature_catalog(cfg: dict, seed: int, N_max: int) -> pd.DataFrame:
    """
    Generate a feature-to-category mapping for N_max coverage features.

    Each feature is associated with exactly two distinct clinical categories
    and an impact factor drawn from Uniform(0.60, 1.00). This structure
    reflects the insurance design setting described in Section 2.1, where
    coverage features span multiple benefit categories.

    The catalogue is an audit artefact; the QUBO cost vector c is derived
    directly via generate_feature_costs_master, not by aggregating claims
    through this catalogue.
    """
    rng = np.random.default_rng(seed + 44_000)
    cat_names, id_to_name = _category_names_12()
    cat_ids = np.arange(1, len(cat_names) + 1, dtype=int)

    cat1 = rng.choice(cat_ids, size=N_max, replace=True)
    cat2 = rng.choice(cat_ids, size=N_max, replace=True)

    # Ensure cat2 != cat1 for every feature
    for i in range(N_max):
        if int(cat2[i]) == int(cat1[i]):
            choices = cat_ids[cat_ids != cat1[i]]
            cat2[i] = rng.choice(choices)

    impact = rng.uniform(0.60, 1.00, size=N_max)

    return pd.DataFrame({
        "feature_id": np.arange(1, N_max + 1, dtype=int),
        "feature_name": [f"f_{i}" for i in range(1, N_max + 1)],
        "cat1_id": cat1.astype(int),
        "cat2_id": cat2.astype(int),
        "impact_factor": impact.astype(float),
    })


# ---------------------------------------------------------------------------
# QUBO-direct artefacts (c, Σ, regulatory set)
# ---------------------------------------------------------------------------

def generate_feature_costs_master(cfg: dict, seed: int, N_max: int) -> np.ndarray:
    """
    Generate the master expected cost vector c of length N_max.

    Costs are drawn from Lognormal(mean, sigma) scaled by c_scale (all
    parameters from config.yaml qubo_generation block). Lognormal costs
    produce realistic heavy-tailed cost heterogeneity consistent with
    feature-level claims costs in insurance.

    The scale c_scale is chosen so that C_scale = sum(c) is comparable in
    magnitude to the premium band values defined in the scenario configs.
    This ensures that penalty weights A_base = scale_factor * C_scale
    provide adequate constraint dominance (Section 2.11).

    Instances at N < N_max use the prefix c[:N].
    """
    rng = np.random.default_rng(seed + 11_000)
    gen_cfg = cfg.get("qubo_generation", {}) or {}

    mean = float(gen_cfg.get("c_lognormal_mean", 0.0))
    sigma = float(gen_cfg.get("c_lognormal_sigma", 0.75))
    scale = float(gen_cfg.get("c_scale", 1.0))

    c = rng.lognormal(mean=mean, sigma=sigma, size=N_max) * scale
    return c.astype(float)


def generate_sigma_master(cfg: dict, seed: int, N_max: int) -> np.ndarray:
    """
    Generate the master covariance matrix Σ of shape (N_max, N_max).

    Σ is constructed as the low-rank factorisation described in Section 2.3:

        Σ = A A^T + ε I

    where A is a random (N_max × rank) matrix with entries drawn from
    N(0, 1), and ε = sigma_noise ensures strict positive definiteness.
    Since A A^T is positive semidefinite and ε I is strictly positive
    definite, Σ is strictly positive definite for all instances, consistent
    with the classical mean–variance formulation [14].

    Optional diagonal normalisation (sigma_normalize_diag = true) rescales
    Σ so that median(diag(Σ)) = 1 before applying sigma_scale. This keeps
    the quadratic risk term lambda_risk * x^T Σ x in a consistent magnitude
    range relative to the linear cost term c^T x across different N.

    Instances at N < N_max use the leading principal submatrix Sigma[:N, :N].
    """
    rng = np.random.default_rng(seed + 22_000)
    gen_cfg = cfg.get("qubo_generation", {}) or {}

    rank = int(gen_cfg.get("sigma_rank", max(5, min(20, N_max // 5))))
    noise = float(gen_cfg.get("sigma_noise", 1e-3))
    sigma_scale = float(gen_cfg.get("sigma_scale", 1.0))
    normalize_diag = bool(gen_cfg.get("sigma_normalize_diag", True))

    # Low-rank factor model: Sigma = A A^T + noise * I
    A = rng.normal(0.0, 1.0, size=(N_max, rank))
    Sigma = A @ A.T
    Sigma = Sigma + noise * np.eye(N_max)

    # Symmetrise numerically (guards against floating-point asymmetry)
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Optional normalisation: rescale so median diagonal entry equals 1
    if normalize_diag:
        d = np.diag(Sigma)
        med = float(np.median(d)) if np.all(d > 0) else 1.0
        if med > 0:
            Sigma = Sigma / med

    Sigma *= sigma_scale
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma.astype(float)


def generate_regulatory_master(cfg: dict, seed: int, N_max: int) -> np.ndarray:
    """
    Generate the master regulatory feature ID set on {1, ..., N_max}.

    Regulatory features are placed using a near-uniform binning strategy:
    the range {1, ..., N_max} is divided into R_size equal-width bins and
    one ID is drawn uniformly from each bin. This ensures that regulatory
    features are spread across the feature catalogue at all sizes, which
    is consistent with the insurance requirement that essential benefits
    span multiple product categories (Section 2.4.3).

    For instances at N < N_max, the active regulatory set is {id : id <= N},
    the natural prefix restriction of the master set.

    R_size is determined by regulatory_share * N_max (rounded) unless
    regulatory_R_size is set explicitly in config.yaml.
    """
    rng = np.random.default_rng(seed + 33_000)
    gen_cfg = cfg.get("qubo_generation", {}) or {}

    regulatory_share = float(gen_cfg.get("regulatory_share", 0.15))
    R_size_override = gen_cfg.get("regulatory_R_size", None)

    if R_size_override is None:
        R_size = int(round(regulatory_share * N_max))
    else:
        R_size = int(R_size_override)

    R_size = max(0, min(R_size, N_max))
    if R_size == 0:
        return np.array([], dtype=int)

    # Near-uniform placement via equal-width bins
    edges = np.linspace(1, N_max + 1, num=R_size + 1, dtype=int)
    ids = []
    for b in range(R_size):
        lo = int(edges[b])
        hi = int(edges[b + 1]) - 1
        if hi < lo:
            hi = lo
        ids.append(int(rng.integers(lo, hi + 1)))

    ids = np.array(sorted(set(ids)), dtype=int)

    # Top-up if bin collisions reduced the count
    if len(ids) < R_size:
        remaining = np.setdiff1d(np.arange(1, N_max + 1, dtype=int), ids)
        extra = rng.choice(remaining, size=(R_size - len(ids)), replace=False)
        ids = np.array(sorted(np.concatenate([ids, extra])), dtype=int)

    return ids.astype(int)


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------

def wipe_dir(out_dir: Path) -> None:
    """Remove all files and subdirectories within out_dir (not the dir itself)."""
    if not out_dir.exists():
        return
    for p in out_dir.glob("*"):
        if p.is_file():
            p.unlink()
        else:
            import shutil
            shutil.rmtree(p)


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic seed data under data/seeds/seed_<seed>/."
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed identifier (e.g., 1000, 2005).")
    parser.add_argument("--N", type=int, default=None,
                        help="Unused; kept for CLI compatibility. N_max is inferred from "
                             "experiment_plan.yaml.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing seed folder if present.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    cfg = load_cfg(project_root)
    plan = try_load_experiment_plan(project_root)

    seed = int(args.seed)
    N_max = get_N_max(cfg, plan, args.N)
    if N_max <= 0:
        raise ValueError("N_max must be > 0.")

    # Warn if N_max exceeds the declared total_features cap (non-fatal)
    total_features_declared = int(cfg.get("features", {}).get("total_features", N_max))
    if total_features_declared > 0 and N_max > total_features_declared:
        print(
            f"WARNING: N_max={N_max} exceeds config.features.total_features="
            f"{total_features_declared}. Proceeding anyway."
        )

    base_dir = project_root / "data" / "seeds"
    base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guard against overwriting an existing seed folder
    # ------------------------------------------------------------------
    out_dir = base_dir / f"seed_{seed}"
    if out_dir.exists():
        if not args.overwrite:
            meta_path = out_dir / "instance_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    existing_N = int(meta.get("N_generated_master", meta.get("N", 0)))
                    if existing_N >= N_max:
                        print(
                            f"Seed folder already exists: {out_dir} "
                            f"(master N={existing_N} >= {N_max}), skipping."
                        )
                        return
                except Exception:
                    pass
            print(f"Seed folder already exists: {out_dir} (use --overwrite to regenerate).")
            return
        wipe_dir(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate all master artefacts at N_max
    # ------------------------------------------------------------------
    members  = generate_members(cfg, seed)
    claims   = generate_claims(cfg, seed, members)
    catalog  = generate_feature_catalog(cfg, seed, N_max)

    c_master      = generate_feature_costs_master(cfg, seed, N_max)
    Sigma_master  = generate_sigma_master(cfg, seed, N_max)
    reg_ids_master = generate_regulatory_master(cfg, seed, N_max)

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------

    # QUBO-direct artefacts (consumed by build_qubo.py)
    pd.DataFrame({"expected_cost": c_master}).to_csv(out_dir / "c_vector.csv", index=False)
    np.savez_compressed(out_dir / "sigma_matrix.npz", sigma=Sigma_master)
    pd.DataFrame({"feature_id": reg_ids_master}).to_csv(out_dir / "regulatory_set.csv", index=False)

    # Audit artefacts (support narrative and optional validation)
    members.to_csv(out_dir / "members.csv", index=False)
    catalog.to_csv(out_dir / "feature_catalog.csv", index=False)
    claims.to_csv(out_dir / "claims.csv", index=False)

    # ------------------------------------------------------------------
    # Write provenance metadata
    # ------------------------------------------------------------------
    gen_cfg = cfg.get("qubo_generation", {}) or {}
    meta: Dict = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "N_generated_master": N_max,
        "prefix_consistent": True,
        "population": {
            "num_members": int(cfg.get("population", {}).get("num_members", 0)),
            "coverage_months": int(cfg.get("population", {}).get("coverage_months", 12)),
        },
        "risk_score": {
            "distribution": str(cfg.get("risk_score", {}).get("distribution", "lognormal")),
            "mean": float(cfg.get("risk_score", {}).get("mean", 0.0)),
            "sigma": float(cfg.get("risk_score", {}).get("sigma", 0.8)),
        },
        "claims_frequency": {
            "base_lambda": float(cfg.get("claims_frequency", {}).get("base_lambda", 1.5)),
        },
        "claims_severity": {
            "distribution": str(cfg.get("claims_severity", {}).get("distribution", "lognormal")),
            "sigma": float(cfg.get("claims_severity", {}).get("sigma", 1.2)),
            "mean_amount_target": float(cfg.get("claims_severity", {}).get("mean_amount", 250.0)),
        },
        "feature_catalog": {
            "categories": 12,
            "two_categories_per_feature": True,
            "impact_factor_range": [0.60, 1.00],
        },
        "qubo_generation_params": {
            "c_lognormal_mean": float(gen_cfg.get("c_lognormal_mean", 0.0)),
            "c_lognormal_sigma": float(gen_cfg.get("c_lognormal_sigma", 0.75)),
            "c_scale": float(gen_cfg.get("c_scale", 1.0)),
            "sigma_rank": int(gen_cfg.get("sigma_rank", max(5, min(20, N_max // 5)))),
            "sigma_noise": float(gen_cfg.get("sigma_noise", 1e-3)),
            "sigma_scale": float(gen_cfg.get("sigma_scale", 1.0)),
            "sigma_normalize_diag": bool(gen_cfg.get("sigma_normalize_diag", True)),
            "regulatory_share": float(gen_cfg.get("regulatory_share", 0.15)),
            "regulatory_R_size": gen_cfg.get("regulatory_R_size", None),
        },
        "notes": (
            "Hybrid generator: member population, claims, and feature catalogue are produced "
            "for auditability and insurance narrative coherence. The QUBO-direct quantities "
            "(c_vector, sigma_matrix, regulatory_set) are generated as controlled synthetic "
            "artefacts whose parameters are governed by config.yaml qubo_generation block. "
            "Covariance matrix generated as Sigma = A A^T + epsilon*I (Section 2.3 of paper); "
            "strictly positive definite by construction. Master instance generated at N_max; "
            "smaller N use prefix slices (c[:N], Sigma[:N,:N], regulatory IDs <= N)."
        ),
    }
    with open(out_dir / "instance_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    print(f"Generated seed data: {out_dir}")
    print(
        "Files: c_vector.csv, sigma_matrix.npz, regulatory_set.csv, "
        "members.csv, feature_catalog.csv, claims.csv, instance_meta.json"
    )
    print(f"Master N_max = {N_max} (prefix-consistent across N grid)")


if __name__ == "__main__":
    main()