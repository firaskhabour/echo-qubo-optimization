#src/generate_date.py
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Config helpers
# -----------------------------
def load_cfg(project_root: Path) -> dict:
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_load_experiment_plan(project_root: Path) -> Optional[dict]:
    plan_path = project_root / "config" / "experiment_plan.yaml"
    if not plan_path.exists():
        return None
    with open(plan_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_N_max(cfg: dict, plan: Optional[dict], N_cli: Optional[int]) -> int:
    """
    We always generate a MASTER instance at N_max to guarantee prefix-consistency across N sweeps.

    Priority:
      1) experiment_plan.yaml max(feature_sizes)
      2) config.yaml features.N_default
      3) CLI --N (fallback)
    """
    if plan and "feature_sizes" in plan and len(plan["feature_sizes"]) > 0:
        return int(max(int(x) for x in plan["feature_sizes"]))

    n_default = int(cfg.get("features", {}).get("N_default", 100))
    if n_default > 0:
        return n_default

    if N_cli is None:
        raise ValueError(
            "Cannot infer N_max: no experiment_plan.yaml, no config features.N_default, and no --N provided."
        )
    return int(N_cli)


# -----------------------------
# Synthetic generation (hybrid)
# -----------------------------
def _category_names_12() -> Tuple[list[str], dict[int, str]]:
    """
    Stable 12-category set for paper clarity.
    We keep both ids and names (ids are what the optimisation uses).
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


def generate_members(cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    U = int(cfg["population"]["num_members"])

    # Age groups using shares
    age_groups = []
    age_keys = list(cfg["age_distribution"].keys())

    for k in age_keys:
        share = float(cfg["age_distribution"][k]["share"])
        age_groups += [k] * int(round(share * U))

    # Fix length drift from rounding
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

    # Risk score: lognormal(mean, sigma) on the log scale (numpy convention)
    risk_cfg = cfg.get("risk_score", {}) or {}
    mean = float(risk_cfg.get("mean", 0.0))
    sigma = float(risk_cfg.get("sigma", 0.8))
    risk = rng.lognormal(mean=mean, sigma=sigma, size=U)

    return pd.DataFrame(
        {
            "member_id": np.arange(1, U + 1),
            "age": ages,
            "age_group": age_groups,
            "region": regions,
            "risk_score": risk,
            "seed": seed,
        }
    )


def generate_claims(cfg: dict, seed: int, members: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight claims generator to support the original 'claims -> feature costs' story.

    - Claim counts: Poisson(base_lambda * risk_score)
    - Categories: 12 fixed categories, drawn with a fixed probability vector
    - Amounts: Lognormal with sigma from config; mu chosen to target a reasonable mean amount

    This is intentionally simple: it is only to support coherent feature cost extraction and
    preserve heavy-tailed severity.
    """
    rng = np.random.default_rng(seed + 55_000)

    U = int(len(members))
    months = int(cfg.get("population", {}).get("coverage_months", 12))

    # Frequency
    lam0 = float(cfg.get("claims_frequency", {}).get("base_lambda", 1.5))
    risk = members["risk_score"].to_numpy(dtype=float)
    lam = np.clip(lam0 * risk, 0.0, None)

    n_claims = rng.poisson(lam=lam, size=U)

    # Categories
    _, id_to_name = _category_names_12()
    cat_ids = np.array(list(id_to_name.keys()), dtype=int)

    # Simple, fixed probability vector (can be refined later; stable for reproducibility)
    # Must sum to 1.
    p = np.array(
        [0.08,  # inpatient (low freq, high cost)
        0.25,  # outpatient (high freq, low cost)  
        0.30,  # pharmacy (very high freq)
        0.02,  # oncology (rare)
        0.01,  # maternity (rare in general pop)
        0.12,  # diagnostics (common)
        0.06,  # emergency
        0.08,  # chronic care
        0.03,  # mental health
        0.03,  # physiotherapy
        0.01,  # dental
        0.01],  # international
        dtype=float,
    )
    p = p / np.sum(p)

    # Severity
    sev_cfg = cfg.get("claims_severity", {}) or {}
    sigma = float(sev_cfg.get("sigma", 1.2))

    # Choose mu to target a mean claim amount (default 250) if not provided
    target_mean = float(sev_cfg.get("mean_amount", 250.0))
    # For lognormal: E[X] = exp(mu + 0.5*sigma^2)
    mu = float(np.log(max(1e-9, target_mean)) - 0.5 * sigma * sigma)

    rows = []
    claim_id = 1

    # Simple claim date model: uniform month bucket (1..months)
    for u in range(U):
        k = int(n_claims[u])
        if k <= 0:
            continue

        cats = rng.choice(cat_ids, size=k, replace=True, p=p)
        amounts = rng.lognormal(mean=mu, sigma=sigma, size=k)

        # Store as an ISO-like month index for auditability (no need for actual dates here)
        months_draw = rng.integers(1, months + 1, size=k)

        member_id = int(members.iloc[u]["member_id"])
        for j in range(k):
            rows.append(
                {
                    "claim_id": claim_id,
                    "member_id": member_id,
                    "category_id": int(cats[j]),
                    "amount": float(amounts[j]),
                    "month": int(months_draw[j]),
                }
            )
            claim_id += 1

    return pd.DataFrame(rows)


def generate_feature_catalog(cfg: dict, seed: int, N_max: int) -> pd.DataFrame:
    """
    Hybrid: match the original conceptual model.
    Each feature affects exactly two categories, and has an impact factor.
    """
    rng = np.random.default_rng(seed + 44_000)
    cat_names, id_to_name = _category_names_12()
    cat_ids = np.arange(1, len(cat_names) + 1, dtype=int)

    cat1 = rng.choice(cat_ids, size=N_max, replace=True)
    cat2 = rng.choice(cat_ids, size=N_max, replace=True)

    # Ensure cat2 != cat1
    for i in range(N_max):
        if int(cat2[i]) == int(cat1[i]):
            choices = cat_ids[cat_ids != cat1[i]]
            cat2[i] = rng.choice(choices)

    impact = rng.uniform(0.60, 1.00, size=N_max)

    return pd.DataFrame(
        {
            "feature_id": np.arange(1, N_max + 1, dtype=int),
            "feature_name": [f"f_{i}" for i in range(1, N_max + 1)],
            "cat1_id": cat1.astype(int),
            "cat2_id": cat2.astype(int),
            "impact_factor": impact.astype(float),
        }
    )


def generate_feature_costs_master(cfg: dict, seed: int, N_max: int) -> np.ndarray:
    """
    Keep existing behaviour: generate expected costs c_i directly.
    This avoids making the entire pipeline depend on claims simulation and extraction.
    The claims + feature catalog are still generated for credibility and optional validation.

    Scale is controlled by config.qubo_generation.c_scale to be comparable to premium bands.
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
    Generate a symmetric PSD covariance-like matrix, MASTER at N_max.

    Low-rank factor:
      Sigma = A A^T + noise I

    Optional normalisation:
      - normalise so median(diag(Sigma)) ~= 1, then multiply by sigma_scale

    NOTE: For the paper, we describe this as a controlled low-rank covariance generator.
    """
    rng = np.random.default_rng(seed + 22_000)
    gen_cfg = cfg.get("qubo_generation", {}) or {}

    rank = int(gen_cfg.get("sigma_rank", max(5, min(20, N_max // 5))))
    noise = float(gen_cfg.get("sigma_noise", 1e-3))
    sigma_scale = float(gen_cfg.get("sigma_scale", 1.0))
    normalize_diag = bool(gen_cfg.get("sigma_normalize_diag", True))

    A = rng.normal(0.0, 1.0, size=(N_max, rank))
    Sigma = A @ A.T
    Sigma = Sigma + noise * np.eye(N_max)

    Sigma = 0.5 * (Sigma + Sigma.T)

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
    Generate MASTER set of regulatory feature IDs on 1..N_max, then for any N we take {id <= N}.
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

    # Bin edges for near-uniform placement
    edges = np.linspace(1, N_max + 1, num=R_size + 1, dtype=int)

    ids = []
    for b in range(R_size):
        lo = int(edges[b])
        hi = int(edges[b + 1]) - 1
        if hi < lo:
            hi = lo
        ids.append(int(rng.integers(lo, hi + 1)))

    ids = np.array(sorted(set(ids)), dtype=int)

    # Top-up if duplicates occurred
    if len(ids) < R_size:
        remaining = np.setdiff1d(np.arange(1, N_max + 1, dtype=int), ids)
        extra = rng.choice(remaining, size=(R_size - len(ids)), replace=False)
        ids = np.array(sorted(np.concatenate([ids, extra])), dtype=int)

    return ids.astype(int)


def wipe_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for p in out_dir.glob("*"):
        if p.is_file():
            p.unlink()
        else:
            import shutil
            shutil.rmtree(p)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic seed data under data/seeds/seed_<seed>/")
    parser.add_argument("--seed", type=int, required=True, help="Seed id (e.g., 1000, 1001, ...)")
    parser.add_argument("--N", type=int, default=None, help="(Ignored for generation scale) kept for compatibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing seed folder if it exists")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(project_root)
    plan = try_load_experiment_plan(project_root)

    seed = int(args.seed)
    N_max = get_N_max(cfg, plan, args.N)
    if N_max <= 0:
        raise ValueError("N_max must be > 0")

    # Optional check against declared total_features (do not hard-fail; keep pipeline flexible)
    total_features_declared = int(cfg.get("features", {}).get("total_features", N_max))
    if total_features_declared > 0 and N_max > total_features_declared:
        print(
            f"WARNING: N_max={N_max} exceeds config.features.total_features={total_features_declared}. "
            "Proceeding anyway."
        )

    base_dir = project_root / "data" / "seeds"
    base_dir.mkdir(parents=True, exist_ok=True)

    out_dir = base_dir / f"seed_{seed}"
    if out_dir.exists():
        if not args.overwrite:
            meta_path = out_dir / "instance_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    existing_N = int(meta.get("N_generated_master", meta.get("N", 0)))
                    if existing_N >= N_max:
                        print(f"Seed folder already exists: {out_dir} (master N={existing_N} >= {N_max}), skipping.")
                        return
                except Exception:
                    pass
            print(f"Seed folder already exists: {out_dir} (use --overwrite to regenerate)")
            return

        wipe_dir(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate MASTER artefacts at N_max
    members = generate_members(cfg, seed)
    claims = generate_claims(cfg, seed, members)
    catalog = generate_feature_catalog(cfg, seed, N_max)

    c_master = generate_feature_costs_master(cfg, seed, N_max)
    Sigma_master = generate_sigma_master(cfg, seed, N_max)
    reg_ids_master = generate_regulatory_master(cfg, seed, N_max)

    # Save core files expected by pipeline (MASTER)
    pd.DataFrame({"expected_cost": c_master}).to_csv(out_dir / "c_vector.csv", index=False)
    np.savez_compressed(out_dir / "sigma_matrix.npz", sigma=Sigma_master)
    pd.DataFrame({"feature_id": reg_ids_master}).to_csv(out_dir / "regulatory_set.csv", index=False)

    # Hybrid-support files
    members.to_csv(out_dir / "members.csv", index=False)
    catalog.to_csv(out_dir / "feature_catalog.csv", index=False)

    # Claims for optional extraction validation
    claims.to_csv(out_dir / "claims.csv", index=False)

    # Metadata (paper-defensible)
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
            "Hybrid generator: claims + feature catalogue are generated for auditability, "
            "but c and Sigma are produced directly as controlled synthetic quantities. "
            "Master instance generated at N_max; smaller N are prefixes (slice c and Sigma; "
            "filter regulatory ids <= N)."
        ),
    }
    with open(out_dir / "instance_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated seed data: {out_dir}")
    print("Files: c_vector.csv, sigma_matrix.npz, regulatory_set.csv, members.csv, feature_catalog.csv, claims.csv, instance_meta.json")
    print(f"Master N_max = {N_max} (prefix-consistent across N sweeps)")


if __name__ == "__main__":
    main()
