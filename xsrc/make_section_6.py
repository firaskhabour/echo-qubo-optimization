import argparse
import math
from pathlib import Path
from typing import Any, Tuple

import pandas as pd


# =========================
# Column schemas supported
# =========================
# New schema (your current results_master.csv)
NEW_COLS = [
    "scenario_id", "scenario_name", "seed", "N", "M", "K", "L_reg_slack", "n_total",
    "energy_greedy", "energy_sa_best", "gap_greedy_to_sa",
    "chosen_premium_band_sa", "chosen_deductible_band_sa",
    "selected_features_count_sa", "selected_reg_features_count_sa",
    "R_size", "Rmin",
    "gurobi_ran", "energy_gurobi", "gap_sa_to_gurobi",
    "gurobi_status", "gurobi_mip_gap", "gurobi_runtime_sec",
]

# Old schema (backward compatibility)
OLD_COLS = [
    "scenario_id", "scenario_name",
    "seed", "N", "M", "K", "L", "n_total",
    "energy_best", "energy_greedy",
    "chosen_premium_band", "chosen_deductible_band",
    "selected_features_count", "selected_reg_features_count",
    "R_size", "Rmin",
]


def _mode_and_share(series: pd.Series) -> Tuple[Any, float]:
    s = series.dropna()
    if len(s) == 0:
        return None, 0.0
    vc = s.value_counts()
    mode_val = vc.index[0]
    share = float(vc.iloc[0]) / float(len(s))
    return mode_val, share


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _df_to_latex(df: pd.DataFrame, caption: str, label: str, *, float_format: str = "%.6f") -> str:
    return df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        float_format=lambda x: float_format % x if isinstance(x, (float, int)) and not isinstance(x, bool) else str(x),
    )


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert either old or new results_master schema into a normalized internal schema:
      energy_best   := SA best energy
      energy_greedy := greedy energy
      chosen_premium_band := SA choice
      chosen_deductible_band := SA choice
      selected_features_count := SA selected features count
      selected_reg_features_count := SA selected regulatory features count
      L := L_reg_slack (if present)
      gurobi_* kept if present
    """
    cols = set(df.columns)

    # Detect new schema
    if "energy_sa_best" in cols:
        # Validate minimally required new columns
        required = {
            "scenario_id", "scenario_name", "seed", "N", "M", "K", "L_reg_slack", "n_total",
            "energy_greedy", "energy_sa_best",
            "chosen_premium_band_sa", "chosen_deductible_band_sa",
            "selected_features_count_sa",
            "R_size", "Rmin",
        }
        missing = sorted(list(required - cols))
        if missing:
            raise ValueError(f"results_master.csv missing required columns (new schema): {missing}")

        out = df.copy()

        # Normalize names to legacy-friendly internal names
        out["energy_best"] = pd.to_numeric(out["energy_sa_best"], errors="coerce")
        out["energy_greedy"] = pd.to_numeric(out["energy_greedy"], errors="coerce")

        out["chosen_premium_band"] = pd.to_numeric(out["chosen_premium_band_sa"], errors="coerce")
        out["chosen_deductible_band"] = pd.to_numeric(out["chosen_deductible_band_sa"], errors="coerce")

        out["selected_features_count"] = pd.to_numeric(out["selected_features_count_sa"], errors="coerce")

        # reg count may be blank for some scenarios; keep numeric
        if "selected_reg_features_count_sa" in out.columns:
            out["selected_reg_features_count"] = pd.to_numeric(out["selected_reg_features_count_sa"], errors="coerce")
        else:
            out["selected_reg_features_count"] = math.nan

        out["L"] = pd.to_numeric(out["L_reg_slack"], errors="coerce")

        # gaps: compute defensively
        out["gap_greedy_to_sa"] = pd.to_numeric(out.get("gap_greedy_to_sa", out["energy_greedy"] - out["energy_best"]), errors="coerce")

        # gurobi fields if present
        if "energy_gurobi" in out.columns:
            out["energy_gurobi"] = pd.to_numeric(out["energy_gurobi"], errors="coerce")
        if "gap_sa_to_gurobi" in out.columns:
            out["gap_sa_to_gurobi"] = pd.to_numeric(out["gap_sa_to_gurobi"], errors="coerce")

        # Coerce core numeric fields
        for c in ["scenario_id", "seed", "N", "M", "K", "n_total", "R_size", "Rmin"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        return out

    # Else assume old schema
    required_old = set(OLD_COLS)
    missing_old = sorted(list(required_old - cols))
    if missing_old:
        raise ValueError(
            "results_master.csv is missing expected columns for BOTH schemas.\n"
            f"Missing old-schema columns: {missing_old}\n"
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    # Normalize numeric
    num_cols = [
        "scenario_id", "seed", "N", "M", "K", "L", "n_total",
        "energy_best", "energy_greedy",
        "chosen_premium_band", "chosen_deductible_band",
        "selected_features_count", "selected_reg_features_count",
        "R_size", "Rmin",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # derive gap
    out["gap_greedy_to_sa"] = out["energy_greedy"] - out["energy_best"]

    return out


# =========================
# Table builders
# =========================
def build_table_6a_sa_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6A (SA): summary by scenario x N
      - runs
      - SA best energy mean/std
      - mean selected features
      - mode premium/deductible (SA) with share
      - mean selected reg features (if available)
    """
    rows = []
    g = df.groupby(["scenario_id", "scenario_name", "N"], dropna=False)
    for (sid, sname, N), sub in g:
        prem_mode, prem_share = _mode_and_share(sub["chosen_premium_band"])
        ded_mode, ded_share = _mode_and_share(sub["chosen_deductible_band"])

        reg_mean = sub["selected_reg_features_count"].mean() if sub["selected_reg_features_count"].notna().any() else math.nan
        R_size = sub["R_size"].dropna().iloc[0] if sub["R_size"].dropna().shape[0] else math.nan
        Rmin = sub["Rmin"].dropna().iloc[0] if sub["Rmin"].dropna().shape[0] else math.nan

        rows.append({
            "scenario_id": int(sid) if not pd.isna(sid) else sid,
            "scenario_name": sname,
            "N": int(N) if not pd.isna(N) else N,
            "runs": int(len(sub)),
            "energy_sa_mean": sub["energy_best"].mean(),
            "energy_sa_std": sub["energy_best"].std(ddof=1),
            "selected_features_mean": sub["selected_features_count"].mean(),
            "selected_reg_features_mean": reg_mean,
            "R_size": R_size,
            "Rmin": Rmin,
            "premium_mode": prem_mode,
            "premium_mode_share": prem_share,
            "deductible_mode": ded_mode,
            "deductible_mode_share": ded_share,
        })
    return pd.DataFrame(rows).sort_values(["scenario_id", "N"]).reset_index(drop=True)


def build_table_6b_gap_greedy_to_sa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6B: greedy vs SA gaps by scenario x N
      gap = energy_greedy - energy_sa_best
    """
    df2 = df.copy()
    df2["gap"] = df2["energy_greedy"] - df2["energy_best"]

    rows = []
    g = df2.groupby(["scenario_id", "scenario_name", "N"], dropna=False)
    for (sid, sname, N), sub in g:
        rows.append({
            "scenario_id": int(sid) if not pd.isna(sid) else sid,
            "scenario_name": sname,
            "N": int(N) if not pd.isna(N) else N,
            "runs": int(len(sub)),
            "gap_mean": sub["gap"].mean(),
            "gap_median": sub["gap"].median(),
            "sa_improve_rate": float((sub["gap"] > 1e-12).mean()),
            "equal_rate": float((sub["gap"].abs() <= 1e-12).mean()),
            "energy_sa_mean": sub["energy_best"].mean(),
            "energy_greedy_mean": sub["energy_greedy"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["scenario_id", "N"]).reset_index(drop=True)


def build_table_6c_model_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6C: footprint by scenario x N
    """
    rows = []
    g = df.groupby(["scenario_id", "scenario_name", "N"], dropna=False)
    for (sid, sname, N), sub in g:
        rows.append({
            "scenario_id": int(sid) if not pd.isna(sid) else sid,
            "scenario_name": sname,
            "N": int(N) if not pd.isna(N) else N,
            "runs": int(len(sub)),
            "M_min": sub["M"].min(), "M_max": sub["M"].max(),
            "K_min": sub["K"].min(), "K_max": sub["K"].max(),
            "L_min": sub["L"].min(), "L_max": sub["L"].max(),
            "n_total_min": sub["n_total"].min(), "n_total_max": sub["n_total"].max(),
        })
    return pd.DataFrame(rows).sort_values(["scenario_id", "N"]).reset_index(drop=True)


def build_table_gurobi_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    New: SA vs Gurobi gaps by scenario x N (only where Gurobi ran and energy_gurobi exists)
      gap_sa_to_gurobi = energy_sa_best - energy_gurobi  (>=0 means SA worse than Gurobi)
    """
    if "energy_gurobi" not in df.columns:
        return pd.DataFrame()

    df2 = df.copy()

    # Determine which rows have usable gurobi
    if "gurobi_ran" in df2.columns:
        df2["gurobi_ran"] = pd.to_numeric(df2["gurobi_ran"], errors="coerce").fillna(0).astype(int)
        df2 = df2[df2["gurobi_ran"] == 1]

    df2["energy_gurobi"] = pd.to_numeric(df2["energy_gurobi"], errors="coerce")
    df2 = df2[df2["energy_gurobi"].notna()]

    if len(df2) == 0:
        return pd.DataFrame()

    df2["gap_sa_to_gurobi_calc"] = df2["energy_best"] - df2["energy_gurobi"]

    rows = []
    g = df2.groupby(["scenario_id", "scenario_name", "N"], dropna=False)
    for (sid, sname, N), sub in g:
        rows.append({
            "scenario_id": int(sid) if not pd.isna(sid) else sid,
            "scenario_name": sname,
            "N": int(N) if not pd.isna(N) else N,
            "runs_gurobi": int(len(sub)),
            "energy_gurobi_mean": sub["energy_gurobi"].mean(),
            "energy_sa_mean": sub["energy_best"].mean(),
            "gap_sa_to_gurobi_mean": sub["gap_sa_to_gurobi_calc"].mean(),
            "gap_sa_to_gurobi_median": sub["gap_sa_to_gurobi_calc"].median(),
            "sa_equals_gurobi_rate": float((sub["gap_sa_to_gurobi_calc"].abs() <= 1e-9).mean()),
            "gurobi_runtime_sec_mean": pd.to_numeric(sub.get("gurobi_runtime_sec", math.nan), errors="coerce").mean()
            if "gurobi_runtime_sec" in sub.columns else math.nan,
        })
    return pd.DataFrame(rows).sort_values(["scenario_id", "N"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Generate Section 6 tables from results_master.csv (new+old schema compatible).")
    p.add_argument("--csv", type=str, default="results/results_master.csv", help="Path to results_master.csv")
    p.add_argument("--outdir", type=str, default="results", help="Output directory")
    p.add_argument("--latex", action="store_true", help="Also emit LaTeX tables")
    p.add_argument("--extra_band_tables", action="store_true", help="Also emit band distribution tables (CSV)")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = (project_root / args.csv).resolve()
    out_dir = (project_root / args.outdir).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df = normalize_schema(df_raw)

    # Build tables
    t6a = build_table_6a_sa_summary(df)
    t6b = build_table_6b_gap_greedy_to_sa(df)
    t6c = build_table_6c_model_size(df)
    tg = build_table_gurobi_gap(df)

    # Save CSVs
    _write_csv(t6a, out_dir / "section6_table_6A_sa_summary_by_scenario_N.csv")
    _write_csv(t6b, out_dir / "section6_table_6B_greedy_vs_sa_gap.csv")
    _write_csv(t6c, out_dir / "section6_table_6C_model_size.csv")

    print("Saved:")
    print(" -", out_dir / "section6_table_6A_sa_summary_by_scenario_N.csv")
    print(" -", out_dir / "section6_table_6B_greedy_vs_sa_gap.csv")
    print(" -", out_dir / "section6_table_6C_model_size.csv")

    if len(tg) > 0:
        _write_csv(tg, out_dir / "section6_table_6D_sa_vs_gurobi_gap.csv")
        print(" -", out_dir / "section6_table_6D_sa_vs_gurobi_gap.csv")
    else:
        print(" - (no gurobi rows found; skipping Table 6D)")

    # Optional band distributions (use SA choices)
    if args.extra_band_tables:
        dist_dir = out_dir / "section6_distributions"
        dist_dir.mkdir(parents=True, exist_ok=True)

        prem = (
            df.pivot_table(
                index=["scenario_id", "scenario_name", "N"],
                columns="chosen_premium_band",
                values="seed",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        prem.columns = [str(c) for c in prem.columns]
        _write_csv(prem, dist_dir / "section6_premium_band_counts.csv")

        ded = (
            df.pivot_table(
                index=["scenario_id", "scenario_name", "N"],
                columns="chosen_deductible_band",
                values="seed",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        ded.columns = [str(c) for c in ded.columns]
        _write_csv(ded, dist_dir / "section6_deductible_band_counts.csv")

        print(" -", dist_dir / "section6_premium_band_counts.csv")
        print(" -", dist_dir / "section6_deductible_band_counts.csv")

    # Optional LaTeX
    if args.latex:
        latex_dir = out_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)

        latex_6a = _df_to_latex(
            t6a[[
                "scenario_name", "N", "runs",
                "energy_sa_mean", "energy_sa_std",
                "selected_features_mean",
                "premium_mode", "premium_mode_share",
                "deductible_mode", "deductible_mode_share",
            ]].copy(),
            caption="Section 6A: Simulated annealing (SA) summary by scenario and problem size $N$ (mean and std).",
            label="tab:sec6a_sa_summary",
            float_format="%.6f",
        )
        latex_6b = _df_to_latex(
            t6b[[
                "scenario_name", "N", "runs",
                "gap_mean", "gap_median",
                "sa_improve_rate", "equal_rate",
            ]].copy(),
            caption="Section 6B: Greedy vs SA gap statistics, where gap = greedy energy minus SA best energy.",
            label="tab:sec6b_greedy_sa_gap",
            float_format="%.6f",
        )
        latex_6c = _df_to_latex(
            t6c[[
                "scenario_name", "N", "runs",
                "L_min", "L_max", "n_total_min", "n_total_max",
            ]].copy(),
            caption="Section 6C: Model footprint by scenario and $N$. $L$ denotes regulatory slack bits.",
            label="tab:sec6c_model_size",
            float_format="%.0f",
        )

        _write_text(latex_6a, latex_dir / "section6_table_6A_sa_summary.tex")
        _write_text(latex_6b, latex_dir / "section6_table_6B_greedy_sa_gap.tex")
        _write_text(latex_6c, latex_dir / "section6_table_6C_model_size.tex")

        print("Saved LaTeX:")
        print(" -", latex_dir / "section6_table_6A_sa_summary.tex")
        print(" -", latex_dir / "section6_table_6B_greedy_sa_gap.tex")
        print(" -", latex_dir / "section6_table_6C_model_size.tex")

        if len(tg) > 0:
            latex_6d = _df_to_latex(
                tg[[
                    "scenario_name", "N", "runs_gurobi",
                    "energy_sa_mean", "energy_gurobi_mean",
                    "gap_sa_to_gurobi_mean", "gap_sa_to_gurobi_median",
                    "sa_equals_gurobi_rate", "gurobi_runtime_sec_mean",
                ]].copy(),
                caption="Section 6D: SA vs Gurobi exact baseline (where available). Gap is SA energy minus Gurobi energy.",
                label="tab:sec6d_sa_gurobi_gap",
                float_format="%.6f",
            )
            _write_text(latex_6d, latex_dir / "section6_table_6D_sa_gurobi_gap.tex")
            print(" -", latex_dir / "section6_table_6D_sa_gurobi_gap.tex")


if __name__ == "__main__":
    main()
