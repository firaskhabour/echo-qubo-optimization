from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "results" / "results_master.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run run_experiments.py first.")

    df = pd.read_csv(results_path)

    # Basic sanity checks
    print("\n=== SANITY CHECKS ===")
    print("Rows:", len(df))
    print("Scenarios:", sorted(df["scenario_id"].unique().tolist()))
    print("Ns:", sorted(df["N"].unique().tolist()))
    print("Any missing selected_reg_features_count_sa:", df["selected_reg_features_count_sa"].isna().any())

    # Feasibility rates
    print("\n=== FEASIBILITY RATES (from checks) ===")
    if "onehot_ok" in df.columns:
        onehot_rate = np.mean(df["onehot_ok"] == True) if df["onehot_ok"].dtype != object else np.mean(df["onehot_ok"].astype(str).str.lower() == "true")
        print(f"onehot_ok rate: {onehot_rate:.3f}")
    if "reg_satisfied" in df.columns:
        reg_rate = np.mean(df["reg_satisfied"] == True) if df["reg_satisfied"].dtype != object else np.mean(df["reg_satisfied"].astype(str).str.lower() == "true")
        print(f"reg_satisfied rate: {reg_rate:.3f}")

    # H1a: Does gap_greedy_to_sa increase with N within each scenario?
    print("\n=== H1a: gap_greedy_to_sa vs N (Spearman) ===")
    for scen in sorted(df["scenario_id"].unique()):
        sub = df[df["scenario_id"] == scen].copy()
        sub = sub.dropna(subset=["gap_greedy_to_sa"])
        corr, pval = spearmanr(sub["N"], sub["gap_greedy_to_sa"])
        print(f"S{scen}: spearman r={corr:.3f}, p={pval:.4g}, n={len(sub)}")

    # H1b: Is S4 harder than S1? Compare gaps
    print("\n=== H1b: S4 vs S1 gap_greedy_to_sa (t-test) ===")
    s1 = df[df["scenario_id"] == 1]["gap_greedy_to_sa"].dropna().values
    s4 = df[df["scenario_id"] == 4]["gap_greedy_to_sa"].dropna().values
    if len(s1) > 2 and len(s4) > 2:
        t_stat, pval = ttest_ind(s4, s1, equal_var=False)
        print(f"S4 mean={np.mean(s4):.3f}, S1 mean={np.mean(s1):.3f} | t={t_stat:.3f}, p={pval:.4g}")
    else:
        print("Not enough data for t-test.")

    # H1c: SA-to-Gurobi gap vs N (only where gurobi_ran==1)
    print("\n=== H1c: gap_sa_to_gurobi vs N (Spearman, gurobi_ran==1) ===")
    gsub = df[df["gurobi_ran"] == 1].dropna(subset=["gap_sa_to_gurobi"]).copy()
    if len(gsub) > 3:
        corr, pval = spearmanr(gsub["N"], gsub["gap_sa_to_gurobi"])
        print(f"spearman r={corr:.3f}, p={pval:.4g}, n={len(gsub)}")
    else:
        print("Not enough Gurobi rows.")

    # Plots
    out_dir = project_root / "results"
    out_dir.mkdir(exist_ok=True)

    # Plot: mean gap by N and scenario
    print("\n=== SAVING PLOTS ===")
    agg = df.groupby(["scenario_id", "N"])["gap_greedy_to_sa"].mean().reset_index()
    plt.figure()
    for scen in sorted(agg["scenario_id"].unique()):
        sub = agg[agg["scenario_id"] == scen]
        plt.plot(sub["N"], sub["gap_greedy_to_sa"], marker="o", label=f"S{scen}")
    plt.xlabel("N")
    plt.ylabel("Mean (energy_greedy - energy_sa_best)")
    plt.title("Mean Greedy-to-SA Improvement vs N")
    plt.legend()
    p1 = out_dir / "plot_gap_greedy_to_sa_vs_N.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", p1.name)

    # Plot: SA-to-Gurobi gap vs N (if available)
    if len(gsub) > 0:
        agg2 = gsub.groupby("N")["gap_sa_to_gurobi"].mean().reset_index()
        plt.figure()
        plt.plot(agg2["N"], agg2["gap_sa_to_gurobi"], marker="o")
        plt.xlabel("N")
        plt.ylabel("Mean (energy_sa_best - energy_gurobi)")
        plt.title("Mean SA-to-Gurobi Gap vs N (N<=50)")
        p2 = out_dir / "plot_gap_sa_to_gurobi_vs_N.png"
        plt.savefig(p2, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", p2.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
