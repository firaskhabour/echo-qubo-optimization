import pandas as pd
import numpy as np

path = "results/results_master.csv"  # adjust
df = pd.read_csv(path)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# 1) Check missing values by column
na = df.isna().mean().sort_values(ascending=False)
print("\nTop NA columns:")
print(na.head(20))

# 2) Check gap consistency
if {"energy_greedy","energy_sa_best","gap_greedy_to_sa"}.issubset(df.columns):
    gap_calc = df["energy_greedy"] - df["energy_sa_best"]
    diff = (gap_calc - df["gap_greedy_to_sa"]).abs()
    print("\nGap consistency:")
    print("Max |calc - logged|:", diff.max())
    print("Rows > 1e-6:", (diff > 1e-6).sum())

# 3) Check Gurobi dominance (if ran and optimal-ish)
if {"gurobi_ran","energy_gurobi","energy_sa_best"}.issubset(df.columns):
    sub = df[df["gurobi_ran"] == 1].copy()
    # Allow tiny tolerance due to float representation
    bad = sub[sub["energy_gurobi"] > sub["energy_sa_best"] + 1e-6]
    print("\nGurobi dominance check (should usually be <= SA):")
    print("Rows where gurobi energy > sa energy:", len(bad))
    if len(bad):
        print(bad[["scenario_id","N","seed","energy_sa_best","energy_gurobi"]].head(10))

# 4) Sanity: improvement rate rough
if {"energy_greedy","energy_sa_best"}.issubset(df.columns):
    improve = (df["energy_sa_best"] < df["energy_greedy"] - 1e-9).mean()
    equal = (np.isclose(df["energy_sa_best"], df["energy_greedy"], atol=1e-9)).mean()
    worse = (df["energy_sa_best"] > df["energy_greedy"] + 1e-9).mean()
    print("\nSA vs Greedy (overall):")
    print("Improve:", improve, "Equal:", equal, "Worse:", worse)

# 5) Detect CSV column shift symptoms: numeric columns parsed as object
obj_cols = [c for c in df.columns if df[c].dtype == "object"]
print("\nObject dtype columns (possible parsing/shift issue):")
print(obj_cols[:50])

# How often did SA actually change features vs greedy?
feature_diff = (df["selected_features_count_sa"] 
                != df["selected_features_count_greedy"]).mean()

print("Share of runs where SA changed feature count:", feature_diff)
