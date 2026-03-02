import argparse
from pathlib import Path

import numpy as np
import yaml


def get_scenario_from_args_or_prompt() -> int:
    parser = argparse.ArgumentParser(description="Check QUBO properties (symmetry, density).")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], help="Scenario number (1..4)")
    args, _ = parser.parse_known_args()

    if args.scenario is not None:
        return args.scenario

    while True:
        s = input("Select scenario to check QUBO (1=S1, 2=S2, 3=S3, 4=S4): ").strip()
        if s in {"1", "2", "3", "4"}:
            return int(s)
        print("Invalid input. Please enter 1, 2, 3, or 4.")


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    scen = get_scenario_from_args_or_prompt()
    cfg_path = PROJECT_ROOT / "config" / f"config_qubo_S{scen}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        qcfg = yaml.safe_load(f)

    SEED = int(qcfg.get("seed", 1000))
    DATA_DIR = PROJECT_ROOT / "data" / "seeds" / f"seed_{SEED}"

    Q_path = DATA_DIR / "qubo_Q.npz"
    if not Q_path.exists():
        raise FileNotFoundError(f"Missing QUBO: {Q_path}")

    Q = np.load(Q_path)["Q"]

    sym_err = float(np.max(np.abs(Q - Q.T)))
    diag = np.diag(Q)
    nz_density = float(np.count_nonzero(Q) / Q.size)

    print(f"Scenario: {qcfg['scenario']['name']} | seed={SEED}")
    print(f"Q shape: {Q.shape}")
    print(f"Max symmetry error: {sym_err}")
    print(f"Diagonal min/max: {float(diag.min())} {float(diag.max())}")
    print(f"Q nonzero density: {nz_density}")


if __name__ == "__main__":
    main()
