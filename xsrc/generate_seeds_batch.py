import sys
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    plan_path = project_root / "config" / "experiment_plan.yaml"
    plan = load_yaml(plan_path)

    feature_sizes = [int(x) for x in plan["feature_sizes"]]
    N_max = max(feature_sizes)

    core_count = int(plan["seeds"]["core_count"])
    stress_count = int(plan["seeds"]["stress_count"])
    core_seed_start = int(plan["seeds"]["core_seed_start"])
    stress_seed_start = int(plan["seeds"]["stress_seed_start"])

    core_seeds = [core_seed_start + i for i in range(core_count)]
    stress_seeds = [stress_seed_start + i for i in range(stress_count)]
    all_seeds = core_seeds + stress_seeds

    py = sys.executable
    gen_script = project_root / "src" / "generate_data.py"

    print(f"Generating seed data for {len(all_seeds)} seeds at N={N_max} ...")
    for seed in all_seeds:
        cmd = [py, str(gen_script), "--seed", str(seed), "--N", str(N_max), "--overwrite"]
        run_cmd(cmd, project_root)
        print(f"  OK seed={seed}")

    print("\nDONE. All seed folders generated.")
    print(f"Base: {project_root / 'data' / 'seeds'}")


if __name__ == "__main__":
    main()
