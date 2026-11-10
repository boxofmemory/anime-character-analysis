from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import load_yaml
from src.common.experiment import create_experiment_run
from src.common.paths import paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal YOLO training entrypoint")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "detector" / "yolo_anime_face.yaml"),
        help="Path to YOLO training config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without starting training",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.validate_only:
        if not Path(config["dataset_yaml"]).exists():
            raise FileNotFoundError(f"YOLO dataset yaml not found: {config['dataset_yaml']}")
        if not Path(config["base_model"]).exists():
            raise FileNotFoundError(f"YOLO base model not found: {config['base_model']}")
        print("YOLO config validation passed")
        print(f"Config: {Path(args.config).resolve()}")
        print(f"Dataset yaml: {config['dataset_yaml']}")
        print(f"Base weights: {config['base_model']}")
        return

    from src.detector import train_yolo_from_config

    run = create_experiment_run(
        "detection",
        str(config.get("experiment_name", "yolo_anime_face_formal")),
        experiment_id_override=str(config.get("experiment_id_override") or "") or None,
    )
    run.write_config_snapshot(config)
    metrics = train_yolo_from_config(config, run)
    run.write_metrics(metrics)
    run.write_summary(_build_summary(metrics, run))
    print(f"YOLO training completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


def _build_summary(metrics: dict, run) -> str:
    last_row = metrics.get("last_row", {})
    return f"""# YOLO Experiment Summary

- Experiment ID: `{metrics['experiment_id']}`
- Dataset yaml: `{metrics['dataset_yaml']}`
- Base model: `{metrics['base_model']}`
- Last metrics row: `{last_row}`
- Best weights: `{metrics['artifacts']['best_weights']}`
- Last weights: `{metrics['artifacts']['last_weights']}`
- Config snapshot: `{run.config_snapshot_path}`
"""


if __name__ == "__main__":
    main()
