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
    parser = argparse.ArgumentParser(description="Formal ArcFace training entrypoint")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "recognition" / "arcface_resnet50.yaml"),
        help="Path to ArcFace training config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without starting training",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.validate_only:
        dataset_dir = Path(config["dataset_dir"])
        if not dataset_dir.exists():
            raise FileNotFoundError(f"ArcFace dataset not found: {dataset_dir}")
        print("ArcFace config validation passed")
        print(f"Config: {Path(args.config).resolve()}")
        print(f"Dataset: {config['dataset_dir']}")
        print(f"Legacy baseline: {paths.legacy / 'train_arcface_v3.py'}")
        return

    from src.recognition import train_arcface_from_config

    run = create_experiment_run(
        "recognition",
        str(config.get("experiment_name", "arcface_resnet50_formal")),
        experiment_id_override=str(config.get("experiment_id_override") or "") or None,
    )
    run.write_config_snapshot(config)
    metrics = train_arcface_from_config(config, run)
    run.write_metrics(metrics)
    run.write_summary(_build_summary(metrics, run))
    print(f"ArcFace training completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


def _build_summary(metrics: dict, run) -> str:
    return f"""# ArcFace Experiment Summary

- Experiment ID: `{metrics['experiment_id']}`
- Model: `resnet50_arcface`
- Dataset: `{metrics['dataset_dir']}`
- Valid classes: `{metrics['valid_classes']}`
- Train samples: `{metrics['train_samples']}`
- Val samples: `{metrics['val_samples']}`
- Best epoch: `{metrics['best_epoch']}`
- Best val acc: `{metrics['best_val_acc']:.4f}`
- Best weights: `{metrics['artifacts']['best_weights']}`
- Last weights: `{metrics['artifacts']['last_weights']}`
- Config snapshot: `{run.config_snapshot_path}`
"""


if __name__ == "__main__":
    main()
