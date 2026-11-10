from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import load_yaml
from src.common.experiment import create_experiment_run
from src.common.paths import paths


LEGACY_SCRIPT = ROOT / "legacy" / "train_arcface_r4_improved.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("train_arcface_r4_improved", LEGACY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load legacy script: {LEGACY_SCRIPT}")
    spec.loader.exec_module(module)
    return module


def _parse_training_log(log_path: Path) -> dict:
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Training log is empty: {log_path}")
    best_row = max(rows, key=lambda row: float(row.get("Prototype Top1(%)", 0.0) or 0.0))
    return {
        "best_epoch": int(float(best_row.get("Epoch", 0) or 0)),
        "best_val_accuracy": round(float(best_row.get("Val Acc ArcHead(%)", 0.0) or 0.0), 4),
        "best_val_acc": round(float(best_row.get("Val Acc ArcHead(%)", 0.0) or 0.0), 4),
        "prototype_top1_best": round(float(best_row.get("Prototype Top1(%)", 0.0) or 0.0), 4),
        "prototype_top5_best": round(float(best_row.get("Prototype Top5(%)", 0.0) or 0.0), 4),
        "recommended_threshold": round(float(best_row.get("Best Threshold", 0.0) or 0.0), 4),
        "last_row": rows[-1],
        "epochs_completed": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal wrapper for the improved ArcFace r4 demo trainer")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "recognition" / "arcface_resnet50_r4_demo.yaml"),
        help="Path to the r4 demo ArcFace config",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_dir = Path(config["dataset_dir"]).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"ArcFace dataset not found: {dataset_dir}")

    run = create_experiment_run(
        "recognition",
        str(config.get("experiment_name", "arcface_r4_demo")),
        experiment_id_override=str(config.get("experiment_id_override") or "") or None,
    )
    run.write_config_snapshot(config)

    module = _load_legacy_module()
    module.DATASET_DIR = str(dataset_dir)
    module.MODEL_SAVE_DIR = str(run.artifacts_dir)
    module.LOG_SAVE_DIR = str(run.artifacts_dir)
    module.MODEL_SAVE_PATH = str(run.artifacts_dir / "best_arcface.pth")
    module.CSV_LOG_PATH = str(run.artifacts_dir / "training_log.csv")
    module.CLASS_INDEX_PATH = str(run.artifacts_dir / "class_index.json")
    module.WRONG_PRED_CSV_PATH = str(run.artifacts_dir / "wrong_predictions.csv")

    overrides = {
        "INPUT_SIZE": config.get("input_size"),
        "BATCH_SIZE": config.get("batch_size"),
        "NUM_EPOCHS": config.get("epochs"),
        "EMBEDDING_SIZE": config.get("embedding_size"),
        "TRAIN_RATIO": config.get("train_ratio"),
        "MIN_IMAGES_PER_CLASS": config.get("min_images_per_class"),
        "RANDOM_SEED": config.get("seed"),
        "NUM_WORKERS": config.get("num_workers"),
        "SCALE": config.get("scale"),
        "FINAL_MARGIN": config.get("margin"),
        "WEIGHT_DECAY": config.get("weight_decay"),
        "EARLY_STOP_PATIENCE": config.get("early_stop_patience"),
    }
    for attr, value in overrides.items():
        if value is not None:
            setattr(module, attr, value)

    module.main()

    parsed = _parse_training_log(run.artifacts_dir / "training_log.csv")
    class_index = json.loads((run.artifacts_dir / "class_index.json").read_text(encoding="utf-8"))
    metrics = {
        "experiment_id": run.experiment_id,
        "model_name": "resnet50_arcface_r4_demo",
        "dataset_dir": str(dataset_dir),
        "valid_classes": int(class_index.get("num_classes") or len(class_index.get("class_names", []))),
        "artifacts": {
            "best_weights": run.artifacts_dir / "best_arcface.pth",
            "class_index": run.artifacts_dir / "class_index.json",
            "training_log": run.artifacts_dir / "training_log.csv",
            "wrong_predictions": run.artifacts_dir / "wrong_predictions.csv",
        },
        **parsed,
    }
    run.write_metrics(metrics)
    run.write_summary(
        "\n".join(
            [
                "# ArcFace r4 Demo Wrapper Summary",
                "",
                f"- Experiment ID: `{run.experiment_id}`",
                f"- Dataset: `{dataset_dir}`",
                f"- Best prototype top1: `{metrics['prototype_top1_best']}`",
                f"- Best validation accuracy: `{metrics['best_val_accuracy']}`",
                f"- Recommended threshold: `{metrics['recommended_threshold']}`",
                f"- Best weights: `{run.artifacts_dir / 'best_arcface.pth'}`",
            ]
        )
    )
    print(f"ArcFace r4 demo training completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


if __name__ == "__main__":
    main()
