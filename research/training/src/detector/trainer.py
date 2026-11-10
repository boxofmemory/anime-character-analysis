from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO

from src.common.experiment import ExperimentRun


def validate_yolo_config(config: dict) -> None:
    if not Path(config["dataset_yaml"]).exists():
        raise FileNotFoundError(f"YOLO dataset yaml not found: {config['dataset_yaml']}")
    if not Path(config["base_model"]).exists():
        raise FileNotFoundError(f"YOLO base model not found: {config['base_model']}")


def train_yolo_from_config(config: dict, run: ExperimentRun) -> dict:
    validate_yolo_config(config)

    model = YOLO(str(config["base_model"]))
    train_kwargs = {
        "data": str(config["dataset_yaml"]),
        "imgsz": int(config.get("imgsz", 640)),
        "epochs": int(config.get("epochs", 120)),
        "batch": int(config.get("batch", 16)),
        "device": config.get("device", 0),
        "patience": int(config.get("patience", 20)),
        "project": str(run.root.parent),
        "name": run.root.name,
        "exist_ok": True,
    }
    if config.get("workers") is not None:
        train_kwargs["workers"] = int(config["workers"])

    model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", run.root))
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if best_path.exists():
        run.copy_artifact(best_path, "best_yolo.pt")
    if last_path.exists():
        run.copy_artifact(last_path, "last_yolo.pt")
    for artifact_name in (
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "args.yaml",
    ):
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            run.copy_artifact(artifact_path, artifact_name)

    metrics = {
        "experiment_id": run.experiment_id,
        "model_name": "yolov8",
        "dataset_yaml": config["dataset_yaml"],
        "base_model": config["base_model"],
        "device": str(config.get("device", 0)),
        "artifacts": {
            "best_weights": best_path if best_path.exists() else None,
            "last_weights": last_path if last_path.exists() else None,
            "results_csv": run_dir / "results.csv",
            "results_image": run.artifacts_dir / "results.png" if (run.artifacts_dir / "results.png").exists() else None,
            "confusion_matrix": run.artifacts_dir / "confusion_matrix.png" if (run.artifacts_dir / "confusion_matrix.png").exists() else None,
            "confusion_matrix_normalized": run.artifacts_dir / "confusion_matrix_normalized.png"
            if (run.artifacts_dir / "confusion_matrix_normalized.png").exists()
            else None,
        },
    }
    metrics.update(_read_results_csv(run_dir / "results.csv"))
    return metrics


def _read_results_csv(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {"history": [], "last_row": {}}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    parsed_rows = [{key: _to_number(value) for key, value in row.items()} for row in rows]
    return {
        "history": parsed_rows,
        "last_row": parsed_rows[-1] if parsed_rows else {},
    }


def _to_number(value: str):
    try:
        if value is None or value == "":
            return value
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except (ValueError, AttributeError):
        return value
