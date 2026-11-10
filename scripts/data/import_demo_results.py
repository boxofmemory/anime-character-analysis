from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "research" / "training" / "experiments"
CATALOG_RECOGNITION_DIR = PROJECT_ROOT / "models" / "catalog" / "recognition"
CATALOG_DETECTION_DIR = PROJECT_ROOT / "models" / "catalog" / "detection"


def _iso_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(payload: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _recognition_experiment_id(weight_path: Path) -> str:
    stamp = datetime.fromtimestamp(weight_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_recognition_arcface_r4_demo"


def _yolo_experiment_id(results_csv: Path) -> str:
    stamp = datetime.fromtimestamp(results_csv.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_yolo_anime_final_v8s_8gb"


def import_recognition(args) -> str:
    weight_path = Path(args.recognition_weights)
    class_index_path = Path(args.recognition_class_index)
    log_path = Path(args.recognition_log)
    wrong_predictions_path = Path(args.recognition_wrong_predictions)

    experiment_id = args.recognition_experiment_id or _recognition_experiment_id(weight_path)
    experiment_dir = EXPERIMENTS_DIR / "recognition" / experiment_id
    artifacts_dir = experiment_dir / "artifacts"
    catalog_weight_path = CATALOG_RECOGNITION_DIR / f"{experiment_id}.pth"

    rows = _load_rows(log_path)
    if not rows:
        raise RuntimeError(f"Recognition log is empty: {log_path}")
    best_row = max(rows, key=lambda row: _to_float(row, "Prototype Top1(%)"))
    last_row = rows[-1]
    class_index = json.loads(class_index_path.read_text(encoding="utf-8"))
    class_count = int(class_index.get("num_classes") or len(class_index.get("class_names", [])))

    _copy_file(weight_path, artifacts_dir / "best_arcface.pth")
    _copy_file(weight_path, catalog_weight_path)
    _copy_file(class_index_path, artifacts_dir / class_index_path.name)
    _copy_file(log_path, artifacts_dir / log_path.name)
    if wrong_predictions_path.exists():
        _copy_file(wrong_predictions_path, artifacts_dir / wrong_predictions_path.name)

    metrics = {
        "experiment_id": experiment_id,
        "model_name": "resnet50_arcface_r4_demo",
        "source_type": "imported_demo",
        "best_epoch": int(_to_float(best_row, "Epoch", 0)),
        "best_val_accuracy": round(_to_float(best_row, "Val Acc ArcHead(%)"), 4),
        "best_val_acc": round(_to_float(best_row, "Val Acc ArcHead(%)"), 4),
        "prototype_top1_best": round(_to_float(best_row, "Prototype Top1(%)"), 4),
        "prototype_top5_best": round(_to_float(best_row, "Prototype Top5(%)"), 4),
        "recommended_threshold": round(_to_float(best_row, "Best Threshold"), 4),
        "valid_classes": class_count,
        "last_row": last_row,
        "artifacts": {
            "best_weights": str((artifacts_dir / "best_arcface.pth").resolve()),
            "class_index": str((artifacts_dir / class_index_path.name).resolve()),
            "training_log": str((artifacts_dir / log_path.name).resolve()),
        },
    }
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(
        experiment_dir / "job.json",
        {
            "job_id": experiment_id,
            "task_type": "recognition",
            "domain": "recognition",
            "title": "ResNet50 + ArcFace Training",
            "display_name": "ArcFace r4 Demo Import",
            "status": "completed",
            "created_at": _iso_timestamp(weight_path),
            "updated_at": _iso_timestamp(log_path),
            "dataset_path": "",
            "primary_resource_path": str(catalog_weight_path.resolve()),
            "secondary_resource_path": "",
        },
    )
    _write_text(
        experiment_dir / "summary.md",
        "\n".join(
            [
                "# ArcFace r4 Imported Demo",
                "",
                f"- Experiment ID: `{experiment_id}`",
                f"- Imported weights: `{weight_path}`",
                f"- Catalog weights: `{catalog_weight_path}`",
                f"- Best prototype top1: `{metrics['prototype_top1_best']}`",
                f"- Best validation accuracy: `{metrics['best_val_accuracy']}`",
                f"- Recommended threshold: `{metrics['recommended_threshold']}`",
                f"- Valid classes: `{class_count}`",
            ]
        ),
    )
    return experiment_id


def import_yolo(args) -> str:
    run_dir = Path(args.yolo_run_dir)
    results_csv = run_dir / "results.csv"
    best_weight = run_dir / "weights" / "best.pt"
    last_weight = run_dir / "weights" / "last.pt"

    experiment_id = args.yolo_experiment_id or _yolo_experiment_id(results_csv)
    experiment_dir = EXPERIMENTS_DIR / "detection" / experiment_id
    artifacts_dir = experiment_dir / "artifacts"
    catalog_weight_path = CATALOG_DETECTION_DIR / f"{experiment_id}.pt"

    rows = _load_rows(results_csv)
    if not rows:
        raise RuntimeError(f"YOLO results are empty: {results_csv}")
    last_row = rows[-1]
    best_row = max(rows, key=lambda row: _to_float(row, "metrics/mAP50(B)"))

    _copy_file(best_weight, artifacts_dir / "best_yolo.pt")
    _copy_file(last_weight, artifacts_dir / "last_yolo.pt")
    _copy_file(best_weight, catalog_weight_path)
    for path in run_dir.iterdir():
        if path.is_file():
            _copy_file(path, artifacts_dir / path.name)

    metrics = {
        "experiment_id": experiment_id,
        "model_name": "yolo_anime_final_v8s_8gb",
        "source_type": "imported_demo",
        "last_row": last_row,
        "best_row": best_row,
        "artifacts": {
            "best_weights": str((artifacts_dir / "best_yolo.pt").resolve()),
            "last_weights": str((artifacts_dir / "last_yolo.pt").resolve()),
            "results_csv": str((artifacts_dir / "results.csv").resolve()),
        },
    }
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(
        experiment_dir / "job.json",
        {
            "job_id": experiment_id,
            "task_type": "yolo",
            "domain": "detection",
            "title": "YOLOv8 Training",
            "display_name": "YOLO anime_final_v8s_8GB Import",
            "status": "completed",
            "created_at": _iso_timestamp(best_weight),
            "updated_at": _iso_timestamp(results_csv),
            "dataset_path": "",
            "primary_resource_path": str(catalog_weight_path.resolve()),
            "secondary_resource_path": "",
        },
    )
    _write_text(
        experiment_dir / "summary.md",
        "\n".join(
            [
                "# YOLO Imported Demo",
                "",
                f"- Experiment ID: `{experiment_id}`",
                f"- Imported run directory: `{run_dir}`",
                f"- Catalog weights: `{catalog_weight_path}`",
                f"- Last mAP50: `{last_row.get('metrics/mAP50(B)', '-')}`",
                f"- Best mAP50: `{best_row.get('metrics/mAP50(B)', '-')}`",
            ]
        ),
    )
    return experiment_id


def parse_args():
    parser = argparse.ArgumentParser(description="Import demo training artifacts into anime-character-analysis.")
    parser.add_argument("--recognition-weights", default=r"./models\resnet50_anime_arcface_best_r4.pth")
    parser.add_argument("--recognition-class-index", default=r"./models\anime_arcface_class_names_r4.json")
    parser.add_argument("--recognition-log", default=r"./models\logs\training_log_r4.csv")
    parser.add_argument("--recognition-wrong-predictions", default=r"./models\logs\wrong_predictions_r4.csv")
    parser.add_argument("--recognition-experiment-id", default="")
    parser.add_argument("--yolo-run-dir", default=r"./test_02\runs\detect\anime_final_v8s_8GB")
    parser.add_argument("--yolo-experiment-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognition_id = import_recognition(args)
    yolo_id = import_yolo(args)
    print(f"Imported recognition experiment: {recognition_id}")
    print(f"Imported YOLO experiment: {yolo_id}")


if __name__ == "__main__":
    main()
