from __future__ import annotations

from pathlib import Path
import sys

from src.common.experiment import ExperimentRun
from src.common.paths import paths


def validate_library_config(config: dict) -> None:
    if not Path(config["dataset_dir"]).exists():
        raise FileNotFoundError(f"Character dataset not found: {config['dataset_dir']}")
    if not Path(config["recognition_model"]).exists():
        raise FileNotFoundError(f"Recognition model not found: {config['recognition_model']}")


def build_known_database_from_config(config: dict, run: ExperimentRun) -> dict:
    validate_library_config(config)
    _ensure_api_root_on_path()
    from video_demo import ResNet50Extractor, build_known_database

    output_path = run.artifacts_dir / config.get("output_name", "known_database.json")
    extractor = ResNet50Extractor(model_path=str(config["recognition_model"]))
    known_db = build_known_database(extractor, str(config["dataset_dir"]), str(output_path))

    return {
        "experiment_id": run.experiment_id,
        "dataset_dir": config["dataset_dir"],
        "recognition_model": config["recognition_model"],
        "known_database_path": output_path,
        "character_count": len(known_db),
        "sample_counts": {name: info.get("count", 0) for name, info in known_db.items()},
        "filtered_classes": _collect_filtered_classes(Path(config["dataset_dir"]), min_images=3),
    }


def _collect_filtered_classes(dataset_dir: Path, min_images: int) -> list[dict]:
    filtered = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        image_count = len([item for item in folder.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if image_count < min_images:
            filtered.append({"class_name": folder.name, "image_count": image_count, "reason": "below_min_images"})
    return filtered


def _ensure_api_root_on_path() -> None:
    api_root = str(paths.api_root)
    if api_root not in sys.path:
        sys.path.insert(0, api_root)
