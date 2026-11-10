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
    parser = argparse.ArgumentParser(description="Run a formal inference demo with explicit experiment outputs")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "tracker" / "inference_demo.yaml"),
        help="Path to inference demo config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without running inference",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    _validate_inference_config(config)
    if args.validate_only:
        print("Inference demo config validation passed")
        print(f"Config: {Path(args.config).resolve()}")
        return

    run = create_experiment_run("inference", str(config.get("experiment_name", "inference_demo_formal")))
    run.write_config_snapshot(config)
    metrics = _run_inference(config, run)
    run.write_metrics(metrics)
    run.write_summary(_build_summary(metrics, run))
    print(f"Inference demo completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


def _run_inference(config: dict, run) -> dict:
    from ultralytics import YOLO

    _ensure_api_root_on_path()
    import video_demo

    video_demo.TRACKER_YAML = str(config["tracker_yaml"])
    extractor = video_demo.ResNet50Extractor(model_path=str(config["recognition_model"]))
    recognizer = video_demo.FaceRecognizer(
        known_db_path=str(config["known_db_path"]),
        stranger_db_path=str(run.artifacts_dir / "stranger_database.json"),
    )
    yolo_model = YOLO(str(config["detector_model"]))
    output_video = run.artifacts_dir / config.get("output_video_name", "annotated_demo.webm")
    output_timeline = run.artifacts_dir / config.get("timeline_name", "timeline_demo.json")
    result = video_demo.process_video(
        video_path=str(config["input_video"]),
        extractor=extractor,
        recognizer=recognizer,
        yolo_model=yolo_model,
        output_path=str(output_video),
        timeline_output_path=str(output_timeline),
    )
    return {
        "experiment_id": run.experiment_id,
        "input_video": config["input_video"],
        "detector_model": config["detector_model"],
        "recognition_model": config["recognition_model"],
        "tracker_yaml": config["tracker_yaml"],
        "known_db_path": config["known_db_path"],
        "output_video": output_video,
        "timeline_output": output_timeline,
        "result": result,
    }


def _validate_inference_config(config: dict) -> None:
    for key in ("input_video", "detector_model", "recognition_model", "tracker_yaml", "known_db_path"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Inference config path not found for {key}: {config[key]}")


def _build_summary(metrics: dict, run) -> str:
    raw_stats = (metrics.get("result") or {}).get("raw_stats", {})
    return f"""# Inference Demo Summary

- Experiment ID: `{metrics['experiment_id']}`
- Input video: `{metrics['input_video']}`
- Detector model: `{metrics['detector_model']}`
- Recognition model: `{metrics['recognition_model']}`
- Tracker yaml: `{metrics['tracker_yaml']}`
- Known DB: `{metrics['known_db_path']}`
- Output video: `{metrics['output_video']}`
- Timeline JSON: `{metrics['timeline_output']}`
- Raw stats: `{raw_stats}`
- Config snapshot: `{run.config_snapshot_path}`
"""


def _ensure_api_root_on_path() -> None:
    api_root = str(paths.api_root)
    if api_root not in sys.path:
        sys.path.insert(0, api_root)


if __name__ == "__main__":
    main()
