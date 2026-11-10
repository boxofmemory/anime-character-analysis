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
    parser = argparse.ArgumentParser(description="Evaluate ByteTrack parameters on sample videos")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "tracker" / "evaluate_bytetrack.yaml"),
        help="Path to tracker evaluation config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without running evaluation",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.validate_only:
        if not Path(config["detector_model"]).exists():
            raise FileNotFoundError(f"Detector model not found: {config['detector_model']}")
        if not list(Path(config["videos_dir"]).glob(config.get("video_glob", "*.mp4"))):
            raise FileNotFoundError(f"No tracker evaluation videos found in {config['videos_dir']}")
        for tracker_config in config["tracker_configs"]:
            if not Path(tracker_config).exists():
                raise FileNotFoundError(f"Tracker config not found: {tracker_config}")
        print("Tracker evaluation config validation passed")
        print(f"Config: {Path(args.config).resolve()}")
        return

    from src.tracker import evaluate_tracker_from_config

    run = create_experiment_run(
        "tracker",
        str(config.get("experiment_name", "bytetrack_eval_formal")),
        experiment_id_override=str(config.get("experiment_id_override") or "") or None,
    )
    run.write_config_snapshot(config)
    metrics = evaluate_tracker_from_config(config, run)
    run.write_metrics(metrics)
    run.write_summary(_build_summary(metrics, run))
    print(f"Tracker evaluation completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


def _build_summary(metrics: dict, run) -> str:
    lines = [
        "# ByteTrack Evaluation Summary",
        "",
        f"- Experiment ID: `{metrics['experiment_id']}`",
        f"- Detector model: `{metrics['detector_model']}`",
        f"- Videos evaluated: `{len(metrics['videos_evaluated'])}`",
        f"- Config snapshot: `{run.config_snapshot_path}`",
        "",
    ]
    for tracker_result in metrics["tracker_results"]:
        lines.extend(
            [
                f"## {Path(tracker_result['tracker_yaml']).name}",
                f"- Total frames: `{tracker_result['total_frames']}`",
                f"- Frames with tracks: `{tracker_result['frames_with_tracks']}`",
                f"- Total boxes: `{tracker_result['total_boxes']}`",
                f"- Unique track ids: `{tracker_result['unique_track_ids']}`",
                f"- Mean track length (frames): `{tracker_result['mean_track_length_frames']}`",
                f"- Avg tracks per active frame: `{tracker_result['avg_tracks_per_active_frame']}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
