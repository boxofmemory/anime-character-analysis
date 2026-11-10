from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.common.experiment import ExperimentRun


def validate_tracker_config(config: dict) -> None:
    if not Path(config["detector_model"]).exists():
        raise FileNotFoundError(f"Detector model not found: {config['detector_model']}")
    video_files = _collect_video_files(config)
    if not video_files:
        raise FileNotFoundError(f"No tracker evaluation videos found in {config['videos_dir']}")
    for tracker_config in config["tracker_configs"]:
        if not Path(tracker_config).exists():
            raise FileNotFoundError(f"Tracker config not found: {tracker_config}")


def evaluate_tracker_from_config(config: dict, run: ExperimentRun) -> dict:
    validate_tracker_config(config)
    video_files = _collect_video_files(config)[: int(config.get("max_videos", 9999))]
    model = YOLO(str(config["detector_model"]))

    tracker_results = []
    for tracker_yaml in config["tracker_configs"]:
        tracker_yaml = Path(tracker_yaml)
        aggregate = {
            "tracker_yaml": tracker_yaml,
            "videos": [],
            "total_frames": 0,
            "frames_with_tracks": 0,
            "total_boxes": 0,
            "unique_track_ids": 0,
            "mean_track_length_frames": 0.0,
        }
        total_lengths: list[int] = []

        for video_path in video_files:
            stats = _evaluate_single_video(
                model=model,
                video_path=video_path,
                tracker_yaml=tracker_yaml,
                conf=float(config.get("confidence", 0.15)),
                iou=float(config.get("iou", 0.6)),
            )
            aggregate["videos"].append(stats)
            aggregate["total_frames"] += stats["total_frames"]
            aggregate["frames_with_tracks"] += stats["frames_with_tracks"]
            aggregate["total_boxes"] += stats["total_boxes"]
            total_lengths.extend(stats["track_lengths"])

        unique_ids = {
            track_id
            for video_stats in aggregate["videos"]
            for track_id in video_stats["unique_track_ids"]
        }
        aggregate["unique_track_ids"] = len(unique_ids)
        aggregate["mean_track_length_frames"] = round(sum(total_lengths) / max(len(total_lengths), 1), 2)
        aggregate["avg_tracks_per_active_frame"] = round(
            aggregate["total_boxes"] / max(aggregate["frames_with_tracks"], 1), 4
        )
        copied_yaml = run.copy_artifact(tracker_yaml, tracker_yaml.name)
        aggregate["artifacts"] = {"tracker_yaml_copy": copied_yaml}
        tracker_results.append(aggregate)

    comparison_chart = _write_tracker_comparison_chart(tracker_results, run.artifacts_dir / "tracker_comparison.png")

    return {
        "experiment_id": run.experiment_id,
        "detector_model": config["detector_model"],
        "videos_evaluated": [Path(video) for video in video_files],
        "tracker_results": tracker_results,
        "artifacts": {
            "comparison_chart": comparison_chart if comparison_chart.exists() else None,
        },
    }


def _collect_video_files(config: dict) -> list[Path]:
    videos_dir = Path(config["videos_dir"])
    return sorted(videos_dir.glob(config.get("video_glob", "*.mp4")))


def _evaluate_single_video(model: YOLO, video_path: Path, tracker_yaml: Path, conf: float, iou: float) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    track_lengths = defaultdict(int)
    unique_track_ids: set[int] = set()
    total_frames = 0
    frames_with_tracks = 0
    total_boxes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        results = model.track(frame, persist=True, verbose=False, conf=conf, iou=iou, tracker=str(tracker_yaml))
        current_ids = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    current_ids.append(track_id)
                    unique_track_ids.add(track_id)
                    track_lengths[track_id] += 1
        if current_ids:
            frames_with_tracks += 1
            total_boxes += len(current_ids)

    cap.release()
    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "frames_with_tracks": frames_with_tracks,
        "total_boxes": total_boxes,
        "unique_track_ids": sorted(unique_track_ids),
        "track_lengths": list(track_lengths.values()),
        "avg_tracks_per_active_frame": round(total_boxes / max(frames_with_tracks, 1), 4),
    }


def _write_tracker_comparison_chart(tracker_results: list[dict], output_path: Path) -> Path:
    if not tracker_results:
        return output_path

    labels = [Path(item["tracker_yaml"]).stem for item in tracker_results]
    values = [item["avg_tracks_per_active_frame"] for item in tracker_results]

    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 2.2), 4.6))
    bars = axis.bar(labels, values, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"][: len(labels)])
    axis.set_title("ByteTrack Parameter Comparison")
    axis.set_ylabel("Avg tracks per active frame")
    axis.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
