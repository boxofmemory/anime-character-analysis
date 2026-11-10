from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "outputs" / "experiments"

FIELDNAMES = [
    "视频ID",
    "识别模型",
    "视频",
    "平滑前片段数",
    "间隙合并后片段数",
    "短片段过滤数",
    "最终片段数",
    "碎片降低率",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a smoothing ablation comparison table.")
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Experiment report JSON files. If omitted, all video_*_experiment_report.json files are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "smoothing_ablation_comparison.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    reports = [_resolve_path(path) for path in args.reports] if args.reports else sorted(EXPERIMENT_DIR.glob("video_*_experiment_report.json"))
    if not reports:
        raise FileNotFoundError(f"No experiment report JSON files found in {EXPERIMENT_DIR}.")

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for report_path in reports:
            writer.writerow(_build_row(report_path))
    print(f"Smoothing ablation comparison exported: {output_path}")


def _build_row(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    smoothing = payload.get("smoothing_ablation", {})
    video_info = payload.get("video_info", {})
    model = video_info.get("model_selection", {}).get("recognition_model", "unknown_model")
    before_count = int(smoothing.get("before_smoothing_segments") or 0)
    final_count = int(smoothing.get("after_filter_segments") or 0)
    reduction_rate = smoothing.get("fragment_reduction_rate")
    if reduction_rate is None:
        reduction_rate = round((before_count - final_count) / before_count, 4) if before_count else 0.0

    return {
        "视频ID": payload.get("video_id") or _video_id_from_name(report_path),
        "识别模型": model,
        "视频": smoothing.get("video") or video_info.get("filename") or "",
        "平滑前片段数": before_count,
        "间隙合并后片段数": int(smoothing.get("after_smoothing_segments") or 0),
        "短片段过滤数": int(smoothing.get("short_filtered_segments") or 0),
        "最终片段数": final_count,
        "碎片降低率": f"{float(reduction_rate) * 100:.2f}%",
    }


def _video_id_from_name(path: Path) -> str:
    name = path.name
    prefix = "video_"
    suffix = "_experiment_report.json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
