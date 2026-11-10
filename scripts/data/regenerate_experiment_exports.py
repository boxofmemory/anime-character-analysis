from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "outputs" / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate experiment CSV files from existing report JSON files.")
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Experiment report JSON files. If omitted, all video_*_experiment_report.json files are used.",
    )
    args = parser.parse_args()

    reports = [_resolve_path(path) for path in args.reports] if args.reports else sorted(EXPERIMENT_DIR.glob("video_*_experiment_report.json"))
    if not reports:
        raise FileNotFoundError(f"No experiment report JSON files found in {EXPERIMENT_DIR}.")

    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        video_id = payload.get("video_id") or _video_id_from_name(report_path)
        output_dir = report_path.parent
        _write_smoothing_csv(output_dir / f"video_{video_id}_smoothing_ablation.csv", payload)
        _write_open_set_case_csv(output_dir / f"video_{video_id}_open_set_case.csv", payload)
        print(f"Regenerated experiment CSV files for video_{video_id}")


def _write_smoothing_csv(output_path: Path, payload: dict) -> None:
    smoothing = payload.get("smoothing_ablation", {})
    video_info = payload.get("video_info", {})
    before_count = int(smoothing.get("before_smoothing_segments") or 0)
    final_count = int(smoothing.get("after_filter_segments") or 0)
    reduction_rate = smoothing.get("fragment_reduction_rate")
    if reduction_rate is None:
        reduction_rate = round((before_count - final_count) / before_count, 4) if before_count else 0.0

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["视频", "平滑前片段数", "间隙合并后片段数", "短片段过滤数", "最终片段数", "碎片降低率"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "视频": smoothing.get("video") or video_info.get("filename") or "",
                "平滑前片段数": before_count,
                "间隙合并后片段数": int(smoothing.get("after_smoothing_segments") or 0),
                "短片段过滤数": int(smoothing.get("short_filtered_segments") or 0),
                "最终片段数": final_count,
                "碎片降低率": f"{float(reduction_rate) * 100:.2f}%",
            }
        )


def _write_open_set_case_csv(output_path: Path, payload: dict) -> None:
    open_set = payload.get("open_set_case", {})
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["项目", "数值"])
        writer.writeheader()
        writer.writerows(
            [
                {"项目": "已知角色数", "数值": open_set.get("known_character_count", 0)},
                {"项目": "预测陌生人簇数", "数值": open_set.get("stranger_cluster_count", 0)},
                {"项目": "评价状态", "数值": "需填写开放集人工标注模板后计算 Purity 与 B-Cubed F1"},
            ]
        )


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
