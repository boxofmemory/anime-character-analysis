from __future__ import annotations

import csv
import json
from pathlib import Path


def export_experiment_report(video_id: int, timeline_data: dict, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = timeline_data.get("experiment_stats") or {}
    smoothing = stats.get("smoothing_ablation") or {}
    open_set = stats.get("open_set_case") or {}

    base_name = f"video_{video_id}"
    report_path = output_dir / f"{base_name}_experiment_report.json"
    smoothing_csv_path = output_dir / f"{base_name}_smoothing_ablation.csv"
    open_set_csv_path = output_dir / f"{base_name}_open_set_case.csv"

    report_payload = {
        "video_id": video_id,
        "video_info": timeline_data.get("video_info", {}),
        "smoothing_ablation": smoothing,
        "open_set_case": open_set,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_smoothing_csv(smoothing_csv_path, video_id, timeline_data, smoothing)
    _write_open_set_case_csv(open_set_csv_path, open_set)

    return {
        "report_json": str(report_path),
        "smoothing_ablation_csv": str(smoothing_csv_path),
        "open_set_case_csv": str(open_set_csv_path),
    }


def _write_smoothing_csv(output_path: Path, video_id: int, timeline_data: dict, smoothing: dict) -> None:
    before_count = int(smoothing.get("before_smoothing_segments") or 0)
    merged_count = int(smoothing.get("after_smoothing_segments") or 0)
    final_count = int(smoothing.get("after_filter_segments") or 0)
    filtered_count = int(smoothing.get("short_filtered_segments") or 0)
    reduction_rate = smoothing.get("fragment_reduction_rate")
    if reduction_rate is None:
        reduction_rate = round((before_count - final_count) / before_count, 4) if before_count else 0.0

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "视频",
                "平滑前片段数",
                "间隙合并后片段数",
                "短片段过滤数",
                "最终片段数",
                "碎片降低率",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "视频": smoothing.get("video") or timeline_data.get("video_info", {}).get("filename") or f"video_{video_id}",
                "平滑前片段数": before_count,
                "间隙合并后片段数": merged_count,
                "短片段过滤数": filtered_count,
                "最终片段数": final_count,
                "碎片降低率": f"{float(reduction_rate) * 100:.2f}%",
            }
        )


def _write_open_set_case_csv(output_path: Path, open_set: dict) -> None:
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
