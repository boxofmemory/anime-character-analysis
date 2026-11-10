from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "outputs" / "experiments"

FIELDNAMES = [
    "video_id",
    "model",
    "cluster_id",
    "prototype_count",
    "appearance_segments",
    "gt_identity",
    "is_mixed",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manual labeling templates for open-set cluster evaluation.")
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Experiment report JSON files, for example data/outputs/experiments/video_22_experiment_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR,
        help="Directory for generated label template CSV files.",
    )
    args = parser.parse_args()

    reports = [_resolve_path(path) for path in args.reports] if args.reports else sorted(EXPERIMENT_DIR.glob("video_*_experiment_report.json"))
    if not reports:
        raise FileNotFoundError(f"No experiment report JSON files found in {EXPERIMENT_DIR}.")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_path in reports:
        output_path = export_template(report_path, output_dir)
        print(f"Label template exported: {output_path}")


def export_template(report_path: Path, output_dir: Path) -> Path:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    video_id = payload.get("video_id") or _video_id_from_name(report_path)
    model = (
        payload.get("video_info", {})
        .get("model_selection", {})
        .get("recognition_model", "unknown_model")
    )
    clusters = payload.get("open_set_case", {}).get("stranger_clusters", {})

    output_path = output_dir / f"video_{video_id}_open_set_labels_template.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for cluster_id, info in sorted(clusters.items(), key=_cluster_sort_key):
            writer.writerow(
                {
                    "video_id": video_id,
                    "model": model,
                    "cluster_id": cluster_id,
                    "prototype_count": int(info.get("prototype_count") or 0),
                    "appearance_segments": int(info.get("appearance_segments") or 0),
                    "gt_identity": "",
                    "is_mixed": "false",
                    "notes": "",
                }
            )
    return output_path


def _cluster_sort_key(item: tuple[str, dict]) -> tuple[int, str]:
    cluster_id, _ = item
    suffix = cluster_id.rsplit("_", 1)[-1]
    return (int(suffix), cluster_id) if suffix.isdigit() else (10**9, cluster_id)


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
