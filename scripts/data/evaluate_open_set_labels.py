from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "outputs" / "experiments"

OUTPUT_FIELDNAMES = [
    "视频ID",
    "识别模型",
    "真实陌生人数量",
    "预测陌生人簇数",
    "错误合并簇数",
    "错误分裂身份数",
    "过分裂比例",
    "Purity",
    "B-Cubed Precision",
    "B-Cubed Recall",
    "B-Cubed F1",
    "有效样本权重",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate manually labeled open-set stranger clusters.")
    parser.add_argument(
        "label_files",
        nargs="*",
        type=Path,
        help="Filled label CSV files. If omitted, video_*_open_set_labels_filled.csv files are evaluated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR,
        help="Directory for metric CSV files.",
    )
    args = parser.parse_args()

    label_files = [_resolve_path(path) for path in args.label_files] if args.label_files else sorted(EXPERIMENT_DIR.glob("video_*_open_set_labels_filled.csv"))
    if not label_files:
        raise FileNotFoundError(f"No open-set label CSV files found in {EXPERIMENT_DIR}.")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label_path in label_files:
        metrics = evaluate_labels(label_path)
        output_path = output_dir / f"video_{metrics['video_id']}_open_set_metrics.csv"
        write_metrics(output_path, metrics)
        rows.append(metrics)
        print(f"Open-set metrics exported: {output_path}")

    if len(rows) > 1:
        comparison_path = output_dir / "open_set_model_comparison.csv"
        write_comparison(comparison_path, rows)
        print(f"Open-set comparison exported: {comparison_path}")


def evaluate_labels(label_path: Path) -> dict:
    rows = _read_rows(label_path)
    if not rows:
        raise ValueError(f"No rows found in {label_path}")

    _fill_gt_identity_from_notes(rows)
    missing = [row.get("cluster_id", "") for row in rows if not row.get("gt_identity", "").strip()]
    if missing:
        raise ValueError(
            f"{label_path} still has unlabeled clusters: {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )

    video_id = rows[0].get("video_id", _video_id_from_name(label_path))
    model = rows[0].get("model", "unknown_model")
    contingency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cluster_to_weight: dict[str, float] = defaultdict(float)
    identity_to_clusters: dict[str, set[str]] = defaultdict(set)
    mixed_clusters = set()

    for row in rows:
        cluster_id = row["cluster_id"].strip()
        identities = _parse_identities(row["gt_identity"])
        if not identities:
            continue
        weight = _row_weight(row)
        share = weight / len(identities)
        for identity in identities:
            contingency[cluster_id][identity] += share
            identity_to_clusters[identity].add(cluster_id)
        cluster_to_weight[cluster_id] += weight
        if _truthy(row.get("is_mixed")) or len(identities) > 1:
            mixed_clusters.add(cluster_id)

    total_weight = sum(cluster_to_weight.values())
    if total_weight <= 0:
        raise ValueError(f"No positive sample weight found in {label_path}")

    identity_weights: dict[str, float] = defaultdict(float)
    for identity_map in contingency.values():
        for identity, value in identity_map.items():
            identity_weights[identity] += value

    purity = sum(max(identity_map.values()) for identity_map in contingency.values()) / total_weight
    bcubed_precision = (
        sum(
            (value * value) / cluster_to_weight[cluster_id]
            for cluster_id, identity_map in contingency.items()
            for value in identity_map.values()
            if cluster_to_weight[cluster_id] > 0
        )
        / total_weight
    )
    bcubed_recall = (
        sum(
            (value * value) / identity_weights[identity]
            for identity_map in contingency.values()
            for identity, value in identity_map.items()
            if identity_weights[identity] > 0
        )
        / total_weight
    )
    bcubed_f1 = _f1(bcubed_precision, bcubed_recall)

    split_identities = {
        identity: clusters
        for identity, clusters in identity_to_clusters.items()
        if len(clusters) > 1
    }
    predicted_clusters = len(contingency)
    real_identities = len(identity_weights)
    over_split_rate = (
        (predicted_clusters - real_identities) / predicted_clusters
        if predicted_clusters > 0 and predicted_clusters > real_identities
        else 0.0
    )

    return {
        "video_id": str(video_id),
        "model": model,
        "real_stranger_count": real_identities,
        "predicted_cluster_count": predicted_clusters,
        "wrong_merged_clusters": len(mixed_clusters),
        "wrong_split_identities": len(split_identities),
        "over_split_rate": over_split_rate,
        "purity": purity,
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": bcubed_f1,
        "total_weight": total_weight,
    }


def write_metrics(output_path: Path, metrics: dict) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["项目", "数值"])
        writer.writeheader()
        writer.writerows(
            [
                {"项目": "识别模型", "数值": metrics["model"]},
                {"项目": "真实陌生人数量", "数值": metrics["real_stranger_count"]},
                {"项目": "预测陌生人簇数", "数值": metrics["predicted_cluster_count"]},
                {"项目": "错误合并簇数", "数值": metrics["wrong_merged_clusters"]},
                {"项目": "错误分裂身份数", "数值": metrics["wrong_split_identities"]},
                {"项目": "过分裂比例", "数值": _fmt_percent(metrics["over_split_rate"])},
                {"项目": "Purity", "数值": _fmt_float(metrics["purity"])},
                {"项目": "B-Cubed Precision", "数值": _fmt_float(metrics["bcubed_precision"])},
                {"项目": "B-Cubed Recall", "数值": _fmt_float(metrics["bcubed_recall"])},
                {"项目": "B-Cubed F1", "数值": _fmt_float(metrics["bcubed_f1"])},
                {"项目": "有效样本权重", "数值": _fmt_float(metrics["total_weight"])},
            ]
        )


def write_comparison(output_path: Path, metrics_rows: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for metrics in sorted(metrics_rows, key=lambda item: int(item["video_id"]) if item["video_id"].isdigit() else item["video_id"]):
            writer.writerow(
                {
                    "视频ID": metrics["video_id"],
                    "识别模型": metrics["model"],
                    "真实陌生人数量": metrics["real_stranger_count"],
                    "预测陌生人簇数": metrics["predicted_cluster_count"],
                    "错误合并簇数": metrics["wrong_merged_clusters"],
                    "错误分裂身份数": metrics["wrong_split_identities"],
                    "过分裂比例": _fmt_percent(metrics["over_split_rate"]),
                    "Purity": _fmt_float(metrics["purity"]),
                    "B-Cubed Precision": _fmt_float(metrics["bcubed_precision"]),
                    "B-Cubed Recall": _fmt_float(metrics["bcubed_recall"]),
                    "B-Cubed F1": _fmt_float(metrics["bcubed_f1"]),
                    "有效样本权重": _fmt_float(metrics["total_weight"]),
                }
            )


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fill_gt_identity_from_notes(rows: list[dict]) -> None:
    fallback_count = 0
    for row in rows:
        if row.get("gt_identity", "").strip():
            continue
        note_value = row.get("notes", "").strip()
        if not note_value:
            continue
        row["gt_identity"] = note_value
        fallback_count += 1
    if fallback_count:
        print(f"Warning: used notes as gt_identity for {fallback_count} rows. You can move these values into gt_identity later.")


def _parse_identities(raw_value: str) -> list[str]:
    identities = [item.strip() for item in re.split(r"[;,|+，；、/]+", raw_value or "") if item.strip()]
    return identities


def _row_weight(row: dict) -> float:
    for field in ("prototype_count", "appearance_segments"):
        try:
            value = float(row.get(field) or 0)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return 1.0


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是", "混合"}


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _video_id_from_name(path: Path) -> str:
    match = re.search(r"video_(.+?)_open_set_labels", path.name)
    return match.group(1) if match else path.stem


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
