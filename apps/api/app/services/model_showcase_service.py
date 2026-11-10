from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from app.core.settings import settings
from app.services.training_service import TrainingService


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _dataset_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    class_dirs = [item for item in path.iterdir() if item.is_dir()]
    image_count = 0
    for class_dir in class_dirs:
        image_count += len(
            [
                item
                for item in class_dir.iterdir()
                if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
    return len(class_dirs), image_count


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: Any, digits: int = 2) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


class ModelShowcaseService:
    TASK_META = {
        "yolo": {
            "key": "yolo",
            "title": "YOLOv8 检测模型",
            "summary": "负责候选目标检测、头像裁剪和视频分析入口中的初始定位，是整条分析链路的起点。",
            "capabilities": ["检测动漫角色头像区域", "支持熟人库扩展时的裁剪预览", "在主分析链路中提供稳定的检测起点"],
            "config_path": settings.training_configs_dir / "detector" / "yolo_anime_face.yaml",
        },
        "recognition": {
            "key": "recognition",
            "title": "ResNet50 + ArcFace 特征提取",
            "summary": "负责熟人识别、角色匹配和已知角色库向量生成，是角色区分能力的核心模块。",
            "capabilities": ["提取动漫角色稳定特征向量", "支持熟人库识别与重建", "适合展示微调后的角色区分能力"],
            "config_path": settings.training_configs_dir / "recognition" / "arcface_resnet50.yaml",
        },
        "tracker": {
            "key": "tracker",
            "title": "ByteTrack 跟踪调参与评估",
            "summary": "负责跨帧 ID 稳定、片段拼接和时间轴连续性，是视频时序展示效果的关键模块。",
            "capabilities": ["减少角色 ID 抖动", "让时间轴片段更完整", "适合展示跟踪参数调优前后的差异"],
            "config_path": settings.training_configs_dir / "tracker" / "evaluate_bytetrack.yaml",
        },
    }

    def get_showcase(self, *, detection_model_path: str | None = None, recognition_model_path: str | None = None, tracker_config_path: str | None = None) -> dict:
        training_service = TrainingService()
        option_map = {item.task_type: item for item in training_service.get_options().tasks}
        jobs = training_service.list_jobs()
        jobs_by_task = {task_type: [job for job in jobs if job.task_type == task_type] for task_type in ["yolo", "recognition", "tracker", "library"]}
        selected_paths = {
            "yolo": self._selected_path(detection_model_path, settings.detection_model_path),
            "recognition": self._selected_path(recognition_model_path, settings.recognition_model_path),
            "tracker": self._selected_path(tracker_config_path, settings.tracker_yaml_path),
        }
        selected_jobs = {
            task_type: self._resolve_selected_job(task_type=task_type, selected_path=selected_paths[task_type], jobs=jobs_by_task.get(task_type, []), training_service=training_service, current_production_experiment_id=(option_map.get(task_type).current_production_experiment_id if option_map.get(task_type) else None))
            for task_type in ["yolo", "recognition", "tracker"]
        }
        cards = [
            self._build_task_card("yolo", selected_paths["yolo"], selected_jobs["yolo"], option_map.get("yolo").current_production_experiment_id if option_map.get("yolo") else None, training_service),
            self._build_task_card("recognition", selected_paths["recognition"], selected_jobs["recognition"], option_map.get("recognition").current_production_experiment_id if option_map.get("recognition") else None, training_service),
            self._build_task_card("tracker", selected_paths["tracker"], selected_jobs["tracker"], option_map.get("tracker").current_production_experiment_id if option_map.get("tracker") else None, training_service),
            self._build_library_card(selected_paths["recognition"], jobs_by_task.get("library", [])),
        ]
        groups = [
            self._build_group(task_type, option_map[task_type].title, option_map[task_type].current_production_experiment_id, jobs_by_task.get(task_type, []), selected_jobs[task_type].job_id if selected_jobs[task_type] else None, training_service)
            for task_type in ["yolo", "recognition", "tracker"] if task_type in option_map
        ]
        return {"cards": cards, "experiment_groups": groups}

    def _build_task_card(self, task_type: str, selected_path: Path, selected_job, current_production_experiment_id: str | None, training_service: TrainingService) -> dict[str, Any]:
        meta = self.TASK_META[task_type]
        detail = training_service.get_job(selected_job.job_id) if selected_job else None
        config_snapshot = detail.config_snapshot if detail else _load_yaml(meta["config_path"])
        metrics = detail.metrics if detail else {}
        return {
            "key": meta["key"],
            "title": meta["title"],
            "artifact_name": selected_path.name,
            "summary": meta["summary"],
            "selection_badge": self._selection_badge(task_type, selected_path),
            "selection_hint": self._selection_hint(task_type, selected_path, selected_job),
            "selected_experiment_id": selected_job.job_id if selected_job else None,
            "capabilities": meta["capabilities"],
            "parameters": self._build_parameters(task_type, config_snapshot),
            "datasets": self._build_datasets(task_type, selected_path, config_snapshot),
            "metrics": self._build_metrics(task_type, selected_job, metrics),
            "metric_sections": self._build_metric_sections(task_type, selected_job, metrics),
            "charts": self._build_charts(task_type, detail),
            "visual_groups": self._build_visual_groups(task_type, detail),
            "current_production_experiment_id": current_production_experiment_id,
        }

    def _build_library_card(self, recognition_model_path: Path, jobs: list) -> dict[str, Any]:
        class_count, image_count = _dataset_stats(settings.small_dataset_dir)
        latest_job = jobs[0] if jobs else None
        library_metrics = _load_json(settings.training_experiments_dir / "library" / "latest" / "metrics.json")
        return {
            "key": "library",
            "title": "熟人库样本集",
            "artifact_name": "small_dataset_112x112",
            "summary": "用于补齐熟人库角色封面和头像展示，也能说明前端熟人库样本的组织方式。",
            "selection_badge": "LIBRARY DATASET",
            "selection_hint": f"当前展示按照特征模型 {recognition_model_path.name} 对应的熟人库视角组织。",
            "selected_experiment_id": latest_job.job_id if latest_job else None,
            "capabilities": ["作为熟人库头像和封面的基础来源", "支持熟人库管理页的视觉展示", "与特征提取模型训练数据形成联动说明"],
            "parameters": [{"label": "默认最少样本", "value": "3"}, {"label": "默认清晰度阈值", "value": "100"}],
            "datasets": [{"label": "熟人库角色数", "value": str(class_count)}, {"label": "熟人库头像数", "value": str(image_count)}, {"label": "关联特征模型", "value": recognition_model_path.name}],
            "metrics": [{"label": "最近建库实验", "value": latest_job.job_id if latest_job else "暂无"}, {"label": "建库角色数", "value": str(library_metrics.get("character_count", class_count))}],
            "metric_sections": [],
            "charts": [],
            "visual_groups": [],
            "current_production_experiment_id": None,
        }

    def _build_group(self, task_type: str, title: str, current_production_experiment_id: str | None, jobs: list, selected_job_id: str | None, training_service: TrainingService) -> dict[str, Any]:
        ordered_jobs = sorted(
            jobs,
            key=lambda job: (0 if job.job_id == selected_job_id else 1, 0 if job.current_production else 1, job.job_id),
            reverse=True,
        )
        if selected_job_id:
            ordered_jobs.sort(key=lambda job: 0 if job.job_id == selected_job_id else 1)
        return {
            "task_type": task_type,
            "title": title,
            "current_production_experiment_id": current_production_experiment_id,
            "experiments": [self._build_experiment(job, selected_job_id, training_service) for job in ordered_jobs[:8]],
        }

    def _build_experiment(self, job, selected_job_id: str | None, training_service: TrainingService) -> dict[str, Any]:
        detail = training_service.get_job(job.job_id)
        return {
            "job_id": job.job_id,
            "task_type": job.task_type,
            "title": job.display_name or job.title,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "current_production": job.current_production,
            "selected": job.job_id == selected_job_id,
            "summary_metrics": job.summary_metrics,
            "artifacts": [{"name": artifact.name, "kind": artifact.kind, "url": artifact.url} for artifact in self._prioritize_artifacts(detail.artifacts)],
        }

    def _resolve_selected_job(self, *, task_type: str, selected_path: Path, jobs: list, training_service: TrainingService, current_production_experiment_id: str | None):
        default_path = self._default_path(task_type)
        if selected_path.resolve() == default_path.resolve() and current_production_experiment_id:
            for job in jobs:
                if job.job_id == current_production_experiment_id and self._job_matches_selected_path(job, selected_path, training_service):
                    return job
        for job in jobs:
            if self._job_matches_selected_path(job, selected_path, training_service):
                return job
        return None

    def _job_matches_selected_path(self, job, selected_path: Path, training_service: TrainingService) -> bool:
        if job.primary_resource_path and self._same_file(Path(job.primary_resource_path), selected_path):
            return True
        try:
            return self._same_file(training_service.get_promotable_artifact_path(job.job_id), selected_path)
        except HTTPException:
            return False

    def _build_parameters(self, task_type: str, config_snapshot: dict[str, Any]) -> list[dict[str, str]]:
        if task_type == "yolo":
            return [{"label": "训练轮次", "value": str(config_snapshot.get("epochs", "-"))}, {"label": "输入尺寸", "value": str(config_snapshot.get("imgsz", "-"))}, {"label": "Batch Size", "value": str(config_snapshot.get("batch", "-"))}, {"label": "Patience", "value": str(config_snapshot.get("patience", "-"))}]
        if task_type == "recognition":
            return [{"label": "训练轮次", "value": str(config_snapshot.get("epochs", "-"))}, {"label": "Embedding", "value": str(config_snapshot.get("embedding_size", "-"))}, {"label": "Batch Size", "value": str(config_snapshot.get("batch_size", "-"))}, {"label": "学习率", "value": str(config_snapshot.get("learning_rate", "-"))}]
        return [{"label": "评估视频数", "value": str(config_snapshot.get("max_videos", "-"))}, {"label": "置信度阈值", "value": str(config_snapshot.get("confidence", "-"))}, {"label": "IOU 阈值", "value": str(config_snapshot.get("iou", "-"))}, {"label": "检测模型", "value": str(Path(config_snapshot.get("detector_model", "-")).name)}]

    def _build_datasets(self, task_type: str, selected_path: Path, config_snapshot: dict[str, Any]) -> list[dict[str, str]]:
        if task_type == "yolo":
            dataset_yaml = Path(config_snapshot.get("dataset_yaml", "")) if config_snapshot.get("dataset_yaml") else None
            return [{"label": "当前权重", "value": selected_path.name}, {"label": "数据集配置", "value": dataset_yaml.name if dataset_yaml else "-"}, {"label": "实验域", "value": "detection"}]
        if task_type == "recognition":
            class_count, image_count = _dataset_stats(settings.arcface_dataset_dir)
            return [{"label": "当前权重", "value": selected_path.name}, {"label": "训练角色数", "value": str(class_count)}, {"label": "训练图像数", "value": str(image_count)}]
        videos_dir = Path(config_snapshot.get("videos_dir", "")) if config_snapshot.get("videos_dir") else None
        return [{"label": "当前配置", "value": selected_path.name}, {"label": "视频目录", "value": videos_dir.name if videos_dir else "-"}, {"label": "检测模型", "value": str(Path(config_snapshot.get("detector_model", "-")).name)}]

    def _build_metrics(self, task_type: str, selected_job, metrics: dict[str, Any]) -> list[dict[str, str]]:
        if task_type == "yolo":
            last_row = metrics.get("last_row", {})
            best_row = metrics.get("best_row", {})
            return [{"label": "关联实验", "value": selected_job.job_id if selected_job else "未关联"}, {"label": "Best mAP50", "value": _fmt_float(self._find_metric_value(best_row, ["mAP50", "map50"]))}, {"label": "Last Precision", "value": _fmt_float(self._find_metric_value(last_row, ["precision"]))}, {"label": "Last Recall", "value": _fmt_float(self._find_metric_value(last_row, ["recall"]))}]
        if task_type == "recognition":
            return [{"label": "关联实验", "value": selected_job.job_id if selected_job else "未关联"}, {"label": "最佳轮次", "value": str(metrics.get("best_epoch", "-"))}, {"label": "最佳验证准确率", "value": _fmt_float(metrics.get("best_val_accuracy", metrics.get("best_val_acc", "-")))}, {"label": "Prototype Top1", "value": _fmt_float(metrics.get("prototype_top1_best", "-"))}]
        return [{"label": "关联实验", "value": selected_job.job_id if selected_job else "未关联"}, {"label": "评估视频数", "value": str(len(metrics.get("videos_evaluated", [])))}, {"label": "配置组数", "value": str(len(metrics.get("tracker_results", [])))}]

    def _build_metric_sections(self, task_type: str, selected_job, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        if not selected_job:
            return []
        if task_type == "yolo":
            last_row = metrics.get("last_row", {})
            best_row = metrics.get("best_row", {})
            return [
                {"title": "最佳结果", "stats": [{"label": "最佳 Epoch", "value": str(best_row.get("epoch", "-"))}, {"label": "Best mAP50", "value": _fmt_float(self._find_metric_value(best_row, ["mAP50", "map50"]))}, {"label": "Best mAP50-95", "value": _fmt_float(self._find_metric_value(best_row, ["mAP50-95", "map50-95"]))}, {"label": "Best Precision", "value": _fmt_float(self._find_metric_value(best_row, ["precision"]))}]},
                {"title": "最终状态", "stats": [{"label": "Final Epoch", "value": str(last_row.get("epoch", "-"))}, {"label": "Last Recall", "value": _fmt_float(self._find_metric_value(last_row, ["recall"]))}, {"label": "Last Box Loss", "value": _fmt_float(last_row.get("val/box_loss"))}, {"label": "Last Cls Loss", "value": _fmt_float(last_row.get("val/cls_loss"))}]},
            ]
        if task_type == "recognition":
            last_row = metrics.get("last_row", {})
            same_sim = _to_float(last_row.get("Same Sim Mean"))
            diff_sim = _to_float(last_row.get("Diff Sim Mean"))
            return [
                {"title": "验证效果", "stats": [{"label": "Best Val Acc", "value": _fmt_float(metrics.get("best_val_accuracy", metrics.get("best_val_acc", "-")))}, {"label": "Prototype Top1", "value": _fmt_float(metrics.get("prototype_top1_best"))}, {"label": "Prototype Top5", "value": _fmt_float(metrics.get("prototype_top5_best"))}, {"label": "推荐阈值", "value": _fmt_float(metrics.get("recommended_threshold"), 3)}]},
                {"title": "判别分离度", "stats": [{"label": "Same Sim Mean", "value": _fmt_float(same_sim, 4)}, {"label": "Diff Sim Mean", "value": _fmt_float(diff_sim, 4)}, {"label": "Similarity Gap", "value": _fmt_float((same_sim - diff_sim) if same_sim is not None and diff_sim is not None else None, 4)}, {"label": "有效类别数", "value": str(metrics.get("valid_classes", "-"))}]},
            ]
        return [{"title": "当前评估", "stats": [{"label": "评估视频数", "value": str(len(metrics.get("videos_evaluated", [])))}, {"label": "配置组数", "value": str(len(metrics.get("tracker_results", [])))}]}]

    def _build_charts(self, task_type: str, detail) -> list[dict[str, Any]]:
        if not detail:
            return []
        if task_type == "yolo":
            rows = _load_csv_rows(self._artifact_local_path(detail, "results.csv"))
            if not rows:
                return []
            epochs = [str(row.get("epoch", index + 1)) for index, row in enumerate(rows)]
            return [
                self._make_chart("yolo-quality", "检测质量曲线", "展示 precision、recall、mAP50 和 mAP50-95 随训练轮次的提升情况。", epochs, [("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)")], rows),
                self._make_chart("yolo-loss", "训练与验证损失", "用 loss 曲线说明训练收敛过程是否稳定。", epochs, [("Train Box", "train/box_loss"), ("Val Box", "val/box_loss"), ("Train Cls", "train/cls_loss"), ("Val Cls", "val/cls_loss")], rows),
            ]
        rows = _load_csv_rows(self._artifact_local_path(detail, "training_log_r4.csv"))
        charts: list[dict[str, Any]] = []
        if rows:
            epochs = [str(row.get("Epoch", index + 1)) for index, row in enumerate(rows)]
            charts.extend([
                self._make_chart("recognition-accuracy", "验证精度曲线", "展示 ArcHead 验证精度与 Prototype Top1/Top5 随 epoch 的变化。", epochs, [("Val Acc", "Val Acc ArcHead(%)"), ("Prototype Top1", "Prototype Top1(%)"), ("Prototype Top5", "Prototype Top5(%)")], rows),
                self._make_chart("recognition-similarity", "相似度分离曲线", "Same/ Diff similarity 的间隔越大，通常说明角色区分越稳。", epochs, [("Same Sim", "Same Sim Mean"), ("Diff Sim", "Diff Sim Mean"), ("Best Threshold", "Best Threshold")], rows),
                self._make_chart("recognition-optimization", "优化过程曲线", "同时观察训练损失和带 margin 的训练精度，说明模型是否稳定收敛。", epochs, [("Train Loss", "Train Loss"), ("Train Acc Margin", "Train Acc Margin(%)"), ("Train Acc Plain", "Train Acc Plain(%)")], rows),
            ])
        wrong_rows = _load_csv_rows(self._artifact_local_path(detail, "wrong_predictions_r4.csv"))
        if wrong_rows:
            top_pairs = Counter(f"{row.get('true_class', '?')} -> {row.get('pred_class', '?')}" for row in wrong_rows if row.get("true_class") and row.get("pred_class")).most_common(8)
            charts.append({"key": "recognition-hard-pairs", "title": "高频混淆角色对", "chart_type": "bar", "summary": "展示最容易混淆的角色对，适合解释当前仍然困难的类别边界。", "x_labels": [pair for pair, _ in top_pairs], "series": [{"name": "Errors", "values": [float(count) for _, count in top_pairs]}], "y_axis_label": "Count"})
        return charts

    def _build_visual_groups(self, task_type: str, detail) -> list[dict[str, Any]]:
        if not detail:
            return []
        images = {artifact.name: artifact for artifact in detail.artifacts if artifact.kind == "image"}
        if task_type == "yolo":
            groups = [
                self._make_visual_group("核心评估图", "这些图最适合直接说明模型检测精度、PR 曲线和混淆情况。", detail, ["results.png", "BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix_normalized.png"], images),
                self._make_visual_group("验证集预测效果", "展示验证集上标签与预测的直接对比，更容易看出检测框质量。", detail, ["val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg", "labels.jpg"], images),
                self._make_visual_group("训练样本示例", "用于解释数据分布、标注质量和训练样本覆盖情况。", detail, ["train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg", "val_batch0_labels.jpg"], images),
            ]
            return [group for group in groups if group["items"]]
        generic_items = [{"title": artifact.name, "image_url": artifact.url, "caption": "实验导出图像"} for artifact in detail.artifacts if artifact.kind == "image"]
        return [{"title": "实验图像", "description": "当前模型关联实验输出的图像产物。", "items": generic_items[:8]}] if generic_items else []

    def _make_chart(self, key: str, title: str, summary: str, x_labels: list[str], series_defs: list[tuple[str, str]], rows: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "key": key,
            "title": title,
            "chart_type": "line",
            "summary": summary,
            "x_labels": x_labels,
            "series": [{"name": label, "values": [_to_float(row.get(column)) for row in rows]} for label, column in series_defs],
            "y_axis_label": None,
        }

    def _make_visual_group(self, title: str, description: str, detail, artifact_names: list[str], images: dict[str, Any]) -> dict[str, Any]:
        items = []
        for name in artifact_names:
            artifact = images.get(name)
            if artifact:
                items.append({"title": name, "image_url": artifact.url, "caption": f"{detail.job_id} - {name}"})
        return {"title": title, "description": description, "items": items}

    def _artifact_local_path(self, detail, artifact_name: str) -> Path:
        for artifact in detail.artifacts:
            if artifact.name == artifact_name:
                return Path(detail.experiment_dir) / artifact.relative_path
        return Path(detail.experiment_dir) / artifact_name

    def _find_metric_value(self, payload: dict[str, Any], keywords: list[str]) -> Any:
        lower_payload = {str(key).lower(): value for key, value in payload.items()}
        for keyword in keywords:
            for key, value in lower_payload.items():
                if keyword.lower() in key:
                    return value
        return "-"

    def _prioritize_artifacts(self, artifacts: list) -> list:
        priority_names = ["results.png", "BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg"]
        lookup = {artifact.name: artifact for artifact in artifacts}
        prioritized_images = [lookup[name] for name in priority_names if name in lookup]
        fallback_images = [artifact for artifact in artifacts if artifact.kind == "image" and artifact.name not in priority_names]
        files = [artifact for artifact in artifacts if artifact.kind != "image"][:8]
        return (prioritized_images + fallback_images[:4])[:8] + files

    def _selected_path(self, raw_path: str | None, default_path: Path) -> Path:
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate.resolve()
        return default_path.resolve()

    def _file_signature(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return str(path.resolve())
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _same_file(self, left: Path, right: Path) -> bool:
        if not left.exists() or not right.exists():
            return False
        if left.resolve() == right.resolve():
            return True
        if left.stat().st_size != right.stat().st_size:
            return False
        return self._file_signature(left) == self._file_signature(right)

    def _default_path(self, task_type: str) -> Path:
        if task_type == "yolo":
            return settings.detection_model_path
        if task_type == "recognition":
            return settings.recognition_model_path
        return settings.tracker_yaml_path

    def _selection_badge(self, task_type: str, selected_path: Path) -> str:
        return "CURRENT PRODUCTION" if selected_path.resolve() == self._default_path(task_type).resolve() else "CURRENT SELECTION"

    def _selection_hint(self, task_type: str, selected_path: Path, selected_job) -> str:
        if selected_path.resolve() == self._default_path(task_type).resolve():
            return "当前分析链路默认使用这一版模型。"
        if selected_job:
            return f"当前界面已切换到 {selected_path.name}，并联动展示实验 {selected_job.job_id} 的指标、曲线和图像成果。"
        return f"当前界面已切换到 {selected_path.name}，但暂未找到完全对应的实验记录。"
