from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any

import yaml
from fastapi import HTTPException

from app.core.settings import settings
from app.schemas.training import (
    TrainingActionResponse,
    TrainingArtifactItem,
    TrainingJobCreateRequest,
    TrainingJobDetail,
    TrainingJobSummary,
    TrainingLogResponse,
    TrainingMetricItem,
    TrainingOptionItem,
    TrainingOptionsResponse,
    TrainingPathPreset,
    TrainingTaskOption,
)


_PROCESS_REGISTRY: dict[str, subprocess.Popen] = {}
_PROCESS_LOCK = threading.Lock()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return slug or "training-job"


def _utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_serialize(payload), handle, allow_unicode=True, sort_keys=False)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize(payload), handle, ensure_ascii=False, indent=2)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_serialize(child) for child in value]
    return value


def _format_time(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _ensure_relative(child: Path, parent: Path) -> Path:
    try:
        return child.resolve().relative_to(parent.resolve())
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Artifact path is outside of the training job directory") from exc


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".pt", ".pth"}:
        return "weights"
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix in {".csv", ".json", ".md", ".log", ".txt"}:
        return "text"
    return "file"


@dataclass(frozen=True)
class TrainingTaskDefinition:
    task_type: str
    domain: str
    title: str
    description: str
    script_name: str
    config_path: Path
    dataset_field: str
    primary_resource_field: str | None
    secondary_resource_field: str | None
    production_target: Path
    recommended_fields: list[str]


class TrainingService:
    TASKS: dict[str, TrainingTaskDefinition] = {
        "yolo": TrainingTaskDefinition(
            task_type="yolo",
            domain="detection",
            title="YOLOv8 训练",
            description="用于目标检测微调，训练完成后可手动晋升为当前生产检测模型。",
            script_name="train_yolo.py",
            config_path=settings.training_configs_dir / "detector" / "yolo_anime_face.yaml",
            dataset_field="dataset_yaml",
            primary_resource_field="base_model",
            secondary_resource_field=None,
            production_target=settings.detection_model_path,
            recommended_fields=["dataset_yaml", "base_model", "epochs", "batch", "imgsz", "device", "patience"],
        ),
        "recognition": TrainingTaskDefinition(
            task_type="recognition",
            domain="recognition",
            title="ResNet50 + ArcFace 训练",
            description="用于角色特征提取模型微调，训练完成后可手动晋升为当前生产识别模型。",
            script_name="train_arcface.py",
            config_path=settings.training_configs_dir / "recognition" / "arcface_resnet50.yaml",
            dataset_field="dataset_dir",
            primary_resource_field="pretrained_backbone",
            secondary_resource_field=None,
            production_target=settings.recognition_model_path,
            recommended_fields=[
                "dataset_dir",
                "pretrained_backbone",
                "epochs",
                "batch_size",
                "learning_rate",
                "embedding_size",
                "device",
                "input_size",
            ],
        ),
        "tracker": TrainingTaskDefinition(
            task_type="tracker",
            domain="tracker",
            title="ByteTrack 调参与评估",
            description="用于跟踪 YAML 参数对比与评估，不直接训练权重，但会生成评估结果并可手动晋升为当前生产跟踪配置。",
            script_name="evaluate_tracker.py",
            config_path=settings.training_configs_dir / "tracker" / "evaluate_bytetrack.yaml",
            dataset_field="videos_dir",
            primary_resource_field="tracker_configs",
            secondary_resource_field="detector_model",
            production_target=settings.tracker_yaml_path,
            recommended_fields=["videos_dir", "confidence", "iou", "max_videos", "detector_model"],
        ),
    }

    def get_options(self) -> TrainingOptionsResponse:
        tasks = []
        manifest = self._load_production_manifest()
        for definition in self.TASKS.values():
            config = _load_yaml(definition.config_path)
            tasks.append(
                TrainingTaskOption(
                    task_type=definition.task_type,
                    title=definition.title,
                    description=definition.description,
                    domain=definition.domain,
                    config_path=str(definition.config_path),
                    default_config=_serialize(config),
                    dataset_field=definition.dataset_field,
                    primary_resource_field=definition.primary_resource_field,
                    secondary_resource_field=definition.secondary_resource_field,
                    dataset_presets=self._build_dataset_presets(definition, config),
                    primary_resource_options=self._build_primary_options(definition, config),
                    secondary_resource_options=self._build_secondary_options(definition, config),
                    recommended_fields=definition.recommended_fields,
                    current_production=definition.production_target.name,
                    current_production_experiment_id=manifest.get(definition.task_type, {}).get("experiment_id"),
                )
            )
        return TrainingOptionsResponse(tasks=tasks)

    def create_job(self, request: TrainingJobCreateRequest) -> TrainingJobDetail:
        definition = self._require_definition(request.task_type)
        config = _load_yaml(definition.config_path)
        job_id = self._generate_job_id(request.task_type, request.display_name or definition.title)
        job_dir = settings.training_experiments_dir / definition.domain / job_id
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = self._build_runtime_config(definition, config, request, job_id, artifacts_dir)
        runtime_config_path = job_dir / "job.runtime.yaml"
        stdout_log_path = job_dir / "stdout.log"
        _dump_yaml(runtime_config_path, runtime_config)

        metadata = {
            "job_id": job_id,
            "task_type": definition.task_type,
            "domain": definition.domain,
            "title": definition.title,
            "display_name": request.display_name or definition.title,
            "status": "pending",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "dataset_path": str(runtime_config.get(definition.dataset_field, request.dataset_path)),
            "primary_resource_path": self._metadata_primary_resource(definition, runtime_config, request),
            "secondary_resource_path": self._metadata_secondary_resource(definition, runtime_config, request),
            "runtime_config_path": str(runtime_config_path),
            "stdout_log_path": str(stdout_log_path),
        }
        self._write_job_metadata(job_dir, metadata)

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("MPLBACKEND", "Agg")
        command = [
            str(settings.training_python_path),
            str(settings.training_scripts_dir / definition.script_name),
            "--config",
            str(runtime_config_path),
        ]
        log_handle = stdout_log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=settings.training_root_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log_handle.close()

        metadata["status"] = "running"
        metadata["pid"] = process.pid
        metadata["updated_at"] = _utc_now()
        self._write_job_metadata(job_dir, metadata)

        with _PROCESS_LOCK:
            _PROCESS_REGISTRY[job_id] = process

        monitor = threading.Thread(target=self._monitor_process, args=(job_id,), daemon=True)
        monitor.start()
        return self.get_job(job_id)

    def list_jobs(self) -> list[TrainingJobSummary]:
        jobs: list[TrainingJobSummary] = []
        for definition in self.TASKS.values():
            domain_dir = settings.training_experiments_dir / definition.domain
            if not domain_dir.exists():
                continue
            for experiment_dir in sorted(
                [item for item in domain_dir.iterdir() if item.is_dir()],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                self._refresh_job_status(experiment_dir, experiment_dir.name)
                jobs.append(self._build_job_summary(experiment_dir, definition.task_type))
        return jobs

    def get_job(self, job_id: str) -> TrainingJobDetail:
        experiment_dir, task_type = self._find_job(job_id)
        self._refresh_job_status(experiment_dir, job_id)
        metadata = self._load_metadata(experiment_dir, task_type)
        metrics = _load_json(experiment_dir / "metrics.json")
        config_snapshot_path = experiment_dir / "config.snapshot.yaml"
        if not config_snapshot_path.exists() and metadata.get("runtime_config_path"):
            config_snapshot_path = Path(metadata["runtime_config_path"])
        config_snapshot = _load_yaml(config_snapshot_path)
        summary_markdown = None
        summary_path = experiment_dir / "summary.md"
        if summary_path.exists():
            summary_markdown = summary_path.read_text(encoding="utf-8")
        log_path = Path(metadata.get("stdout_log_path", experiment_dir / "stdout.log"))
        artifacts = self._collect_artifacts(experiment_dir, job_id)
        summary = self._build_job_summary(experiment_dir, task_type)
        return TrainingJobDetail(
            **summary.model_dump(),
            config_snapshot=_serialize(config_snapshot),
            metrics=_serialize(metrics),
            artifacts=artifacts,
            summary_markdown=summary_markdown,
            log_size_bytes=log_path.stat().st_size if log_path.exists() else None,
        )

    def get_log(self, job_id: str, max_bytes: int = 60_000) -> TrainingLogResponse:
        detail = self.get_job(job_id)
        experiment_dir, _ = self._find_job(job_id)
        log_path = experiment_dir / "stdout.log"
        if not log_path.exists():
            return TrainingLogResponse(job_id=job_id, status=detail.status, text="")
        with log_path.open("rb") as handle:
            if log_path.stat().st_size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            text = handle.read().decode("utf-8", errors="ignore")
        return TrainingLogResponse(job_id=job_id, status=detail.status, text=text)

    def cancel_job(self, job_id: str) -> TrainingActionResponse:
        experiment_dir, task_type = self._find_job(job_id)
        metadata = self._load_metadata(experiment_dir, task_type)
        if metadata.get("status") != "running":
            raise HTTPException(status_code=400, detail="Only running jobs can be cancelled.")
        process = self._get_process(job_id)
        if process is None:
            raise HTTPException(status_code=404, detail="Training process not found.")
        process.terminate()
        metadata["status"] = "cancelled"
        metadata["updated_at"] = _utc_now()
        self._write_job_metadata(experiment_dir, metadata)
        with _PROCESS_LOCK:
            _PROCESS_REGISTRY.pop(job_id, None)
        return TrainingActionResponse(message="Training job cancelled.", job=self.get_job(job_id))

    def promote_job(self, job_id: str) -> TrainingActionResponse:
        experiment_dir, task_type = self._find_job(job_id)
        detail = self.get_job(job_id)
        if detail.status != "completed":
            raise HTTPException(status_code=400, detail="Only completed experiments can be promoted.")

        definition = self._require_definition(task_type)
        source_artifact = self._resolve_promotion_source(experiment_dir, task_type)
        if not source_artifact.exists():
            raise HTTPException(status_code=404, detail="Promotion artifact not found.")

        definition.production_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_artifact, definition.production_target)

        manifest = self._load_production_manifest()
        manifest[task_type] = {
            "experiment_id": job_id,
            "promoted_at": _utc_now(),
            "source_artifact": str(source_artifact),
            "production_path": str(definition.production_target),
        }
        _dump_json(settings.training_production_manifest_path, manifest)

        return TrainingActionResponse(
            message=f"{definition.title} has been promoted to the current production model.",
            job=self.get_job(job_id),
        )

    def get_artifact_path(self, job_id: str, relative_path: str) -> Path:
        experiment_dir, _ = self._find_job(job_id)
        artifact_path = (experiment_dir / relative_path).resolve()
        _ensure_relative(artifact_path, experiment_dir)
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return artifact_path

    def get_promotable_artifact_path(self, job_id: str) -> Path:
        experiment_dir, task_type = self._find_job(job_id)
        return self._resolve_promotion_source(experiment_dir, task_type)

    def _refresh_job_status(self, experiment_dir: Path, job_id: str) -> None:
        metadata = self._load_metadata(experiment_dir, None)
        status = metadata.get("status")
        if status not in {"pending", "running"}:
            return

        process = self._get_process(job_id)
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return

        metadata["status"] = "completed" if return_code == 0 and (experiment_dir / "metrics.json").exists() else "failed"
        metadata["return_code"] = return_code
        metadata["updated_at"] = _utc_now()
        self._write_job_metadata(experiment_dir, metadata)
        with _PROCESS_LOCK:
            _PROCESS_REGISTRY.pop(job_id, None)

    def _monitor_process(self, job_id: str) -> None:
        process = self._get_process(job_id)
        if process is None:
            return
        process.wait()
        experiment_dir, _ = self._find_job(job_id)
        self._refresh_job_status(experiment_dir, job_id)

    def _build_runtime_config(
        self,
        definition: TrainingTaskDefinition,
        base_config: dict[str, Any],
        request: TrainingJobCreateRequest,
        job_id: str,
        artifacts_dir: Path,
    ) -> dict[str, Any]:
        config = dict(base_config)
        overrides = dict(request.config_overrides or {})
        config["experiment_name"] = request.display_name or definition.title
        config["experiment_id_override"] = job_id

        if definition.task_type == "tracker":
            tracker_yaml_overrides = overrides.pop("tracker_yaml_overrides", {}) or {}
            tracker_source = Path(
                request.base_model_or_tracker_config
                or (config.get("tracker_configs") or [settings.tracker_yaml_path])[0]
            ).resolve()
            if not tracker_source.exists():
                raise HTTPException(status_code=400, detail=f"Tracker config not found: {tracker_source}")
            tracker_yaml = _load_yaml(tracker_source)
            tracker_yaml.update(tracker_yaml_overrides)
            runtime_tracker_yaml = artifacts_dir / "tracker_runtime.yaml"
            _dump_yaml(runtime_tracker_yaml, tracker_yaml)
            config["tracker_configs"] = [runtime_tracker_yaml]
            config["detector_model"] = str(
                Path(request.secondary_resource_path or config.get("detector_model") or settings.detection_model_path).resolve()
            )
        else:
            if definition.primary_resource_field and request.base_model_or_tracker_config:
                config[definition.primary_resource_field] = str(Path(request.base_model_or_tracker_config).resolve())

        config[definition.dataset_field] = str(Path(request.dataset_path).resolve())
        config.update(overrides)

        self._validate_runtime_config(definition, config)
        return config

    def _validate_runtime_config(self, definition: TrainingTaskDefinition, config: dict[str, Any]) -> None:
        dataset_path = Path(config[definition.dataset_field])
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset path not found: {dataset_path}")

        if definition.primary_resource_field and definition.task_type != "tracker":
            primary_path = Path(config[definition.primary_resource_field])
            if not primary_path.exists():
                raise HTTPException(status_code=400, detail=f"Base resource not found: {primary_path}")

        if definition.task_type == "tracker":
            detector_model = Path(config["detector_model"])
            if not detector_model.exists():
                raise HTTPException(status_code=400, detail=f"Detector model not found: {detector_model}")
            for tracker_config in config.get("tracker_configs", []):
                if not Path(tracker_config).exists():
                    raise HTTPException(status_code=400, detail=f"Tracker config not found: {tracker_config}")

    def _build_dataset_presets(self, definition: TrainingTaskDefinition, config: dict[str, Any]) -> list[TrainingPathPreset]:
        candidates: list[tuple[str, str]] = [(f"正式配置", str(config.get(definition.dataset_field, "")))]
        if definition.task_type == "recognition":
            candidates.extend(
                [
                    ("ArcFace 数据集", str(settings.arcface_dataset_dir)),
                    ("熟人库小样本集", str(settings.small_dataset_dir)),
                ]
            )
        elif definition.task_type == "tracker":
            candidates.append(("样例视频目录", str(settings.data_dir / "inputs" / "sample_videos")))
        return self._dedupe_presets(candidates, "path")

    def _build_primary_options(self, definition: TrainingTaskDefinition, config: dict[str, Any]) -> list[TrainingOptionItem]:
        candidates: list[Path] = []
        if definition.task_type == "yolo":
            candidates.extend(sorted(settings.training_weights_dir.joinpath("detection").glob("*.pt")))
            candidates.extend(sorted(settings.models_dir.joinpath("production", "detection").glob("*.pt")))
            candidates.extend(sorted(settings.catalog_detection_dir.glob("*.pt")))
        elif definition.task_type == "recognition":
            candidates.extend(sorted(settings.training_weights_dir.joinpath("recognition").glob("*.pth")))
            candidates.extend(sorted(settings.models_dir.joinpath("production", "recognition").glob("*.pth")))
            candidates.extend(sorted(settings.catalog_recognition_dir.glob("*.pth")))
            candidates.extend(sorted(settings.models_dir.joinpath("archive", "recognition").glob("*.pth")))
        elif definition.task_type == "tracker":
            candidates.extend(sorted(settings.training_configs_dir.joinpath("tracker").glob("*.yaml")))
            candidates.extend(sorted(settings.models_dir.joinpath("production", "tracking").glob("*.yaml")))
            candidates.extend(sorted(settings.catalog_tracking_dir.glob("*.yaml")))

        configured = config.get(definition.primary_resource_field or "", None)
        if isinstance(configured, list):
            candidates = [Path(item) for item in configured] + candidates
        elif configured:
            candidates.insert(0, Path(configured))

        return self._dedupe_option_paths(candidates)

    def _build_secondary_options(self, definition: TrainingTaskDefinition, config: dict[str, Any]) -> list[TrainingOptionItem]:
        if definition.task_type != "tracker":
            return []
        candidates = [Path(config.get("detector_model", settings.detection_model_path))]
        candidates.extend(sorted(settings.training_weights_dir.joinpath("detection").glob("*.pt")))
        candidates.extend(sorted(settings.models_dir.joinpath("production", "detection").glob("*.pt")))
        candidates.extend(sorted(settings.catalog_detection_dir.glob("*.pt")))
        return self._dedupe_option_paths(candidates)

    def _dedupe_option_paths(self, candidates: list[Path]) -> list[TrainingOptionItem]:
        items: list[TrainingOptionItem] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            resolved = str(candidate.resolve())
            if resolved in seen or not candidate.exists():
                continue
            seen.add(resolved)
            items.append(TrainingOptionItem(key=_slugify(candidate.stem), label=candidate.name, path=resolved))
        return items

    def _dedupe_presets(self, candidates: list[tuple[str, str]], kind: str) -> list[TrainingPathPreset]:
        presets: list[TrainingPathPreset] = []
        seen: set[str] = set()
        for label, raw_path in candidates:
            if not raw_path:
                continue
            path = Path(raw_path).resolve()
            if str(path) in seen:
                continue
            seen.add(str(path))
            presets.append(TrainingPathPreset(label=label, path=str(path), kind=kind, exists=path.exists()))
        return presets

    def _generate_job_id(self, task_type: str, display_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{_slugify(task_type)}_{_slugify(display_name)}"

    def _find_job(self, job_id: str) -> tuple[Path, str]:
        for definition in self.TASKS.values():
            experiment_dir = settings.training_experiments_dir / definition.domain / job_id
            if experiment_dir.exists():
                return experiment_dir, definition.task_type
        raise HTTPException(status_code=404, detail="Training job not found.")

    def _load_metadata(self, experiment_dir: Path, fallback_task_type: str | None) -> dict[str, Any]:
        metadata = _load_json(experiment_dir / "job.json")
        if metadata:
            return metadata
        task_type = fallback_task_type or self._task_type_from_domain(experiment_dir.parent.name)
        metrics = _load_json(experiment_dir / "metrics.json")
        return {
            "job_id": experiment_dir.name,
            "task_type": task_type,
            "title": self._require_definition(task_type).title,
            "display_name": experiment_dir.name,
            "status": "completed" if metrics else "failed",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "dataset_path": str(_load_yaml(experiment_dir / "config.snapshot.yaml").get(self._require_definition(task_type).dataset_field, "")),
            "primary_resource_path": "",
            "secondary_resource_path": "",
        }

    def _write_job_metadata(self, experiment_dir: Path, payload: dict[str, Any]) -> None:
        _dump_json(experiment_dir / "job.json", payload)
        _dump_json(experiment_dir / "status.json", payload)

    def _build_job_summary(self, experiment_dir: Path, task_type: str) -> TrainingJobSummary:
        metadata = self._load_metadata(experiment_dir, task_type)
        metrics = _load_json(experiment_dir / "metrics.json")
        manifest = self._load_production_manifest()
        current_production = manifest.get(task_type, {}).get("experiment_id") == experiment_dir.name

        return TrainingJobSummary(
            job_id=experiment_dir.name,
            task_type=task_type,
            title=self._require_definition(task_type).title,
            status=metadata.get("status", "pending"),
            created_at=_format_time(metadata.get("created_at")),
            updated_at=_format_time(metadata.get("updated_at")),
            experiment_dir=str(experiment_dir),
            display_name=metadata.get("display_name"),
            dataset_path=metadata.get("dataset_path"),
            primary_resource_path=metadata.get("primary_resource_path"),
            secondary_resource_path=metadata.get("secondary_resource_path"),
            current_production=current_production,
            summary_metrics=self._build_summary_metrics(task_type, metrics),
        )

    def _build_summary_metrics(self, task_type: str, metrics: dict[str, Any]) -> list[TrainingMetricItem]:
        if task_type == "yolo":
            last_row = metrics.get("last_row", {})
            return [
                TrainingMetricItem(label="Last mAP50", value=str(self._find_metric_value(last_row, ["mAP50", "map50"]))),
                TrainingMetricItem(label="Precision", value=str(self._find_metric_value(last_row, ["precision"]))),
                TrainingMetricItem(label="Recall", value=str(self._find_metric_value(last_row, ["recall"]))),
            ]
        if task_type == "recognition":
            return [
                TrainingMetricItem(label="Best Val Acc", value=str(metrics.get("best_val_accuracy", metrics.get("best_val_acc", "-")))),
                TrainingMetricItem(label="Best Epoch", value=str(metrics.get("best_epoch", "-"))),
                TrainingMetricItem(label="Valid Classes", value=str(metrics.get("valid_classes", "-"))),
            ]
        return [
            TrainingMetricItem(label="Videos", value=str(len(metrics.get("videos_evaluated", [])))),
            TrainingMetricItem(label="Track Configs", value=str(len(metrics.get("tracker_results", [])))),
        ]

    def _find_metric_value(self, payload: dict[str, Any], keywords: list[str]) -> Any:
        lower_payload = {str(key).lower(): value for key, value in payload.items()}
        for keyword in keywords:
            for key, value in lower_payload.items():
                if keyword.lower() in key:
                    return value
        return "-"

    def _collect_artifacts(self, experiment_dir: Path, job_id: str) -> list[TrainingArtifactItem]:
        files: list[Path] = []
        for candidate in [experiment_dir / "artifacts", experiment_dir]:
            if not candidate.exists():
                continue
            for path in candidate.rglob("*"):
                if path.is_file() and path.name not in {"job.json", "status.json", "stdout.log"}:
                    files.append(path)

        deduped: dict[str, Path] = {}
        for path in files:
            relative = _ensure_relative(path, experiment_dir)
            deduped[str(relative)] = path

        artifacts = []
        for relative, path in sorted(deduped.items()):
            artifacts.append(
                TrainingArtifactItem(
                    name=path.name,
                    relative_path=relative,
                    kind=_artifact_kind(path),
                    size_bytes=path.stat().st_size,
                    updated_at=_format_time(datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")),
                    url=f"/api/training/jobs/{job_id}/artifacts/{relative.replace(os.sep, '/')}",
                )
            )
        return artifacts

    def _require_definition(self, task_type: str) -> TrainingTaskDefinition:
        definition = self.TASKS.get(task_type)
        if not definition:
            raise HTTPException(status_code=400, detail=f"Unsupported training task type: {task_type}")
        return definition

    def _task_type_from_domain(self, domain: str) -> str:
        for definition in self.TASKS.values():
            if definition.domain == domain:
                return definition.task_type
        raise HTTPException(status_code=400, detail=f"Unsupported experiment domain: {domain}")

    def _resolve_promotion_source(self, experiment_dir: Path, task_type: str) -> Path:
        artifacts_dir = experiment_dir / "artifacts"
        candidates: dict[str, list[str]] = {
            "yolo": ["best_yolo.pt", "last_yolo.pt"],
            "recognition": ["best_arcface.pth", "last_arcface.pth"],
            "tracker": ["tracker_runtime.yaml", "bytetrack_anime.yaml"],
        }
        for name in candidates.get(task_type, []):
            path = artifacts_dir / name
            if path.exists():
                return path
        if task_type == "tracker":
            for yaml_file in artifacts_dir.glob("*.yaml"):
                return yaml_file
        raise HTTPException(status_code=404, detail="No promotable artifact found for this experiment.")

    def _metadata_primary_resource(self, definition: TrainingTaskDefinition, config: dict[str, Any], request: TrainingJobCreateRequest) -> str | None:
        if definition.task_type == "tracker":
            return request.base_model_or_tracker_config or ""
        if not definition.primary_resource_field:
            return None
        return str(config.get(definition.primary_resource_field, ""))

    def _metadata_secondary_resource(self, definition: TrainingTaskDefinition, config: dict[str, Any], request: TrainingJobCreateRequest) -> str | None:
        if not definition.secondary_resource_field:
            return None
        return str(request.secondary_resource_path or config.get(definition.secondary_resource_field, ""))

    def _load_production_manifest(self) -> dict[str, Any]:
        return _load_json(settings.training_production_manifest_path)

    def _get_process(self, job_id: str) -> subprocess.Popen | None:
        with _PROCESS_LOCK:
            return _PROCESS_REGISTRY.get(job_id)
