from __future__ import annotations

import hashlib
import json
import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.models import Video
from app.repositories.video_repository import VideoRepository
from app.schemas.video import (
    AnalysisModelOption,
    AnalysisModelOptionsResponse,
    AnalysisModelSelector,
    ImageRecognitionResponse,
)
from app.services.training_service import TrainingService
from app.services.experiment_export import export_experiment_report
from app.services.video_pipeline import run_image_pipeline, run_video_pipeline


class VideoService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = VideoRepository(db)

    def save_upload(self, file: UploadFile) -> tuple[Video, Path]:
        extension = Path(file.filename or "upload.mp4").suffix or ".mp4"
        filename = f"{uuid.uuid4().hex}{extension}"
        save_path = settings.videos_dir / filename
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return self.repository.create_video(filename), save_path

    def enqueue_processing(
        self,
        background_tasks: BackgroundTasks,
        video_id: int,
        file_path: Path,
        model_selection: dict[str, str] | None = None,
    ) -> None:
        selection = model_selection or {}
        self._validate_analysis_model_selection(selection)
        background_tasks.add_task(
            run_ai_and_save_to_db,
            video_id,
            str(file_path),
            selection.get("detection_model_path"),
            selection.get("recognition_model_path"),
            selection.get("tracker_config_path"),
            bool(selection.get("data_export_enabled")),
        )

    def get_status(self, video_id: int) -> Video:
        video = self.repository.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="鎵句笉鍒拌瑙嗛")
        return video

    def get_timeline(self, video_id: int) -> dict:
        video = self.get_status(video_id)
        if video.status != "completed":
            raise HTTPException(status_code=400, detail="瑙嗛灏氭湭澶勭悊瀹屾垚")

        appearances = self.repository.list_video_appearances(video_id)
        timeline: dict[str, dict] = {}
        for appearance in appearances:
            name = appearance.character.name
            timeline.setdefault(name, {"type": appearance.character.type, "segments": []})
            timeline[name]["segments"].append(
                {"start": appearance.start_time, "end": appearance.end_time}
            )

        timeline_file = settings.timelines_dir / f"timeline_{video_id}.json"
        model_selection = None
        if timeline_file.exists():
            try:
                payload = json.loads(timeline_file.read_text(encoding="utf-8"))
                model_selection = payload.get("video_info", {}).get("model_selection")
            except (json.JSONDecodeError, OSError):
                model_selection = None

        return {
            "video_id": video_id,
            "timeline": timeline,
            "video_info": {
                "duration_sec": video.duration,
                "fps": video.fps,
                "model_selection": model_selection,
            },
        }

    def get_download_path(self, video_id: int) -> Path:
        video = self.get_status(video_id)
        if video.status != "completed":
            raise HTTPException(status_code=400, detail="瑙嗛灏氭湭澶勭悊瀹屾垚")

        output_path = self._annotated_video_path(video.filename)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="鎵句笉鍒版爣娉ㄨ棰戞枃浠?")
        return output_path

    def get_history(self) -> list[dict]:
        videos = self.repository.list_completed_videos()
        return [
            {
                "id": video.id,
                "filename": video.filename,
                "duration": video.duration,
                "status": video.status,
                "has_annotated_video": self._annotated_video_path(video.filename).exists(),
            }
            for video in videos
        ]

    def rename_video(self, video_id: int, new_name: str) -> None:
        video = self.get_status(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="鎵句笉鍒拌瑙嗛")

        old_stem = Path(video.filename).stem
        new_stem = Path(new_name).stem

        old_vid_path = settings.videos_dir / video.filename
        new_vid_path = settings.videos_dir / new_name
        if old_vid_path.exists() and str(old_vid_path) != str(new_vid_path):
            old_vid_path.rename(new_vid_path)

        old_webm_path = self._annotated_video_path(f"{old_stem}.webm")
        new_webm_path = self._annotated_video_path(f"{new_stem}.webm")
        if old_webm_path.exists() and str(old_webm_path) != str(new_webm_path):
            old_webm_path.rename(new_webm_path)

        video.filename = new_name
        self.repository.commit()

    def delete_video(self, video_id: int) -> None:
        video = self.get_status(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="鎵句笉鍒拌瑙嗛")

        vid_path = settings.videos_dir / video.filename
        webm_path = self._annotated_video_path(video.filename)
        json_path = settings.timelines_dir / f"timeline_{video_id}.json"

        for path in [vid_path, webm_path, json_path]:
            if path.exists():
                path.unlink()

        self.repository.db.delete(video)
        self.repository.commit()

    def _annotated_video_path(self, filename: str) -> Path:
        return settings.annotated_dir / f"annotated_{Path(filename).stem}.webm"

    def _validate_analysis_model_selection(self, selection: dict[str, str]) -> None:
        checks = {
            "detection_model_path": ".pt",
            "recognition_model_path": ".pth",
            "tracker_config_path": ".yaml",
        }
        for field, suffix in checks.items():
            raw_path = selection.get(field)
            if not raw_path:
                continue
            path = Path(raw_path).resolve()
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"Selected model path does not exist: {path}")
            if path.suffix.lower() != suffix:
                raise HTTPException(status_code=400, detail=f"Selected {field} must be a {suffix} file.")

    def get_analysis_model_options(self) -> AnalysisModelOptionsResponse:
        training_service = TrainingService()

        return AnalysisModelOptionsResponse(
            detection=self._build_selector(
                key="detection_model_path",
                label="检测模型",
                task_type="yolo",
                current_path=settings.detection_model_path,
                training_service=training_service,
            ),
            recognition=self._build_selector(
                key="recognition_model_path",
                label="特征模型",
                task_type="recognition",
                current_path=settings.recognition_model_path,
                training_service=training_service,
            ),
            tracker=self._build_selector(
                key="tracker_config_path",
                label="跟踪配置",
                task_type="tracker",
                current_path=settings.tracker_yaml_path,
                training_service=training_service,
            ),
        )

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        detection_model_path: str | None = None,
        recognition_model_path: str | None = None,
    ) -> ImageRecognitionResponse:
        self._validate_analysis_model_selection(
            {
                "detection_model_path": detection_model_path or "",
                "recognition_model_path": recognition_model_path or "",
            }
        )
        payload = run_image_pipeline(
            image_bytes=image_bytes,
            detection_model_path=detection_model_path,
            recognition_model_path=recognition_model_path,
        )
        return ImageRecognitionResponse(**payload)

    def _build_selector(
        self,
        *,
        key: str,
        label: str,
        task_type: str,
        current_path: Path,
        training_service: TrainingService,
    ) -> AnalysisModelSelector:
        options: list[AnalysisModelOption] = []
        seen: set[str] = set()

        manifest = training_service._load_production_manifest().get(task_type, {})  # noqa: SLF001 - local service reuse
        current_experiment_id = manifest.get("experiment_id")
        current_resolved = str(current_path.resolve())
        options.append(
            AnalysisModelOption(
                key=f"{task_type}-production",
                label=f"当前生产 · {current_path.name}",
                value=current_resolved,
                source="production",
                experiment_id=current_experiment_id,
                summary="当前默认分析配置",
                current_production=True,
            )
        )
        seen.add(current_resolved)

        for candidate in self._selector_candidate_paths(task_type, current_path):
            resolved = str(candidate.resolve())
            if not candidate.exists() or resolved in seen:
                continue
            options.append(self._build_curated_option(task_type, candidate, training_service))
            seen.add(resolved)

        return AnalysisModelSelector(
            key=key,
            label=label,
            current_value=current_resolved,
            options=options,
        )

        task_option = task_options.get(task_type)
        for item in (task_option.primary_resource_options if task_option else []):
            if item.path in seen:
                continue
            options.append(
                AnalysisModelOption(
                    key=f"{task_type}-{item.key}",
                    label=item.label,
                    value=item.path,
                    source="registered",
                    summary="已登记的可选模型/配置",
                )
            )
            seen.add(item.path)

        for extra_path in extra_paths or []:
            resolved = str(extra_path.resolve())
            if not extra_path.exists() or resolved in seen:
                continue
            options.append(
                AnalysisModelOption(
                    key=f"{task_type}-{extra_path.stem}",
                    label=f"历史归档 · {extra_path.name}",
                    value=resolved,
                    source="archive",
                    summary="保留的旧版本模型",
                )
            )
            seen.add(resolved)

        for job in completed_jobs:
            if job.task_type != task_type:
                continue
            try:
                artifact_path = training_service.get_promotable_artifact_path(job.job_id)
            except HTTPException:
                continue
            resolved = str(artifact_path.resolve())
            if resolved in seen:
                continue
            summary = " / ".join(f"{metric.label}: {metric.value}" for metric in job.summary_metrics[:2]) or "实验模型版本"
            options.append(
                AnalysisModelOption(
                    key=f"{task_type}-{job.job_id}",
                    label=f"{job.display_name or job.job_id} · {artifact_path.name}",
                    value=resolved,
                    source="experiment",
                    experiment_id=job.job_id,
                    summary=summary,
                    current_production=job.current_production,
                )
            )
            seen.add(resolved)

        return AnalysisModelSelector(
            key=key,
            label=label,
            current_value=current_resolved,
            options=options,
        )

    def _selector_candidate_paths(self, task_type: str, current_path: Path) -> list[Path]:
        candidates: list[Path] = []
        if task_type == "yolo":
            candidates.extend(sorted(current_path.parent.glob("*.pt")))
            candidates.extend(sorted(settings.catalog_detection_dir.glob("*.pt")))
        elif task_type == "recognition":
            candidates.extend(sorted(current_path.parent.glob("*.pth")))
            candidates.extend(sorted(settings.catalog_recognition_dir.glob("*.pth")))
            candidates.extend(sorted(settings.models_dir.joinpath("archive", "recognition").glob("*.pth")))
        elif task_type == "tracker":
            candidates.extend(sorted(current_path.parent.glob("*.yaml")))
            candidates.extend(sorted(settings.catalog_tracking_dir.glob("*.yaml")))

        unique: list[Path] = []
        seen: set[str] = set()
        seen_signatures: set[str] = set()
        for candidate in [current_path, *candidates]:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            signature = self._candidate_signature(candidate)
            if signature in seen_signatures:
                continue
            seen.add(resolved)
            seen_signatures.add(signature)
            unique.append(candidate)
        return unique

    def _build_curated_option(
        self,
        task_type: str,
        candidate: Path,
        training_service: TrainingService,
    ) -> AnalysisModelOption:
        resolved = str(candidate.resolve())
        current_resolved = str(self._current_task_path(task_type).resolve())
        experiment_id = self._match_experiment_id_for_selector(task_type, candidate, training_service)

        if resolved == current_resolved:
            return AnalysisModelOption(
                key=f"{task_type}-production",
                label=f"褰撳墠鐢熶骇 路 {candidate.name}",
                value=resolved,
                source="production",
                experiment_id=experiment_id,
                summary="褰撳墠榛樿鍒嗘瀽閰嶇疆",
                current_production=True,
            )

        source = "available"
        label_prefix = "鍙垏鎹㈢増鏈?路 "
        summary = "姝ｅ紡鐗堝墠绔彲鐩存帴閫夋嫨鐨勬ā鍨?閰嶇疆"
        if self._is_within(candidate, settings.models_dir.joinpath("catalog")):
            source = "catalog"
            label_prefix = "瀵煎叆鎴愭灉 路 "
            summary = "浠庡疄楠屾垨澶栭儴缁撴灉瀵煎叆鐨勫睍绀虹増鏈?        "
        elif self._is_within(candidate, settings.models_dir.joinpath("archive")):
            source = "archive"
            label_prefix = "鍘嗗彶鐗堟湰 路 "
            summary = "淇濈暀鐨勬棫鐗堟湰妯″瀷锛岄€傚悎瀵规瘮灞曠ず"

        return AnalysisModelOption(
            key=f"{task_type}-{candidate.stem}",
            label=f"{label_prefix}{candidate.name}",
            value=resolved,
            source=source,
            experiment_id=experiment_id,
            summary=summary,
            current_production=False,
        )

    def _current_task_path(self, task_type: str) -> Path:
        if task_type == "yolo":
            return settings.detection_model_path
        if task_type == "recognition":
            return settings.recognition_model_path
        return settings.tracker_yaml_path

    def _match_experiment_id_for_selector(
        self,
        task_type: str,
        candidate: Path,
        training_service: TrainingService,
    ) -> str | None:
        manifest = training_service._load_production_manifest()  # noqa: SLF001 - local service reuse
        preferred_job_id = None
        if candidate.resolve() == self._current_task_path(task_type).resolve():
            preferred_job_id = manifest.get(task_type, {}).get("experiment_id")

        for job in training_service.list_jobs():
            if job.task_type != task_type:
                continue
            if preferred_job_id and job.job_id == preferred_job_id and self._job_matches_candidate(job, candidate, training_service):
                return job.job_id
            if job.primary_resource_path and self._same_file(candidate, Path(job.primary_resource_path)):
                return job.job_id
            try:
                if self._same_file(candidate, training_service.get_promotable_artifact_path(job.job_id)):
                    return job.job_id
            except HTTPException:
                continue
        return None

    def _job_matches_candidate(self, job, candidate: Path, training_service: TrainingService) -> bool:
        if job.primary_resource_path and self._same_file(candidate, Path(job.primary_resource_path)):
            return True
        try:
            return self._same_file(candidate, training_service.get_promotable_artifact_path(job.job_id))
        except HTTPException:
            return False

    def _candidate_signature(self, candidate: Path) -> str:
        if not candidate.exists() or not candidate.is_file():
            return str(candidate.resolve())
        hasher = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _same_file(self, left: Path, right: Path) -> bool:
        if not left.exists() or not right.exists():
            return False
        try:
            if left.resolve() == right.resolve():
                return True
        except OSError:
            return False
        if left.stat().st_size != right.stat().st_size:
            return False
        return self._candidate_signature(left) == self._candidate_signature(right)

    def _is_within(self, candidate: Path, root: Path) -> bool:
        try:
            candidate.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False


def run_ai_and_save_to_db(
    video_id: int,
    file_path: str,
    detection_model_path: str | None = None,
    recognition_model_path: str | None = None,
    tracker_config_path: str | None = None,
    data_export_enabled: bool = False,
) -> None:
    from app.core.database import SessionLocal

    db = SessionLocal()
    repository = VideoRepository(db)
    try:
        video = repository.get_video(video_id)
        if not video:
            return

        video.status = "processing"
        repository.commit()

        def update_progress(percent: float) -> None:
            video.progress = percent
            repository.commit()

        output_path = settings.annotated_dir / f"annotated_{Path(file_path).stem}.webm"
        timeline_output_path = settings.timelines_dir / f"timeline_{video_id}.json"
        timeline_data = run_video_pipeline(
            file_path=file_path,
            progress_callback=update_progress,
            output_path=str(output_path),
            timeline_output_path=str(timeline_output_path),
            detection_model_path=detection_model_path,
            recognition_model_path=recognition_model_path,
            tracker_config_path=tracker_config_path,
        )
        if not timeline_data:
            video.status = "failed"
            repository.commit()
            return

        if data_export_enabled:
            export_paths = export_experiment_report(video_id, timeline_data, settings.experiment_exports_dir)
            timeline_data.setdefault("experiment_stats", {})["export_paths"] = export_paths
            timeline_output_path.write_text(json.dumps(timeline_data, ensure_ascii=False, indent=2), encoding="utf-8")

        video.duration = timeline_data["video_info"]["duration_sec"]
        video.fps = timeline_data["video_info"]["fps"]

        for character_name, character_info in timeline_data["characters"].items():
            character = repository.get_or_create_character(
                name=character_name,
                character_type=character_info["type"],
                video_id=video_id,
            )
            for segment in character_info["appearances"]:
                repository.add_appearance(
                    video_id=video_id,
                    character_id=character.id,
                    start_time=segment["start"],
                    end_time=segment["end"],
                )

        video.progress = 100.0
        video.status = "completed"
        repository.commit()
    except Exception:
        traceback.print_exc()
        db.rollback()
        video = repository.get_video(video_id)
        if video:
            video.status = "failed"
            repository.commit()
    finally:
        db.close()
