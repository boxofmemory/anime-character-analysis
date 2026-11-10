from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.models import Appearance, Character, Video
from app.services.library_service import LibraryService


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DashboardService:
    def __init__(self, db: Session):
        self.db = db

    def get_summary(self) -> dict:
        library_service = LibraryService(self.db)
        known_db = _load_json(library_service.known_db_path, {})
        manifest = library_service._ensure_manifest()
        characters = list(manifest.get("characters", {}).values())

        completed_videos = (
            self.db.query(Video)
            .filter(Video.status == "completed")
            .order_by(Video.created_at.desc())
            .limit(7)
            .all()
        )
        completed_video_count = self.db.query(func.count(Video.id)).filter(Video.status == "completed").scalar() or 0
        last_video = (
            self.db.query(Video)
            .filter(Video.status == "completed")
            .order_by(Video.updated_at.desc())
            .first()
        )
        total_samples = sum(item.get("sample_count", 0) for item in characters)

        trends = []
        for video in reversed(completed_videos):
            appearances = (
                self.db.query(Appearance)
                .filter(Appearance.video_id == video.id)
                .all()
            )
            trends.append(
                {
                    "video_id": video.id,
                    "filename": video.filename,
                    "label": Path(video.filename).stem[:18],
                    "duration": video.duration,
                    "has_annotated_video": (settings.annotated_dir / f"annotated_{Path(video.filename).stem}.webm").exists(),
                    "character_count": len({item.character_id for item in appearances}),
                    "total_segments": len(appearances),
                }
            )

        recent_characters = []
        for item in sorted(
            characters,
            key=lambda value: value.get("updated_at") or "",
            reverse=True,
        )[:5]:
            recent_characters.append(
                {
                    "name": item["name"],
                    "sample_count": item.get("sample_count", 0),
                    "updated_at": item.get("updated_at"),
                    "portrait_url": library_service._asset_url(item.get("portrait_rel_path")),
                }
            )

        recent_updates = []
        for item in recent_characters[:3]:
            recent_updates.append(
                {
                    "title": f"熟人库更新: {item['name']}",
                    "detail": f"当前样本数 {item['sample_count']} 张",
                    "updated_at": item["updated_at"],
                    "kind": "library",
                }
            )
        for video in completed_videos[:2]:
            char_count = (
                self.db.query(func.count(func.distinct(Appearance.character_id)))
                .filter(Appearance.video_id == video.id)
                .scalar()
                or 0
            )
            recent_updates.append(
                {
                    "title": f"视频分析完成: {Path(video.filename).stem}",
                    "detail": f"识别到 {char_count} 个角色",
                    "updated_at": video.updated_at.isoformat() if video.updated_at else None,
                    "kind": "video",
                }
            )

        recent_videos = []
        for video in completed_videos[:5]:
            character_count = (
                self.db.query(func.count(func.distinct(Appearance.character_id)))
                .filter(Appearance.video_id == video.id)
                .scalar()
                or 0
            )
            recent_videos.append(
                {
                    "id": video.id,
                    "filename": video.filename,
                    "duration": video.duration,
                    "updated_at": video.updated_at.isoformat() if video.updated_at else None,
                    "character_count": character_count,
                    "has_annotated_video": (settings.annotated_dir / f"annotated_{Path(video.filename).stem}.webm").exists(),
                }
            )

        total_duration = sum(v.duration for v in completed_videos if v.duration)
        avg_video_duration = total_duration / len(completed_videos) if completed_videos else 0.0

        total_appearances_count = 0
        known_appearances_count = 0
        for video in completed_videos:
            appearances = self.db.query(Appearance).filter(Appearance.video_id == video.id).all()
            total_appearances_count += len(appearances)
            for app in appearances:
                if app.character.type == "known":
                    known_appearances_count += 1
        
        known_recognition_rate = (known_appearances_count / total_appearances_count * 100) if total_appearances_count else 0.0
        avg_characters_per_video = sum(v["character_count"] for v in recent_videos) / len(recent_videos) if recent_videos else 0.0

        return {
            "stats": [
                {
                    "label": "熟人库角色数",
                    "value": str(len(known_db)),
                    "hint": "当前正式版已登记角色",
                },
                {
                    "label": "熟人库样本数",
                    "value": str(total_samples),
                    "hint": "按裁剪后头像样本统计",
                },
                {
                    "label": "已处理视频数",
                    "value": str(completed_video_count),
                    "hint": "数据库中已完成分析的视频",
                },
                {
                    "label": "最近分析时间",
                    "value": last_video.updated_at.strftime("%Y-%m-%d %H:%M") if last_video and last_video.updated_at else "暂无",
                    "hint": "最后一次完成的视频任务",
                },
            ],
            "models": {
                "yolo_model": settings.detection_model_path.name,
                "recognition_model": settings.recognition_model_path.name,
                "tracker_config": settings.tracker_yaml_path.name,
            },
            "trends": trends,
            "recent_characters": recent_characters,
            "recent_videos": recent_videos,
            "recent_updates": recent_updates,
            "insights": {
                "avg_video_duration": avg_video_duration,
                "known_recognition_rate": known_recognition_rate,
                "avg_characters_per_video": avg_characters_per_video,
                "total_appearances": total_appearances_count,
            },
        }

    def _asset_url(self, relative_path: str | None) -> str | None:
        if not relative_path:
            return None
        normalized = relative_path.replace("\\", "/")
        return f"/api/library/assets/{normalized}"
