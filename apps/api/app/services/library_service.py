from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from ultralytics import YOLO

from app.core.settings import settings
from app.models import Video
from app.services.video_pipeline import get_extractor
from video_demo import build_known_database


DEFAULT_CONFIDENCE = 0.5
DEFAULT_MIN_FACE_SIZE = 30
DEFAULT_MIN_SHARPNESS = 100.0
DEFAULT_MIN_SAMPLES = 3
MAX_WEB_IMAGES = 12


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or f"character_{uuid.uuid4().hex[:8]}"


def _read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix or ".jpg", image)[1].tofile(str(path))


def _check_sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _select_main_face(boxes, confidences, image_width: int, image_height: int):
    if len(boxes) == 0:
        return None, 0.0

    center_x = image_width / 2
    center_y = image_height / 2
    max_distance = max(np.sqrt(center_x**2 + center_y**2), 1.0)
    best_score = -1.0
    best_box = None
    best_conf = 0.0

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1.0)
        area_score = area / max(image_width * image_height, 1.0)
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        distance = np.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2)
        position_score = 1 - (distance / max_distance)
        confidence = float(confidences[idx]) if idx < len(confidences) else 0.5
        total_score = area_score * 0.5 + position_score * 0.3 + confidence * 0.2
        if total_score > best_score:
            best_score = total_score
            best_box = box
            best_conf = confidence

    return best_box, best_conf


def _crop_face_safely(image, box, margin_ratio: float = 0.2):
    x1, y1, x2, y2 = box
    image_height, image_width = image.shape[:2]
    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1 + 2 * margin_ratio)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    crop_x1 = int(center_x - side / 2)
    crop_y1 = int(center_y - side / 2)
    crop_x2 = int(center_x + side / 2)
    crop_y2 = int(center_y + side / 2)

    if crop_x2 > image_width:
        shift = crop_x2 - image_width
        crop_x1 -= shift
        crop_x2 -= shift
    if crop_x1 < 0:
        shift = -crop_x1
        crop_x1 += shift
        crop_x2 += shift
    if crop_y2 > image_height:
        shift = crop_y2 - image_height
        crop_y1 -= shift
        crop_y2 -= shift
    if crop_y1 < 0:
        shift = -crop_y1
        crop_y1 += shift
        crop_y2 += shift

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_width, crop_x2)
    crop_y2 = min(image_height, crop_y2)
    return image[crop_y1:crop_y2, crop_x1:crop_x2]


class LibraryService:
    def __init__(
        self,
        db: Session,
        recognition_model_key: str | None = None,
        recognition_model_path: str | None = None,
    ):
        self.db = db
        try:
            resolved_model_path = (
                settings.resolve_model_key_path(recognition_model_key, settings.recognition_model_path)
                if recognition_model_key
                else Path(recognition_model_path or settings.recognition_model_path).resolve()
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        self.library_paths = settings.library_profile_paths(resolved_model_path)
        self.library_paths.ensure_dirs()
        self.recognition_model_path = self.library_paths.recognition_model_path
        self.library_root_dir = self.library_paths.root_dir
        self.known_db_path = self.library_paths.known_db_path
        self.stranger_db_path = self.library_paths.stranger_db_path
        self.library_manifest_path = self.library_paths.manifest_path
        self.library_sources_dir = self.library_paths.sources_dir
        self.library_cropped_dir = self.library_paths.cropped_dir
        self.library_portraits_dir = self.library_paths.portraits_dir
        self.library_imports_dir = self.library_paths.imports_dir
        self.recognition_model_key = self._model_key_from_path(self.recognition_model_path)
        self._ensure_manifest()

    def list_characters(self) -> list[dict]:
        manifest = self._ensure_manifest()
        characters = []
        for entry in manifest["characters"].values():
            characters.append(
                {
                    "name": entry["name"],
                    "sample_count": int(entry.get("sample_count", 0)),
                    "created_at": entry.get("created_at"),
                    "updated_at": entry.get("updated_at"),
                    "portrait_url": self._asset_url(entry.get("portrait_rel_path")),
                    "source_types": entry.get("source_types", []),
                }
            )
        characters.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return characters

    def get_character_detail(self, name: str) -> dict:
        manifest = self._ensure_manifest()
        entry = manifest["characters"].get(name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"熟人库中不存在 {name}")

        images = self._materialize_entry_images(entry)

        return {
            "name": entry["name"],
            "sample_count": int(entry.get("sample_count", len(images))),
            "created_at": entry.get("created_at"),
            "updated_at": entry.get("updated_at"),
            "portrait_url": self._asset_url(entry.get("portrait_rel_path")),
            "source_types": entry.get("source_types", []),
            "images": images,
        }

    def get_options(self) -> dict:
        return {
            "yolo_models": self._collect_model_options(
                [
                    settings.models_dir / "production" / "detection",
                    settings.catalog_detection_dir,
                ],
                ".pt",
            ),
            "recognition_models": self._collect_model_options(
                [
                    settings.models_dir / "production" / "recognition",
                    settings.catalog_recognition_dir,
                    settings.models_dir / "archive" / "recognition",
                ],
                ".pth",
            ),
            "tracker_configs": self._collect_model_options(
                [
                    settings.models_dir / "production" / "tracking",
                    settings.catalog_tracking_dir,
                ],
                ".yaml",
            ),
            "crop_defaults": {
                "confidence": DEFAULT_CONFIDENCE,
                "min_face_size": DEFAULT_MIN_FACE_SIZE,
                "min_sharpness": DEFAULT_MIN_SHARPNESS,
                "min_samples": DEFAULT_MIN_SAMPLES,
            },
            "web_import_available": bool(settings.danbooru_username and settings.danbooru_api_key),
            "web_import_hint": (
                "已检测到 Danbooru 凭据，可直接使用实验素材入口。"
                if settings.danbooru_username and settings.danbooru_api_key
                else "未配置 Danbooru 凭据，实验素材入口仍可尝试匿名抓取，但稳定性会较弱。"
            ),
        }

    def import_character(
        self,
        *,
        character_name: str,
        source_type: str,
        yolo_model_key: str | None,
        files: list[UploadFile] | None,
        finalize: bool,
        session_id: str | None,
        accepted_ids: list[str] | None,
        history_video_id: int | None,
        capture_interval: float,
        max_images: int,
        confidence: float,
        min_face_size: int,
        min_sharpness: float,
        min_samples: int,
    ) -> dict:
        if finalize:
            if not session_id:
                raise HTTPException(status_code=400, detail="缺少预览会话编号")
            return self._finalize_import(character_name, session_id, accepted_ids)

        session_id = uuid.uuid4().hex
        session_dir = self.library_imports_dir / session_id
        originals_dir = session_dir / "originals"
        cropped_dir = session_dir / "cropped"
        originals_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self._collect_sources(
            source_type=source_type,
            character_name=character_name,
            files=files or [],
            originals_dir=originals_dir,
            history_video_id=history_video_id,
            capture_interval=capture_interval,
            max_images=max_images,
        )
        if not source_paths:
            raise HTTPException(status_code=400, detail="没有可用于入库的图片素材")

        yolo_model = YOLO(str(self._resolve_model_path(yolo_model_key, settings.detection_model_path)))
        previews = self._build_previews(
            source_paths=source_paths,
            cropped_dir=cropped_dir,
            yolo_model=yolo_model,
            confidence=confidence,
            min_face_size=min_face_size,
            min_sharpness=min_sharpness,
        )
        if not previews:
            raise HTTPException(status_code=400, detail="未得到有效裁剪结果，请调整素材或参数后重试")

        metadata = {
            "session_id": session_id,
            "character_name": character_name,
            "source_type": source_type,
            "yolo_model_key": yolo_model_key,
            "created_at": _now_iso(),
            "previews": previews,
            "min_samples": min_samples,
        }
        _save_json(session_dir / "metadata.json", metadata)
        return {
            "stage": "preview",
            "session_id": session_id,
            "previews": previews,
            "message": "裁剪预览已生成，请确认后加入熟人库",
        }

    def import_from_web(
        self,
        *,
        character_name: str,
        tag_query: str,
        max_images: int,
        yolo_model_key: str | None,
        confidence: float,
        min_face_size: int,
        min_sharpness: float,
        min_samples: int,
    ) -> dict:
        if not tag_query.strip():
            raise HTTPException(status_code=400, detail="实验素材来源需要提供查询标签")

        session_id = uuid.uuid4().hex
        session_dir = self.library_imports_dir / session_id
        originals_dir = session_dir / "originals"
        cropped_dir = session_dir / "cropped"
        originals_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self._download_web_sources(tag_query=tag_query, originals_dir=originals_dir, max_images=max_images)
        if not source_paths:
            raise HTTPException(status_code=400, detail="未抓取到可用图片，请更换标签后重试")

        yolo_model = YOLO(str(self._resolve_model_path(yolo_model_key, settings.detection_model_path)))
        previews = self._build_previews(
            source_paths=source_paths,
            cropped_dir=cropped_dir,
            yolo_model=yolo_model,
            confidence=confidence,
            min_face_size=min_face_size,
            min_sharpness=min_sharpness,
        )
        if not previews:
            raise HTTPException(status_code=400, detail="网络素材未得到有效裁剪结果")

        metadata = {
            "session_id": session_id,
            "character_name": character_name,
            "source_type": "web_import",
            "created_at": _now_iso(),
            "tag_query": tag_query,
            "previews": previews,
            "min_samples": min_samples,
        }
        _save_json(session_dir / "metadata.json", metadata)
        return {
            "stage": "preview",
            "session_id": session_id,
            "previews": previews,
            "message": "实验素材已抓取，请确认裁剪结果后加入熟人库",
        }

    def rebuild_character(self, name: str) -> dict:
        manifest = self._ensure_manifest()
        entry = manifest["characters"].get(name)
        if not entry:
            raise HTTPException(status_code=404, detail=f"熟人库中不存在 {name}")

        folder = entry["folder"]
        cropped_dir = self.library_cropped_dir / folder
        if not cropped_dir.exists():
            raise HTTPException(status_code=404, detail="该角色没有可重建的裁剪样本")

        sample_images = self._list_images(cropped_dir)
        entry["sample_count"] = len(sample_images)
        entry["updated_at"] = _now_iso()
        self._refresh_portrait(entry, sample_images)
        self._persist_manifest(manifest)
        self._rebuild_known_database()
        return self.get_character_detail(name)

    def delete_character(self, name: str) -> dict:
        manifest = self._ensure_manifest()
        entry = manifest["characters"].pop(name, None)
        if not entry:
            raise HTTPException(status_code=404, detail=f"熟人库中不存在 {name}")

        folder = entry["folder"]
        for path in (
            self.library_sources_dir / folder,
            self.library_cropped_dir / folder,
        ):
            if path.exists():
                shutil.rmtree(path)

        portrait_rel_path = entry.get("portrait_rel_path")
        if portrait_rel_path:
            portrait_path = self.library_root_dir / portrait_rel_path
            if portrait_path.exists():
                portrait_path.unlink()

        self._persist_manifest(manifest)
        self._rebuild_known_database()
        return {"status": "success", "message": f"已删除角色 {name}"}

    def resolve_asset_path(self, asset_path: str) -> Path:
        candidate = (self.library_root_dir / asset_path).resolve()
        root = self.library_root_dir.resolve()
        if root not in candidate.parents and candidate != root:
            raise HTTPException(status_code=400, detail="非法资源路径")
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="资源不存在")
        return candidate

    def _collect_model_options(self, directories: list[Path], suffix: str) -> list[dict]:
        options = []
        seen: set[str] = set()
        seen_signatures: set[str] = set()
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.glob(f"*{suffix}")):
                key = path.relative_to(settings.models_dir).as_posix()
                if key in seen:
                    continue
                signature = self._model_signature(path)
                if signature in seen_signatures:
                    continue
                seen.add(key)
                seen_signatures.add(signature)
                options.append(
                    {
                        "key": key,
                        "label": path.name,
                    }
                )
        return options

    def _model_signature(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return str(path.resolve())
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _resolve_model_path(self, model_key: str | None, fallback: Path) -> Path:
        if not model_key:
            return fallback
        model_path = (settings.models_dir / model_key).resolve()
        root = settings.models_dir.resolve()
        if root not in model_path.parents and model_path != root:
            raise HTTPException(status_code=400, detail="非法模型路径")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_key}")
        return model_path

    def _collect_sources(
        self,
        *,
        source_type: str,
        character_name: str,
        files: list[UploadFile],
        originals_dir: Path,
        history_video_id: int | None,
        capture_interval: float,
        max_images: int,
    ) -> list[Path]:
        if source_type in {"local_upload", "batch_import"}:
            return self._save_uploaded_files(files, originals_dir)
        if source_type == "history_capture":
            if history_video_id is None:
                raise HTTPException(status_code=400, detail="缺少历史视频编号")
            return self._extract_frames_from_history(history_video_id, originals_dir, capture_interval, max_images)
        raise HTTPException(status_code=400, detail=f"暂不支持的素材来源: {source_type}")

    def _save_uploaded_files(self, files: list[UploadFile], originals_dir: Path) -> list[Path]:
        saved_paths = []
        for index, upload in enumerate(files, start=1):
            suffix = Path(upload.filename or "").suffix or ".jpg"
            target = originals_dir / f"upload_{index:03d}{suffix}"
            with target.open("wb") as handle:
                handle.write(upload.file.read())
            saved_paths.append(target)
        return saved_paths

    def _extract_frames_from_history(
        self,
        video_id: int,
        originals_dir: Path,
        capture_interval: float,
        max_images: int,
    ) -> list[Path]:
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"找不到历史视频 {video_id}")

        video_path = settings.videos_dir / video.filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="历史视频文件不存在")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="无法读取历史视频")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(int(fps * max(capture_interval, 0.5)), 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        current_frame = 0
        saved_paths = []

        while current_frame < total_frames and len(saved_paths) < max(max_images, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ok, frame = cap.read()
            if not ok:
                break
            target = originals_dir / f"frame_{len(saved_paths) + 1:03d}.jpg"
            _write_image(target, frame)
            saved_paths.append(target)
            current_frame += frame_step

        cap.release()
        return saved_paths

    def _download_web_sources(self, *, tag_query: str, originals_dir: Path, max_images: int) -> list[Path]:
        params = {"tags": tag_query, "limit": min(max(max_images, 1), MAX_WEB_IMAGES)}
        request = Request(
            f"https://danbooru.donmai.us/posts.json?{urlencode(params)}",
            headers=self._danbooru_headers(),
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"网络素材抓取失败: {exc}") from exc

        saved_paths = []
        for post in payload:
            file_url = None
            for key in ("file_url", "large_file_url", "sample_url", "preview_file_url", "preview_url"):
                value = post.get(key)
                if isinstance(value, str) and value:
                    file_url = value
                    break
            if not file_url:
                continue

            suffix = Path(file_url).suffix or ".jpg"
            target = originals_dir / f"web_{post.get('id', uuid.uuid4().hex)}{suffix}"
            try:
                with urlopen(Request(file_url, headers=self._danbooru_headers()), timeout=30) as response:
                    target.write_bytes(response.read())
                saved_paths.append(target)
            except Exception:
                continue
            if len(saved_paths) >= max(max_images, 1):
                break
        return saved_paths

    def _danbooru_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "AnimeCharacterAnalysis/1.0"}
        if settings.danbooru_username and settings.danbooru_api_key:
            token = base64.b64encode(
                f"{settings.danbooru_username}:{settings.danbooru_api_key}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers

    def _build_previews(
        self,
        *,
        source_paths: Iterable[Path],
        cropped_dir: Path,
        yolo_model: YOLO,
        confidence: float,
        min_face_size: int,
        min_sharpness: float,
    ) -> list[dict]:
        previews = []
        for index, image_path in enumerate(source_paths, start=1):
            image = _read_image(image_path)
            if image is None or image.size == 0:
                continue

            results = yolo_model(image, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.array([])
            confidences = (
                results[0].boxes.conf.cpu().numpy()
                if results and results[0].boxes is not None and results[0].boxes.conf is not None
                else np.array([])
            )
            valid_boxes = []
            valid_confidences = []
            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                conf_value = float(confidences[box_index]) if box_index < len(confidences) else 0.0
                if conf_value < confidence:
                    continue
                if width < min_face_size or height < min_face_size:
                    continue
                valid_boxes.append(box)
                valid_confidences.append(conf_value)
            if not valid_boxes:
                continue

            main_face, face_confidence = _select_main_face(
                valid_boxes,
                valid_confidences,
                image.shape[1],
                image.shape[0],
            )
            if main_face is None:
                continue
            face_crop = _crop_face_safely(image, main_face)
            if face_crop.size == 0:
                continue
            sharpness = _check_sharpness(face_crop)
            if sharpness < min_sharpness:
                continue

            resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
            crop_target = cropped_dir / f"crop_{index:03d}.jpg"
            _write_image(crop_target, resized)
            quality_score = round(sharpness * 0.6 + face_confidence * 100 * 0.4, 2)
            previews.append(
                {
                    "id": crop_target.stem,
                    "original_rel_path": image_path.relative_to(self.library_root_dir).as_posix(),
                    "cropped_rel_path": crop_target.relative_to(self.library_root_dir).as_posix(),
                    "original_url": self._asset_url(image_path.relative_to(self.library_root_dir).as_posix()),
                    "cropped_url": self._asset_url(crop_target.relative_to(self.library_root_dir).as_posix()),
                    "quality_score": quality_score,
                    "sharpness": round(sharpness, 2),
                    "confidence": round(face_confidence, 3),
                }
            )

        previews.sort(key=lambda item: item["quality_score"], reverse=True)
        return previews

    def _finalize_import(self, character_name: str, session_id: str, accepted_ids: list[str] | None) -> dict:
        session_dir = self.library_imports_dir / session_id
        metadata_path = session_dir / "metadata.json"
        metadata = _load_json(metadata_path, None)
        if not metadata:
            raise HTTPException(status_code=404, detail="找不到对应的导入预览会话")

        manifest = self._ensure_manifest()
        entry = manifest["characters"].get(character_name)
        if not entry:
            entry = {
                "name": character_name,
                "folder": _slugify(character_name),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "source_types": [],
                "sample_count": 0,
                "portrait_rel_path": None,
            }
            manifest["characters"][character_name] = entry

        selected = set(accepted_ids or [])
        previews = metadata.get("previews", [])
        if selected:
            previews = [item for item in previews if item["id"] in selected]
        if not previews:
            raise HTTPException(status_code=400, detail="没有选中的有效裁剪样本")

        folder = entry["folder"]
        source_dir = self.library_sources_dir / folder
        cropped_dir = self.library_cropped_dir / folder
        source_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        min_samples = int(metadata.get("min_samples", DEFAULT_MIN_SAMPLES))
        existing_samples = len(self._list_images(cropped_dir))
        if existing_samples + len(previews) < min_samples:
            raise HTTPException(
                status_code=400,
                detail=f"当前样本数不足 {min_samples} 张，请补充素材后再入库",
            )

        saved_records = []
        for index, item in enumerate(previews, start=1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_source = self.library_root_dir / item["original_rel_path"]
            cropped_source = self.library_root_dir / item["cropped_rel_path"]
            original_target = source_dir / f"{timestamp}_{index:03d}{original_source.suffix}"
            cropped_target = cropped_dir / f"{timestamp}_{index:03d}.jpg"
            shutil.copy2(original_source, original_target)
            shutil.copy2(cropped_source, cropped_target)
            saved_records.append(
                {
                    "id": f"{timestamp}_{index:03d}",
                    "cropped_rel_path": cropped_target.relative_to(self.library_root_dir).as_posix(),
                    "original_rel_path": original_target.relative_to(self.library_root_dir).as_posix(),
                    "quality_score": item.get("quality_score"),
                    "sharpness": item.get("sharpness"),
                    "confidence": item.get("confidence"),
                    "source_type": metadata.get("source_type", "import"),
                    "created_at": _now_iso(),
                }
            )

        sample_images = self._list_images(cropped_dir)
        entry["sample_count"] = len(sample_images)
        entry["updated_at"] = _now_iso()
        source_type = metadata.get("source_type", "import")
        if source_type not in entry["source_types"]:
            entry["source_types"].append(source_type)
        entry.setdefault("images", [])
        existing_by_crop = {
            item.get("cropped_rel_path"): item
            for item in entry.get("images", [])
            if item.get("cropped_rel_path")
        }
        for item in saved_records:
            existing_by_crop[item["cropped_rel_path"]] = item
        entry["images"] = sorted(
            existing_by_crop.values(),
            key=lambda value: value.get("quality_score") or 0,
            reverse=True,
        )
        self._refresh_portrait(entry, sample_images)
        self._persist_manifest(manifest)
        self._rebuild_known_database()
        shutil.rmtree(session_dir, ignore_errors=True)
        return {
            "stage": "completed",
            "character": self.get_character_detail(character_name),
            "message": f"已将 {character_name} 加入熟人库",
        }

    def _refresh_portrait(self, entry: dict, sample_images: list[Path]) -> None:
        if not sample_images:
            external_portrait = self._find_external_portrait_path(entry)
            if external_portrait:
                portrait_target = self.library_portraits_dir / f"{entry['folder']}.jpg"
                shutil.copy2(external_portrait, portrait_target)
                entry["portrait_rel_path"] = portrait_target.relative_to(self.library_root_dir).as_posix()
                return
            entry["portrait_rel_path"] = None
            return
        portrait_source = self._pick_best_portrait_path(entry, sample_images)
        portrait_target = self.library_portraits_dir / f"{entry['folder']}.jpg"
        shutil.copy2(portrait_source, portrait_target)
        entry["portrait_rel_path"] = portrait_target.relative_to(self.library_root_dir).as_posix()

    def _rebuild_known_database(self) -> None:
        build_known_database(
            extractor=get_extractor(str(self.recognition_model_path)),
            dataset_dir=str(self.library_cropped_dir),
            output_path=str(self.known_db_path),
        )

    def _ensure_manifest(self) -> dict:
        manifest = _load_json(self.library_manifest_path, {"characters": {}, "updated_at": _now_iso()})
        known_db = _load_json(self.known_db_path, {})
        changed = False

        for name, value in known_db.items():
            entry = manifest["characters"].get(name)
            if not entry:
                entry = {
                    "name": name,
                    "folder": _slugify(name),
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "source_types": [],
                    "sample_count": int(value.get("count", 0) if isinstance(value, dict) else 0),
                    "portrait_rel_path": None,
                }
                manifest["characters"][name] = entry
                changed = True
            portrait_rel_path = entry.get("portrait_rel_path")
            if portrait_rel_path:
                portrait_path = self.library_root_dir / portrait_rel_path
                if not portrait_path.exists():
                    entry["portrait_rel_path"] = None
                    changed = True
            sample_images = self._list_images(self.library_cropped_dir / entry["folder"])
            if sample_images and entry.get("sample_count") != len(sample_images):
                entry["sample_count"] = len(sample_images)
                changed = True
            if sample_images and not entry.get("images"):
                entry["images"] = self._rebuild_image_manifest(entry)
                changed = True
            if sample_images and not entry.get("portrait_rel_path"):
                self._refresh_portrait(entry, sample_images)
                changed = True
            if not sample_images and not entry.get("portrait_rel_path"):
                external_portrait = self._find_external_portrait_path(entry)
                if external_portrait:
                    self._refresh_portrait(entry, sample_images)
                    changed = True

        manifest["updated_at"] = _now_iso()
        if changed or not self.library_manifest_path.exists():
            self._persist_manifest(manifest)
        return manifest

    def _persist_manifest(self, manifest: dict) -> None:
        manifest["updated_at"] = _now_iso()
        _save_json(self.library_manifest_path, manifest)

    def _asset_url(self, relative_path: str | None) -> str | None:
        if not relative_path:
            return None
        normalized = relative_path.replace("\\", "/")
        model_key = quote(self.recognition_model_key, safe="")
        return f"/api/library/assets/{normalized}?recognition_model_key={model_key}"

    def _model_key_from_path(self, model_path: Path) -> str:
        try:
            return model_path.resolve().relative_to(settings.models_dir.resolve()).as_posix()
        except ValueError:
            return str(model_path.resolve())

    def _list_images(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )

    def _materialize_entry_images(self, entry: dict) -> list[dict]:
        if not entry.get("images"):
            self._rebuild_image_manifest(entry)

        items = []
        for item in entry.get("images", []):
            cropped_rel_path = item.get("cropped_rel_path")
            if not cropped_rel_path:
                continue
            items.append(
                {
                    "id": item.get("id") or Path(cropped_rel_path).stem,
                    "url": self._asset_url(cropped_rel_path),
                    "original_url": self._asset_url(item.get("original_rel_path")),
                    "quality_score": item.get("quality_score"),
                    "sharpness": item.get("sharpness"),
                    "confidence": item.get("confidence"),
                    "source_type": item.get("source_type"),
                    "created_at": item.get("created_at"),
                }
            )
        return items

    def _rebuild_image_manifest(self, entry: dict) -> list[dict]:
        folder = entry["folder"]
        source_dir = self.library_sources_dir / folder
        cropped_dir = self.library_cropped_dir / folder
        source_images = self._list_images(source_dir)
        cropped_images = self._list_images(cropped_dir)
        rebuilt = []
        for index, cropped_path in enumerate(cropped_images):
            original_path = source_images[index] if index < len(source_images) else None
            rebuilt.append(
                {
                    "id": cropped_path.stem,
                    "cropped_rel_path": cropped_path.relative_to(self.library_root_dir).as_posix(),
                    "original_rel_path": original_path.relative_to(self.library_root_dir).as_posix() if original_path else None,
                    "quality_score": None,
                    "sharpness": None,
                    "confidence": None,
                    "source_type": (entry.get("source_types") or [None])[-1],
                    "created_at": entry.get("updated_at") or entry.get("created_at"),
                }
            )
        entry["images"] = rebuilt
        return rebuilt

    def _pick_best_portrait_path(self, entry: dict, sample_images: list[Path]) -> Path:
        image_records = entry.get("images") or []
        best_rel_path = None
        best_score = -1.0
        for item in image_records:
            cropped_rel_path = item.get("cropped_rel_path")
            score = item.get("quality_score")
            if cropped_rel_path and score is not None and score > best_score:
                best_score = score
                best_rel_path = cropped_rel_path
        if best_rel_path:
            candidate = self.library_root_dir / best_rel_path
            if candidate.exists():
                return candidate
        return sample_images[0]

    def _find_external_portrait_path(self, entry: dict) -> Path | None:
        candidate_names = {
            entry.get("name", ""),
            entry.get("folder", ""),
            _slugify(entry.get("name", "")),
            _slugify(entry.get("folder", "")),
        }
        candidate_names = {item.lower() for item in candidate_names if item}

        for dataset_dir in settings.external_avatar_dataset_dirs:
            if not dataset_dir.exists():
                continue
            folders = {
                child.name.lower(): child
                for child in dataset_dir.iterdir()
                if child.is_dir()
            }
            for candidate_name in candidate_names:
                folder = folders.get(candidate_name)
                if not folder:
                    continue
                images = self._list_images(folder)
                if images:
                    return images[0]
        return None
