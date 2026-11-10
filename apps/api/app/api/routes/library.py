from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.library import (
    LibraryCharacterDetail,
    LibraryCharacterSummary,
    LibraryImportResponse,
    LibraryOptionsResponse,
)
from app.services.library_service import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_MIN_SHARPNESS,
    LibraryService,
)

router = APIRouter(tags=["library"])


@router.get("/api/library/options", response_model=LibraryOptionsResponse)
def get_library_options(
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).get_options()


@router.get("/api/library/characters", response_model=list[LibraryCharacterSummary])
def list_library_characters(
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).list_characters()


@router.get("/api/library/characters/{name}", response_model=LibraryCharacterDetail)
def get_library_character(
    name: str,
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).get_character_detail(name)


@router.post("/api/library/characters/import", response_model=LibraryImportResponse)
def import_library_character(
    character_name: str = Form(...),
    source_type: str = Form(...),
    yolo_model_key: str | None = Form(default=None),
    finalize: bool = Form(default=False),
    session_id: str | None = Form(default=None),
    accepted_ids: str | None = Form(default=None),
    history_video_id: int | None = Form(default=None),
    capture_interval: float = Form(default=2.0),
    max_images: int = Form(default=10),
    confidence: float = Form(default=DEFAULT_CONFIDENCE),
    min_face_size: int = Form(default=DEFAULT_MIN_FACE_SIZE),
    min_sharpness: float = Form(default=DEFAULT_MIN_SHARPNESS),
    min_samples: int = Form(default=3),
    recognition_model_key: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    selected_ids = json.loads(accepted_ids) if accepted_ids else None
    return LibraryService(db, recognition_model_key=recognition_model_key).import_character(
        character_name=character_name,
        source_type=source_type,
        yolo_model_key=yolo_model_key,
        files=files,
        finalize=finalize,
        session_id=session_id,
        accepted_ids=selected_ids,
        history_video_id=history_video_id,
        capture_interval=capture_interval,
        max_images=max_images,
        confidence=confidence,
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
        min_samples=min_samples,
    )


@router.post("/api/library/characters/import-from-web", response_model=LibraryImportResponse)
def import_library_character_from_web(
    character_name: str = Form(...),
    tag_query: str = Form(...),
    max_images: int = Form(default=8),
    yolo_model_key: str | None = Form(default=None),
    confidence: float = Form(default=DEFAULT_CONFIDENCE),
    min_face_size: int = Form(default=DEFAULT_MIN_FACE_SIZE),
    min_sharpness: float = Form(default=DEFAULT_MIN_SHARPNESS),
    min_samples: int = Form(default=3),
    recognition_model_key: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).import_from_web(
        character_name=character_name,
        tag_query=tag_query,
        max_images=max_images,
        yolo_model_key=yolo_model_key,
        confidence=confidence,
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
        min_samples=min_samples,
    )


@router.post("/api/library/characters/{name}/rebuild", response_model=LibraryCharacterDetail)
def rebuild_library_character(
    name: str,
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).rebuild_character(name)


@router.delete("/api/library/characters/{name}")
def delete_library_character(
    name: str,
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    return LibraryService(db, recognition_model_key=recognition_model_key).delete_character(name)


@router.get("/api/library/assets/{asset_path:path}")
def get_library_asset(
    asset_path: str,
    recognition_model_key: str | None = None,
    db: Session = Depends(get_db),
):
    asset = LibraryService(db, recognition_model_key=recognition_model_key).resolve_asset_path(asset_path)
    return FileResponse(str(asset))
