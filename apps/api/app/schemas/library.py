from __future__ import annotations

from pydantic import BaseModel


class LibraryCharacterSummary(BaseModel):
    name: str
    sample_count: int
    created_at: str | None = None
    updated_at: str | None = None
    portrait_url: str | None = None
    source_types: list[str] = []


class LibraryImageItem(BaseModel):
    id: str
    url: str
    original_url: str | None = None
    quality_score: float | None = None
    sharpness: float | None = None
    confidence: float | None = None
    source_type: str | None = None
    created_at: str | None = None


class LibraryCharacterDetail(BaseModel):
    name: str
    sample_count: int
    created_at: str | None = None
    updated_at: str | None = None
    portrait_url: str | None = None
    source_types: list[str] = []
    images: list[LibraryImageItem] = []


class LibraryOptionItem(BaseModel):
    key: str
    label: str


class LibraryCropDefaults(BaseModel):
    confidence: float
    min_face_size: int
    min_sharpness: float
    min_samples: int


class LibraryOptionsResponse(BaseModel):
    yolo_models: list[LibraryOptionItem]
    recognition_models: list[LibraryOptionItem]
    tracker_configs: list[LibraryOptionItem]
    crop_defaults: LibraryCropDefaults
    web_import_available: bool
    web_import_hint: str | None = None
    model_showcase: list[dict] = []


class LibraryImportPreviewItem(BaseModel):
    id: str
    original_url: str
    cropped_url: str
    quality_score: float
    sharpness: float
    confidence: float


class LibraryImportResponse(BaseModel):
    stage: str
    session_id: str | None = None
    character: LibraryCharacterDetail | None = None
    previews: list[LibraryImportPreviewItem] = []
    message: str
