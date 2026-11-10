from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


VideoStatus = Literal["pending", "processing", "completed", "failed"]
CharacterType = Literal["known", "stranger"]


class UploadVideoResponse(BaseModel):
    message: str
    video_id: int
    status: VideoStatus


class AnalysisModelOption(BaseModel):
    key: str
    label: str
    value: str
    source: str
    experiment_id: str | None = None
    summary: str | None = None
    current_production: bool = False


class AnalysisModelSelector(BaseModel):
    key: str
    label: str
    current_value: str | None = None
    options: list[AnalysisModelOption]


class AnalysisModelOptionsResponse(BaseModel):
    detection: AnalysisModelSelector
    recognition: AnalysisModelSelector
    tracker: AnalysisModelSelector


class ImageRecognitionDetection(BaseModel):
    name: str
    type: CharacterType | Literal["unmatched"]
    similarity: float | None = None
    confidence: float | None = None
    bbox: list[int]
    candidates: list[dict[str, float | str]] = Field(default_factory=list)
    reason: str | None = None


class ImageRecognitionResponse(BaseModel):
    detections: list[ImageRecognitionDetection]
    annotated_image_base64: str
    image_width: int
    image_height: int
    model_selection: dict[str, str]
    diagnostics: dict = Field(default_factory=dict)


class VideoStatusResponse(BaseModel):
    video_id: int
    filename: str
    status: VideoStatus
    progress: float


class TimelineSegment(BaseModel):
    start: float
    end: float


class TimelineCharacter(BaseModel):
    type: CharacterType
    segments: list[TimelineSegment]


class TimelineVideoInfo(BaseModel):
    duration_sec: float | None = None
    fps: float | None = None
    model_selection: dict[str, str] | None = None


class TimelineResponse(BaseModel):
    video_id: int
    timeline: dict[str, TimelineCharacter]
    video_info: TimelineVideoInfo | None = None


class HistoryItem(BaseModel):
    id: int
    filename: str
    duration: float | None = None
    status: VideoStatus
    has_annotated_video: bool = False


class HealthResponse(BaseModel):
    name: str
    version: str
    status: str

class RenameVideoRequest(BaseModel):
    filename: str
