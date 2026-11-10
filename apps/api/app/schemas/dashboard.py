from __future__ import annotations

from pydantic import BaseModel


class DashboardStatItem(BaseModel):
    label: str
    value: str
    hint: str | None = None


class DashboardModelInfo(BaseModel):
    yolo_model: str
    recognition_model: str
    tracker_config: str


class DashboardTrendPoint(BaseModel):
    video_id: int
    filename: str
    label: str
    duration: float | None = None
    has_annotated_video: bool = False
    character_count: int
    total_segments: int


class DashboardRecentCharacter(BaseModel):
    name: str
    sample_count: int
    updated_at: str | None = None
    portrait_url: str | None = None


class DashboardRecentUpdate(BaseModel):
    title: str
    detail: str
    updated_at: str | None = None
    kind: str


class DashboardRecentVideo(BaseModel):
    id: int
    filename: str
    duration: float | None = None
    updated_at: str | None = None
    character_count: int
    has_annotated_video: bool = False


class DashboardInsights(BaseModel):
    avg_video_duration: float
    known_recognition_rate: float
    avg_characters_per_video: float
    total_appearances: int


class DashboardSummaryResponse(BaseModel):
    stats: list[DashboardStatItem]
    models: DashboardModelInfo
    trends: list[DashboardTrendPoint]
    recent_characters: list[DashboardRecentCharacter]
    recent_videos: list[DashboardRecentVideo]
    recent_updates: list[DashboardRecentUpdate]
    insights: DashboardInsights
