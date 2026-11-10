from __future__ import annotations

from fastapi import APIRouter

from app.schemas.model_showcase import ModelShowcaseResponse
from app.services.model_showcase_service import ModelShowcaseService

router = APIRouter(tags=["model-showcase"])


@router.get("/api/model-showcase", response_model=ModelShowcaseResponse)
def get_model_showcase(
    detection_model_path: str | None = None,
    recognition_model_path: str | None = None,
    tracker_config_path: str | None = None,
):
    return ModelShowcaseService().get_showcase(
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path,
        tracker_config_path=tracker_config_path,
    )
