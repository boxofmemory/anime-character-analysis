from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.training import (
    TrainingActionResponse,
    TrainingJobCreateRequest,
    TrainingJobDetail,
    TrainingJobListResponse,
    TrainingLogResponse,
    TrainingOptionsResponse,
)
from app.services.training_service import TrainingService


router = APIRouter(tags=["training"])


@router.get("/api/training/options", response_model=TrainingOptionsResponse)
def get_training_options():
    return TrainingService().get_options()


@router.post("/api/training/jobs", response_model=TrainingJobDetail)
def create_training_job(payload: TrainingJobCreateRequest):
    return TrainingService().create_job(payload)


@router.get("/api/training/jobs", response_model=TrainingJobListResponse)
def list_training_jobs():
    return TrainingJobListResponse(jobs=TrainingService().list_jobs())


@router.get("/api/training/jobs/{job_id}", response_model=TrainingJobDetail)
def get_training_job(job_id: str):
    return TrainingService().get_job(job_id)


@router.get("/api/training/jobs/{job_id}/log", response_model=TrainingLogResponse)
def get_training_log(job_id: str):
    return TrainingService().get_log(job_id)


@router.post("/api/training/jobs/{job_id}/cancel", response_model=TrainingActionResponse)
def cancel_training_job(job_id: str):
    return TrainingService().cancel_job(job_id)


@router.post("/api/training/jobs/{job_id}/promote", response_model=TrainingActionResponse)
def promote_training_job(job_id: str):
    return TrainingService().promote_job(job_id)


@router.get("/api/training/jobs/{job_id}/artifacts/{artifact_path:path}")
def download_training_artifact(job_id: str, artifact_path: str):
    path = TrainingService().get_artifact_path(job_id, artifact_path)
    return FileResponse(path)
