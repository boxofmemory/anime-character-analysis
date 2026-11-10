from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.video import (
    AnalysisModelOptionsResponse,
    HealthResponse,
    HistoryItem,
    ImageRecognitionResponse,
    TimelineResponse,
    UploadVideoResponse,
    VideoStatusResponse,
    RenameVideoRequest,
)
from app.services.video_service import VideoService

router = APIRouter(tags=["videos"])


@router.get("/", response_model=HealthResponse)
def read_root():
    return {"name": "Anime Character Analysis API", "version": "1.0.0", "status": "ok"}


@router.post("/api/upload", response_model=UploadVideoResponse)
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_model_path: str | None = Form(default=None),
    recognition_model_path: str | None = Form(default=None),
    tracker_config_path: str | None = Form(default=None),
    data_export_enabled: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    service = VideoService(db)
    video, save_path = service.save_upload(file)
    service.enqueue_processing(
        background_tasks,
        video.id,
        save_path,
        {
            "detection_model_path": detection_model_path,
            "recognition_model_path": recognition_model_path,
            "tracker_config_path": tracker_config_path,
            "data_export_enabled": data_export_enabled,
        },
    )
    return {
        "message": "视频上传成功，后台正在分析",
        "video_id": video.id,
        "status": "pending",
    }


@router.get("/api/analysis/model-options", response_model=AnalysisModelOptionsResponse)
def get_analysis_model_options(db: Session = Depends(get_db)):
    return VideoService(db).get_analysis_model_options()


@router.post("/api/image-recognition", response_model=ImageRecognitionResponse)
def recognize_single_image(
    file: UploadFile = File(...),
    detection_model_path: str | None = Form(default=None),
    recognition_model_path: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    image_bytes = file.file.read()
    return VideoService(db).recognize_image(
        image_bytes,
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path,
    )


@router.get("/api/videos/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: int, db: Session = Depends(get_db)):
    video = VideoService(db).get_status(video_id)
    return {
        "video_id": video.id,
        "filename": video.filename,
        "status": video.status,
        "progress": video.progress,
    }


@router.get("/api/videos/{video_id}/timeline", response_model=TimelineResponse)
def get_video_timeline(video_id: int, db: Session = Depends(get_db)):
    return VideoService(db).get_timeline(video_id)


@router.get("/api/videos/{video_id}/download")
def download_annotated_video(video_id: int, db: Session = Depends(get_db)):
    output_path = VideoService(db).get_download_path(video_id)
    return FileResponse(str(output_path), media_type="video/webm")


@router.get("/api/history", response_model=list[HistoryItem])
def get_video_history(db: Session = Depends(get_db)):
    return VideoService(db).get_history()


@router.put("/api/videos/{video_id}/rename")
def rename_video(video_id: int, req: RenameVideoRequest, db: Session = Depends(get_db)):
    VideoService(db).rename_video(video_id, req.filename)
    return {"message": "重命名成功"}


@router.delete("/api/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    VideoService(db).delete_video(video_id)
    return {"message": "删除成功"}
