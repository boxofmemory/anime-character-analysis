from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.character import RenameStrangerRequest, RenameStrangerResponse
from app.services.character_service import CharacterService

router = APIRouter(tags=["characters"])


@router.post("/api/characters/rename-stranger", response_model=RenameStrangerResponse)
def rename_stranger(payload: RenameStrangerRequest, db: Session = Depends(get_db)):
    return CharacterService(db, payload.recognition_model_path).rename_stranger(payload.old_name, payload.new_name)


@router.post("/api/stranger/rename", response_model=RenameStrangerResponse)
def rename_stranger_legacy(payload: RenameStrangerRequest, db: Session = Depends(get_db)):
    return CharacterService(db, payload.recognition_model_path).rename_stranger(payload.old_name, payload.new_name)
