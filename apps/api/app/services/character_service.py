from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.models import Appearance, Character


_CHARACTER_FILE_LOCK = threading.Lock()


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class CharacterService:
    def __init__(self, db: Session, recognition_model_path: str | None = None):
        self.db = db
        self.library_paths = settings.library_profile_paths(recognition_model_path)
        self.library_paths.ensure_dirs()

    def rename_stranger(self, old_name: str, new_name: str) -> dict:
        if not old_name.startswith("Stranger_"):
            raise HTTPException(status_code=400, detail="Only Stranger_* entries can be promoted.")

        with _CHARACTER_FILE_LOCK:
            stranger_db = _load_json(self.library_paths.stranger_db_path, {})
            known_db = _load_json(self.library_paths.known_db_path, {})

            if old_name not in stranger_db:
                raise HTTPException(status_code=404, detail=f"Stranger library entry not found: {old_name}")

            stranger_data = stranger_db.pop(old_name)
            feature_list = (
                stranger_data.get("mean", stranger_data)
                if isinstance(stranger_data, dict)
                else stranger_data
            )

            created = new_name not in known_db
            fused = not created

            if fused:
                existing = known_db[new_name]
                existing_features = (
                    existing.get("mean", existing) if isinstance(existing, dict) else existing
                )
                old_vec = np.array(existing_features, dtype=np.float32)
                new_vec = np.array(feature_list, dtype=np.float32)
                merged = (old_vec + new_vec) / 2.0
                norm = np.linalg.norm(merged)
                if norm > 0:
                    merged = merged / norm
                if isinstance(existing, dict):
                    existing["mean"] = merged.tolist()
                    existing["count"] = existing.get("count", 1) + 1
                else:
                    known_db[new_name] = merged.tolist()
            else:
                known_db[new_name] = (
                    stranger_data
                    if isinstance(stranger_data, dict)
                    else {"mean": feature_list, "count": 1}
                )

            _save_json(self.library_paths.stranger_db_path, stranger_db)
            _save_json(self.library_paths.known_db_path, known_db)

        self._sync_database(old_name=old_name, new_name=new_name, fused=fused)

        action = "merged into" if fused else "promoted to"
        return {
            "status": "success",
            "message": f"{old_name} has been {action} {new_name}",
            "created": created,
            "fused": fused,
        }

    def _sync_database(self, old_name: str, new_name: str, fused: bool) -> None:
        old_character = self.db.query(Character).filter(Character.name == old_name).first()
        if not old_character:
            return

        target_character = self.db.query(Character).filter(Character.name == new_name).first()

        if fused and target_character:
            self.db.query(Appearance).filter(Appearance.character_id == old_character.id).update(
                {"character_id": target_character.id},
                synchronize_session=False,
            )
            self.db.delete(old_character)
        else:
            old_character.name = new_name
            old_character.type = "known"

        self.db.commit()
