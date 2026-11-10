from sqlalchemy.orm import Session

from app.models import Appearance, Character, Video


class VideoRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_video(self, filename: str) -> Video:
        video = Video(filename=filename, status="pending", progress=0.0)
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_video(self, video_id: int) -> Video | None:
        return self.db.query(Video).filter(Video.id == video_id).first()

    def list_completed_videos(self, limit: int = 50) -> list[Video]:
        return (
            self.db.query(Video)
            .filter(Video.status == "completed")
            .order_by(Video.id.desc())
            .limit(limit)
            .all()
        )

    def list_video_appearances(self, video_id: int) -> list[Appearance]:
        return self.db.query(Appearance).filter(Appearance.video_id == video_id).all()

    def get_character_by_name(self, name: str) -> Character | None:
        return self.db.query(Character).filter(Character.name == name).first()

    def get_or_create_character(self, name: str, character_type: str, video_id: int) -> Character:
        character = self.get_character_by_name(name)
        if character:
            return character
        character = Character(name=name, type=character_type, first_seen_video_id=video_id)
        self.db.add(character)
        self.db.flush()
        return character

    def add_appearance(self, video_id: int, character_id: int, start_time: float, end_time: float) -> None:
        self.db.add(
            Appearance(
                video_id=video_id,
                character_id=character_id,
                start_time=start_time,
                end_time=end_time,
            )
        )

    def commit(self) -> None:
        self.db.commit()
