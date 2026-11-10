from sqlalchemy import Column, Enum, Float, ForeignKey, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    status = Column(Enum("pending", "processing", "completed", "failed"), default="pending")
    progress = Column(Float, default=0.0)
    duration = Column(Float)
    fps = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    appearances = relationship("Appearance", back_populates="video", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    type = Column(Enum("known", "stranger"), nullable=False)
    first_seen_video_id = Column(Integer, ForeignKey("videos.id", ondelete="SET NULL"))
    created_at = Column(TIMESTAMP, server_default=func.now())

    appearances = relationship("Appearance", back_populates="character", cascade="all, delete-orphan")


class Appearance(Base):
    __tablename__ = "appearances"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    character_id = Column(Integer, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    video = relationship("Video", back_populates="appearances")
    character = relationship("Character", back_populates="appearances")
