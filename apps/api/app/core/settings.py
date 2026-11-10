from __future__ import annotations

import os
import sys
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(*candidates: Path) -> None:
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


CURRENT_FILE = Path(__file__).resolve()
API_ROOT = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[4]

_load_dotenv(PROJECT_ROOT / ".env", API_ROOT / ".env")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w\-.]+", "_", value.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "model"


@dataclass(frozen=True)
class LibraryProfilePaths:
    profile_key: str
    recognition_model_path: Path
    root_dir: Path
    known_faces_dir: Path
    known_db_path: Path
    strangers_dir: Path
    stranger_db_path: Path
    manifest_path: Path
    sources_dir: Path
    cropped_dir: Path
    portraits_dir: Path
    imports_dir: Path

    def ensure_dirs(self) -> None:
        for path in (
            self.root_dir,
            self.known_faces_dir,
            self.strangers_dir,
            self.sources_dir,
            self.cropped_dir,
            self.portraits_dir,
            self.imports_dir,
            self.known_db_path.parent,
            self.stranger_db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    api_root: Path = API_ROOT
    app_name: str = os.getenv("APP_NAME", "Anime Character Analysis API")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:CHANGE_ME@localhost:3306/anime_tracker",
    )
    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "*")
    known_threshold: float = float(os.getenv("KNOWN_THRESHOLD", "0.50"))
    stranger_threshold: float = float(os.getenv("STRANGER_THRESHOLD", "0.55"))
    track_buffer: int = int(os.getenv("TRACK_BUFFER", "90"))
    min_track_frames: int = int(os.getenv("MIN_TRACK_FRAMES", "3"))
    max_gap_frames: int = int(os.getenv("MAX_GAP_FRAMES", "15"))
    min_duration_sec: float = float(os.getenv("MIN_DURATION_SEC", "0.5"))
    danbooru_username: str = os.getenv("DANBOORU_USERNAME", "")
    danbooru_api_key: str = os.getenv("DANBOORU_API_KEY", "")
    small_dataset_dir_raw: str = os.getenv("SMALL_DATASET_DIR", str(PROJECT_ROOT / "datasets" / "small_dataset_112x112"))
    arcface_dataset_dir_raw: str = os.getenv("ARCFACE_DATASET_DIR", str(PROJECT_ROOT / "datasets" / "arcface_dataset_112x112"))
    training_python_raw: str = os.getenv("TRAINING_PYTHON", sys.executable)

    @property
    def cors_origins(self) -> list[str]:
        if self.cors_origins_raw.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def model_catalog_dir(self) -> Path:
        return self.models_dir / "catalog"

    @property
    def catalog_detection_dir(self) -> Path:
        return self.model_catalog_dir / "detection"

    @property
    def catalog_recognition_dir(self) -> Path:
        return self.model_catalog_dir / "recognition"

    @property
    def catalog_tracking_dir(self) -> Path:
        return self.model_catalog_dir / "tracking"

    @property
    def detection_model_path(self) -> Path:
        return self.models_dir / "production" / "detection" / "yolov8s_anime.pt"

    @property
    def recognition_model_path(self) -> Path:
        return self.models_dir / "production" / "recognition" / "resnet50_anime_arcface_best_r4.pth"

    @property
    def tracker_yaml_path(self) -> Path:
        return self.models_dir / "production" / "tracking" / "bytetrack_anime.yaml"

    @property
    def archived_recognition_model_path(self) -> Path:
        return self.models_dir / "archive" / "recognition" / "resnet50_anime_arcface.pth"

    @property
    def known_db_path(self) -> Path:
        return self.data_dir / "libraries" / "known_faces" / "known_database.json"

    @property
    def stranger_db_path(self) -> Path:
        return self.data_dir / "libraries" / "strangers" / "stranger_database.json"

    @property
    def library_root_dir(self) -> Path:
        return self.data_dir / "libraries"

    @property
    def known_faces_dir(self) -> Path:
        return self.library_root_dir / "known_faces"

    @property
    def library_manifest_path(self) -> Path:
        return self.known_faces_dir / "library_manifest.json"

    @property
    def library_sources_dir(self) -> Path:
        return self.known_faces_dir / "source"

    @property
    def library_cropped_dir(self) -> Path:
        return self.known_faces_dir / "cropped"

    @property
    def library_portraits_dir(self) -> Path:
        return self.known_faces_dir / "portraits"

    @property
    def library_imports_dir(self) -> Path:
        return self.known_faces_dir / "imports"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "inputs" / "videos"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def annotated_dir(self) -> Path:
        return self.outputs_dir / "inference" / "annotated"

    @property
    def timelines_dir(self) -> Path:
        return self.outputs_dir / "inference" / "timelines"

    @property
    def experiment_exports_dir(self) -> Path:
        return self.outputs_dir / "experiments"

    @property
    def training_outputs_dir(self) -> Path:
        return self.outputs_dir / "training"

    @property
    def sample_video_path(self) -> Path:
        return self.data_dir / "inputs" / "sample_videos" / "demo.mp4"

    @property
    def timeline_json_path(self) -> Path:
        return self.timelines_dir / "timeline.json"

    @property
    def output_video_path(self) -> Path:
        return self.annotated_dir / "output_annotated.webm"

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "research" / "dataset"

    @property
    def training_root_dir(self) -> Path:
        return self.project_root / "research" / "training"

    @property
    def training_configs_dir(self) -> Path:
        return self.training_root_dir / "configs"

    @property
    def training_experiments_dir(self) -> Path:
        return self.training_root_dir / "experiments"

    @property
    def training_scripts_dir(self) -> Path:
        return self.training_root_dir / "scripts"

    @property
    def training_weights_dir(self) -> Path:
        return self.training_root_dir / "weights"

    @property
    def training_python_path(self) -> Path:
        return Path(self.training_python_raw)

    @property
    def training_production_manifest_path(self) -> Path:
        return self.models_dir / "manifests" / "training_production.json"

    @property
    def small_dataset_dir(self) -> Path:
        return Path(self.small_dataset_dir_raw)

    @property
    def arcface_dataset_dir(self) -> Path:
        return Path(self.arcface_dataset_dir_raw)

    @property
    def external_avatar_dataset_dirs(self) -> list[Path]:
        candidates = [self.small_dataset_dir, self.arcface_dataset_dir]
        return [path for path in candidates if path.exists()]

    def resolve_model_key_path(self, model_key: str | None, fallback: Path) -> Path:
        if not model_key:
            return fallback
        model_path = (self.models_dir / model_key).resolve()
        root = self.models_dir.resolve()
        if root not in model_path.parents and model_path != root:
            raise ValueError(f"Invalid model key path: {model_key}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model asset not found: {model_key}")
        return model_path

    def library_profile_paths(self, recognition_model_path: str | Path | None = None) -> LibraryProfilePaths:
        resolved = Path(recognition_model_path or self.recognition_model_path).resolve()
        default_resolved = self.recognition_model_path.resolve()
        profile_key = self._recognition_profile_key(resolved)
        profile_root_dir = self.library_root_dir / "profiles" / profile_key
        profile_known_faces_dir = profile_root_dir / "known_faces"
        profile_strangers_dir = profile_root_dir / "strangers"
        if resolved == default_resolved:
            if (
                profile_known_faces_dir.exists()
                and (
                    (profile_known_faces_dir / "known_database.json").exists()
                    or (profile_known_faces_dir / "library_manifest.json").exists()
                )
            ):
                return LibraryProfilePaths(
                    profile_key=profile_key,
                    recognition_model_path=resolved,
                    root_dir=profile_root_dir,
                    known_faces_dir=profile_known_faces_dir,
                    known_db_path=profile_known_faces_dir / "known_database.json",
                    strangers_dir=profile_strangers_dir,
                    stranger_db_path=profile_strangers_dir / "stranger_database.json",
                    manifest_path=profile_known_faces_dir / "library_manifest.json",
                    sources_dir=profile_known_faces_dir / "source",
                    cropped_dir=profile_known_faces_dir / "cropped",
                    portraits_dir=profile_known_faces_dir / "portraits",
                    imports_dir=profile_known_faces_dir / "imports",
                )
            return LibraryProfilePaths(
                profile_key=profile_key,
                recognition_model_path=resolved,
                root_dir=self.library_root_dir,
                known_faces_dir=self.known_faces_dir,
                known_db_path=self.known_db_path,
                strangers_dir=self.library_root_dir / "strangers",
                stranger_db_path=self.stranger_db_path,
                manifest_path=self.library_manifest_path,
                sources_dir=self.library_sources_dir,
                cropped_dir=self.library_cropped_dir,
                portraits_dir=self.library_portraits_dir,
                imports_dir=self.library_imports_dir,
            )

        return LibraryProfilePaths(
            profile_key=profile_key,
            recognition_model_path=resolved,
            root_dir=profile_root_dir,
            known_faces_dir=profile_known_faces_dir,
            known_db_path=profile_known_faces_dir / "known_database.json",
            strangers_dir=profile_strangers_dir,
            stranger_db_path=profile_strangers_dir / "stranger_database.json",
            manifest_path=profile_known_faces_dir / "library_manifest.json",
            sources_dir=profile_known_faces_dir / "source",
            cropped_dir=profile_known_faces_dir / "cropped",
            portraits_dir=profile_known_faces_dir / "portraits",
            imports_dir=profile_known_faces_dir / "imports",
        )

    def _recognition_profile_key(self, model_path: Path) -> str:
        slug = _slugify(model_path.stem)
        digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:8]
        return f"{slug}--{digest}"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.data_dir / "mysql",
            self.models_dir / "production" / "detection",
            self.models_dir / "production" / "recognition",
            self.models_dir / "production" / "tracking",
            self.catalog_detection_dir,
            self.catalog_recognition_dir,
            self.catalog_tracking_dir,
            self.models_dir / "archive" / "recognition",
            self.models_dir / "manifests",
            self.library_root_dir / "profiles",
            self.videos_dir,
            self.annotated_dir,
            self.timelines_dir,
            self.experiment_exports_dir,
            self.training_outputs_dir,
            self.known_db_path.parent,
            self.stranger_db_path.parent,
            self.library_sources_dir,
            self.library_cropped_dir,
            self.library_portraits_dir,
            self.library_imports_dir,
            self.data_dir / "inputs" / "sample_videos",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_runtime_dirs()


def validate_runtime_assets() -> None:
    required_assets = {
        "YOLO model": settings.detection_model_path,
        "Recognition model": settings.recognition_model_path,
        "ByteTrack config": settings.tracker_yaml_path,
    }
    missing = [f"{label}: {path}" for label, path in required_assets.items() if not path.exists()]
    if missing:
        raise RuntimeError("Missing required runtime assets:\n" + "\n".join(missing))
