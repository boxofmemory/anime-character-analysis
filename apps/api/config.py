from app.core.settings import settings

BASE_DIR = str(settings.api_root)
MODEL_PATH = str(settings.recognition_model_path)
YOLO_MODEL_PATH = str(settings.detection_model_path)
TRACKER_YAML = str(settings.tracker_yaml_path)
DATASET_DIR = str(settings.dataset_dir)
KNOWN_DB_PATH = str(settings.known_db_path)
STRANGER_DB_PATH = str(settings.stranger_db_path)
VIDEO_PATH = str(settings.sample_video_path)
OUTPUT_DIR = str(settings.outputs_dir)
ANNOTATED_DIR = str(settings.annotated_dir)
VIDEOS_DIR = str(settings.videos_dir)
OUTPUT_VIDEO_PATH = str(settings.output_video_path)
TIMELINE_JSON_PATH = str(settings.timeline_json_path)
KNOWN_THRESH = settings.known_threshold
STRANGER_THRESH = settings.stranger_threshold
TRACK_BUFFER = settings.track_buffer
MIN_TRACK_FRAMES = settings.min_track_frames
MAX_GAP_FRAMES = settings.max_gap_frames
MIN_DURATION_SEC = settings.min_duration_sec
DATABASE_URL = settings.database_url
