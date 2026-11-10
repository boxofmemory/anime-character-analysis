from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.settings import settings
from video_demo import FaceRecognizer, ResNet50Extractor, process_video


def get_extractor(model_path: str | None = None) -> ResNet50Extractor:
    resolved_path = Path(model_path or settings.recognition_model_path).resolve()
    cache = getattr(get_extractor, "_instances", {})
    mtime = resolved_path.stat().st_mtime if resolved_path.exists() else 0.0
    cache_key = (str(resolved_path), mtime)
    if cache_key not in cache:
        cache[cache_key] = ResNet50Extractor(model_path=str(resolved_path))
        if len(cache) > 6:
            latest_keys = list(cache.keys())[-6:]
            cache = {key: cache[key] for key in latest_keys}
    get_extractor._instances = cache
    return cache[cache_key]


def build_yolo_model(model_path: str | None = None) -> YOLO:
    resolved_path = Path(model_path or settings.detection_model_path).resolve()
    return YOLO(str(resolved_path))


def build_recognizer(recognition_model_path: str | None = None) -> FaceRecognizer:
    library_paths = settings.library_profile_paths(recognition_model_path)
    library_paths.ensure_dirs()
    return FaceRecognizer(
        known_db_path=str(library_paths.known_db_path),
        stranger_db_path=str(library_paths.stranger_db_path),
    )


def _ensure_recognition_compatibility(extractor: ResNet50Extractor, recognizer: FaceRecognizer) -> None:
    database_dim = recognizer.get_embedding_dim()
    if database_dim is not None and database_dim != extractor.output_dim:
        raise ValueError(
            "Recognition model output dimension "
            f"({extractor.output_dim}) does not match current face library vectors ({database_dim}). "
            "Please rebuild the known face database and related feature libraries with the selected model."
        )


def _rank_known_candidates(recognizer: FaceRecognizer, embedding: np.ndarray, limit: int = 5) -> list[dict]:
    candidates = []
    for name, data in recognizer.known_db.items():
        vector_data = data.get("mean", data.get("vector")) if isinstance(data, dict) else data
        if vector_data is None:
            continue
        db_vec = np.array(vector_data, dtype=np.float32)
        candidates.append(
            {
                "name": name,
                "similarity": round(float(np.dot(embedding, db_vec)), 4),
            }
        )
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    return candidates[:limit]


def run_video_pipeline(
    file_path: str,
    progress_callback=None,
    output_path: str | None = None,
    timeline_output_path: str | None = None,
    detection_model_path: str | None = None,
    recognition_model_path: str | None = None,
    tracker_config_path: str | None = None,
) -> dict:
    detection_path = str(Path(detection_model_path or settings.detection_model_path).resolve())
    recognition_path = str(Path(recognition_model_path or settings.recognition_model_path).resolve())
    tracker_path = str(Path(tracker_config_path or settings.tracker_yaml_path).resolve())
    library_paths = settings.library_profile_paths(recognition_path)
    library_paths.ensure_dirs()
    extractor = get_extractor(recognition_path)
    recognizer = build_recognizer(recognition_path)
    _ensure_recognition_compatibility(extractor, recognizer)
    return process_video(
        video_path=file_path,
        extractor=extractor,
        recognizer=recognizer,
        yolo_model=build_yolo_model(detection_path),
        output_path=output_path,
        timeline_output_path=timeline_output_path,
        progress_callback=progress_callback,
        tracker_yaml_path=tracker_path,
        stranger_db_path=str(library_paths.stranger_db_path),
        model_info={
            "detection_model": Path(detection_path).name,
            "detection_model_path": detection_path,
            "recognition_model": Path(recognition_path).name,
            "recognition_model_path": recognition_path,
            "tracker_config": Path(tracker_path).name,
            "tracker_config_path": tracker_path,
            "recognition_library_profile": library_paths.profile_key,
        },
    )


def run_image_pipeline(
    image_bytes: bytes,
    detection_model_path: str | None = None,
    recognition_model_path: str | None = None,
) -> dict:
    detection_path = str(Path(detection_model_path or settings.detection_model_path).resolve())
    recognition_path = str(Path(recognition_model_path or settings.recognition_model_path).resolve())
    library_paths = settings.library_profile_paths(recognition_path)
    library_paths.ensure_dirs()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")

    recognizer = build_recognizer(recognition_path)
    extractor = get_extractor(recognition_path)
    _ensure_recognition_compatibility(extractor, recognizer)
    yolo_model = build_yolo_model(detection_path)

    detections = []
    results = yolo_model(image, verbose=False, conf=0.15)
    if results and len(results) > 0 and results[0].boxes is not None:
        height, width = image.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            confidence = float(box.conf.item()) if box.conf is not None else None
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (112, 112))
            embedding = extractor.extract(face_resized)
            if embedding is None:
                continue

            candidates = _rank_known_candidates(recognizer, embedding)
            name, similarity, match_type = recognizer.query(embedding)
            reason = None
            if match_type == "new":
                display_name = "\u672a\u5339\u914d\u89d2\u8272"
                draw_name = "Unmatched"
                display_type = "unmatched"
                color = (180, 180, 180)
                if candidates:
                    reason = (
                        f"\u6700\u9ad8\u5019\u9009 {candidates[0]['name']} "
                        f"\u76f8\u4f3c\u5ea6 {candidates[0]['similarity']:.2f} "
                        f"\u4f4e\u4e8e\u9608\u503c {recognizer.KNOWN_THRESH:.2f}"
                    )
                else:
                    reason = "\u5f53\u524d\u89d2\u8272\u5e93\u6ca1\u6709\u53ef\u7528\u7684\u5df2\u77e5\u89d2\u8272\u7279\u5f81"
            elif match_type == "known":
                display_name = name
                draw_name = name
                display_type = "known"
                color = (0, 200, 0)
            else:
                display_name = name
                draw_name = name
                display_type = "stranger"
                color = (0, 200, 255)

            label = draw_name if similarity is None else f"{draw_name} ({similarity:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(y1 - text_h - 10, 0)
            cv2.rectangle(image, (x1, top), (x1 + text_w + 8, top + text_h + 8), color, -1)
            cv2.putText(image, label, (x1 + 4, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detections.append(
                {
                    "name": display_name,
                    "type": display_type,
                    "similarity": round(similarity, 4) if similarity is not None else None,
                    "confidence": round(confidence, 4) if confidence is not None else None,
                    "bbox": [x1, y1, x2, y2],
                    "candidates": candidates,
                    "reason": reason,
                }
            )

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Unable to encode annotated image")

    return {
        "detections": detections,
        "annotated_image_base64": base64.b64encode(encoded.tobytes()).decode("utf-8"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model_selection": {
            "detection_model": Path(detection_path).name,
            "detection_model_path": detection_path,
            "recognition_model": Path(recognition_path).name,
            "recognition_model_path": recognition_path,
            "recognition_library_profile": library_paths.profile_key,
        },
        "diagnostics": {
            "known_count": len(recognizer.known_db),
            "stranger_count": len(recognizer.stranger_db),
            "known_threshold": recognizer.KNOWN_THRESH,
            "stranger_threshold": recognizer.STRANGER_THRESH,
            "recognition_library_profile": library_paths.profile_key,
            "known_db_path": str(library_paths.known_db_path),
            "known_db_exists": library_paths.known_db_path.exists(),
        },
    }
