"""
Video analysis pipeline used by the formal API.

This version keeps the demo algorithm approachable while
fixing a few issues that would be obvious during a live demo:

- timeline frames are collected by track id and mapped to the final identity
- preview recognition does not pollute stranger prototypes
- stranger DB is loaded and merged instead of being overwritten each run
- timeline output can be written per-video to avoid concurrent overwrite
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from config import (
    DATASET_DIR,
    KNOWN_DB_PATH,
    KNOWN_THRESH,
    MAX_GAP_FRAMES,
    MIN_DURATION_SEC,
    MIN_TRACK_FRAMES,
    MODEL_PATH,
    OUTPUT_VIDEO_PATH,
    STRANGER_DB_PATH,
    STRANGER_THRESH,
    TIMELINE_JSON_PATH,
    TRACK_BUFFER,
    TRACKER_YAML,
    VIDEO_PATH,
    YOLO_MODEL_PATH,
)


_STRANGER_DB_LOCK = threading.Lock()


class ResNet50Extractor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self._build_backbone()
        self.bn = nn.BatchNorm1d(2048)
        self.embedding_head = None
        self.output_dim = 2048
        if model_path and Path(model_path).exists():
            self._load_finetuned_weights(model_path)
        else:
            print(f"Recognition weights not found, using ImageNet backbone: {model_path}")
        self.backbone.to(self.device)
        self.bn.to(self.device)
        if self.embedding_head is not None:
            self.embedding_head.to(self.device)
        self.backbone.eval()
        self.bn.eval()
        if self.embedding_head is not None:
            self.embedding_head.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_backbone(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(resnet.children())[:-1])

    def _load_finetuned_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        backbone_state, bn_state, embedding_state = self._resolve_checkpoint_states(checkpoint)
        self.backbone.load_state_dict(backbone_state, strict=False)
        if embedding_state:
            out_features = embedding_state["1.weight"].shape[0]
            self.embedding_head = nn.Sequential(
                nn.Identity(),
                nn.Linear(2048, out_features, bias=False),
                nn.BatchNorm1d(out_features),
            )
            self.embedding_head.load_state_dict(embedding_state, strict=False)
            self.output_dim = out_features
        elif bn_state:
            self.bn.load_state_dict(bn_state, strict=False)

    def _resolve_checkpoint_states(self, checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint, None, None

        if "backbone" in checkpoint:
            return checkpoint["backbone"], checkpoint.get("bn"), None

        if "model_state" in checkpoint:
            return self._split_model_state(checkpoint["model_state"])

        if "state_dict" in checkpoint:
            return self._split_model_state(checkpoint["state_dict"])

        return self._split_model_state(checkpoint)

    def _split_model_state(self, state_dict):
        if not isinstance(state_dict, dict):
            return state_dict, None, None

        normalized = {}
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith("module.") else key
            normalized[clean_key] = value

        if any(key.startswith("resnet.") for key in normalized):
            backbone_state = {}
            bn_state = {}
            embedding_state = {}
            for key, value in normalized.items():
                if key.startswith("resnet."):
                    mapped_key = self._map_resnet_key(key[len("resnet."):])
                    if mapped_key:
                        backbone_state[mapped_key] = value
                elif key.startswith("backbone."):
                    backbone_state[key[len("backbone."):]] = value
                elif key.startswith("bn."):
                    bn_state[key[len("bn."):]] = value
                elif key.startswith("embedding."):
                    embedding_state[key[len("embedding."):]] = value
            return backbone_state, bn_state or None, embedding_state or None

        if any(key.startswith("backbone.") for key in normalized):
            backbone_state = {
                key[len("backbone."):]: value
                for key, value in normalized.items()
                if key.startswith("backbone.")
            }
            bn_state = {
                key[len("bn."):]: value
                for key, value in normalized.items()
                if key.startswith("bn.")
            }
            embedding_state = {
                key[len("embedding."):]: value
                for key, value in normalized.items()
                if key.startswith("embedding.")
            }
            return backbone_state, bn_state or None, embedding_state or None

        return normalized, None, None

    def _map_resnet_key(self, key):
        if key == "conv1.weight":
            return "0.weight"
        if key.startswith("bn1."):
            return f"1.{key[4:]}"
        if key.startswith("layer1."):
            return f"4.{key[7:]}"
        if key.startswith("layer2."):
            return f"5.{key[7:]}"
        if key.startswith("layer3."):
            return f"6.{key[7:]}"
        if key.startswith("layer4."):
            return f"7.{key[7:]}"
        return None

    @torch.no_grad()
    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            feat = self.backbone(tensor).view(-1).unsqueeze(0)
            if self.embedding_head is not None:
                feat = self.embedding_head(feat)
            else:
                feat = self.bn(feat)
            feat = feat / torch.norm(feat, dim=1, keepdim=True)
            return feat.cpu().numpy().flatten()
        except Exception:
            return None


class FaceRecognizer:
    def __init__(self, known_db_path=None, stranger_db_path=None):
        self.known_db = {}
        self.stranger_db = {}
        self.stranger_counter = 1
        self.KNOWN_THRESH = KNOWN_THRESH
        self.STRANGER_THRESH = STRANGER_THRESH
        if known_db_path and Path(known_db_path).exists():
            self._load_known_db(known_db_path)
        if stranger_db_path and Path(stranger_db_path).exists():
            self._load_stranger_db(stranger_db_path)

    def _load_known_db(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.known_db = json.load(f)

    def _load_stranger_db(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.stranger_db = json.load(f)
        existing_ids = []
        for name in self.stranger_db:
            if name.startswith("Stranger_"):
                suffix = name.split("_", 1)[1]
                if suffix.isdigit():
                    existing_ids.append(int(suffix))
        self.stranger_counter = max(existing_ids, default=0) + 1

    def save_stranger_db(self, path):
        with _STRANGER_DB_LOCK:
            merged = {}
            path_obj = Path(path)
            if path_obj.exists():
                with path_obj.open("r", encoding="utf-8") as f:
                    merged = json.load(f)
            merged.update(self.stranger_db)
            with path_obj.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

    def query(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        best_name, best_sim = self._find_best_match(embedding, self.known_db)
        if best_name is not None and best_sim >= self.KNOWN_THRESH:
            return best_name, best_sim, "known"

        best_sid, best_ssim = self._find_best_match(embedding, self.stranger_db)
        if best_sid is not None and best_ssim >= self.STRANGER_THRESH:
            return best_sid, best_ssim, "stranger"

        return "new", 0.0, "new"

    def finalize_track(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        best_name, best_sim = self._find_best_match(embedding, self.known_db)
        if best_name is not None and best_sim >= self.KNOWN_THRESH:
            return best_name, best_sim, "known"

        best_sid, best_ssim = self._find_best_match(embedding, self.stranger_db)
        if best_sid is not None and best_ssim >= self.STRANGER_THRESH:
            old_vec = np.array(self.stranger_db[best_sid]["mean"], dtype=np.float32)
            count = self.stranger_db[best_sid]["count"]
            new_vec = (old_vec * count + embedding) / (count + 1)
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            self.stranger_db[best_sid]["mean"] = new_vec.tolist()
            self.stranger_db[best_sid]["count"] += 1
            return best_sid, best_ssim, "stranger"

        new_id = f"Stranger_{self.stranger_counter}"
        self.stranger_db[new_id] = {"mean": embedding.tolist(), "count": 1}
        self.stranger_counter += 1
        return new_id, 0.0, "new"

    def _find_best_match(self, query_vec: np.ndarray, database: dict) -> Tuple[Optional[str], float]:
        if not database:
            return None, -1.0
        best_name, best_sim = None, -1.0
        for name, data in database.items():
            vector_data = data.get("mean", data.get("vector")) if isinstance(data, dict) else data
            if vector_data is None:
                continue
            db_vec = np.array(vector_data, dtype=np.float32)
            sim = float(np.dot(query_vec, db_vec))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim

    def get_stats(self):
        return {
            "known_count": len(self.known_db),
            "stranger_count": len(self.stranger_db),
            "stranger_details": {k: v["count"] for k, v in self.stranger_db.items()},
        }

    def get_embedding_dim(self):
        for database in (self.known_db, self.stranger_db):
            for data in database.values():
                vector_data = data.get("mean", data.get("vector")) if isinstance(data, dict) else data
                if vector_data:
                    return len(vector_data)
        return None


def build_known_database(extractor: ResNet50Extractor, dataset_dir: str, output_path: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return {}
    known_db = {}
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    for folder in folders:
        char_name = folder.name
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        if len(images) < 3:
            continue
        vectors = []
        for img_path in images[:50]:
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            emb = extractor.extract(img)
            if emb is not None:
                vectors.append(emb)
        if len(vectors) >= 3:
            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec)
            known_db[char_name] = {"mean": mean_vec.tolist(), "count": len(vectors)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(known_db, f, ensure_ascii=False, indent=2)
    return known_db


class TrackFeatureAggregator:
    def __init__(self, buffer=90, min_frames=3, preview_frames=5):
        self.buffer = buffer
        self.min_frames = min_frames
        self.preview_frames = preview_frames
        self.tracks = {}
        self.max_features = 50

    def update(self, track_id, frame_idx, embedding):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "features": [],
                "last_frame": -999,
                "result": None,
                "previewed": False,
                "last_preview_frame": 0,
            }
        track = self.tracks[track_id]
        track["features"].append(embedding)
        track["last_frame"] = frame_idx
        if len(track["features"]) > self.max_features:
            track["features"] = track["features"][-self.max_features :]

        finished_results = []
        preview_result = None

        features_count = len(track["features"])
        if features_count >= self.preview_frames:
            if not track["previewed"] or (frame_idx - track["last_preview_frame"]) >= 10:
                avg_feat = np.mean(track["features"][-self.preview_frames :], axis=0)
                avg_feat = avg_feat / np.linalg.norm(avg_feat)
                preview_result = (track_id, avg_feat, False)
                track["previewed"] = True
                track["last_preview_frame"] = frame_idx

        finished = []
        for tid, tracked in self.tracks.items():
            if frame_idx - tracked["last_frame"] > self.buffer and tracked["result"] is None:
                if len(tracked["features"]) >= self.min_frames:
                    avg_feat = np.mean(tracked["features"], axis=0)
                    avg_feat = avg_feat / np.linalg.norm(avg_feat)
                    tracked["result"] = (tid, avg_feat)
                finished.append(tid)
        for tid in finished:
            if self.tracks[tid]["result"] is not None:
                res_tid, res_feat = self.tracks[tid]["result"]
                finished_results.append((res_tid, res_feat, True))
            del self.tracks[tid]
        return finished_results, preview_result

    def force_flush(self):
        results = []
        for tid, tracked in list(self.tracks.items()):
            if len(tracked["features"]) >= self.min_frames:
                avg_feat = np.mean(tracked["features"], axis=0)
                avg_feat = avg_feat / np.linalg.norm(avg_feat)
                results.append((tid, avg_feat, True))
        self.tracks.clear()
        return results


def _frames_to_segments(frames: list[int], fps: float, max_gap_frames: int) -> List[dict]:
    if not frames:
        return []
    segments = []
    start = frames[0]
    prev = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx - prev <= max_gap_frames:
            prev = frame_idx
            continue
        segments.append({"start": round(start / fps, 2), "end": round(prev / fps, 2)})
        start = frame_idx
        prev = frame_idx
    segments.append({"start": round(start / fps, 2), "end": round(prev / fps, 2)})
    return segments


def generate_timeline_with_stats(records: Dict[str, set], fps: float) -> Tuple[Dict[str, List[dict]], dict]:
    result = {}
    by_character = {}
    totals = {
        "before_smoothing_segments": 0,
        "after_smoothing_segments": 0,
        "after_filter_segments": 0,
        "short_filtered_segments": 0,
    }
    for name, frame_set in records.items():
        frames = sorted(frame_set)
        if not frames:
            continue
        raw_segments = _frames_to_segments(frames, fps, max_gap_frames=1)
        smoothed_segments = _frames_to_segments(frames, fps, max_gap_frames=MAX_GAP_FRAMES)
        filtered = [seg for seg in smoothed_segments if (seg["end"] - seg["start"]) >= MIN_DURATION_SEC]
        short_filtered = len(smoothed_segments) - len(filtered)

        totals["before_smoothing_segments"] += len(raw_segments)
        totals["after_smoothing_segments"] += len(smoothed_segments)
        totals["after_filter_segments"] += len(filtered)
        totals["short_filtered_segments"] += short_filtered
        by_character[name] = {
            "before_smoothing_segments": len(raw_segments),
            "after_smoothing_segments": len(smoothed_segments),
            "after_filter_segments": len(filtered),
            "short_filtered_segments": short_filtered,
        }
        if filtered:
            result[name] = filtered

    before_count = totals["before_smoothing_segments"]
    after_count = totals["after_filter_segments"]
    totals["fragment_reduction_rate"] = round((before_count - after_count) / before_count, 4) if before_count else 0.0
    totals["by_character"] = by_character
    return result, totals


def generate_timeline(records: Dict[str, set], fps: float) -> Dict[str, List[dict]]:
    result, _ = generate_timeline_with_stats(records, fps)
    return result


def _finalize_timeline_by_track(track_frames: Dict[int, set], track_labels: Dict[int, tuple], fps: float):
    named_records: Dict[str, set] = {}
    for tid, frame_set in track_frames.items():
        if tid not in track_labels:
            continue
        final_name, _, match_type = track_labels[tid]
        if match_type == "tracking":
            continue
        named_records.setdefault(final_name, set()).update(frame_set)
    timeline, smoothing_stats = generate_timeline_with_stats(named_records, fps)
    return timeline, named_records, smoothing_stats


def process_video(
    video_path,
    extractor,
    recognizer,
    yolo_model,
    output_path=None,
    timeline_output_path=None,
    progress_callback=None,
    tracker_yaml_path=None,
    stranger_db_path=None,
    model_info=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_w, out_h = w // 2, h // 2
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (out_w, out_h))

    aggregator = TrackFeatureAggregator(buffer=TRACK_BUFFER, min_frames=MIN_TRACK_FRAMES)
    track_labels = {}
    track_frames: Dict[int, set] = {}
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.15,
            iou=0.6,
            tracker=tracker_yaml_path or TRACKER_YAML,
        )
        valid_data = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    tid = int(box.id.item())
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                        valid_data.append(((x1, y1, x2, y2), tid))

        for (x1, y1, x2, y2), tid in valid_data:
            track_frames.setdefault(tid, set()).add(frame_idx)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_resized = cv2.resize(face, (112, 112))
            embedding = extractor.extract(face_resized)
            if embedding is None:
                continue

            finished_tracks, preview_track = aggregator.update(tid, frame_idx, embedding)
            if preview_track is not None:
                p_tid, p_emb, _ = preview_track
                name, sim, match_type = recognizer.query(p_emb)
                preview_name = f"Preview_{p_tid}" if match_type == "new" else name
                preview_type = "stranger" if match_type == "new" else match_type
                track_labels[p_tid] = (preview_name, sim, preview_type)

            for f_tid, f_emb, _ in finished_tracks:
                name, sim, match_type = recognizer.finalize_track(f_emb)
                track_labels[f_tid] = (name, sim, match_type)

            if tid in track_labels:
                name, sim, match_type = track_labels[tid]
            else:
                name, sim, match_type = f"ID_{tid}", 0.0, "tracking"

            if match_type == "known":
                color, label = (0, 200, 0), f"{name} ({sim:.2f})"
            elif match_type in {"stranger", "new"}:
                color, label = (0, 200, 255), f"{name} ({sim:.2f})"
            else:
                color, label = (200, 200, 200), "Tracking..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        stats = recognizer.get_stats()
        status = (
            f"Frame: {frame_idx}/{total_frames} | "
            f"Known: {stats['known_count']} | "
            f"Strangers: {stats['stranger_count']} | "
            f"Tracks: {len(aggregator.tracks)}"
        )
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer:
            frame_out = cv2.resize(frame, (out_w, out_h))
            writer.write(frame_out)

        if frame_idx % 30 == 0 and progress_callback:
            percent = round((frame_idx / total_frames) * 100, 2)
            progress_callback(percent)

    remaining = aggregator.force_flush()
    for f_tid, f_emb, _ in remaining:
        name, sim, match_type = recognizer.finalize_track(f_emb)
        track_labels[f_tid] = (name, sim, match_type)

    cap.release()
    if writer:
        writer.release()

    timeline_data, named_records, smoothing_stats = _finalize_timeline_by_track(track_frames, track_labels, fps)
    stats = recognizer.get_stats()
    known_names = [name for name in timeline_data if not name.startswith("Stranger_")]
    stranger_names = [name for name in timeline_data if name.startswith("Stranger_")]
    timeline_output = {
        "video_info": {
            "filename": Path(video_path).name,
            "fps": fps,
            "duration_sec": round(total_frames / fps, 2),
            "model_selection": model_info or {},
        },
        "characters": {},
        "raw_stats": {
            "total_frames_recorded": sum(len(v) for v in named_records.values()),
            "unique_names_raw": len(track_frames),
            "unique_names_after_denoise": len(timeline_data),
        },
        "experiment_stats": {
            "smoothing_ablation": {
                "video": Path(video_path).name,
                **smoothing_stats,
            },
            "open_set_case": {
                "known_character_count": len(known_names),
                "stranger_cluster_count": len(stranger_names),
                "known_characters": known_names,
                "stranger_clusters": {
                    name: {
                        "appearance_segments": len(timeline_data.get(name, [])),
                        "prototype_count": stats.get("stranger_details", {}).get(name, 0),
                    }
                    for name in stranger_names
                },
                "manual_verified_correct_stranger_clusters": None,
                "wrong_merged_clusters": None,
                "wrong_split_clusters": None,
            },
        },
    }

    for name, segments in timeline_data.items():
        total_dur = sum(seg["end"] - seg["start"] for seg in segments)
        timeline_output["characters"][name] = {
            "type": "known" if not name.startswith("Stranger_") else "stranger",
            "appearances": segments,
            "total_duration_sec": round(total_dur, 2),
        }

    if timeline_output_path:
        os.makedirs(os.path.dirname(timeline_output_path), exist_ok=True)
        with open(timeline_output_path, "w", encoding="utf-8") as f:
            json.dump(timeline_output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s")
    if stats["stranger_count"] > 0:
        recognizer.save_stranger_db(stranger_db_path or STRANGER_DB_PATH)
    return timeline_output


def main():
    print("Formal video pipeline")
    extractor = ResNet50Extractor(model_path=MODEL_PATH)

    if not Path(KNOWN_DB_PATH).exists():
        build_known_database(extractor, DATASET_DIR, KNOWN_DB_PATH)

    recognizer = FaceRecognizer(
        known_db_path=KNOWN_DB_PATH,
        stranger_db_path=STRANGER_DB_PATH,
    )

    from ultralytics import YOLO

    yolo_model = YOLO(YOLO_MODEL_PATH)
    if not Path(VIDEO_PATH).exists():
        print(f"Video not found: {VIDEO_PATH}")
        return
    process_video(
        VIDEO_PATH,
        extractor,
        recognizer,
        yolo_model,
        OUTPUT_VIDEO_PATH,
        TIMELINE_JSON_PATH,
        stranger_db_path=STRANGER_DB_PATH,
    )


if __name__ == "__main__":
    main()
