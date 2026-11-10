from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = PROJECT_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.settings import settings  # noqa: E402
from app.services.video_pipeline import build_yolo_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 3-4 real face crops from one ByteTrack track as a PPT-ready image sheet."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", default="", help="Output sheet image path.")
    parser.add_argument("--track-id", type=int, default=None, help="Specific track id to export. Auto-pick if omitted.")
    parser.add_argument("--count", type=int, default=4, help="Number of crops in the sheet.")
    parser.add_argument("--scan-frames", type=int, default=900, help="Maximum frames to scan.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    parser.add_argument("--min-gap", type=int, default=12, help="Minimum frame gap between selected crops.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence.")
    parser.add_argument("--detection-model-path", default=str(settings.detection_model_path))
    parser.add_argument("--tracker-config-path", default=str(settings.tracker_yaml_path))
    parser.add_argument(
        "--captions",
        default="",
        help="Optional captions separated by |. Example: side face|front face|expression change|lighting change",
    )
    return parser.parse_args()


def crop_with_margin(frame: np.ndarray, xyxy: tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    side = int(max(box_w, box_h) * (1 + margin * 2))
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(width, nx1 + side)
    ny2 = min(height, ny1 + side)
    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size else None


def collect_track_crops(args: argparse.Namespace) -> dict[int, list[dict]]:
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    yolo_model = build_yolo_model(args.detection_model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    tracks: dict[int, list[dict]] = defaultdict(list)
    frame_idx = 0
    scanned = 0
    while scanned < args.scan_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(args.stride, 1) != 0:
            frame_idx += 1
            continue

        results = yolo_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=args.conf,
            tracker=args.tracker_config_path,
        )
        scanned += 1
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            if ids is not None:
                for box, track_id in zip(boxes, ids.cpu().numpy().astype(int)):
                    if args.track_id is not None and track_id != args.track_id:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    crop = crop_with_margin(frame, (x1, y1, x2, y2))
                    if crop is None:
                        continue
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    confidence = float(box.conf.item()) if box.conf is not None else 0.0
                    tracks[track_id].append(
                        {
                            "frame_idx": frame_idx,
                            "crop": crop,
                            "area": area,
                            "confidence": confidence,
                        }
                    )
        frame_idx += 1

    cap.release()
    return tracks


def pick_track(tracks: dict[int, list[dict]], requested_track_id: int | None, count: int) -> tuple[int, list[dict]]:
    if not tracks:
        raise RuntimeError("No tracked face crops were found.")
    if requested_track_id is not None:
        crops = tracks.get(requested_track_id, [])
        if not crops:
            raise RuntimeError(f"Track {requested_track_id} was not found.")
        return requested_track_id, crops

    ranked = sorted(
        tracks.items(),
        key=lambda item: (len(item[1]) >= count, len(item[1]), np.mean([crop["area"] for crop in item[1]])),
        reverse=True,
    )
    return ranked[0]


def select_diverse_crops(crops: list[dict], count: int, min_gap: int) -> list[dict]:
    ordered = sorted(crops, key=lambda item: (item["confidence"], item["area"]), reverse=True)
    if not ordered:
        return []

    selected = [ordered[0]]
    while len(selected) < count and len(selected) < len(ordered):
        best_item = None
        best_score = -1.0
        for item in ordered:
            if item in selected:
                continue
            frame_gap = min(abs(item["frame_idx"] - prev["frame_idx"]) for prev in selected)
            if frame_gap < min_gap and len(ordered) >= count:
                continue
            visual_gap = min(_crop_distance(item["crop"], prev["crop"]) for prev in selected)
            quality = item["confidence"] * 0.2 + min(item["area"] / 20000, 1.0) * 0.1
            score = visual_gap * 0.7 + min(frame_gap / max(min_gap, 1), 2.0) * 0.2 + quality
            if score > best_score:
                best_item = item
                best_score = score
        if best_item is None:
            min_gap = max(1, min_gap // 2)
            continue
        selected.append(best_item)

    return sorted(selected, key=lambda item: item["frame_idx"])


def _crop_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_small = cv2.resize(left, (32, 32), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (32, 32), interpolation=cv2.INTER_AREA)
    left_lab = cv2.cvtColor(left_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    right_lab = cv2.cvtColor(right_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.abs(left_lab - right_lab)) / 255.0)


def make_sheet(items: list[dict], track_id: int, output_path: Path, captions: list[str] | None = None) -> None:
    tile = 220
    label_h = 54
    gap = 14
    padding = 18
    width = padding * 2 + len(items) * tile + (len(items) - 1) * gap
    height = padding * 2 + tile + label_h
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)

    for index, item in enumerate(items):
        crop = item["crop"]
        resized = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
        x = padding + index * (tile + gap)
        y = padding
        sheet[y : y + tile, x : x + tile] = resized
        label = captions[index] if captions and index < len(captions) else f"track {track_id}  frame {item['frame_idx']}"
        sub_label = f"track {track_id} / frame {item['frame_idx']}"
        _draw_text(sheet, label, (x, y + tile + 8), size=17, color=(30, 30, 30))
        _draw_text(sheet, sub_label, (x, y + tile + 32), size=13, color=(110, 110, 110))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def _draw_text(image: np.ndarray, text: str, position: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(size)
    draw.text(position, text, fill=color, font=font)
    image[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _load_font(size: int):
    for font_path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else settings.experiment_exports_dir / "ppt_track_crop_sheet.png"
    tracks = collect_track_crops(args)
    track_id, crops = pick_track(tracks, args.track_id, args.count)
    selected = select_diverse_crops(crops, max(1, args.count), max(1, args.min_gap))
    captions = [caption.strip() for caption in args.captions.split("|") if caption.strip()] if args.captions else None
    make_sheet(selected, track_id, output, captions)
    print(f"Exported: {output}")
    print(f"Track id: {track_id}")
    print("Frames:", ", ".join(str(item["frame_idx"]) for item in selected))


if __name__ == "__main__":
    main()
