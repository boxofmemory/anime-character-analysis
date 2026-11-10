"""
danbooru_clean_dataset_builder.py
=================================
面向动漫角色识别的数据集构建脚本：
1. 从 Danbooru API 批量下载候选图。
2. 根据 Danbooru 返回的角色标签过滤多角色/错标签候选。
3. 使用 YOLO 动漫人脸检测模型裁剪人脸。
4. 默认拒绝多脸图，避免把其他角色裁入当前角色文件夹。
5. 进行质量过滤、清晰度过滤、近似去重，并输出 CSV 报告。

推荐流程：
    python danbooru_clean_dataset_builder.py download
    python danbooru_clean_dataset_builder.py crop
    python danbooru_clean_dataset_builder.py all

queries 文件支持两种格式：
    格式 1：直接写 Danbooru 查询语句，第一项作为角色 tag 和文件夹名
        makima_(chainsaw_man)
        power_(chainsaw_man)

    格式 2：target_tag | folder_name | extra_tags
        makima_(chainsaw_man)|makima|long_hair yellow_eyes
        power_(chainsaw_man)|power|blonde_hair horns

注意：
    - 不要把 API Key 写死在代码里。请用环境变量 DANBOORU_USERNAME 和 DANBOORU_API_KEY。
    - 默认只保留 Danbooru 角色标签中“仅包含目标角色”的帖子。
    - 默认 YOLO 检测到多张有效人脸时不自动选择主脸，而是放入 review_multi_face。
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ================================================================
# 配置区：根据你的电脑修改这里
# ================================================================
QUERIES_FILE = "正确的熟人库_final.txt"

# 原始候选图输出目录
RAW_OUTPUT_DIR = r"./datasets/anime_face_dataset\danbooru_candidates"

# 清洗后的人脸训练集输出目录。后续 r3 可以把 DATASET_DIR 指向这里。
CLEAN_OUTPUT_DIR = r"./datasets/anime_face_dataset\anime_face_dataset_clean_224"

# 需要人工复核或被拒绝的样本目录
REVIEW_OUTPUT_DIR = r"./datasets/anime_face_dataset\anime_face_dataset_review"

# CSV 报告目录
REPORT_DIR = r"./datasets/anime_face_dataset\anime_face_dataset_reports"

# 你的动漫人脸检测模型
YOLO_MODEL_PATH = "yolov8s_anime.pt"

# 每个角色先抓取多少候选图。建议远大于最终目标数量。
CANDIDATES_PER_CHAR = 200

# 每个角色最终最多保留多少张干净人脸。
TARGET_CLEAN_IMAGES_PER_CHAR = 80

# Danbooru 查询增强
REQUIRE_SOLO_TAG = True
STRICT_SINGLE_CHARACTER_TAG = True
MIN_DANBOORU_SCORE = 5
ALLOWED_RATINGS = {"g"}  # g=general。若你确认只做本地研究且需要更多图，可改为 {"g", "s"}

# 查询时主动排除的 tag。可按需增减。
EXCLUDE_TAGS = [
    "comic",
    "4koma",
    "multiple_girls",
    "multiple_boys",
    "multiple_others",
    "text",
    "speech_bubble",
    "translation_request",
    "censored",
]

# 图片下载设置
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = 0.6
SLEEP_BETWEEN_DOWNLOADS = 0.3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_GIF = True
MIN_IMAGE_WIDTH = 256
MIN_IMAGE_HEIGHT = 256

# 人脸裁剪参数
TARGET_SIZE = 224
MARGIN_RATIO = 0.25
MIN_CONFIDENCE = 0.55
MIN_FACE_SIZE = 45
MAX_ASPECT_RATIO = 1.45
MIN_FACE_AREA_RATIO = 0.015
MIN_SHARPNESS = 80.0
REJECT_MULTI_FACE = True

# 近似去重。值越小越严格；4~8 通常比较合适。
ENABLE_DHASH_DEDUP = True
DHASH_DUP_THRESHOLD = 5

HEADERS = {
    "User-Agent": "AnimeCharacterDatasetBuilder/2.0 (research use)"
}

DANBOORU_API = "https://danbooru.donmai.us/posts.json"


# ================================================================
# 数据结构
# ================================================================
@dataclass
class QueryTask:
    target_tag: str
    folder_name: str
    query_tags: str


@dataclass
class FilterResult:
    keep: bool
    reason: str


# ================================================================
# 通用工具
# ================================================================
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip().strip(".")
    return name or "unknown"


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_url_ext(url: str) -> str:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return ext


def safe_read_image_cv2(path: str | Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def safe_write_image_cv2(path: str | Path, img: np.ndarray) -> bool:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def copy_to_dir(src: str | Path, dst_dir: str | Path, prefix: str = "") -> Optional[Path]:
    try:
        src = Path(src)
        dst_dir = ensure_dir(dst_dir)
        dst = dst_dir / f"{prefix}{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def read_tasks(path: str | Path) -> List[QueryTask]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到查询文件：{path}")

    tasks: List[QueryTask] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                target_tag = parts[0]
                folder_name = sanitize_filename(parts[1] if len(parts) > 1 and parts[1] else target_tag)
                extra = parts[2] if len(parts) > 2 else ""
                query_tags = f"{target_tag} {extra}".strip()
            else:
                tokens = line.split()
                target_tag = tokens[0]
                folder_name = sanitize_filename(target_tag)
                query_tags = line

            tasks.append(QueryTask(target_tag=target_tag, folder_name=folder_name, query_tags=query_tags))

    return tasks


# ================================================================
# Danbooru 下载部分
# ================================================================
def build_final_query(task: QueryTask) -> str:
    """
    Danbooru 查询阶段只保留最少条件，避免 422。
    rating、score、多角色等严格过滤放到 filter_post_by_metadata() 本地完成。
    """
    tags = [task.target_tag]

    if REQUIRE_SOLO_TAG:
        tags.append("solo")

    return " ".join(tags)


def get_best_file_url(post: Dict) -> Optional[str]:
    for key in ["file_url", "large_file_url", "sample_url"]:
        url = post.get(key)
        if isinstance(url, str) and url:
            return url
    return None


def parse_tags(post: Dict, key: str) -> List[str]:
    value = post.get(key, "")
    if not isinstance(value, str):
        return []
    return value.split()


def filter_post_by_metadata(post: Dict, target_tag: str) -> FilterResult:
    url = get_best_file_url(post)
    if not url:
        return FilterResult(False, "no_file_url")

    ext = get_url_ext(url)
    if SKIP_GIF and ext == ".gif":
        return FilterResult(False, "gif_skipped")
    if ext not in IMAGE_EXTS:
        return FilterResult(False, f"bad_ext:{ext}")

    width = int(post.get("image_width") or 0)
    height = int(post.get("image_height") or 0)
    if width and height and (width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT):
        return FilterResult(False, "image_too_small")

    rating = str(post.get("rating", "")).lower()
    if ALLOWED_RATINGS and rating not in ALLOWED_RATINGS:
        return FilterResult(False, f"rating:{rating}")

    try:
        score = int(post.get("score") or 0)
    except Exception:
        score = 0
    if MIN_DANBOORU_SCORE is not None and score < MIN_DANBOORU_SCORE:
        return FilterResult(False, "low_score")

    char_tags = parse_tags(post, "tag_string_character")
    general_tags = parse_tags(post, "tag_string_general")

    if target_tag not in char_tags:
        return FilterResult(False, "target_character_tag_missing")

    if STRICT_SINGLE_CHARACTER_TAG:
        other_chars = [t for t in char_tags if t != target_tag]
        if other_chars:
            return FilterResult(False, "multiple_character_tags:" + ",".join(other_chars[:5]))

    if REQUIRE_SOLO_TAG and "solo" not in general_tags:
        return FilterResult(False, "solo_tag_missing")

    return FilterResult(True, "keep")


def fetch_posts(session: requests.Session, query: str, max_posts: int) -> List[Dict]:
    posts: List[Dict] = []
    page = 1

    while len(posts) < max_posts:
        params = {
            "tags": query,
            "limit": min(100, max_posts - len(posts)),
            "page": page,
        }
        resp = session.get(DANBOORU_API, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        posts.extend(batch)
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return posts[:max_posts]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    username = os.getenv("DANBOORU_USERNAME", "").strip()
    api_key = os.getenv("DANBOORU_API_KEY", "").strip()
    if username and api_key:
        session.auth = (username, api_key)
        print(f"已启用 Danbooru 认证：{username}")
    else:
        print("未检测到 DANBOORU_USERNAME / DANBOORU_API_KEY，将以匿名方式访问。")

    return session


def download_one_image(session: requests.Session, url: str, save_path: Path) -> bool:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"下载失败：{url} | {e}")
        return False


def download_candidates():
    ensure_dir(RAW_OUTPUT_DIR)
    ensure_dir(REPORT_DIR)
    report_path = Path(REPORT_DIR) / "download_report.csv"

    tasks = read_tasks(QUERIES_FILE)
    session = make_session()

    print(f"共加载 {len(tasks)} 个角色任务。")
    print(f"候选图输出目录：{Path(RAW_OUTPUT_DIR).absolute()}")
    print(f"每个角色候选上限：{CANDIDATES_PER_CHAR}")

    with open(report_path, "w", newline="", encoding="utf-8-sig") as rf:
        writer = csv.writer(rf)
        writer.writerow([
            "folder", "target_tag", "post_id", "status", "reason", "rating", "score",
            "character_tags", "general_tags", "file_path", "source_url"
        ])

        for idx, task in enumerate(tasks, 1):
            final_query = build_final_query(task)
            char_dir = ensure_dir(Path(RAW_OUTPUT_DIR) / task.folder_name)
            meta_dir = ensure_dir(char_dir / "_metadata")

            existing_images = [p for p in char_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            print(f"\n[{idx}/{len(tasks)}] {task.folder_name}")
            print(f"查询：{final_query}")
            print(f"已有候选图：{len(existing_images)}")

            if len(existing_images) >= CANDIDATES_PER_CHAR:
                print("候选数量已达上限，跳过下载。")
                continue

            try:
                posts = fetch_posts(session, final_query, CANDIDATES_PER_CHAR * 3)
            except Exception as e:
                print(f"API 请求失败：{e}")
                writer.writerow([task.folder_name, task.target_tag, "", "api_error", str(e), "", "", "", "", "", ""])
                continue

            kept_count = len(existing_images)
            seen_post_ids = {p.stem.split("_")[0] for p in existing_images}

            for post in posts:
                post_id = str(post.get("id", "unknown"))
                url = get_best_file_url(post) or ""
                filter_result = filter_post_by_metadata(post, task.target_tag)
                char_tags = post.get("tag_string_character", "")
                general_tags = post.get("tag_string_general", "")
                rating = post.get("rating", "")
                score = post.get("score", "")

                if not filter_result.keep:
                    writer.writerow([
                        task.folder_name, task.target_tag, post_id, "rejected_metadata",
                        filter_result.reason, rating, score, char_tags, general_tags, "", url
                    ])
                    continue

                if post_id in seen_post_ids:
                    writer.writerow([
                        task.folder_name, task.target_tag, post_id, "skipped_existing",
                        "post_id_exists", rating, score, char_tags, general_tags, "", url
                    ])
                    continue

                ext = get_url_ext(url) or ".jpg"
                save_path = char_dir / f"{post_id}_{task.folder_name}{ext}"

                ok = download_one_image(session, url, save_path)
                if ok:
                    kept_count += 1
                    seen_post_ids.add(post_id)
                    meta_path = meta_dir / f"{post_id}.json"
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(post, mf, ensure_ascii=False, indent=2)

                    writer.writerow([
                        task.folder_name, task.target_tag, post_id, "downloaded",
                        "keep", rating, score, char_tags, general_tags, str(save_path), url
                    ])
                    print(f"  下载：{save_path.name}  [{kept_count}/{CANDIDATES_PER_CHAR}]")
                    time.sleep(SLEEP_BETWEEN_DOWNLOADS)

                else:
                    writer.writerow([
                        task.folder_name, task.target_tag, post_id, "download_error",
                        "request_failed", rating, score, char_tags, general_tags, "", url
                    ])

                if kept_count >= CANDIDATES_PER_CHAR:
                    break

    print(f"\n下载完成。报告：{report_path}")


# ================================================================
# 人脸裁剪与清洗部分
# ================================================================
def check_sharpness_bgr(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash_bgr(img: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance_int(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def is_duplicate_hash(new_hash: int, existing_hashes: List[int], threshold: int) -> bool:
    for old in existing_hashes:
        if hamming_distance_int(new_hash, old) <= threshold:
            return True
    return False


def crop_face_square(img: np.ndarray, box: np.ndarray, margin_ratio: float) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [float(v) for v in box]
    h, w = img.shape[:2]

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    side = max(bw, bh) * (1.0 + 2.0 * margin_ratio)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = int(round(cx + side / 2.0))
    sy2 = int(round(cy + side / 2.0))

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > w:
        shift = sx2 - w
        sx1 -= shift
        sx2 = w
    if sy2 > h:
        shift = sy2 - h
        sy1 -= shift
        sy2 = h

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(w, sx2)
    sy2 = min(h, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    crop = img[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None
    return crop


def filter_boxes(boxes: np.ndarray, confs: np.ndarray, img_w: int, img_h: int) -> List[Tuple[np.ndarray, float, str]]:
    valid = []
    image_area = img_w * img_h

    for i, box in enumerate(boxes):
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1

        if conf < MIN_CONFIDENCE:
            continue
        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
            continue
        aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
        if aspect > MAX_ASPECT_RATIO:
            continue
        area_ratio = (bw * bh) / max(image_area, 1)
        if area_ratio < MIN_FACE_AREA_RATIO:
            continue

        valid.append((box, conf, "valid"))

    return valid


def annotate_faces(img: np.ndarray, valid_boxes: List[Tuple[np.ndarray, float, str]]) -> np.ndarray:
    canvas = img.copy()
    for i, (box, conf, _) in enumerate(valid_boxes, 1):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(canvas, f"face{i}:{conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_existing_hashes(clean_char_dir: Path) -> List[int]:
    hashes: List[int] = []
    if not clean_char_dir.exists():
        return hashes
    for img_path in list_image_files(clean_char_dir):
        img = safe_read_image_cv2(img_path)
        if img is None:
            continue
        try:
            hashes.append(dhash_bgr(img))
        except Exception:
            pass
    return hashes


def crop_and_clean_faces():
    if YOLO is None:
        raise RuntimeError("未安装 ultralytics。请先运行：pip install ultralytics")

    raw_root = Path(RAW_OUTPUT_DIR)
    clean_root = ensure_dir(CLEAN_OUTPUT_DIR)
    review_root = ensure_dir(REVIEW_OUTPUT_DIR)
    report_root = ensure_dir(REPORT_DIR)
    crop_report_path = report_root / "crop_report.csv"

    if not raw_root.exists():
        raise FileNotFoundError(f"找不到候选图目录：{raw_root}")

    print(f"加载 YOLO 模型：{YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    char_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and not p.name.startswith("_")])
    print(f"共发现 {len(char_dirs)} 个角色候选目录。")
    print(f"清洗后输出目录：{clean_root}")
    print(f"复核目录：{review_root}")

    with open(crop_report_path, "w", newline="", encoding="utf-8-sig") as rf:
        writer = csv.writer(rf)
        writer.writerow([
            "folder", "image", "status", "reason", "num_valid_faces", "conf", "sharpness", "output_path"
        ])

        global_stats = CounterDict()

        for char_idx, char_dir in enumerate(char_dirs, 1):
            folder_name = char_dir.name
            clean_char_dir = ensure_dir(clean_root / folder_name)
            review_multi_dir = ensure_dir(review_root / "review_multi_face" / folder_name)
            review_no_face_dir = ensure_dir(review_root / "rejected_no_face" / folder_name)
            review_low_quality_dir = ensure_dir(review_root / "rejected_low_quality" / folder_name)
            review_duplicate_dir = ensure_dir(review_root / "rejected_duplicate" / folder_name)

            existing_clean = list_image_files(clean_char_dir)
            existing_hashes = load_existing_hashes(clean_char_dir) if ENABLE_DHASH_DEDUP else []

            images = list_image_files(char_dir)
            print(f"\n[{char_idx}/{len(char_dirs)}] {folder_name}: 候选 {len(images)} 张，已清洗 {len(existing_clean)} 张")

            stats = CounterDict()

            for img_path in tqdm(images, desc=folder_name, ncols=100):
                if len(list_image_files(clean_char_dir)) >= TARGET_CLEAN_IMAGES_PER_CHAR:
                    break

                img = safe_read_image_cv2(img_path)
                if img is None:
                    stats["read_error"] += 1
                    global_stats["read_error"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", "read_error", 0, "", "", ""])
                    continue

                img_h, img_w = img.shape[:2]

                try:
                    results = model(img, verbose=False)
                    if len(results) == 0 or results[0].boxes is None:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        confs = np.array([], dtype=np.float32)
                    else:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                except Exception as e:
                    stats["detect_error"] += 1
                    global_stats["detect_error"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", f"detect_error:{e}", 0, "", "", ""])
                    continue

                valid_boxes = filter_boxes(boxes, confs, img_w, img_h)

                if len(valid_boxes) == 0:
                    copy_to_dir(img_path, review_no_face_dir)
                    stats["no_valid_face"] += 1
                    global_stats["no_valid_face"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", "no_valid_face", 0, "", "", ""])
                    continue

                if REJECT_MULTI_FACE and len(valid_boxes) > 1:
                    annotated = annotate_faces(img, valid_boxes)
                    annotated_path = review_multi_dir / f"multi_{img_path.stem}.jpg"
                    safe_write_image_cv2(annotated_path, annotated)
                    copy_to_dir(img_path, review_multi_dir, prefix="orig_")
                    stats["multi_face"] += 1
                    global_stats["multi_face"] += 1
                    writer.writerow([folder_name, img_path.name, "review", "multi_face", len(valid_boxes), "", "", str(annotated_path)])
                    continue

                box, conf, _ = valid_boxes[0]
                crop = crop_face_square(img, box, MARGIN_RATIO)
                if crop is None:
                    copy_to_dir(img_path, review_low_quality_dir)
                    stats["bad_crop"] += 1
                    global_stats["bad_crop"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", "bad_crop", len(valid_boxes), f"{conf:.4f}", "", ""])
                    continue

                sharpness = check_sharpness_bgr(crop)
                if sharpness < MIN_SHARPNESS:
                    copy_to_dir(img_path, review_low_quality_dir)
                    stats["low_sharpness"] += 1
                    global_stats["low_sharpness"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", "low_sharpness", len(valid_boxes), f"{conf:.4f}", f"{sharpness:.2f}", ""])
                    continue

                face_resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

                if ENABLE_DHASH_DEDUP:
                    new_hash = dhash_bgr(face_resized)
                    if is_duplicate_hash(new_hash, existing_hashes, DHASH_DUP_THRESHOLD):
                        copy_to_dir(img_path, review_duplicate_dir)
                        stats["duplicate"] += 1
                        global_stats["duplicate"] += 1
                        writer.writerow([folder_name, img_path.name, "rejected", "duplicate_dhash", len(valid_boxes), f"{conf:.4f}", f"{sharpness:.2f}", ""])
                        continue
                    existing_hashes.append(new_hash)

                save_name = f"face_{img_path.stem}.jpg"
                save_path = clean_char_dir / save_name
                ok = safe_write_image_cv2(save_path, face_resized)
                if ok:
                    stats["saved"] += 1
                    global_stats["saved"] += 1
                    writer.writerow([folder_name, img_path.name, "saved", "clean", len(valid_boxes), f"{conf:.4f}", f"{sharpness:.2f}", str(save_path)])
                else:
                    stats["save_error"] += 1
                    global_stats["save_error"] += 1
                    writer.writerow([folder_name, img_path.name, "rejected", "save_error", len(valid_boxes), f"{conf:.4f}", f"{sharpness:.2f}", ""])

            print(f"  统计：{dict(stats)}")

    print("\n裁剪清洗完成。")
    print(f"报告：{crop_report_path}")
    print(f"干净数据集：{clean_root}")
    print(f"复核目录：{review_root}")


class CounterDict(dict):
    def __missing__(self, key):
        self[key] = 0
        return 0


# ================================================================
# 数据集统计
# ================================================================
def dataset_stats():
    root = Path(CLEAN_OUTPUT_DIR)
    if not root.exists():
        print(f"清洗数据集目录不存在：{root}")
        return

    rows = []
    for char_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        count = len(list_image_files(char_dir))
        rows.append((char_dir.name, count))

    if not rows:
        print("没有找到清洗后的类别。")
        return

    counts = [c for _, c in rows]
    print("\n清洗后数据集统计：")
    print(f"类别数：{len(rows)}")
    print(f"总图片数：{sum(counts)}")
    print(f"每类最少：{min(counts)}")
    print(f"每类最多：{max(counts)}")
    print(f"每类平均：{sum(counts) / len(counts):.2f}")

    report_path = Path(REPORT_DIR) / "clean_dataset_stats.csv"
    ensure_dir(REPORT_DIR)
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "count"])
        writer.writerows(rows)

    print(f"统计报告：{report_path}")
    print("样本数最少的前 20 类：")
    for name, count in sorted(rows, key=lambda x: x[1])[:20]:
        print(f"  {name}: {count}")


# ================================================================
# 命令行入口
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Danbooru 动漫角色数据集半自动清洗工具")
    parser.add_argument(
        "mode",
        choices=["download", "crop", "all", "stats"],
        help="download=下载候选图；crop=裁剪清洗；all=下载+裁剪；stats=统计清洗后数据集",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in {"download", "all"}:
        download_candidates()

    if args.mode in {"crop", "all"}:
        crop_and_clean_faces()

    if args.mode == "stats":
        dataset_stats()

    if args.mode == "all":
        dataset_stats()


if __name__ == "__main__":
    main()
