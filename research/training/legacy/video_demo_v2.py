"""
video_demo.py (终极缝合版)
==============
主程序：构建熟人库 + 视频识别 + 增量聚类

改进点：
1. 修复了与 bytetrack_anime.yaml 的 buffer 冲突（TRACK_BUFFER 提升至 90）
2. 新增"预识别"机制（角色出现 5 帧立刻出名字，不用等离开画面）
3. 强制使用绝对路径加载 tracker yaml（防止静默回退默认配置）
4. 修复动漫角色特殊字符路径导致的读取失败问题

依赖：pip install torch torchvision opencv-python numpy ultralytics
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ===================== 配置区 =====================
# 模型路径（train_arcface.py 生成的，如果没跑训练，会自动用 ImageNet 预训练）
MODEL_PATH = "resnet50_anime_arcface.pth"

# 数据集路径（用于构建熟人库）
DATASET_DIR = r"./datasets/arcface_112x112"

# 熟人库保存路径
KNOWN_DB_PATH = "known_database_01.json"

# 输入视频路径
VIDEO_PATH = r"./test_02\Chainsaw_Man.mp4"  # ← 改成你的视频路径

# 输出视频路径
OUTPUT_VIDEO_PATH = "output_annotated.mp4"

# YOLO 人脸检测模型路径
YOLO_MODEL_PATH = "yolov8s_anime.pt"  # ← 改成你的模型路径

# ⚠️ 关键修改：ByteTrack 配置文件的绝对路径（防止 Ultralytics 找不到导致用默认配置）
TRACKER_YAML = r"./test04\bytetrack_anime.yaml"  # ← 必须改成你的绝对路径！

# 识别阈值
KNOWN_THRESH = 0.50  # 熟人识别阈值
STRANGER_THRESH = 0.55  # 陌生人聚类阈值

# ⚠️ 关键修改：聚合器缓冲必须大于 yaml 里的 track_buffer (60)
TRACK_BUFFER = 90  # 聚合器判定 track 结束的帧数
MIN_TRACK_FRAMES = 3  # 一个 track 至少出现几帧才进行识别


# ================================================


# ================================================================
#  1. 特征提取器：ResNet50（加载微调权重或 ImageNet 预训练）
# ================================================================
class ResNet50Extractor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self._build_backbone()
        self.bn = nn.BatchNorm1d(2048)

        if model_path and Path(model_path).exists():
            self._load_finetuned_weights(model_path)
            print(f"✅ 加载微调模型: {model_path}")
        else:
            print(f"⚠️  未找到微调模型，使用 ImageNet 预训练权重")

        self.backbone.to(self.device)
        self.bn.to(self.device)
        self.backbone.eval()
        self.bn.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"   特征维度: 2048 | 运行设备: {self.device}")

    def _build_backbone(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(resnet.children())[:-1])

    def _load_finetuned_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
        else:
            self.backbone.load_state_dict(checkpoint)
        if 'bn' in checkpoint:
            self.bn.load_state_dict(checkpoint['bn'])
        if 'val_acc' in checkpoint:
            print(f"   模型验证准确率: {checkpoint['val_acc']:.2f}%")

    @torch.no_grad()
    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            feat = self.backbone(tensor).view(-1).unsqueeze(0)
            feat = self.bn(feat)
            feat = feat / torch.norm(feat, dim=1, keepdim=True)
            return feat.cpu().numpy().flatten()
        except Exception as e:
            return None


# ================================================================
#  2. 开集识别引擎：熟人库 + 增量聚类
# ================================================================
class FaceRecognizer:
    def __init__(self, known_db_path=None):
        self.known_db = {}
        self.stranger_db = {}
        self.stranger_counter = 1
        self.KNOWN_THRESH = KNOWN_THRESH
        self.STRANGER_THRESH = STRANGER_THRESH

        if known_db_path and Path(known_db_path).exists():
            self._load_known_db(known_db_path)

    def _load_known_db(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.known_db = json.load(f)
        print(f"📂 已加载熟人库: {len(self.known_db)} 个角色")

    def save_stranger_db(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stranger_db, f, ensure_ascii=False, indent=2)

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        # 第一问：查熟人库
        best_name, best_sim = self._find_best_match(embedding, self.known_db)
        if best_name is not None and best_sim >= self.KNOWN_THRESH:
            return best_name, best_sim, "known"

        # 第二问：查陌生人库
        #stranger_vectors = {k: v["vector"] for k, v in self.stranger_db.items()}
        #best_sid, best_ssim = self._find_best_match(embedding, stranger_vectors)
        best_sid, best_ssim = self._find_best_match(embedding, self.stranger_db)

        if best_sid is not None and best_ssim >= self.STRANGER_THRESH:
            # 增量聚类核心
            old_vec = np.array(self.stranger_db[best_sid]["vector"])
            count = self.stranger_db[best_sid]["count"]
            new_vec = (old_vec * count + embedding) / (count + 1)
            new_vec = new_vec / np.linalg.norm(new_vec)
            self.stranger_db[best_sid]["vector"] = new_vec.tolist()
            self.stranger_db[best_sid]["count"] += 1
            return best_sid, best_ssim, "stranger"

        # 第三问：纯新人
        new_id = f"Stranger_{self.stranger_counter}"
        self.stranger_db[new_id] = {"vector": embedding.tolist(), "count": 1}
        self.stranger_counter += 1
        return new_id, 0.0, "new"

    def _find_best_match(self, query_vec: np.ndarray, database: dict) -> Tuple[Optional[str], float]:
        if not database: return None, -1.0
        best_name, best_sim = None, -1.0
        for name, data in database.items():
            db_vec = np.array(data["mean"] if "mean" in data else data["vector"])
            sim = float(np.dot(query_vec, db_vec))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim

    def get_stats(self):
        return {
            "known_count": len(self.known_db),
            "stranger_count": len(self.stranger_db),
            "stranger_details": {k: v["count"] for k, v in self.stranger_db.items()}
        }


# ================================================================
#  3. 构建熟人库（线下准备阶段）
# ================================================================
def build_known_database(extractor: ResNet50Extractor, dataset_dir: str, output_path: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists(): return {}

    print(f"\n{'=' * 60}\n🚀 构建熟人特征库\n{'=' * 60}")
    known_db = {}
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

    for idx, folder in enumerate(folders, 1):
        char_name = folder.name
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        if len(images) < 3: continue

        vectors = []
        for img_path in images[:50]:
            # ⚠️ 关键修改：解决日文/特殊字符路径报错问题
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0: continue

            emb = extractor.extract(img)
            if emb is not None: vectors.append(emb)

        if len(vectors) >= 3:
            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec)
            known_db[char_name] = {"mean": mean_vec.tolist(), "count": len(vectors)}
            print(f"   [{idx}/{len(folders)}] ✅ {char_name}: {len(vectors)} 张特征")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(known_db, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 熟人库构建完成！共 {len(known_db)} 个角色，保存至 {output_path}\n{'=' * 60}")
    return known_db


# ================================================================
#  4. Track 级特征聚合器（升级版：支持画面内预识别）
# ================================================================
class TrackFeatureAggregator:
    def __init__(self, buffer=90, min_frames=3, preview_frames=5):
        self.buffer = buffer
        self.min_frames = min_frames
        self.preview_frames = preview_frames  # 积累几帧触发预识别
        self.tracks = {}

    def update(self, track_id, frame_idx, embedding):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"features": [], "last_frame": -999, "result": None, "previewed": False}

        track = self.tracks[track_id]
        track["features"].append(embedding)
        track["last_frame"] = frame_idx

        finished_results = []
        preview_result = None

        # 1. 预识别逻辑：角色还在画面中，但特征够用了，先给个名字
        if not track["previewed"] and len(track["features"]) >= self.preview_frames:
            avg_feat = np.mean(track["features"][-self.preview_frames:], axis=0)
            avg_feat = avg_feat / np.linalg.norm(avg_feat)
            preview_result = (track_id, avg_feat, False)  # False 表示非最终结果
            track["previewed"] = True

        # 2. 最终结算逻辑：角色离开画面超过 buffer 帧
        finished = []
        for tid, t in self.tracks.items():
            if frame_idx - t["last_frame"] > self.buffer and t["result"] is None:
                if len(t["features"]) >= self.min_frames:
                    avg_feat = np.mean(t["features"], axis=0)
                    avg_feat = avg_feat / np.linalg.norm(avg_feat)
                    t["result"] = (tid, avg_feat)
                finished.append(tid)

        for tid in finished:
            if self.tracks[tid]["result"] is not None:
                res_tid, res_feat = self.tracks[tid]["result"]
                finished_results.append((res_tid, res_feat, True))  # True 表示最终结果
            del self.tracks[tid]

        return finished_results, preview_result

    def force_flush(self):
        results = []
        for tid, t in self.tracks.items():
            if t["result"] is None and len(t["features"]) >= self.min_frames:
                avg_feat = np.mean(t["features"], axis=0)
                avg_feat = avg_feat / np.linalg.norm(avg_feat)
                results.append((tid, avg_feat, True))
        self.tracks.clear()
        return results


# ================================================================
#  5. 视频处理主流程
# ================================================================
def process_video(video_path, extractor, recognizer, yolo_model, output_path=None):
    print(f"\n{'=' * 60}\n🎬 开始处理视频: {video_path}\n{'=' * 60}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"❌ 无法打开视频"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   分辨率: {w}x{h} | 帧率: {fps:.1f} | 总帧数: {total_frames}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if output_path else None
    aggregator = TrackFeatureAggregator(buffer=TRACK_BUFFER, min_frames=MIN_TRACK_FRAMES)
    track_labels = {}
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 1. YOLO 检测 + ByteTrack 追踪
        results = yolo_model.track(frame, persist=True, verbose=False, conf=0.15, iou=0.6, tracker=TRACKER_YAML)
        valid_data = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for (x1, y1, x2, y2), tid in zip(boxes_xyxy, track_ids):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    valid_data.append(((x1, y1, x2, y2), tid))

        # 2. 提取特征 + 识别
        for (x1, y1, x2, y2), tid in valid_data:
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            face_resized = cv2.resize(face, (112, 112))
            embedding = extractor.extract(face_resized)
            if embedding is None: continue

            # 接收预识别和最终结算
            finished_tracks, preview_track = aggregator.update(tid, frame_idx, embedding)

            if preview_track is not None:
                p_tid, p_emb, _ = preview_track
                name, sim, match_type = recognizer.recognize(p_emb)
                track_labels[p_tid] = (name, sim, match_type)

            for f_tid, f_emb, _ in finished_tracks:
                name, sim, match_type = recognizer.recognize(f_emb)
                track_labels[f_tid] = (name, sim, match_type)

            if tid in track_labels:
                name, sim, match_type = track_labels[tid]
            else:
                name, sim, match_type = f"ID_{tid}", 0.0, "tracking"

            # 3. 画面标注
            if match_type == "known":
                color, label = (0, 200, 0), f"{name} ({sim:.2f})"
            elif match_type == "stranger":
                color, label = (0, 200, 255), f"{name} ({sim:.2f})"
            elif match_type == "new":
                color, label = (0, 100, 255), f"{name} [NEW]"
            else:
                color, label = (200, 200, 200), "Tracking..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. 状态栏
        stats = recognizer.get_stats()
        status = f"Frame: {frame_idx}/{total_frames} | Known: {stats['known_count']} | Strangers: {stats['stranger_count']} | Tracks: {len(aggregator.tracks)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer: writer.write(frame)

        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(
                f"   进度: {frame_idx}/{total_frames} ({frame_idx / total_frames * 100:.1f}%) | 速度: {fps_actual:.1f} fps | 剩余: {eta:.0f}s")

        #cv2.imshow("Anime Face Recognition", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 视频结束，结算残余 track
    remaining = aggregator.force_flush()
    for f_tid, f_emb, _ in remaining:
        name, sim, match_type = recognizer.recognize(f_emb)
        track_labels[f_tid] = (name, sim, match_type)

    cap.release()
    if writer: writer.release()
    #cv2.destroyAllWindows()

    # 打印统计
    elapsed = time.time() - start_time
    stats = recognizer.get_stats()
    print(f"\n{'=' * 60}\n📊 处理完毕！\n{'=' * 60}")
    print(f"   总帧数: {frame_idx} | 耗时: {elapsed:.1f}s | 速度: {frame_idx / elapsed:.1f} fps")
    print(f"   熟人库: {stats['known_count']} 个角色 | 追踪到的陌生人: {stats['stranger_count']} 个")

    if stats['stranger_details']:
        print(f"\n   陌生人详情:")
        for sid, count in sorted(stats['stranger_details'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {sid}: 出现 {count} 次")

    if stats['stranger_count'] > 0:
        recognizer.save_stranger_db("stranger_database.json")
        print(f"\n   陌生人库已保存: stranger_database.json")
    if output_path: print(f"   标注视频已保存: {output_path}")
    print(f"{'=' * 60}")


# ================================================================
#  6. 主程序入口
# ================================================================
def main():
    print("=" * 60)
    print("    动漫视频角色识别系统 (终极缝合版)")
    print("    ResNet50 + ArcFace + 增量聚类")
    print("=" * 60)

    print("\n📥 步骤 1: 加载 ResNet50 特征提取模型...")
    extractor = ResNet50Extractor(model_path=MODEL_PATH)

    print(f"\n📚 步骤 2: 熟人库...")
    if not Path(KNOWN_DB_PATH).exists():
        build_known_database(extractor, DATASET_DIR, KNOWN_DB_PATH)
    else:
        print(f"   发现已有的熟人库: {KNOWN_DB_PATH}")

    print(f"\n🧠 步骤 3: 初始化识别引擎...")
    recognizer = FaceRecognizer(known_db_path=KNOWN_DB_PATH)

    print(f"\n🔍 步骤 4: 加载 YOLO 模型...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH) if Path(YOLO_MODEL_PATH).exists() else YOLO("yolov8n.pt")
    except ImportError:
        print("❌ 未安装 ultralytics");
        return

    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ 找不到视频: {VIDEO_PATH}");
        return

    process_video(VIDEO_PATH, extractor, recognizer, yolo_model, OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
"""
video_demo.py (终极缝合版)
==============
主程序：构建熟人库 + 视频识别 + 增量聚类

改进点：
1. 修复了与 bytetrack_anime.yaml 的 buffer 冲突（TRACK_BUFFER 提升至 90）
2. 新增"预识别"机制（角色出现 5 帧立刻出名字，不用等离开画面）
3. 强制使用绝对路径加载 tracker yaml（防止静默回退默认配置）
4. 修复动漫角色特殊字符路径导致的读取失败问题

依赖：pip install torch torchvision opencv-python numpy ultralytics
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ===================== 配置区 =====================
# 模型路径（train_arcface.py 生成的，如果没跑训练，会自动用 ImageNet 预训练）
MODEL_PATH = "resnet50_anime_arcface.pth"

# 数据集路径（用于构建熟人库）
DATASET_DIR = r"./datasets/arcface_112x112"

# 熟人库保存路径
KNOWN_DB_PATH = "known_database.json"

# 输入视频路径
VIDEO_PATH = r"./data/inputs/sample_videos/demo.mp4"  # ← 改成你的视频路径

# 输出视频路径
OUTPUT_VIDEO_PATH = "output_annotated.mp4"

# YOLO 人脸检测模型路径
YOLO_MODEL_PATH = r"runs/detect/naime-small-v121/weights/best.pt"  # ← 改成你的模型路径

# ⚠️ 关键修改：ByteTrack 配置文件的绝对路径（防止 Ultralytics 找不到导致用默认配置）
TRACKER_YAML = r"./models/production/tracking/bytetrack_anime.yaml"  # ← 必须改成你的绝对路径！

# 识别阈值
KNOWN_THRESH = 0.50  # 熟人识别阈值
STRANGER_THRESH = 0.55  # 陌生人聚类阈值

# ⚠️ 关键修改：聚合器缓冲必须大于 yaml 里的 track_buffer (60)
TRACK_BUFFER = 90  # 聚合器判定 track 结束的帧数
MIN_TRACK_FRAMES = 3  # 一个 track 至少出现几帧才进行识别


# ================================================


# ================================================================
#  1. 特征提取器：ResNet50（加载微调权重或 ImageNet 预训练）
# ================================================================
class ResNet50Extractor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self._build_backbone()
        self.bn = nn.BatchNorm1d(2048)

        if model_path and Path(model_path).exists():
            self._load_finetuned_weights(model_path)
            print(f"✅ 加载微调模型: {model_path}")
        else:
            print(f"⚠️  未找到微调模型，使用 ImageNet 预训练权重")

        self.backbone.to(self.device)
        self.bn.to(self.device)
        self.backbone.eval()
        self.bn.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"   特征维度: 2048 | 运行设备: {self.device}")

    def _build_backbone(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(resnet.children())[:-1])

    def _load_finetuned_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
        else:
            self.backbone.load_state_dict(checkpoint)
        if 'bn' in checkpoint:
            self.bn.load_state_dict(checkpoint['bn'])
        if 'val_acc' in checkpoint:
            print(f"   模型验证准确率: {checkpoint['val_acc']:.2f}%")

    @torch.no_grad()
    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            feat = self.backbone(tensor).view(-1).unsqueeze(0)
            feat = self.bn(feat)
            feat = feat / torch.norm(feat, dim=1, keepdim=True)
            return feat.cpu().numpy().flatten()
        except Exception as e:
            return None


# ================================================================
#  2. 开集识别引擎：熟人库 + 增量聚类
# ================================================================
class FaceRecognizer:
    def __init__(self, known_db_path=None):
        self.known_db = {}
        self.stranger_db = {}
        self.stranger_counter = 1
        self.KNOWN_THRESH = KNOWN_THRESH
        self.STRANGER_THRESH = STRANGER_THRESH

        if known_db_path and Path(known_db_path).exists():
            self._load_known_db(known_db_path)

    def _load_known_db(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.known_db = json.load(f)
        print(f"📂 已加载熟人库: {len(self.known_db)} 个角色")

    def save_stranger_db(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stranger_db, f, ensure_ascii=False, indent=2)

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        # 第一问：查熟人库
        best_name, best_sim = self._find_best_match(embedding, self.known_db)
        if best_name is not None and best_sim >= self.KNOWN_THRESH:
            return best_name, best_sim, "known"

        # 第二问：查陌生人库
        stranger_vectors = {k: v["vector"] for k, v in self.stranger_db.items()}
        best_sid, best_ssim = self._find_best_match(embedding, stranger_vectors)

        if best_sid is not None and best_ssim >= self.STRANGER_THRESH:
            # 增量聚类核心
            old_vec = np.array(self.stranger_db[best_sid]["vector"])
            count = self.stranger_db[best_sid]["count"]
            new_vec = (old_vec * count + embedding) / (count + 1)
            new_vec = new_vec / np.linalg.norm(new_vec)
            self.stranger_db[best_sid]["vector"] = new_vec.tolist()
            self.stranger_db[best_sid]["count"] += 1
            return best_sid, best_ssim, "stranger"

        # 第三问：纯新人
        new_id = f"Stranger_{self.stranger_counter}"
        self.stranger_db[new_id] = {"vector": embedding.tolist(), "count": 1}
        self.stranger_counter += 1
        return new_id, 0.0, "new"

    def _find_best_match(self, query_vec: np.ndarray, database: dict) -> Tuple[Optional[str], float]:
        if not database: return None, -1.0
        best_name, best_sim = None, -1.0
        for name, data in database.items():
            db_vec = np.array(data["mean"] if "mean" in data else data["vector"])
            sim = float(np.dot(query_vec, db_vec))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim

    def get_stats(self):
        return {
            "known_count": len(self.known_db),
            "stranger_count": len(self.stranger_db),
            "stranger_details": {k: v["count"] for k, v in self.stranger_db.items()}
        }


# ================================================================
#  3. 构建熟人库（线下准备阶段）
# ================================================================
def build_known_database(extractor: ResNet50Extractor, dataset_dir: str, output_path: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists(): return {}

    print(f"\n{'=' * 60}\n🚀 构建熟人特征库\n{'=' * 60}")
    known_db = {}
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

    for idx, folder in enumerate(folders, 1):
        char_name = folder.name
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        if len(images) < 3: continue

        vectors = []
        for img_path in images[:50]:
            # ⚠️ 关键修改：解决日文/特殊字符路径报错问题
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0: continue

            emb = extractor.extract(img)
            if emb is not None: vectors.append(emb)

        if len(vectors) >= 3:
            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec)
            known_db[char_name] = {"mean": mean_vec.tolist(), "count": len(vectors)}
            print(f"   [{idx}/{len(folders)}] ✅ {char_name}: {len(vectors)} 张特征")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(known_db, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 熟人库构建完成！共 {len(known_db)} 个角色，保存至 {output_path}\n{'=' * 60}")
    return known_db


# ================================================================
#  4. Track 级特征聚合器（升级版：支持画面内预识别）
# ================================================================
class TrackFeatureAggregator:
    def __init__(self, buffer=90, min_frames=3, preview_frames=5):
        self.buffer = buffer
        self.min_frames = min_frames
        self.preview_frames = preview_frames  # 积累几帧触发预识别
        self.tracks = {}

    def update(self, track_id, frame_idx, embedding):
        if track_id not in self.tracks:
            self.tracks[track_id] = {"features": [], "last_frame": -999, "result": None, "previewed": False}

        track = self.tracks[track_id]
        track["features"].append(embedding)
        track["last_frame"] = frame_idx

        finished_results = []
        preview_result = None

        # 1. 预识别逻辑：角色还在画面中，但特征够用了，先给个名字
        if not track["previewed"] and len(track["features"]) >= self.preview_frames:
            avg_feat = np.mean(track["features"][-self.preview_frames:], axis=0)
            avg_feat = avg_feat / np.linalg.norm(avg_feat)
            preview_result = (track_id, avg_feat, False)  # False 表示非最终结果
            track["previewed"] = True

        # 2. 最终结算逻辑：角色离开画面超过 buffer 帧
        finished = []
        for tid, t in self.tracks.items():
            if frame_idx - t["last_frame"] > self.buffer and t["result"] is None:
                if len(t["features"]) >= self.min_frames:
                    avg_feat = np.mean(t["features"], axis=0)
                    avg_feat = avg_feat / np.linalg.norm(avg_feat)
                    t["result"] = (tid, avg_feat)
                finished.append(tid)

        for tid in finished:
            if self.tracks[tid]["result"] is not None:
                res_tid, res_feat = self.tracks[tid]["result"]
                finished_results.append((res_tid, res_feat, True))  # True 表示最终结果
            del self.tracks[tid]

        return finished_results, preview_result

    def force_flush(self):
        results = []
        for tid, t in self.tracks.items():
            if t["result"] is None and len(t["features"]) >= self.min_frames:
                avg_feat = np.mean(t["features"], axis=0)
                avg_feat = avg_feat / np.linalg.norm(avg_feat)
                results.append((tid, avg_feat, True))
        self.tracks.clear()
        return results


# ================================================================
#  5. 视频处理主流程
# ================================================================
def process_video(video_path, extractor, recognizer, yolo_model, output_path=None):
    print(f"\n{'=' * 60}\n🎬 开始处理视频: {video_path}\n{'=' * 60}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"❌ 无法打开视频"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   分辨率: {w}x{h} | 帧率: {fps:.1f} | 总帧数: {total_frames}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if output_path else None
    aggregator = TrackFeatureAggregator(buffer=TRACK_BUFFER, min_frames=MIN_TRACK_FRAMES)
    track_labels = {}
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 1. YOLO 检测 + ByteTrack 追踪
        results = yolo_model.track(frame, persist=True, verbose=False, conf=0.15, iou=0.6, tracker=TRACKER_YAML)
        valid_data = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for (x1, y1, x2, y2), tid in zip(boxes_xyxy, track_ids):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    valid_data.append(((x1, y1, x2, y2), tid))

        # 2. 提取特征 + 识别
        for (x1, y1, x2, y2), tid in valid_data:
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            face_resized = cv2.resize(face, (112, 112))
            embedding = extractor.extract(face_resized)
            if embedding is None: continue

            # 接收预识别和最终结算
            finished_tracks, preview_track = aggregator.update(tid, frame_idx, embedding)

            if preview_track is not None:
                p_tid, p_emb, _ = preview_track
                name, sim, match_type = recognizer.recognize(p_emb)
                track_labels[p_tid] = (name, sim, match_type)

            for f_tid, f_emb, _ in finished_tracks:
                name, sim, match_type = recognizer.recognize(f_emb)
                track_labels[f_tid] = (name, sim, match_type)

            if tid in track_labels:
                name, sim, match_type = track_labels[tid]
            else:
                name, sim, match_type = f"ID_{tid}", 0.0, "tracking"

            # 3. 画面标注
            if match_type == "known":
                color, label = (0, 200, 0), f"{name} ({sim:.2f})"
            elif match_type == "stranger":
                color, label = (0, 200, 255), f"{name} ({sim:.2f})"
            elif match_type == "new":
                color, label = (0, 100, 255), f"{name} [NEW]"
            else:
                color, label = (200, 200, 200), "Tracking..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. 状态栏
        stats = recognizer.get_stats()
        status = f"Frame: {frame_idx}/{total_frames} | Known: {stats['known_count']} | Strangers: {stats['stranger_count']} | Tracks: {len(aggregator.tracks)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer: writer.write(frame)

        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(
                f"   进度: {frame_idx}/{total_frames} ({frame_idx / total_frames * 100:.1f}%) | 速度: {fps_actual:.1f} fps | 剩余: {eta:.0f}s")

        cv2.imshow("Anime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 视频结束，结算残余 track
    remaining = aggregator.force_flush()
    for f_tid, f_emb, _ in remaining:
        name, sim, match_type = recognizer.recognize(f_emb)
        track_labels[f_tid] = (name, sim, match_type)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    # 打印统计
    elapsed = time.time() - start_time
    stats = recognizer.get_stats()
    print(f"\n{'=' * 60}\n📊 处理完毕！\n{'=' * 60}")
    print(f"   总帧数: {frame_idx} | 耗时: {elapsed:.1f}s | 速度: {frame_idx / elapsed:.1f} fps")
    print(f"   熟人库: {stats['known_count']} 个角色 | 追踪到的陌生人: {stats['stranger_count']} 个")

    if stats['stranger_details']:
        print(f"\n   陌生人详情:")
        for sid, count in sorted(stats['stranger_details'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {sid}: 出现 {count} 次")

    if stats['stranger_count'] > 0:
        recognizer.save_stranger_db("stranger_database.json")
        print(f"\n   陌生人库已保存: stranger_database.json")
    if output_path: print(f"   标注视频已保存: {output_path}")
    print(f"{'=' * 60}")


# ================================================================
#  6. 主程序入口
# ================================================================
def main():
    print("=" * 60)
    print("    动漫视频角色识别系统 (终极缝合版)")
    print("    ResNet50 + ArcFace + 增量聚类")
    print("=" * 60)

    print("\n📥 步骤 1: 加载 ResNet50 特征提取模型...")
    extractor = ResNet50Extractor(model_path=MODEL_PATH)

    print(f"\n📚 步骤 2: 熟人库...")
    if not Path(KNOWN_DB_PATH).exists():
        build_known_database(extractor, DATASET_DIR, KNOWN_DB_PATH)
    else:
        print(f"   发现已有的熟人库: {KNOWN_DB_PATH}")

    print(f"\n🧠 步骤 3: 初始化识别引擎...")
    recognizer = FaceRecognizer(known_db_path=KNOWN_DB_PATH)

    print(f"\n🔍 步骤 4: 加载 YOLO 模型...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH) if Path(YOLO_MODEL_PATH).exists() else YOLO("yolov8n.pt")
    except ImportError:
        print("❌ 未安装 ultralytics");
        return

    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ 找不到视频: {VIDEO_PATH}");
        return

    process_video(VIDEO_PATH, extractor, recognizer, yolo_model, OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
