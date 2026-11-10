"""
video_demo.py
==============
主程序：构建熟人库 + 视频识别 + 增量聚类

功能：
1. 加载微调后的 ResNet50 模型（提取 2048 维特征）
2. 扫描数据集，为每个角色计算"平均特征向量"，生成熟人库
3. 处理上传的动漫视频：
   - YOLO 检测人脸
   - ByteTrack 追踪同一张脸
   - ResNet50 提取特征
   - 三连问：查熟人库 → 查陌生人库 → 新建档案
   - 增量聚类：更新陌生人的特征（滑动平均）
4. 输出标注视频 + 识别统计报告

依赖：pip install torch torchvision opencv-python numpy ultralytics tqdm
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
# 模型路径（train_arcface.py 生成的）
MODEL_PATH = "resnet50_anime_arcface.pth"

# 数据集路径（用于构建熟人库）
DATASET_DIR = r"./datasets/arcface_112x112"

# 熟人库保存路径
KNOWN_DB_PATH = "known_database.json"

# 输入视频路径
VIDEO_PATH = r"./data/inputs/sample_videos/demo.mp4"  # ← 改成你的视频路径

# 输出视频路径
OUTPUT_VIDEO_PATH = "output_annotated.mp4"

# YOLO 人脸检测模型路径（你已有的）
YOLO_MODEL_PATH = r"your_yolo_face_model.pt"  # ← 改成你的模型路径

# 识别阈值
KNOWN_THRESH = 0.50       # 熟人识别阈值（特征相似度 > 此值判定为熟人）
STRANGER_THRESH = 0.55    # 陌生人聚类阈值（> 此值判定为见过的陌生人）

# ByteTrack 相关
TRACK_BUFFER = 30          # 追踪缓冲帧数（同一 track_id 多少帧内认为是同一个人）
MIN_TRACK_FRAMES = 3       # 一个 track 至少出现几帧才进行识别
# ================================================


# ================================================================
#  1. 特征提取器：ResNet50（加载微调权重或 ImageNet 预训练）
# ================================================================

class ResNet50Extractor:
    """
    ResNet50 特征提取器
    
    输入: 人脸图片 (numpy BGR, 任意尺寸)
    输出: 2048 维 L2 归一化特征向量
    
    流程:
    图片 → resize(224x224) → 标准化 → ResNet50骨干 → GAP → [2048维]
    → BatchNorm → L2归一化 → 输出
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 构建模型结构
        self.backbone = self._build_backbone()
        self.bn = nn.BatchNorm1d(2048)
        
        # 加载权重
        if model_path and Path(model_path).exists():
            self._load_finetuned_weights(model_path)
            print(f"✅ 加载微调模型: {model_path}")
        else:
            print(f"⚠️  未找到微调模型，使用 ImageNet 预训练权重")
            print(f"   提示: 运行 train_arcface.py 可以获得更好的效果")
        
        self.backbone.to(self.device)
        self.bn.to(self.device)
        self.backbone.eval()
        self.bn.eval()
        
        # 预处理流水线
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        print(f"   特征维度: 2048")
        print(f"   运行设备: {self.device}")
    
    def _build_backbone(self):
        """构建 ResNet50 骨干（去掉最后的 FC 层）"""
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 去掉最后的全连接层
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        return backbone

    def _load_finetuned_weights(self, model_path):
        """加载微调后的权重"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
        else:
            self.backbone.load_state_dict(checkpoint)

        if 'bn' in checkpoint:
            self.bn.load_state_dict(checkpoint['bn'])

        # 打印模型信息
        if 'val_acc' in checkpoint:
            print(f"   模型验证准确率: {checkpoint['val_acc']:.2f}%")
        if 'class_names' in checkpoint:
            print(f"   训练类别数: {len(checkpoint['class_names'])}")

    @torch.no_grad()
    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        从单张人脸图片中提取 2048 维特征向量

        参数:
            face_img: numpy 数组 (BGR格式, 任意尺寸)
        返回:
            2048 维 L2 归一化特征向量，失败返回 None
        """
        try:
            # BGR → RGB
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 预处理
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            # ResNet50 提取特征
            feat = self.backbone(tensor)      # [1, 2048, 1, 1]
            feat = feat.view(-1)               # [2048]
            feat = feat.unsqueeze(0)           # [1, 2048] (BatchNorm 需要)

            # BatchNorm
            feat = self.bn(feat)               # [1, 2048]

            # L2 归一化
            feat = feat / torch.norm(feat, dim=1, keepdim=True)

            return feat.cpu().numpy().flatten()
        except Exception as e:
            return None


# ================================================================
#  2. 开集识别引擎：熟人库 + 增量聚类
# ================================================================

class FaceRecognizer:
    """
    开集人脸识别引擎（核心业务逻辑）

    "三连问"流程:
    第一问: 你是熟人吗？→ 查已知库 → 余弦相似度 > KNOWN_THRESH → 返回名字
    第二问: 之前见过吗？→ 查陌生人库 → > STRANGER_THRESH → 更新特征（增量聚类）
    第三问: 纯新人！→ 新建 Stranger_X 档案
    """

    def __init__(self, known_db_path=None):
        self.known_db = {}          # 熟人库 {"角色名": {"mean": [2048维], "count": 20}}
        self.stranger_db = {}       # 陌生人库 {"Stranger_1": {"vector": [2048维], "count": 5}}
        self.stranger_counter = 1

        self.KNOWN_THRESH = KNOWN_THRESH
        self.STRANGER_THRESH = STRANGER_THRESH

        # 加载熟人库
        if known_db_path and Path(known_db_path).exists():
            self._load_known_db(known_db_path)

    def _load_known_db(self, path):
        """加载已保存的熟人库"""
        with open(path, 'r', encoding='utf-8') as f:
            self.known_db = json.load(f)
        print(f"📂 已加载熟人库: {len(self.known_db)} 个角色")

    def save_known_db(self, path):
        """保存熟人库"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.known_db, f, ensure_ascii=False, indent=2)

    def save_stranger_db(self, path):
        """保存陌生人库"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stranger_db, f, ensure_ascii=False, indent=2)

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        """
        核心识别逻辑：传入 2048 维特征向量，返回识别结果

        参数:
            embedding: 2048 维 L2 归一化特征向量

        返回:
            (名字, 相似度, 类型)
            类型: 'known' (熟人) / 'stranger' (见过的陌生人) / 'new' (纯新人)
        """
        # 第一问：查熟人库
        best_name, best_sim = self._find_best_match(embedding, self.known_db)
        if best_name is not None and best_sim >= self.KNOWN_THRESH:
            return best_name, best_sim, "known"

        # 第二问：查陌生人库
        stranger_vectors = {k: v["vector"] for k, v in self.stranger_db.items()}
        best_sid, best_ssim = self._find_best_match(embedding, stranger_vectors)

        if best_sid is not None and best_ssim >= self.STRANGER_THRESH:
            # ★ 增量聚类核心：滑动平均更新特征质心
            old_vec = np.array(self.stranger_db[best_sid]["vector"])
            count = self.stranger_db[best_sid]["count"]

            # 新特征 = (旧特征 × 旧数量 + 新特征) / (旧数量 + 1)
            new_vec = (old_vec * count + embedding) / (count + 1)
            # 重新 L2 归一化（保持在单位球面上）
            new_vec = new_vec / np.linalg.norm(new_vec)

            self.stranger_db[best_sid]["vector"] = new_vec.tolist()
            self.stranger_db[best_sid]["count"] += 1

            return best_sid, best_ssim, "stranger"

        # 第三问：纯新人，新建档案
        new_id = f"Stranger_{self.stranger_counter}"
        self.stranger_db[new_id] = {
            "vector": embedding.tolist(),
            "count": 1
        }
        self.stranger_counter += 1

        return new_id, 0.0, "new"

    def _find_best_match(self, query_vec: np.ndarray, database: dict) -> Tuple[Optional[str], float]:
        """
        在数据库中查找与 query_vec 最相似的特征

        因为向量已经 L2 归一化，余弦相似度 = 点积（np.dot）
        """
        if not database:
            return None, -1.0

        best_name = None
        best_sim = -1.0

        for name, data in database.items():
            # 兼容两种数据结构
            db_vec = data["mean"] if "mean" in data else data["vector"]
            db_vec = np.array(db_vec)

            # 余弦相似度 = 点积（因为都已归一化）
            sim = float(np.dot(query_vec, db_vec))

            if sim > best_sim:
                best_sim = sim
                best_name = name

        return best_name, best_sim

    def get_stats(self):
        """获取统计信息"""
        return {
            "known_count": len(self.known_db),
            "stranger_count": len(self.stranger_db),
            "stranger_details": {
                k: v["count"] for k, v in self.stranger_db.items()
            }
        }


# ================================================================
#  3. 构建熟人库（线下准备阶段）
# ================================================================

def build_known_database(extractor: ResNet50Extractor, dataset_dir: str, output_path: str):
    """
    扫描数据集，为每个角色提取特征并计算平均向量

    流程:
    对每个角色文件夹:
        1. 读取该角色的所有图片
        2. 用 ResNet50 提取每张图的 2048 维特征
        3. 对所有特征取平均值，得到该角色的"标准身份证号"
        4. L2 归一化
    保存为 JSON 文件
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ 数据集不存在: {dataset_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"🚀 构建熟人特征库")
    print(f"   数据集: {dataset_dir}")
    print(f"{'='*60}")

    known_db = {}
    folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

    for idx, folder in enumerate(folders, 1):
        char_name = folder.name
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + \
                 list(folder.glob("*.jpeg")) + list(folder.glob("*.webp"))

        if len(images) < 3:
            print(f"   [{idx}/{len(folders)}] ⏭️  {char_name}: 图片太少 ({len(images)}), 跳过")
            continue

        # 提取所有图片的特征
        vectors = []
        for img_path in images[:50]:  # 每个角色最多取 50 张
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue

            emb = extractor.extract(img)
            if emb is not None:
                vectors.append(emb)

        if len(vectors) >= 3:
            # 计算平均特征向量（"标准身份证号"）
            mean_vec = np.mean(vectors, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec)  # L2 归一化

            known_db[char_name] = {
                "mean": mean_vec.tolist(),
                "count": len(vectors)
            }
            print(f"   [{idx}/{len(folders)}] ✅ {char_name}: {len(vectors)} 张特征")
        else:
            print(f"   [{idx}/{len(folders)}] ⚠️  {char_name}: 有效特征太少, 跳过")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(known_db, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 熟人库构建完成！")
    print(f"   角色数量: {len(known_db)}")
    print(f"   保存路径: {output_path}")
    print(f"{'='*60}")

    return known_db


# ================================================================
#  4. Track 级特征聚合（多帧特征融合，提升稳定性）
# ================================================================

class TrackFeatureAggregator:
    """
    Track 级特征聚合器

    作用:
    ByteTrack 给每一帧的人脸分配一个 track_id。
    同一个 track_id 的多帧人脸可能角度不同、质量不同。
    这个类把同一个 track_id 的多帧特征取平均，
    得到更稳定、更可靠的"Track-level 特征"。

    过滤策略:
    - 只在 track 结束时（连续 TRACK_BUFFER 帧没出现）才输出最终特征
    - 至少需要 MIN_TRACK_FRAMES 帧才输出（过滤闪现的误检）
    """

    def __init__(self, buffer=30, min_frames=3):
        self.buffer = buffer
        self.min_frames = min_frames

        # 存储每个 track_id 的特征和帧信息
        # {track_id: {"features": [...], "last_frame": int, "result": None}}
        self.tracks = {}

    def update(self, track_id, frame_idx, embedding):
        """
        更新某个 track 的特征

        返回:
            如果该 track 结束了（满足输出条件），返回 (track_id, averaged_embedding)
            否则返回 None
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "features": [],
                "last_frame": -999,
                "result": None
            }

        track = self.tracks[track_id]
        track["features"].append(embedding)
        track["last_frame"] = frame_idx

        # 检查是否有旧的 track 可以结算
        finished = []
        for tid, t in self.tracks.items():
            if frame_idx - t["last_frame"] > self.buffer and t["result"] is None:
                # Track 已结束
                if len(t["features"]) >= self.min_frames:
                    # 计算平均特征
                    avg_feat = np.mean(t["features"], axis=0)
                    avg_feat = avg_feat / np.linalg.norm(avg_feat)
                    t["result"] = (tid, avg_feat)
                    finished.append(tid)
                else:
                    # 帧数不够，丢弃
                    finished.append(tid)

        # 清理已结算的 track
        results = []
        for tid in finished:
            if self.tracks[tid]["result"] is not None:
                results.append(self.tracks[tid]["result"])
            del self.tracks[tid]

        return results

    def force_flush(self):
        """强制结算所有未完成的 track"""
        results = []
        for tid, t in self.tracks.items():
            if t["result"] is None and len(t["features"]) >= self.min_frames:
                avg_feat = np.mean(t["features"], axis=0)
                avg_feat = avg_feat / np.linalg.norm(avg_feat)
                results.append((tid, avg_feat))
        self.tracks.clear()
        return results


# ================================================================
#  5. 视频处理主流程
# ================================================================

def process_video(video_path, extractor, recognizer, yolo_model, output_path=None):
    """
    处理上传的动漫视频

    流程:
    读取视频 → 逐帧处理:
      → YOLO 检测人脸 → 获取 track_id
      → 裁剪人脸 → ResNet50 提取特征
      → Track 级特征聚合（多帧融合）
      → 开集识别（三连问）
      → 画面标注
    → 保存输出视频
    → 打印统计报告
    """
    print(f"\n{'='*60}")
    print(f"🎬 开始处理视频: {video_path}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   分辨率: {w}x{h}")
    print(f"   帧率: {fps:.1f} fps")
    print(f"   总帧数: {total_frames}")

    # 创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Track 特征聚合器
    aggregator = TrackFeatureAggregator(buffer=TRACK_BUFFER, min_frames=MIN_TRACK_FRAMES)

    # 实时显示的 track 标签 {track_id: (name, sim, type)}
    track_labels = {}

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ---- 1. YOLO 检测 + ByteTrack 追踪 (安全且优雅的写法) ----
        results = yolo_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.15,
            iou=0.6,
            tracker="bytetrack_anime.yaml"
        )

        valid_data = []  # 存储格式: [(x1, y1, x2, y2, track_id), ...]

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for (x1, y1, x2, y2), tid in zip(boxes_xyxy, track_ids):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

                # 过滤掉太小的框
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    valid_data.append((x1, y1, x2, y2, tid))

        # ---- 2. 提取特征 + 识别 ----
        for (x1, y1, x2, y2), tid in valid_data:
            # 裁剪人脸
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (112, 112))

            # 提取特征
            embedding = extractor.extract(face_resized)
            if embedding is None:
                continue

            # Track 级特征聚合
            finished_tracks = aggregator.update(tid, frame_idx, embedding)

            # 处理已结算的 track（识别结果）
            for finished_tid, avg_embedding in finished_tracks:
                name, sim, match_type = recognizer.recognize(avg_embedding)
                track_labels[finished_tid] = (name, sim, match_type)

            # 在画面上显示（用已有的标签或显示"识别中..."）
            if tid in track_labels:
                name, sim, match_type = track_labels[tid]
            else:
                name, sim, match_type = f"ID_{tid}", 0.0, "tracking"

            # ---- 3. 画面标注 ----
            if match_type == "known":
                color = (0, 200, 0)       # 绿色
                label = f"{name} ({sim:.2f})"
            elif match_type == "stranger":
                color = (0, 200, 255)     # 黄色
                label = f"{name} ({sim:.2f})"
            elif match_type == "new":
                color = (0, 100, 255)     # 蓝色
                label = f"{name} [NEW]"
            else:  # tracking
                color = (200, 200, 200)   # 灰色（正在追踪中）
                label = f"Tracking..."

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 画标签背景
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

            # 画文字
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # ---- 4. 状态栏 ----
        stats = recognizer.get_stats()
        status = (f"Frame: {frame_idx}/{total_frames} | "
                  f"Known: {stats['known_count']} | "
                  f"Strangers: {stats['stranger_count']} | "
                  f"Tracks: {len(aggregator.tracks)}")
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- 5. 写入输出视频 ----
        if writer:
            writer.write(frame)

        # 显示进度
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"   进度: {frame_idx}/{total_frames} "
                  f"({frame_idx/total_frames*100:.1f}%) "
                  f"| 速度: {fps_actual:.1f} fps "
                  f"| 剩余: {eta:.0f}s")

        cv2.imshow("Anime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- 结算剩余的 track ----
    remaining = aggregator.force_flush()
    for finished_tid, avg_embedding in remaining:
        name, sim, match_type = recognizer.recognize(avg_embedding)
        track_labels[finished_tid] = (name, sim, match_type)

    # ---- 清理 ----
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ---- 打印统计报告 ----
    elapsed = time.time() - start_time
    stats = recognizer.get_stats()

    print(f"\n{'='*60}")
    print(f"📊 处理完毕！")
    print(f"{'='*60}")
    print(f"   总帧数: {frame_idx}")
    print(f"   处理时间: {elapsed:.1f} 秒")
    print(f"   平均速度: {frame_idx/elapsed:.1f} fps")
    print(f"   熟人库: {stats['known_count']} 个角色")
    print(f"   追踪到的陌生人: {stats['stranger_count']} 个")

    if stats['stranger_details']:
        print(f"\n   陌生人详情:")
        for sid, count in sorted(stats['stranger_details'].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"     {sid}: 出现 {count} 次")

    # 保存陌生人库
    if stats['stranger_count'] > 0:
        stranger_path = "stranger_database.json"
        recognizer.save_stranger_db(stranger_path)
        print(f"\n   陌生人库已保存: {stranger_path}")

    if output_path:
        print(f"\n   标注视频已保存: {output_path}")

    print(f"{'='*60}")


# ================================================================
#  6. 主程序入口
# ================================================================

def main():
    print("=" * 60)
    print("    动漫视频角色识别系统 Demo")
    print("    ResNet50 + ArcFace + 增量聚类")
    print("=" * 60)

    # ---- 步骤 1: 加载特征提取模型 ----
    print("\n📥 步骤 1: 加载 ResNet50 特征提取模型...")
    extractor = ResNet50Extractor(model_path=MODEL_PATH)

    # ---- 步骤 2: 构建/加载熟人库 ----
    print(f"\n📚 步骤 2: 熟人库...")
    if not Path(KNOWN_DB_PATH).exists():
        known_db = build_known_database(extractor, DATASET_DIR, KNOWN_DB_PATH)
    else:
        print(f"   发现已有的熟人库: {KNOWN_DB_PATH}")
        recognizer_temp = FaceRecognizer(KNOWN_DB_PATH)

    # ---- 步骤 3: 初始化识别引擎 ----
    print(f"\n🧠 步骤 3: 初始化识别引擎...")
    recognizer = FaceRecognizer(known_db_path=KNOWN_DB_PATH)

    # ---- 步骤 4: 加载 YOLO 人脸检测模型 ----
    print(f"\n🔍 步骤 4: 加载 YOLO 人脸检测模型...")
    try:
        from ultralytics import YOLO
        if Path(YOLO_MODEL_PATH).exists():
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"   ✅ YOLO 模型已加载: {YOLO_MODEL_PATH}")
        else:
            # 如果没有指定的模型，用 YOLOv8n 做通用目标检测（fallback）
            print(f"   ⚠️  未找到指定模型: {YOLO_MODEL_PATH}")
            print(f"   使用 YOLOv8n 通用模型（可能需要调整）")
            yolo_model = YOLO("yolov8n.pt")
    except ImportError:
        print(f"   ❌ 未安装 ultralytics: pip install ultralytics")
        return

    # ---- 步骤 5: 处理视频 ----
    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ 找不到视频文件: {VIDEO_PATH}")
        print(f"   请修改配置区的 VIDEO_PATH 为你的视频路径")
        return

    process_video(
        video_path=VIDEO_PATH,
        extractor=extractor,
        recognizer=recognizer,
        yolo_model=yolo_model,
        output_path=OUTPUT_VIDEO_PATH
    )


if __name__ == "__main__":
    main()
