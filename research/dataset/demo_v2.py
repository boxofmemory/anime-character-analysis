# ===================================================================
# 数据资产化：自动保存结果视频（答辩用）、轨迹JSON（统计用）、人脸小图（识别用）。
# 性能优化：一次性将 Tensor 转为 Numpy，减少 CPU/GPU 通信；引入内存清理机制。
# 交互增强：增加 暂停 (Space)、单帧步进 (N)、退出 (Q) 功能，方便调试每一帧。
# 鲁棒性：修复了 polylines 绘图崩溃、空帧报错等所有隐患。
# 新增优化：断点续传、质量筛选、热力轨迹、多线程IO、确定性颜色
# ===================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys
from datetime import datetime

# ================= 配置区域 =================
VIDEO_PATH = r"./test_02\Chainsaw_Man.mp4"
MODEL_PATH = "yolov8s_anime.pt"  # 微调模型
TRACKER_CFG = "bytetrack_anime.yaml"  # 配置文件

# 输出设置
OUTPUT_DIR = "output_data"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")  # 存人脸小图
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, "result.mp4")  # 存结果视频
JSON_OUT_PATH = os.path.join(OUTPUT_DIR, "tracks.json")  # 存轨迹数据
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.json")  # 断点文件

# 参数设置
CONF_THRESH = 0.1  # 检测置信度
IOU_THRESH = 0.5  # NMS阈值
MAX_TRACK_HISTORY = 30  # 画轨迹只画最近30帧
CLEANUP_INTERVAL = 30  # 每30帧清理一次失踪人口
MAX_MISSING_FRAMES = 60  # 超过60帧没出现就删掉记录

# 小图保存设置
CROP_INTERVAL = 5  # 每5帧保存一张
MIN_FACE_AREA = 5000  # 最小人脸面积（像素）
MIN_LAPLACIAN_VAR = 100  # 最小清晰度（拉普拉斯方差）
MAX_JSON_BUFFER = 5000  # JSON数据内存缓冲区大小，超过则刷盘

# 断点设置
CHECKPOINT_INTERVAL = 1000  # 每1000帧保存进度
RESUME_FROM_CHECKPOINT = False  # 是否从断点恢复


# ================= 初始化环境 =================
def setup_environment():
    """初始化输出环境"""
    if os.path.exists(OUTPUT_DIR) and not RESUME_FROM_CHECKPOINT:
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(CROP_DIR, exist_ok=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 输出目录: {os.path.abspath(OUTPUT_DIR)}")


def load_model_safe():
    """安全加载模型，带错误处理"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载模型: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        print("请确保模型文件路径正确，或使用默认YOLOv8模型:")
        print('  model = YOLO("yolov8s.pt")  # 使用官方预训练模型')
        sys.exit(1)

    try:
        model = YOLO(MODEL_PATH)
        print(f"模型加载成功: {model.task}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


def setup_video_io():
    """设置视频输入输出"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {VIDEO_PATH}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 尝试使用H264编码，失败则回退到mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

    if not out_writer.isOpened():
        print("H264编码器不可用，回退到mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

    return cap, out_writer, width, height, fps, total_frames


# ================= 核心功能类 =================
class TrackManager:
    """管理跟踪状态、颜色和轨迹"""

    def __init__(self):
        self.track_history = defaultdict(list)
        self.track_last_seen = {}
        self.track_colors = {}
        self.created_dirs = set()

    def get_color(self, idx):
        """基于ID生成确定性颜色，保证相同ID颜色一致且区分度高"""
        if idx not in self.track_colors:
            # 使用黄金角偏移的色相，保证颜色分布均匀
            hue = (idx * 137.508) % 360
            hsv = np.uint8([[[hue / 2, 200, 255]]])  # hue: 0-179 in OpenCV
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self.track_colors[idx] = [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        return self.track_colors[idx]

    def update_track(self, track_id, center, frame_idx):
        """更新轨迹"""
        self.track_history[track_id].append(center)
        self.track_last_seen[track_id] = frame_idx
        if len(self.track_history[track_id]) > MAX_TRACK_HISTORY:
            self.track_history[track_id].pop(0)

    def cleanup_old_tracks(self, current_frame):
        """清理长时间未出现的轨迹"""
        active_ids = []
        removed_count = 0
        for tid in list(self.track_history.keys()):
            last_frame = self.track_last_seen.get(tid, 0)
            if current_frame - last_frame > MAX_MISSING_FRAMES:
                del self.track_history[tid]
                removed_count += 1
            else:
                active_ids.append(tid)
        return active_ids, removed_count

    def get_heatmap_trajectory(self, track_id):
        """获取带透明度权重的轨迹点"""
        pts = self.track_history[track_id]
        if len(pts) < 2:
            return None
        # 返回点列表和对应的权重（越新权重越高）
        weights = np.linspace(0.3, 1.0, len(pts))
        return pts, weights


class DataSaver:
    """管理数据保存，支持异步IO和断点续传"""

    def __init__(self):
        self.all_tracks_data = []
        self.frame_count_since_flush = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.json_buffer_file = os.path.join(OUTPUT_DIR, "tracks_buffer.jsonl")

    def add_track_data(self, data):
        """添加轨迹数据到缓冲区"""
        self.all_tracks_data.append(data)
        self.frame_count_since_flush += 1

        # 缓冲区满时刷盘，防止内存爆炸
        if len(self.all_tracks_data) >= MAX_JSON_BUFFER:
            self.flush_buffer()

    def flush_buffer(self):
        """将缓冲区数据写入临时文件"""
        if not self.all_tracks_data:
            return

        with open(self.json_buffer_file, 'a', encoding='utf-8') as f:
            for item in self.all_tracks_data:
                f.write(json.dumps(item) + '\n')

        self.all_tracks_data = []
        self.frame_count_since_flush = 0

    def save_crop_async(self, face_img, filepath):
        """异步保存裁剪图片"""

        def save_task(img, path):
            try:
                cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e:
                print(f"保存图片失败 {path}: {e}")

        self.executor.submit(save_task, face_img.copy(), filepath)

    def save_checkpoint(self, frame_idx, track_manager):
        """保存检查点"""
        checkpoint = {
            'frame_idx': frame_idx,
            'track_last_seen': track_manager.track_last_seen,
            'track_colors': track_manager.track_colors,
            'timestamp': datetime.now().isoformat()
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f)

    def finalize(self):
        """收尾工作：合并所有数据"""
        self.executor.shutdown(wait=True)

        # 合并缓冲区数据
        final_data = []
        if os.path.exists(self.json_buffer_file):
            with open(self.json_buffer_file, 'r', encoding='utf-8') as f:
                for line in f:
                    final_data.append(json.loads(line.strip()))

        # 添加内存中的剩余数据
        final_data.extend(self.all_tracks_data)

        # 保存最终JSON
        with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2)

        # 清理临时文件
        if os.path.exists(self.json_buffer_file):
            os.remove(self.json_buffer_file)
            print(f"已清理临时文件")

        return len(final_data)


class QualityFilter:
    """人脸质量筛选器"""

    def __init__(self):
        self.quality_cache = {}

    def check_quality(self, face_img, track_id):
        """检查人脸质量，返回是否通过"""
        # 尺寸检查
        h, w = face_img.shape[:2]
        area = h * w
        if area < MIN_FACE_AREA:
            return False, "too_small"

        # 清晰度检查（每10帧计算一次，缓存结果）
        if track_id not in self.quality_cache or self.quality_cache[track_id]['count'] % 10 == 0:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            self.quality_cache[track_id] = {'var': lap_var, 'count': 0}

        self.quality_cache[track_id]['count'] += 1

        if self.quality_cache[track_id]['var'] < MIN_LAPLACIAN_VAR:
            return False, "too_blur"

        return True, "ok"


# ================= 可视化功能 =================
def draw_heatmap_trajectory(frame, track_manager, track_id):
    """绘制热力轨迹（越新越亮、越粗）"""
    result = track_manager.get_heatmap_trajectory(track_id)
    if result is None:
        return frame

    pts, weights = result
    color = track_manager.get_color(track_id)

    # 创建透明叠加层
    overlay = frame.copy()

    for i in range(1, len(pts)):
        pt1 = tuple(map(int, pts[i - 1]))
        pt2 = tuple(map(int, pts[i]))

        # 动态粗细和透明度
        thickness = int(1 + weights[i] * 4)
        alpha = weights[i]

        # 在overlay上画线
        cv2.line(overlay, pt1, pt2, color, thickness)

        # 混合到原图
        frame = cv2.addWeighted(frame, 1 - alpha * 0.3, overlay, alpha * 0.3, 0)
        overlay = frame.copy()  # 更新overlay

    return frame


def draw_info_panel(frame, frame_idx, total_frames, fps, active_ids, paused, processing_time):
    """绘制信息面板"""
    h, w = frame.shape[:2]

    # 背景条
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)

    # 顶部信息
    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
    info_text = f"Frame: {frame_idx}/{total_frames} ({progress:.1f}%) | FPS: {fps:.1f} | Active IDs: {active_ids} | Time: {processing_time:.3f}s"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 底部控制提示
    control_text = "SPACE: Pause | N: Step | Q: Quit | S: Screenshot"
    cv2.putText(frame, control_text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 暂停指示器
    if paused:
        # 半透明遮罩
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # 大文字提示
        text = "PAUSED (Press 'N' to step, SPACE to resume)"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, text, ((w - text_w) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame


# ================= 主程序 =================
def main():
    # 初始化
    setup_environment()
    model = load_model_safe()
    cap, out_writer, width, height, fps, total_frames = setup_video_io()

    # 初始化管理器
    track_manager = TrackManager()
    data_saver = DataSaver()
    quality_filter = QualityFilter()

    # 状态变量
    paused = False
    step_mode = False
    frame_idx = 0
    start_time_total = time.time()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始处理 {total_frames} 帧...")
    print("控制键: [SPACE]暂停/继续 | [N]单帧步进 | [Q]退出 | [S]截图")
    print("-" * 60)

    try:
        while True:
            # --- 交互控制 ---
            key = cv2.waitKey(1 if not paused else 0) & 0xFF  # 暂停时不等待

            if key == ord('q'):
                print("\n用户请求退出...")
                break
            elif key == ord(' '):
                paused = not paused
                step_mode = False
                print(f"{'暂停' if paused else '继续'}处理...")
            elif key == ord('n'):
                if paused:
                    step_mode = True
                    paused = False  # 临时解除暂停
                    print("单帧步进...")
            elif key == ord('s'):
                # 截图功能
                screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{frame_idx:06d}.jpg")
                cv2.imwrite(screenshot_path, display_frame if 'display_frame' in locals() else annotated_frame)
                print(f"截图已保存: {screenshot_path}")

            # 如果暂停中且不是步进模式，继续循环等待
            if paused and not step_mode:
                continue

            # --- 读取帧 ---
            ret, frame = cap.read()
            if not ret:
                print(f"\n视频读取结束，共处理 {frame_idx} 帧")
                break

            frame_idx += 1
            loop_start = time.time()

            # --- 核心推理 ---
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False
            )

            # --- 处理检测结果 ---
            if results[0].boxes.id is not None:
                # 一次性转Numpy（性能优化）
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                boxes_xywh = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                # YOLO自带可视化作为底图
                annotated_frame = results[0].plot()

                # 遍历每个目标
                for box_xyxy, box_xywh, track_id, conf in zip(
                        boxes_xyxy, boxes_xywh, track_ids, confs
                ):
                    track_id = int(track_id)
                    x, y, w, h = box_xywh
                    center = (float(x), float(y))

                    # 更新轨迹
                    track_manager.update_track(track_id, center, frame_idx)

                    # 坐标边界保护
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # 保存高质量小图
                    if x2 > x1 and y2 > y1 and frame_idx % CROP_INTERVAL == 0:
                        face_img = frame[y1:y2, x1:x2]

                        # 质量筛选
                        is_quality_ok, reason = quality_filter.check_quality(face_img, track_id)

                        if is_quality_ok:
                            # 确保目录存在（缓存优化）
                            if track_id not in track_manager.created_dirs:
                                id_dir = os.path.join(CROP_DIR, str(track_id))
                                os.makedirs(id_dir, exist_ok=True)
                                track_manager.created_dirs.add(track_id)

                            # 异步保存
                            crop_path = os.path.join(CROP_DIR, str(track_id), f"frame_{frame_idx:06d}.jpg")
                            data_saver.save_crop_async(face_img, crop_path)

                    # 记录数据
                    data_saver.add_track_data({
                        "frame": frame_idx,
                        "id": track_id,
                        "box": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "center": [float(x), float(y)]
                    })

                    # 绘制热力轨迹
                    annotated_frame = draw_heatmap_trajectory(
                        annotated_frame, track_manager, track_id
                    )

                    # 绘制ID标签（带背景）
                    label = f"ID:{track_id} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    label_x = int(x - w / 2)
                    label_y = int(y - h / 2) - 10

                    # 标签背景
                    cv2.rectangle(
                        annotated_frame,
                        (label_x - 5, label_y - text_h - 5),
                        (label_x + text_w + 5, label_y + 5),
                        track_manager.get_color(track_id),
                        -1
                    )
                    cv2.putText(
                        annotated_frame, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
            else:
                annotated_frame = frame.copy()

            # --- 内存清理 ---
            if frame_idx % CLEANUP_INTERVAL == 0:
                active_ids, removed = track_manager.cleanup_old_tracks(frame_idx)
                if removed > 0:
                    print(f"  [帧 {frame_idx}] 清理 {removed} 个失踪ID，当前活跃: {len(active_ids)}")

            # --- 断点保存 ---
            if frame_idx % CHECKPOINT_INTERVAL == 0:
                data_saver.flush_buffer()
                data_saver.save_checkpoint(frame_idx, track_manager)
                elapsed = time.time() - start_time_total
                progress = frame_idx / total_frames * 100
                eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                print(
                    f"  [检查点] 帧 {frame_idx}/{total_frames} ({progress:.1f}%) | 已用: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")

            # --- 计算性能指标 ---
            proc_time = time.time() - loop_start
            curr_fps = 1.0 / (proc_time + 1e-6)

            # --- 绘制UI ---
            active_ids = len(track_manager.track_history)
            annotated_frame = draw_info_panel(
                annotated_frame, frame_idx, total_frames,
                curr_fps, active_ids, paused, proc_time
            )

            # --- 写入视频 ---
            out_writer.write(annotated_frame)

            # --- 显示（带缩放保护）---
            display_h = min(720, height)
            scale = display_h / height
            display_w = int(width * scale)
            display_frame = cv2.resize(annotated_frame, (display_w, display_h))
            cv2.imshow("Advanced Tracking System v2.0", display_frame)

            # --- 步进模式恢复暂停 ---
            if step_mode:
                paused = True
                step_mode = False

    except KeyboardInterrupt:
        print("\n收到中断信号，正在保存进度...")
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ================= 收尾工作 =================
        print("\n" + "=" * 60)
        print("正在清理资源...")

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

        # 保存最终数据
        print("正在合并数据文件...")
        total_records = data_saver.finalize()

        # 清理检查点
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

        # 统计信息
        elapsed_total = time.time() - start_time_total
        avg_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0

        print(f"\n处理完成统计:")
        print(f"  - 总帧数: {frame_idx}")
        print(f"  - 总时间: {elapsed_total:.2f}秒")
        print(f"  - 平均FPS: {avg_fps:.2f}")
        print(f"  - 轨迹记录: {total_records} 条")
        print(f"  - 活跃ID数: {len(track_manager.track_history)}")
        print(f"  - 保存人脸: {len(track_manager.created_dirs)} 个ID")
        print(f"\n输出文件:")
        print(f"  - 视频: {os.path.abspath(VIDEO_OUT_PATH)}")
        print(f"  - 数据: {os.path.abspath(JSON_OUT_PATH)}")
        print(f"  - 人脸: {os.path.abspath(CROP_DIR)}")
        print("=" * 60)


if __name__ == "__main__":
    main()