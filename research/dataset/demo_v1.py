# ===================================================================
# 数据资产化：自动保存结果视频（答辩用）、轨迹JSON（统计用）、人脸小图（识别用）。
# 性能优化：一次性将 Tensor 转为 Numpy，减少 CPU/GPU 通信；引入内存清理机制。
# 交互增强：增加 暂停 (Space)、单帧步进 (N)、退出 (Q) 功能，方便调试每一帧。
# 鲁棒性：修复了 polylines 绘图崩溃、空帧报错等所有你提到的隐患。
# ===================================================================





import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import json
import os
import shutil

# ================= 配置区域 =================
VIDEO_PATH = r"./test_02\Chainsaw_Man.mp4"
MODEL_PATH = "yolov8s_anime.pt"  # 你的微调模型
TRACKER_CFG = "bytetrack_anime.yaml"  # 你的配置文件

# 输出设置
OUTPUT_DIR = "output_data"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")  # 存人脸小图
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, "result.mp4")  # 存结果视频
JSON_OUT_PATH = os.path.join(OUTPUT_DIR, "tracks.json")  # 存轨迹数据

# 参数设置
CONF_THRESH = 0.1  # 检测置信度
IOU_THRESH = 0.5  # NMS阈值
MAX_TRACK_HISTORY = 30  # 画轨迹只画最近30帧
CLEANUP_INTERVAL = 30  # 每30帧清理一次失踪人口
MAX_MISSING_FRAMES = 60  # 超过60帧没出现就删掉记录

# ================= 初始化环境 =================
# 1. 创建输出文件夹 (如果存在则清空，保证每次跑都是新的)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(CROP_DIR, exist_ok=True)

# 2. 加载模型
print(f"Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 3. 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 4. 初始化视频写入器 (答辩演示必备)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (width, height))

# ================= 状态变量 =================
track_history = defaultdict(lambda: [])  # 记录轨迹点
track_last_seen = {}  # 记录最后出现的帧号 (用于清理)
track_colors = {}  # 记录每个ID的颜色
all_tracks_data = []  # 用于保存JSON
paused = False  # 暂停状态标记


def get_color(idx):
    """为每个ID生成固定的随机颜色"""
    if idx not in track_colors:
        # 生成随机颜色 (B, G, R)
        track_colors[idx] = np.random.randint(0, 255, size=3).tolist()
    return track_colors[idx]


print(f"Start processing {total_frames} frames...")
print("按 'SPACE' 暂停/继续，按 'N' 单帧步进，按 'Q' 退出")

try:
    frame_idx = 0
    while True:
        # --- 交互控制 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User requested exit.")
            break
        elif key == ord(' '):  # 空格键暂停
            paused = not paused
        elif key == ord('n'):  # N键单帧步进（仅在暂停时有效）
            pass  # 逻辑向下执行

        # 如果暂停中且没有按N，则无限循环等待
        if paused and key != ord('n'):
            continue

        # --- 读取帧 ---
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- 计时 ---
        start_time = time.time()

        # --- 核心推理 (YOLO + ByteTrack) ---
        # verbose=False 不打印每帧的详细日志，保持控制台干净
        results = model.track(frame, persist=True, tracker=TRACKER_CFG,
                              conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

        # --- 性能优化：一次性转 CPU Numpy ---
        # 避免在循环里频繁 device copy
        if results[0].boxes.id is not None:
            # 拿到所有数据
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # 用于裁剪
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()  # 用于画轨迹
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # 你的 "annotated_frame" (YOLO自带画图，作为底图)
            annotated_frame = results[0].plot()

            # --- 遍历每个目标 ---
            for box_xyxy, box_xywh, track_id, conf in zip(boxes_xyxy, boxes_xywh, track_ids, confs):
                track_id = int(track_id)  # 确保是 Python int

                # 1. 更新最后出现时间
                track_last_seen[track_id] = frame_idx

                # 2. 记录轨迹
                x, y, w, h = box_xywh
                center = (float(x), float(y))
                track_history[track_id].append(center)
                if len(track_history[track_id]) > MAX_TRACK_HISTORY:
                    track_history[track_id].pop(0)

                # 3. 【核心功能】保存人脸小图 (为识别做准备)
                # 坐标边界保护，防止超出图片范围报错
                x1, y1, x2, y2 = map(int, box_xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x2 > x1 and y2 > y1:  # 确保框有效
                    face_img = frame[y1:y2, x1:x2]
                    # 按ID存文件夹: output/crops/101/frame_001.jpg
                    id_dir = os.path.join(CROP_DIR, str(track_id))
                    os.makedirs(id_dir, exist_ok=True)
                    crop_name = f"frame_{frame_idx:06d}.jpg"
                    # 这里可以加逻辑：每隔5帧保存一张，节省空间
                    if frame_idx % 5 == 0:
                        cv2.imwrite(os.path.join(id_dir, crop_name), face_img)

                # 4. 【核心功能】记录数据到内存 (为JSON做准备)
                all_tracks_data.append({
                    "frame": frame_idx,
                    "id": track_id,
                    "box": [x1, y1, x2, y2],
                    "conf": float(conf)
                })

                # 5. 绘制轨迹 (鲁棒性绘制)
                color = get_color(track_id)
                pts_list = track_history[track_id]
                # 只有点数 >= 2 才画线，避免报错
                if len(pts_list) >= 2:
                    # 这里的 reshape 是 cv2.polylines 要求的固定格式
                    pts = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

                # 绘制更大的ID
                cv2.putText(annotated_frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        else:
            # 如果没检测到人，就直接用原图
            annotated_frame = frame

        # --- 内存清理机制 ---
        # 每隔一定帧数，清理掉很久没出现过的 ID 轨迹，防止内存爆炸
        if frame_idx % CLEANUP_INTERVAL == 0:
            active_ids = []
            for tid in list(track_history.keys()):
                last_frame = track_last_seen.get(tid, 0)
                if frame_idx - last_frame > MAX_MISSING_FRAMES:
                    del track_history[tid]  # 删掉轨迹
                    # del track_last_seen[tid] # last_seen保留也没事，占不了多少内存
                else:
                    active_ids.append(tid)

        # --- 界面信息 ---
        proc_time = time.time() - start_time
        curr_fps = 1.0 / (proc_time + 1e-6)
        info_text = f"Frame: {frame_idx}/{total_frames} | FPS: {curr_fps:.1f} | IDs: {len(track_history)}"
        cv2.putText(annotated_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if paused:
            cv2.putText(annotated_frame, "PAUSED (Press 'N' to step)", (width // 2 - 150, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # --- 保存与显示 ---
        # 1. 写入视频文件
        out_writer.write(annotated_frame)

        # 2. 屏幕显示 (带缩放保护)
        display_h = 720
        scale = display_h / height
        display_w = int(width * scale)
        display_frame = cv2.resize(annotated_frame, (display_w, display_h))
        cv2.imshow("Advanced Tracking System", display_frame)

finally:
    # ================= 收尾工作 =================
    print("Cleaning up...")
    cap.release()
    out_writer.release()  # 记得保存视频！
    cv2.destroyAllWindows()

    # 保存轨迹数据到 JSON
    print(f"Saving tracking data to {JSON_OUT_PATH}...")
    with open(JSON_OUT_PATH, 'w') as f:
        json.dump(all_tracks_data, f)

    print("Done! Check 'output_data' folder.")