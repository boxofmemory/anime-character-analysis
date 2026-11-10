# 导入 YOLO 模型类（来自 Ultralytics 官方库）
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2

# 加载模型
model = YOLO("yolov8s_anime.pt")  # 确保这里是你最好的模型路径

# 打开视频
cap = cv2.VideoCapture(r"./test_02\Chainsaw_Man.mp4")

# 建立轨迹默认字典
track_history = defaultdict(lambda: [])

# 开始逐帧处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 追踪 (保持你的参数不变)
    results = model.track(frame, persist=True, conf=0.15, iou=0.6, tracker="bytetrack_anime.yaml")

    # ================= 关键修改 1：定义统一画板 =================
    # 先让 YOLO 生成一张带绿框和基础ID的图，我们就在这张图上继续画轨迹
    annotated_frame = results[0].plot()
    # ==========================================================

    # 获取检测框 (x, y, w, h)
    boxes = results[0].boxes.xywh.cpu()

    # 检查是否检测并跟踪到了目标
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 打印ID (保留你的功能)
        print("Track IDs:", track_ids)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]

            # 记录中心点
            track.append((float(x), float(y)))
            if len(track) > 30:  # 只保留最近30帧
                track.pop(0)

            # ================= 关键修改 2：在 annotated_frame 上画画 =================
            # 画轨迹线 (注意这里改成了 annotated_frame)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # 画更清晰的 ID 标签 (你的自定义标签)
            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # ======================================================================

    # ================= 关键修改 3：删除覆盖代码并统一缩放 =================
    # 我删除了你原本在这里的 for r in results: frame = r.plot()...
    # 因为那段代码会把你上面画的轨迹全部覆盖掉！

    # 缩放显示 (防止视频太大爆屏幕)
    # 只要 annotated_frame 存在，就进行缩放
    scale = 1280 / annotated_frame.shape[1]
    dim = (1280, int(annotated_frame.shape[0] * scale))
    display_frame = cv2.resize(annotated_frame, dim)

    # 显示处理后的视频
    cv2.imshow("YOLOv8 Inference", display_frame)
    # ===================================================================

    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()