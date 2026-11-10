# 导入 YOLO 模型类（来自 Ultralytics 官方库）
from ultralytics import YOLO

from collections import defaultdict
import numpy as np

# 导入 OpenCV，用于视频读取、显示和图像处理
import cv2

# 加载训练好的 YOLOv8 模型（这里是针对动漫风格优化的小模型）
# 可替换为你的 best.pt（如：model = YOLO("runs/detect/train/weights/best.pt")）
model = YOLO("yolov8s_anime.pt")

# 打开本地视频文件进行读取
# 使用原始字符串 r"..." 避免 Windows 路径中的反斜杠转义问题
cap = cv2.VideoCapture(r"./test_02\Chainsaw_Man.mp4")

# 建立轨迹默认字典，用于画轨迹
track_history = defaultdict(lambda:[])

# 开始逐帧处理视频
while True:
    # 从视频流中读取一帧
    # ret: 是否成功读取帧（True/False）
    # frame: 当前帧的图像数据（NumPy 数组）
    ret, frame = cap.read()

    # 如果视频结束或读取出错，跳出循环
    if not ret:
        break

    # 使用 YOLOv8 模型对当前帧进行 **带跟踪的目标检测**
    # 参数说明：
    #   - frame: 输入图像
    #   - persist=True: 保持跨帧的跟踪状态（确保 ID 连续）
    #   - tracker="bytetrack.yaml": 指定使用 ByteTrack 跟踪器（需在 ultralytics/cfg/trackers/ 下有该配置文件）
    # 新增： 1. conf=0.1(default:0.25)-只保留模型预测置信度 ≥ 0.1 的检测框。
    #       2. iou=0.5(default:0.7)-若两个框 IoU > 0.5，只保留置信度高的那个。
    results = model.track(frame, persist=True, conf=0.1, iou=0.5, tracker="bytetrack_anime.yaml")

    # 无论有没有检测到人，先生成一张带框的基础图
    annotated_frame = results[0].plot()

    # 先用 YOLO 自带 plot 画框和 ID
    frame = results[0].plot()

    # 获取检测框和ID
    boxes = results[0].boxes.xywh.cpu()

    # 检查是否检测并跟踪到了目标（是否有分配的 track ID）
    if results[0].boxes.id is not None:
        # 将 GPU 上的 ID 张量转为 CPU，并转换为整数列表
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # 打印当前帧中所有被跟踪目标的 ID
        print("Track IDs:", track_ids)

        for box,track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            # 记录中心点
            track.append((float(x), float(y)))
            if len(track) > 30:  # 只保留最近30帧的轨迹
                track.pop(0)

                # 缩放到宽 1280 (保持比例)
                scale = 1280 / annotated_frame.shape[1]
                dim = (1280, int(annotated_frame.shape[0] * scale))
                display_frame = cv2.resize(annotated_frame, dim)

            # 画轨迹线
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # 画更清晰的 ID 标签
            cv2.putText(frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # 【可选】旧版画框方式（已注释）
    # annotated_frame = results[0].plot()  # 自动绘制检测框、类别、置信度和 Track ID
    # cv2.imshow("Tracking", annotated_frame)

    # 新版推荐方式：遍历 results（虽然通常只有1个结果），对每帧绘制检测+跟踪结果
    for r in results:
        frame = r.plot()  # 在原图上叠加边界框、标签、置信度和 Track ID（YOLOv8.1+ 已内置支持）

    # 绘制自带的框（可选，为了对比）
    frame = results[0].plot()

    # 显示处理后的视频帧窗口，标题为 "YOLOv8 Inference"
    cv2.imshow("YOLOv8 Inference", frame)

    # 等待 1 毫秒，同时检测键盘按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 'q' 键，退出循环（停止播放）
    if key == ord('q'):
        break

# 释放视频资源（关闭文件句柄）
cap.release()

# 关闭所有 OpenCV 创建的窗口
cv2.destroyAllWindows()