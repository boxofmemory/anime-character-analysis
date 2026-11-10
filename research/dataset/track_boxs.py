from ultralytics import YOLO
import cv2
import torch

# -----------------------------
# 1. 加载模型（可替换为你自己的 .pt 文件）
# -----------------------------
model = YOLO("yolov8s_anime.pt")

# -----------------------------
# 2. 准备一张测试图像（可用本地图片或从视频读一帧）
# -----------------------------
# 方式一：用 OpenCV 读一张图（推荐）
img_path = "img_test.png"  # ← 改成你自己的图片路径
frame = cv2.imread(img_path)

# 如果没有图片，也可以生成一个空白图（但可能检测不到目标）
# frame = (255 * torch.rand(480, 640, 3)).numpy().astype('uint8')

if frame is None:
    print(f"⚠️ 图像 {img_path} 无法加载，请检查路径！")
    exit()

# -----------------------------
# 3. 运行跟踪（关键步骤）
# -----------------------------
print("🚀 正在运行 model.track() ...")
results = model.track(frame, persist=True, tracker="bytetrack_anime.yaml")

# -----------------------------
# 4. 探索 results 结构（重点！）
# -----------------------------
print("\n🔍 === 整体结构 ===")
print(f"type(results): {type(results)}")  # 通常是 list
print(f"len(results): {len(results)}")  # 一般为 1（每帧一个结果）

r = results[0]  # 取第一个（也是唯一一个）Result 对象
print(f"\ntype(r): {type(r)}")  # <class 'ultralytics.engine.results.Results'>

# -----------------------------
# 5. 查看 boxes（最核心部分）
# -----------------------------
print("\n📦 === r.boxes 信息 ===")
print(f"type(r.boxes): {type(r.boxes)}")  # <class 'ultralytics.engine.results.Boxes'>

if r.boxes is not None and len(r.boxes) > 0:
    print(f"检测到 {len(r.boxes)} 个目标")

    # 坐标 [x1, y1, x2, y2]
    print(f"\n- r.boxes.xyxy: \n{r.boxes.xyxy}")  # shape: (N, 4)
    print(f"  类型: {r.boxes.xyxy.dtype}, 设备: {r.boxes.xyxy.device}")

    # 置信度
    print(f"\n- r.boxes.conf: \n{r.boxes.conf}")  # shape: (N,)

    # 类别 ID
    print(f"\n- r.boxes.cls: \n{r.boxes.cls}")  # shape: (N,)

    # 👉 跟踪 ID（只有 track() 才有！）
    print(f"\n- r.boxes.id: \n{r.boxes.id}")  # shape: (N,)，可能为 None
    if r.boxes.id is not None:
        print(f"  Track IDs (CPU 列表): {r.boxes.id.int().cpu().tolist()}")

else:
    print(" 未检测到任何目标！请换一张含目标的图。")

# -----------------------------
# 6. 其他有用字段（可选了解）
# -----------------------------
print("\n === 其他字段 ===")
print(f"r.orig_img.shape: {r.orig_img.shape}")  # 原始输入图像 (H, W, C)
print(f"r.path: {r.path}")  # 图像路径（如果是文件）
print(f"r.names: {r.names}")  # 类别名字典，如 {0: 'person', 1: 'bicycle', ...}

# -----------------------------
# 7. 可视化（确认是否跟踪成功）
# -----------------------------
annotated = r.plot()  # 自动画框 + 标签 + Track ID
cv2.imshow("YOLOv8 Track Result", annotated)
print("\n🖼 已显示结果图，按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()