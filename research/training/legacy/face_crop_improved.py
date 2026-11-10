"""
动漫人脸裁剪与预处理脚本（改进版）
=====================================
改进点：
1. 加入位置权重（中心优先策略）
2. 修复边界裁剪问题（保证正方形）
3. 加入质量过滤（置信度、大小、长宽比）
4. 加入模糊度检测
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ===================== 配置区 =====================
YOLO_MODEL_PATH = "yolov8s_anime.pt"

INPUT_DIR = r"./datasets/small"
OUTPUT_DIR = r"./datasets/small_112x112"

TARGET_SIZE = (112, 112)
MARGIN_RATIO = 0.2

# 质量过滤阈值
MIN_CONFIDENCE = 0.5      # YOLOv8 置信度阈值
MIN_FACE_SIZE = 30        # 最小人脸尺寸
MAX_ASPECT_RATIO = 1.5    # 最大长宽比
MIN_SHARPNESS = 100       # 最小清晰度（Laplacian 方差）

# ================================================

def cv2_imread_safe(file_path):
    """安全读取图片（兼容特殊字符路径）"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def cv2_imwrite_safe(file_path, img):
    """安全保存图片（兼容特殊字符路径）"""
    cv2.imencode('.jpg', img)[1].tofile(file_path)

def check_sharpness(img):
    """检测图片清晰度（Laplacian 方差）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def select_main_face(boxes, confidences, img_w, img_h):
    """
    选择主角人脸（综合面积和位置）
    """
    if len(boxes) == 0:
        return None
    
    img_center_x = img_w / 2
    img_center_y = img_h / 2
    
    best_score = -1
    best_box = boxes[0]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # 面积得分
        area = (x2 - x1) * (y2 - y1)
        area_score = area / (img_w * img_h)
        
        # 位置得分
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        distance = np.sqrt((box_center_x - img_center_x)**2 + 
                          (box_center_y - img_center_y)**2)
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        position_score = 1 - (distance / max_distance)
        
        # 置信度得分
        conf_score = confidences[i] if i < len(confidences) else 0.5
        
        # 综合得分
        total_score = (area_score * 0.5 + 
                      position_score * 0.3 + 
                      conf_score * 0.2)
        
        if total_score > best_score:
            best_score = total_score
            best_box = box
    
    return best_box

def crop_face_safely(img, box, margin_ratio=0.2):
    """
    安全裁剪人脸（保证正方形，处理边界）
    """
    x1, y1, x2, y2 = box
    img_h, img_w = img.shape[:2]
    
    # 计算正方形边长
    w = x2 - x1
    h = y2 - y1
    side = max(w, h) * (1 + 2 * margin_ratio)
    
    # 计算中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算正方形边界
    sq_x1 = int(center_x - side / 2)
    sq_y1 = int(center_y - side / 2)
    sq_x2 = int(center_x + side / 2)
    sq_y2 = int(center_y + side / 2)
    
    # 调整到图像边界内（保持正方形）
    # 处理 x 方向
    if sq_x2 > img_w:
        shift = sq_x2 - img_w
        sq_x1 -= shift
        sq_x2 -= shift
    if sq_x1 < 0:
        shift = -sq_x1
        sq_x1 += shift
        sq_x2 += shift
    
    # 处理 y 方向
    if sq_y2 > img_h:
        shift = sq_y2 - img_h
        sq_y1 -= shift
        sq_y2 -= shift
    if sq_y1 < 0:
        shift = -sq_y1
        sq_y1 += shift
        sq_y2 += shift
    
    # 最终边界检查
    sq_x1 = max(0, sq_x1)
    sq_y1 = max(0, sq_y1)
    sq_x2 = min(img_w, sq_x2)
    sq_y2 = min(img_h, sq_y2)
    
    # 裁剪
    face_crop = img[sq_y1:sq_y2, sq_x1:sq_x2]
    
    return face_crop

def process_dataset():
    print("🚀 正在加载 YOLOv8 人脸检测模型...")
    model = YOLO(YOLO_MODEL_PATH)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    char_folders = [f for f in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, f))]
    total_chars = len(char_folders)
    
    stats = {
        "processed": 0,
        "no_face": 0,
        "low_quality": 0,
        "saved": 0,
        "errors": 0
    }

    for idx, char_name in enumerate(char_folders, 1):
        input_char_dir = os.path.join(INPUT_DIR, char_name)
        output_char_dir = os.path.join(OUTPUT_DIR, char_name)
        os.makedirs(output_char_dir, exist_ok=True)
        
        images = [img for img in os.listdir(input_char_dir) 
                 if img.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
        
        print(f"[{idx}/{total_chars}] {char_name}: {len(images)} 张")
        
        for img_name in images:
            img_path = os.path.join(input_char_dir, img_name)
            stats["processed"] += 1
            
            try:
                # 1. 读取图片
                img = cv2_imread_safe(img_path)
                if img is None:
                    stats["errors"] += 1
                    continue
                
                img_h, img_w = img.shape[:2]
                
                # 2. YOLOv8 推理
                results = model(img, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
                
                if len(boxes) == 0:
                    stats["no_face"] += 1
                    continue
                
                # 3. 质量过滤
                valid_boxes = []
                valid_confs = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # 检查置信度
                    if i < len(confidences) and confidences[i] < MIN_CONFIDENCE:
                        continue
                    
                    # 检查尺寸
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue
                    
                    # 检查长宽比
                    aspect_ratio = max(w/h, h/w)
                    if aspect_ratio > MAX_ASPECT_RATIO:
                        continue
                    
                    valid_boxes.append(box)
                    valid_confs.append(confidences[i] if i < len(confidences) else 0.5)
                
                if len(valid_boxes) == 0:
                    stats["low_quality"] += 1
                    continue
                
                # 4. 选择主角人脸
                main_face = select_main_face(valid_boxes, valid_confs, img_w, img_h)
                if main_face is None:
                    stats["no_face"] += 1
                    continue
                
                # 5. 裁剪人脸
                face_crop = crop_face_safely(img, main_face, MARGIN_RATIO)
                
                if face_crop.size == 0:
                    stats["errors"] += 1
                    continue
                
                # 6. 清晰度检查
                sharpness = check_sharpness(face_crop)
                if sharpness < MIN_SHARPNESS:
                    stats["low_quality"] += 1
                    continue
                
                # 7. 缩放并保存
                face_resized = cv2.resize(face_crop, TARGET_SIZE, 
                                         interpolation=cv2.INTER_AREA)
                
                save_path = os.path.join(output_char_dir, f"face_{img_name}")
                cv2_imwrite_safe(save_path, face_resized)
                stats["saved"] += 1
                
            except Exception as e:
                print(f"      ❌ 处理 {img_name} 时出错: {e}")
                stats["errors"] += 1

    print("\n" + "="*60)
    print("✅ 数据集裁剪预处理完毕！")
    print(f"总计处理图片: {stats['processed']} 张")
    print(f"成功截取人脸: {stats['saved']} 张")
    print(f"未检测到人脸: {stats['no_face']} 张")
    print(f"质量不合格:   {stats['low_quality']} 张")
    print(f"处理错误:     {stats['errors']} 张")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n⚠️  建议：仍需人工检查，删除误检和切错的人脸！")

if __name__ == "__main__":
    process_dataset()
