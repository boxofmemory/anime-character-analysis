"""
train_arcface.py
================
使用动漫人脸数据集微调 ResNet50 + ArcFace Loss
运行一次即可，生成微调后的模型权重文件

依赖：pip install torch torchvision tqdm
"""

import os
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ===================== 配置区 =====================
# 数据集路径（每个角色一个子文件夹，里面是裁剪好的人脸图）
DATASET_DIR = r"./datasets/arcface_112x112"

# 模型保存路径
MODEL_SAVE_DIR = r"./test04"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_anime_arcface.pth")

# 训练参数
BATCH_SIZE = 64          # 如果显存不够，改成 32 或 16
NUM_EPOCHS = 20          # 训练轮数
LEARNING_RATE = 0.01     # 初始学习率
WEIGHT_DECAY = 5e-4      # 权重衰减
SCALE = 30.0             # ArcFace scale 参数
MARGIN = 0.5             # ArcFace margin 参数（角度间隔）

# 每个角色最少需要多少张图才纳入训练
MIN_IMAGES_PER_CLASS = 5

# 数据划分：训练集比例（剩下的自动成为验证集）
TRAIN_RATIO = 0.8
# ================================================


# ================================================================
#  1. ArcFace 核心组件：ArcMarginProduct
# ================================================================

class ArcMarginProduct(nn.Module):
    """
    ArcFace 的核心数学组件
    
    作用：
    - 把 ResNet50 输出的 2048 维特征，投影到 N 个类别上
    - 在"正确类别"的角度上加上一个惩罚（margin）
    - 强迫模型让"同一个人的特征"聚集在一起，"不同人的特征"分散开
    
    数学原理：
    - 常规做法：输出 = W * f （W 是分类权重，f 是特征）
    - ArcFace：输出 = W * f，但对正确类别加上 margin
      cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
    
    训练结束后，这个层会被丢弃，只保留前面的 ResNet50 骨干
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super().__init__()
        self.in_features = in_features  # 2048
        self.out_features = out_features  # 450（角色数量）
        self.scale = scale
        self.margin = margin
        
        # 分类权重矩阵 [num_classes, 2048]
        # 相当于每个类别在 2048 维空间中的一个"代表点"
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        """
        参数:
            features: [Batch, 2048]  L2归一化后的特征
            labels:   [Batch]        每张图对应的角色索引
        返回:
            logits:   [Batch, num_classes]  带margin的缩放余弦值
        """
        # 1. 对特征做 L2 归一化（投影到单位超球面）
        features_norm = F.normalize(features, dim=1)
        # 2. 对权重做 L2 归一化
        weight_norm = F.normalize(self.weight, dim=1)
        
        # 3. 计算余弦相似度 [Batch, num_classes]
        cos_theta = F.linear(features_norm, weight_norm)
        
        # 4. 对正确类别加上 angular margin
        #    cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        target_cos = cos_theta[torch.arange(len(labels), device=labels.device), labels]
        sin_theta = torch.sqrt((1.0 - target_cos.pow(2)).clamp(min=1e-7))
        cos_theta_m = (target_cos * math.cos(self.margin) - sin_theta * math.sin(self.margin))
        
        # 5. 把正确类别的值替换为加了 margin 的值
        cos_theta.scatter_(1, labels.unsqueeze(1), cos_theta_m.unsqueeze(1))
        
        # 6. 乘以 scale 放大
        logits = cos_theta * self.scale
        
        return logits


# ================================================================
#  2. 完整模型：ResNet50 骨干 + ArcFace 分类头
# ================================================================

class ResNet50ArcFace(nn.Module):
    """
    完整的训练模型
    
    结构:
    输入(112x112x3) → ResNet50骨干(去掉FC) → GAP → [2048维]
    → BatchNorm → ArcMarginProduct → [num_classes维] → CrossEntropyLoss
    
    推理时:
    输入(112x112x3) → ResNet50骨干 → GAP → [2048维] → BN → L2归一化 → 输出
    """
    def __init__(self, num_classes, embedding_size=2048, scale=30.0, margin=0.5):
        super().__init__()
        
        # ResNet50 骨干网络（去掉最后的全连接层）
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 去掉最后的 FC 层（1000类分类层）
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 全局平均池化已经内置在 ResNet50 的 AdaptiveAvgPool2d 中了
        # backbone 输出: [Batch, 2048, 1, 1]
        
        # BatchNorm1d：让特征分布更稳定（训练技巧）
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # ArcFace 分类头（只在训练时使用）
        self.arcface = ArcMarginProduct(embedding_size, num_classes, scale, margin)
        
        # 冻结骨干网络的前两层（加速训练，防止过拟合）
        # ResNet50 的 children 顺序: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False
    
    def forward(self, x, labels=None):
        """
        训练模式：返回 arcface logits（用于计算 CrossEntropyLoss）
        推理模式：返回 L2 归一化的 2048 维特征向量
        """
        # 1. 骨干网络提取特征
        x = self.backbone(x)     # [Batch, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [Batch, 2048]
        
        # 2. BatchNorm
        x = self.bn(x)           # [Batch, 2048]
        
        if self.training and labels is not None:
            # 训练模式：通过 ArcFace 层输出 logits
            logits = self.arcface(x, labels)
            return logits
        else:
            # 推理模式：L2 归一化后直接返回特征
            x = x / x.norm(dim=1, keepdim=True)
            return x


# ================================================================
#  3. 数据集加载器
# ================================================================

class AnimeFaceDataset(Dataset):
    """
    动漫人脸数据集加载器
    
    目录结构:
    ./datasets/arcface_112x112\
        ├── levi\
        │   ├── face_001.jpg
        │   └── face_002.jpg
        ├── naruto\
        │   └── ...
        └── ...
    """
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []   # [(img_path, class_idx), ...]
        self.class_names = []
        
        if class_to_idx is not None:
            # 使用已有的类别映射（验证集）
            self.class_to_idx = class_to_idx
        else:
            # 扫描文件夹，建立类别映射（训练集）
            self.class_to_idx = {}
        
        # 扫描所有角色文件夹
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            class_name = folder.name
            
            # 如果不在映射中，自动添加（训练集模式）
            if class_name not in self.class_to_idx:
                idx = len(self.class_to_idx)
                self.class_to_idx[class_name] = idx
                self.class_names.append(class_name)
            
            # 收集该角色的所有图片
            idx = self.class_to_idx[class_name]
            for img_path in sorted(folder.glob("*.*")):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 安全读取图片（兼容中文路径）
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ================================================================
#  4. 主训练流程
# ================================================================

def main():
    print("=" * 60)
    print("     ResNet50 + ArcFace 微调训练")
    print("=" * 60)
    
    # 检查数据集
    dataset_path = Path(DATASET_DIR)
    if not dataset_path.exists():
        print(f"\n❌ 找不到数据集: {DATASET_DIR}")
        return
    
    # 检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("   ⚠️  未检测到 GPU，将使用 CPU 训练（速度较慢）")
    
    # ---- 数据增强 ----
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 标准输入
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ---- 加载完整数据集（先扫描所有类别） ----
    print(f"\n📂 扫描数据集: {DATASET_DIR}")
    full_dataset = AnimeFaceDataset(DATASET_DIR, transform=None)
    
    # 过滤掉图片太少的类别
    class_counts = {}
    for _, label in full_dataset.samples:
        class_name = list(full_dataset.class_to_idx.keys())[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    valid_classes = {name: idx for name, idx in full_dataset.class_to_idx.items() 
                     if class_counts.get(name, 0) >= MIN_IMAGES_PER_CLASS}
    
    # 重新映射类别索引（连续编号）
    new_class_to_idx = {}
    for i, (name, _) in enumerate(sorted(valid_classes.items())):
        new_class_to_idx[name] = i
    
    print(f"   总类别数: {len(full_dataset.class_to_idx)}")
    print(f"   有效类别数（>= {MIN_IMAGES_PER_CLASS} 张）: {len(new_class_to_idx)}")
    
    # ---- 划分训练集和验证集 ----
    print("\n✂️  划分训练集和验证集...")
    
    # 按类别划分（确保每个类都有训练和验证样本）
    train_samples = []
    val_samples = []
    
    for class_name in sorted(new_class_to_idx.keys()):
        old_idx = full_dataset.class_to_idx[class_name]
        class_imgs = [(path, label) for path, label in full_dataset.samples 
                      if label == old_idx]
        
        # 打乱后按比例划分
        import random
        random.shuffle(class_imgs)
        split = int(len(class_imgs) * TRAIN_RATIO)
        
        new_idx = new_class_to_idx[class_name]
        for path, _ in class_imgs[:split]:
            train_samples.append((path, new_idx))
        for path, _ in class_imgs[split:]:
            val_samples.append((path, new_idx))
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    # 创建 Dataset
    train_dataset = AnimeFaceDataset.__new__(AnimeFaceDataset)
    train_dataset.root_dir = dataset_path
    train_dataset.transform = train_transform
    train_dataset.samples = train_samples
    train_dataset.class_to_idx = new_class_to_idx
    train_dataset.class_names = sorted(new_class_to_idx.keys())
    
    val_dataset = AnimeFaceDataset.__new__(AnimeFaceDataset)
    val_dataset.root_dir = dataset_path
    val_dataset.transform = val_transform
    val_dataset.samples = val_samples
    val_dataset.class_to_idx = new_class_to_idx
    val_dataset.class_names = sorted(new_class_to_idx.keys())
    
    print(f"   训练集: {len(train_dataset)} 张图片")
    print(f"   验证集: {len(val_dataset)} 张图片")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    # ---- 创建模型 ----
    num_classes = len(new_class_to_idx)
    print(f"\n🔧 创建模型: ResNet50 + ArcFace ({num_classes} 个类别)")
    
    model = ResNet50ArcFace(
        num_classes=num_classes,
        embedding_size=2048,
        scale=SCALE,
        margin=MARGIN
    )
    model = model.to(device)
    
    # ---- 损失函数和优化器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                          momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # ---- 训练循环 ----
    print(f"\n🏋️  开始训练 ({NUM_EPOCHS} 轮)...")
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # -- 训练阶段 --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播（训练模式：返回 arcface logits）
            logits = model(images, labels)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # -- 验证阶段 --
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播（推理模式：返回归一化特征）
                features = model(images)
                
                # 用特征和权重矩阵计算 logits 来预测
                # 这是简化做法：用特征间的余弦相似度找最近邻
                # 实际验证时用 arcface 层的权重来做分类
                features_norm = F.normalize(features, dim=1)
                weight_norm = F.normalize(model.arcface.weight, dim=1)
                logits = F.linear(features_norm, weight_norm) * model.arcface.scale
                _, predicted = logits.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start
        
        print(f"\n   📊 Epoch {epoch} 总结:")
        print(f"      Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"      Val Acc: {val_acc:.2f}%  |  LR: {lr:.6f}  |  Time: {elapsed:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            
            # 只保存骨干网络的权重（用于推理时提取特征）
            torch.save({
                'backbone': model.backbone.state_dict(),
                'bn': model.bn.state_dict(),
                'class_names': sorted(new_class_to_idx.keys()),
                'embedding_size': 2048,
                'val_acc': val_acc,
            }, MODEL_SAVE_PATH)
            
            print(f"      ✅ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
        print("-" * 60)
    
    # ---- 训练完成 ----
    print(f"\n🎉 训练完成！")
    print(f"   最佳验证准确率: {best_val_acc:.2f}%")
    print(f"   模型保存至: {MODEL_SAVE_PATH}")
    print(f"\n📌 下一步: 运行 video_demo.py 处理视频")


if __name__ == "__main__":
    main()
