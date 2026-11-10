"""
train_arcface_v2.py（强化训练版）
==============================
改进清单：
1. 解冻 layer2+3+4（原来只训练 layer3+4）
2. 大幅增强数据增强（应对动漫光影剧变）
3. 训练 60 轮 + 早停机制（原来只训练 20 轮）
4. MARGIN = 0.5（保持原值，解决"同脸不同人"）
5. LEARNING_RATE = 0.001（稳定微调）
6. BATCH_SIZE = 32（防止显存溢出）

依赖：pip install torch torchvision tqdm
"""

import os
import time
import math
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ===================== 配置区 =====================
DATASET_DIR = r"./datasets/arcface_112x112"
MODEL_SAVE_DIR = r"./models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_anime_arcface.pth")

BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
SCALE = 30.0
MARGIN = 0.5
MIN_IMAGES_PER_CLASS = 5
TRAIN_RATIO = 0.8
EARLY_STOP_PATIENCE = 10
# ================================================


# ================================================================
#  ArcFace 核心组件
# ================================================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(features_norm, weight_norm)
        target_cos = cos_theta[torch.arange(len(labels), device=labels.device), labels]
        sin_theta = torch.sqrt((1.0 - target_cos.pow(2)).clamp(min=1e-7))
        cos_theta_m = (target_cos * math.cos(self.margin) - sin_theta * math.sin(self.margin))
        #cos_theta.scatter_(1, labels.unsqueeze(1), cos_theta_m)
        # 把正确类别的值替换为加了 margin 的值
        # 加上 .unsqueeze(1) 保证维度对齐为 [Batch, 1]
        cos_theta.scatter_(1, labels.unsqueeze(1), cos_theta_m.unsqueeze(1))
        logits = cos_theta * self.scale
        return logits



# ================================================================
#  完整模型
# ================================================================
class ResNet50ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size=2048, scale=30.0, margin=0.5):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcMarginProduct(embedding_size, num_classes, scale, margin)

        # 只冻结 layer1（底层边缘检测，动漫和真人通用）
        # 解冻 layer2+3+4（纹理/语义层，动漫风格和真人完全不同，必须重新学）
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "conv1" in name or "bn1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        if self.training and labels is not None:
            return self.arcface(x, labels)
        else:
            return x / x.norm(dim=1, keepdim=True)


# ================================================================
#  数据集加载器
# ================================================================
class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = []
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {}
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue
            class_name = folder.name
            if class_name not in self.class_to_idx:
                idx = len(self.class_to_idx)
                self.class_to_idx[class_name] = idx
                self.class_names.append(class_name)
            idx = self.class_to_idx[class_name]
            for img_path in sorted(folder.glob("*.*")):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ================================================================
#  主训练流程
# ================================================================
def main():
    print("=" * 60)
    print("     ResNet50 + ArcFace 强化训练版 (v2)")
    print("=" * 60)

    dataset_path = Path(DATASET_DIR)
    if not dataset_path.exists():
        print(f"\n  找不到数据集: {DATASET_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        #mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   显存: {mem:.1f} GB")

    # ---- 数据增强（大幅加强） ----
    # 原理：动漫角色可能出现在白天(高亮度)、夜晚(低亮度)、
    #       回忆杀(灰度)、战斗场景(高对比度)
    #       模型训练时必须见过这些变化，否则推理时遇到就认不出
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- 扫描数据集 ----
    print(f"\n  扫描数据集: {DATASET_DIR}")
    full_dataset = AnimeFaceDataset(DATASET_DIR, transform=None)

    class_counts = {}
    for _, label in full_dataset.samples:
        class_name = list(full_dataset.class_to_idx.keys())[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    valid_classes = {name: idx for name, idx in full_dataset.class_to_idx.items()
                     if class_counts.get(name, 0) >= MIN_IMAGES_PER_CLASS}

    new_class_to_idx = {}
    for i, (name, _) in enumerate(sorted(valid_classes.items())):
        new_class_to_idx[name] = i

    print(f"   总类别数: {len(full_dataset.class_to_idx)}")
    print(f"   有效类别数: {len(new_class_to_idx)}")

    # ---- 划分训练集和验证集 ----
    print("\n  划分训练集和验证集...")
    train_samples = []
    val_samples = []

    for class_name in sorted(new_class_to_idx.keys()):
        old_idx = full_dataset.class_to_idx[class_name]
        class_imgs = [(path, label) for path, label in full_dataset.samples
                      if label == old_idx]
        random.shuffle(class_imgs)
        split = int(len(class_imgs) * TRAIN_RATIO)
        new_idx = new_class_to_idx[class_name]
        for path, _ in class_imgs[:split]:
            train_samples.append((path, new_idx))
        for path, _ in class_imgs[split:]:
            val_samples.append((path, new_idx))

    random.shuffle(train_samples)
    random.shuffle(val_samples)

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

    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                           shuffle=True, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    # ---- 创建模型 ----
    num_classes = len(new_class_to_idx)
    print(f"\n  创建模型: ResNet50 + ArcFace ({num_classes} 类, margin={MARGIN})")

    model = ResNet50ArcFace(
        num_classes=num_classes,
        embedding_size=2048,
        scale=SCALE,
        margin=MARGIN
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params/1e6:.1f}M")
    print(f"   可训练参数量: {trainable_params/1e6:.1f}M")
    print(f"   冻结参数量: {(total_params-trainable_params)/1e6:.1f}M")

    # ---- 优化器和调度器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ---- 训练循环（含早停） ----
    print(f"\n  开始训练 (最多 {NUM_EPOCHS} 轮, LR={LEARNING_RATE}, "
          f"早停耐心={EARLY_STOP_PATIENCE})...")
    print("-" * 60)

    best_val_acc = 0.0
    patience_counter = 0

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
            logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_acc = 100. * train_correct / train_total

        # -- 验证阶段 --
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = model(images)
                features_norm = F.normalize(features, dim=1)
                weight_norm = F.normalize(model.arcface.weight, dim=1)
                logits = F.linear(features_norm, weight_norm) * model.arcface.scale
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        print(f"\n   Epoch {epoch}: "
              f"TrainAcc={train_acc:.1f}% | ValAcc={val_acc:.1f}% | "
              f"LR={lr:.6f} | {elapsed:.0f}s")

        # -- 保存最佳模型 --
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                'backbone': model.backbone.state_dict(),
                'bn': model.bn.state_dict(),
                'class_names': sorted(new_class_to_idx.keys()),
                'embedding_size': 2048,
                'val_acc': val_acc,
                'epoch': epoch,
            }, MODEL_SAVE_PATH)
            print(f"      新最佳! ValAcc={val_acc:.1f}% 已保存")
        else:
            patience_counter += 1
            print(f"      未提升 ({patience_counter}/{EARLY_STOP_PATIENCE})")

        scheduler.step()

        # -- 早停判断 --
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n      早停! 连续 {EARLY_STOP_PATIENCE} 轮无提升，停止训练")
            break

        print("-" * 60)

    # ---- 训练完成 ----
    print(f"\n  训练完成!")
    print(f"   最佳验证准确率: {best_val_acc:.1f}%")
    print(f"   模型保存至: {MODEL_SAVE_PATH}")
    print(f"\n  下一步: 用新模型重新构建熟人库 -> 跑 video_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
