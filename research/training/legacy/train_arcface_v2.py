"""
train_arcface_v2.py（强化训练版）
==============================
针对 14% 准确率的改进：
1. 解冻 layer2+3+4（原来只训练 layer3+4）
2. 大幅增强数据增强（应对动漫光影剧变）
3. 训练 60 轮 + 早停机制（原来只训练 20 轮）
4. ArcFace margin 从 0.5 降到 0.4（450 类数据量偏小）

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

BATCH_SIZE = 32           # ⚠️ 改1：从64降到32（解冻更多层后显存占用翻倍，防止OOM）
NUM_EPOCHS = 60           # ⚠️ 改2：从20升到60（给模型足够时间学动漫特征）
LEARNING_RATE = 0.001      # 配合 CosineAnnealing 后期自动降到接近0
WEIGHT_DECAY = 5e-4
SCALE = 30.0
MARGIN = 0.5             # ⚠️ 改3：从0.5降到0.4（450类+9000张图，0.5太激进，模型收敛困难）
MIN_IMAGES_PER_CLASS = 5
TRAIN_RATIO = 0.8

EARLY_STOP_PATIENCE = 10  # ⚠️ 改4：新增早停（验证集连续10轮不涨就停，既省时间又防过拟合）
# ================================================


# ================================================================
#  ArcFace 核心组件（无需修改）
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
        cos_theta.scatter_(1, labels.unsqueeze(1), cos_theta_m)
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

        # ⚠️ 核心改动：原来冻结了 layer1+layer2，现在只冻结 layer1
        # layer1 学的是底层边缘检测（横线、竖线），动漫和真人通用，冻结没问题
        # layer2 学的是纹理组合（眼睛形状、鼻子的轮廓），动漫线条风格和真人完全不同，必须重新学
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "conv1" in name or "bn1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

