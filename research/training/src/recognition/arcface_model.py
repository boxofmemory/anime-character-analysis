from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(features_norm, weight_norm)
        target_cos = cos_theta[torch.arange(len(labels), device=labels.device), labels]
        sin_theta = torch.sqrt((1.0 - target_cos.pow(2)).clamp(min=1e-7))
        cos_theta_m = target_cos * math.cos(self.margin) - sin_theta * math.sin(self.margin)
        cos_theta.scatter_(1, labels.unsqueeze(1), cos_theta_m.unsqueeze(1))
        return cos_theta * self.scale


class ResNet50ArcFace(nn.Module):
    def __init__(self, num_classes: int, embedding_size: int = 2048, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcMarginProduct(embedding_size, num_classes, scale, margin)

        for name, param in self.backbone.named_parameters():
            param.requires_grad = not any(tag in name for tag in ("conv1", "bn1", "layer1"))

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        if self.training and labels is not None:
            return self.arcface(x, labels)
        return x / x.norm(dim=1, keepdim=True)
