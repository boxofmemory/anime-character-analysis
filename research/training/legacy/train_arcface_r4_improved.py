"""
train_arcface_r3_improved.py
============================
ResNet50 + ArcFace 动漫人脸识别训练改进版

主要目标：
1. 解决 r2 中 layer3/layer4 可能没有真正解冻的问题。
2. 降低 ArcFace 训练难度，使用 margin warmup，避免一开始 loss 过大、Train Acc 长期为 0。
3. 针对 248 类 / 3009 张图这类小样本多类别数据，加入类别均衡采样。
4. 提高样本过滤门槛，减少每类样本过少导致的 prototype 不稳定。
5. 早停不再过早触发，至少训练到高层/中层微调阶段之后再判断。
6. 保存 prototype、class_names、推荐阈值，方便接入 video_demo.py。

数据集格式：
DATASET_DIR/
    character_1/
        001.jpg
        002.jpg
    character_2/
        001.jpg
        002.jpg

依赖：
pip install torch torchvision tqdm pillow
"""

import os
import csv
import json
import math
import time
import random
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ================================================================
# 配置区
# ================================================================
DATASET_DIR = r"./anime_cnn_kaggle\dataset"
MODEL_SAVE_DIR = r"./models"
LOG_SAVE_DIR = r"./models\logs"

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_anime_arcface_best_r4.pth")
CSV_LOG_PATH = os.path.join(LOG_SAVE_DIR, "training_log_r4.csv")
CLASS_INDEX_PATH = os.path.join(MODEL_SAVE_DIR, "anime_arcface_class_names_r4.json")
WRONG_PRED_CSV_PATH = os.path.join(LOG_SAVE_DIR, "wrong_predictions_r4.csv")

INPUT_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 70
EMBEDDING_SIZE = 512
TRAIN_RATIO = 0.8
MIN_IMAGES_PER_CLASS = 10
RANDOM_SEED = 42
NUM_WORKERS = 0

SCALE = 30.0
FINAL_MARGIN = 0.35
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05

EARLY_STOP_PATIENCE = 18
MIN_EPOCHS_BEFORE_EARLY_STOP = 35
PROTO_EVAL_EVERY = 1

USE_BALANCED_SAMPLER = True
VERIFY_IMAGES = True
USE_AMP = True

THRESHOLD_START = 0.20
THRESHOLD_END = 0.90
THRESHOLD_STEP = 0.02

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ================================================================
# 基础工具
# ================================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dirs():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_SAVE_DIR, exist_ok=True)


def safe_load_image(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def verify_image_file(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_resnet50(pretrained=True):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet50(weights=weights)
    except Exception:
        return models.resnet50(pretrained=pretrained)


# ================================================================
# ArcFace
# ================================================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.update_margin(m)

    def update_margin(self, margin):
        self.m = float(margin)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits

    def logits_without_margin(self, features):
        logits = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        return logits * self.s


class ResNet50ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()
        self.resnet = get_resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=SCALE,
            m=FINAL_MARGIN,
        )

    def extract_feature(self, x):
        x = self.resnet(x)
        x = self.embedding(x)
        return F.normalize(x, dim=1)

    def forward(self, x, labels=None):
        features = self.extract_feature(x)
        if labels is not None:
            return self.arcface(features, labels)
        return features

    def classify_without_margin(self, x):
        features = self.extract_feature(x)
        return self.arcface.logits_without_margin(features)


# ================================================================
# 数据集
# ================================================================
class AnimeFaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = safe_load_image(img_path)
        if image is None:
            image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path


# ================================================================
# 数据扫描与划分
# ================================================================
def scan_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    class_dirs = [p for p in sorted(dataset_path.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"数据集目录下没有类别文件夹：{dataset_dir}")

    valid_items = []
    skipped_small = []
    skipped_bad_images = 0

    for class_dir in class_dirs:
        image_paths = []
        for p in sorted(class_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTENSIONS:
                continue
            if VERIFY_IMAGES and not verify_image_file(p):
                skipped_bad_images += 1
                continue
            image_paths.append(str(p))

        if len(image_paths) >= MIN_IMAGES_PER_CLASS:
            valid_items.append((class_dir.name, image_paths))
        else:
            skipped_small.append((class_dir.name, len(image_paths)))

    if len(valid_items) < 2:
        raise RuntimeError("有效类别少于 2 个，无法训练。请降低 MIN_IMAGES_PER_CLASS 或检查数据集。")

    class_names = [name for name, _ in valid_items]
    class_to_samples = defaultdict(list)
    all_samples = []

    for label, (_, paths) in enumerate(valid_items):
        for path in paths:
            class_to_samples[label].append(path)
            all_samples.append((path, label))

    counts = [len(paths) for _, paths in valid_items]
    print("\n数据集扫描完成：")
    print(f"  原始类别数：{len(class_dirs)}")
    print(f"  有效类别数：{len(class_names)}")
    print(f"  总图片数：{len(all_samples)}")
    print(f"  每类最少图片数：{MIN_IMAGES_PER_CLASS}")
    print(f"  每类图片数：min={min(counts)}, max={max(counts)}, avg={sum(counts) / len(counts):.2f}")
    print(f"  跳过样本过少类别：{len(skipped_small)}")
    print(f"  跳过坏图：{skipped_bad_images}")
    if skipped_small:
        print(f"  样本过少示例：{skipped_small[:10]}")

    return class_names, class_to_samples


def stratified_split(class_to_samples, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for label, paths in class_to_samples.items():
        paths = list(paths)
        rng.shuffle(paths)
        split_idx = int(len(paths) * train_ratio)
        split_idx = max(1, min(split_idx, len(paths) - 1))

        train_samples.extend((p, label) for p in paths[:split_idx])
        val_samples.extend((p, label) for p in paths[split_idx:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_balanced_sampler(samples):
    label_counts = Counter(label for _, label in samples)
    weights = [1.0 / label_counts[label] for _, label in samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(samples),
        replacement=True,
    )


def build_dataloaders():
    class_names, class_to_samples = scan_dataset(DATASET_DIR)
    train_samples, val_samples = stratified_split(class_to_samples, TRAIN_RATIO, RANDOM_SEED)
    train_transform, eval_transform = build_transforms()

    train_dataset = AnimeFaceDataset(train_samples, transform=train_transform)
    val_dataset = AnimeFaceDataset(val_samples, transform=eval_transform)
    prototype_dataset = AnimeFaceDataset(train_samples, transform=eval_transform)

    if USE_BALANCED_SAMPLER:
        sampler = build_balanced_sampler(train_samples)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    prototype_loader = DataLoader(
        prototype_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print("\n数据划分：")
    print(f"  训练集：{len(train_samples)} 张")
    print(f"  验证集：{len(val_samples)} 张")
    print(f"  类别数：{len(class_names)}")
    print(f"  类别均衡采样：{USE_BALANCED_SAMPLER}")

    return train_loader, val_loader, prototype_loader, class_names


# ================================================================
# 解冻策略
# ================================================================
def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_train_stage(epoch, model, optimizer):
    if epoch <= 3:
        stage_name = "阶段一：冻结 ResNet，仅训练 embedding + ArcFace，margin=0"
        set_requires_grad(model.resnet, False)
        set_requires_grad(model.embedding, True)
        set_requires_grad(model.arcface, True)
        margin = 0.00
        bb_lr = 0.0
        emb_lr = 1e-3
        head_lr = 1e-3

    elif epoch <= 10:
        stage_name = "阶段二：解冻 layer4，低 margin 适配动漫高层语义"
        set_requires_grad(model.resnet, False)
        set_requires_grad(model.resnet.layer4, True)
        set_requires_grad(model.embedding, True)
        set_requires_grad(model.arcface, True)
        margin = 0.15
        bb_lr = 1e-4
        emb_lr = 8e-4
        head_lr = 8e-4

    elif epoch <= 25:
        stage_name = "阶段三：解冻 layer3 + layer4，逐步拉开角色特征"
        set_requires_grad(model.resnet, False)
        set_requires_grad(model.resnet.layer3, True)
        set_requires_grad(model.resnet.layer4, True)
        set_requires_grad(model.embedding, True)
        set_requires_grad(model.arcface, True)
        margin = 0.25
        bb_lr = 5e-5
        emb_lr = 5e-4
        head_lr = 5e-4

    else:
        stage_name = "阶段四：解冻 layer2 + layer3 + layer4，小学习率精调"
        set_requires_grad(model.resnet, False)
        set_requires_grad(model.resnet.layer2, True)
        set_requires_grad(model.resnet.layer3, True)
        set_requires_grad(model.resnet.layer4, True)
        set_requires_grad(model.embedding, True)
        set_requires_grad(model.arcface, True)
        margin = FINAL_MARGIN
        bb_lr = 1e-5
        emb_lr = 1e-4
        head_lr = 1e-4

    model.arcface.update_margin(margin)
    optimizer.param_groups[0]["lr"] = bb_lr
    optimizer.param_groups[1]["lr"] = emb_lr
    optimizer.param_groups[2]["lr"] = head_lr

    print(f"\n=> {stage_name}")
    print(f"   margin={margin:.2f}, backbone_lr={bb_lr:.8f}, embedding_lr={emb_lr:.8f}, head_lr={head_lr:.8f}")
    print(f"   当前可训练参数量：{count_trainable_params(model) / 1e6:.2f} M")

    return margin, bb_lr, emb_lr, head_lr


# ================================================================
# 训练与评估
# ================================================================
def make_criterion():
    try:
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    except TypeError:
        return nn.CrossEntropyLoss()


def unpack_batch(batch, device):
    images, labels, paths = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels, paths


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct_margin = 0
    total_correct_plain = 0
    total_count = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch} Train", ncols=120)
    for batch in progress:
        images, labels, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        use_amp_now = USE_AMP and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp_now):
            logits_margin = model(images, labels)
            loss = criterion(logits_margin, labels)

        if scaler is not None and use_amp_now:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            features = model(images)
            logits_plain = model.arcface.logits_without_margin(features)
            pred_margin = logits_margin.argmax(dim=1)
            pred_plain = logits_plain.argmax(dim=1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct_margin += pred_margin.eq(labels).sum().item()
        total_correct_plain += pred_plain.eq(labels).sum().item()
        total_count += bs

        progress.set_postfix({
            "loss": f"{total_loss / max(total_count, 1):.4f}",
            "plain_acc": f"{100.0 * total_correct_plain / max(total_count, 1):.2f}%",
        })

    avg_loss = total_loss / max(total_count, 1)
    train_acc_margin = 100.0 * total_correct_margin / max(total_count, 1)
    train_acc_plain = 100.0 * total_correct_plain / max(total_count, 1)
    return avg_loss, train_acc_margin, train_acc_plain


@torch.no_grad()
def validate_arc_head(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_count = 0

    for batch in tqdm(val_loader, desc="Validate ArcHead", ncols=120):
        images, labels, _ = unpack_batch(batch, device)
        logits = model.classify_without_margin(images)
        preds = logits.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_count += labels.size(0)

    return 100.0 * total_correct / max(total_count, 1)


@torch.no_grad()
def extract_all_features(model, loader, device, desc):
    model.eval()
    features_list = []
    labels_list = []
    paths_list = []

    for batch in tqdm(loader, desc=desc, ncols=120):
        images, labels, paths = unpack_batch(batch, device)
        features = model(images)
        features_list.append(features.cpu())
        labels_list.append(labels.cpu())
        paths_list.extend(list(paths))

    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0), paths_list


def build_class_prototypes(features, labels, num_classes):
    prototypes = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() == 0:
            proto = torch.zeros(features.size(1))
        else:
            proto = features[mask].mean(dim=0)
            proto = F.normalize(proto, dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes, dim=0)
    return F.normalize(prototypes, dim=1)


def find_best_threshold(same_scores, diff_scores):
    best_threshold = 0.5
    best_score = -1.0

    t = THRESHOLD_START
    while t <= THRESHOLD_END + 1e-8:
        known_accept = (same_scores >= t).float().mean().item()
        diff_reject = (diff_scores < t).float().mean().item()
        score = 0.5 * known_accept + 0.5 * diff_reject
        if score > best_score:
            best_score = score
            best_threshold = t
        t += THRESHOLD_STEP

    return round(best_threshold, 4), round(best_score, 4)


@torch.no_grad()
def evaluate_with_prototypes(model, prototype_loader, val_loader, num_classes, class_names, device, epoch):
    train_features, train_labels, _ = extract_all_features(model, prototype_loader, device, "Extract Train Features")
    prototypes = build_class_prototypes(train_features, train_labels, num_classes)

    val_features, val_labels, val_paths = extract_all_features(model, val_loader, device, "Extract Val Features")
    similarity = torch.matmul(val_features, prototypes.t())

    top1_scores, top1_preds = similarity.max(dim=1)
    top1_acc = 100.0 * top1_preds.eq(val_labels).sum().item() / max(len(val_labels), 1)

    topk = min(5, num_classes)
    topk_preds = similarity.topk(k=topk, dim=1).indices
    top5_correct = topk_preds.eq(val_labels.view(-1, 1)).any(dim=1).sum().item()
    top5_acc = 100.0 * top5_correct / max(len(val_labels), 1)

    same_scores = similarity[torch.arange(len(val_labels)), val_labels]
    sim_without_true = similarity.clone()
    sim_without_true[torch.arange(len(val_labels)), val_labels] = -999.0
    diff_scores = sim_without_true.max(dim=1).values

    best_threshold, threshold_score = find_best_threshold(same_scores, diff_scores)

    write_wrong_predictions(
        epoch=epoch,
        val_paths=val_paths,
        val_labels=val_labels,
        top1_preds=top1_preds,
        top1_scores=top1_scores,
        same_scores=same_scores,
        class_names=class_names,
    )

    metrics = {
        "prototype_top1_acc": top1_acc,
        "prototype_top5_acc": top5_acc,
        "same_class_similarity_mean": same_scores.mean().item(),
        "diff_class_similarity_mean": diff_scores.mean().item(),
        "best_threshold": best_threshold,
        "threshold_score": threshold_score,
    }

    return metrics, prototypes


def write_wrong_predictions(epoch, val_paths, val_labels, top1_preds, top1_scores, same_scores, class_names):
    wrong_rows = []
    for path, true_label, pred_label, pred_score, true_score in zip(
        val_paths,
        val_labels.tolist(),
        top1_preds.tolist(),
        top1_scores.tolist(),
        same_scores.tolist(),
    ):
        if true_label != pred_label:
            wrong_rows.append([
                epoch,
                path,
                class_names[true_label],
                class_names[pred_label],
                f"{pred_score:.4f}",
                f"{true_score:.4f}",
            ])

    wrong_rows = sorted(wrong_rows, key=lambda x: float(x[4]), reverse=True)[:200]
    need_header = not os.path.exists(WRONG_PRED_CSV_PATH)

    with open(WRONG_PRED_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["epoch", "image_path", "true_class", "pred_class", "pred_score", "true_score"])
        writer.writerows(wrong_rows)


# ================================================================
# 日志与保存
# ================================================================
def init_csv_log():
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch",
            "Margin",
            "Train Loss",
            "Train Acc Margin(%)",
            "Train Acc Plain(%)",
            "Val Acc ArcHead(%)",
            "Prototype Top1(%)",
            "Prototype Top5(%)",
            "Same Sim Mean",
            "Diff Sim Mean",
            "Best Threshold",
            "Backbone LR",
            "Embedding LR",
            "Head LR",
            "Time(s)",
        ])


def append_csv_log(epoch, margin, train_loss, train_acc_margin, train_acc_plain, val_acc, proto_metrics, lrs, elapsed):
    bb_lr, emb_lr, head_lr = lrs
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{margin:.2f}",
            f"{train_loss:.4f}",
            f"{train_acc_margin:.2f}",
            f"{train_acc_plain:.2f}",
            f"{val_acc:.2f}",
            f"{proto_metrics['prototype_top1_acc']:.2f}",
            f"{proto_metrics['prototype_top5_acc']:.2f}",
            f"{proto_metrics['same_class_similarity_mean']:.4f}",
            f"{proto_metrics['diff_class_similarity_mean']:.4f}",
            f"{proto_metrics['best_threshold']:.4f}",
            f"{bb_lr:.8f}",
            f"{emb_lr:.8f}",
            f"{head_lr:.8f}",
            f"{elapsed:.1f}",
        ])


def save_best_model(model, class_names, epoch, val_acc, proto_metrics, prototypes):
    save_dict = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_size": INPUT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "scale": SCALE,
        "final_margin": FINAL_MARGIN,
        "val_acc_arc_head": val_acc,
        "prototype_metrics": proto_metrics,
        "prototypes": prototypes.cpu(),
        "recommended_threshold": proto_metrics["best_threshold"],
    }
    torch.save(save_dict, MODEL_SAVE_PATH)

    with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "num_classes": len(class_names),
                "model_path": MODEL_SAVE_PATH,
                "input_size": INPUT_SIZE,
                "embedding_size": EMBEDDING_SIZE,
                "recommended_threshold": proto_metrics["best_threshold"],
                "prototype_metrics": proto_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ================================================================
# 主流程
# ================================================================
def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()
    init_csv_log()

    print("=" * 76)
    print("ResNet50 + ArcFace 动漫人脸识别训练脚本 r3_improved")
    print("=" * 76)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    train_loader, val_loader, prototype_loader, class_names = build_dataloaders()
    num_classes = len(class_names)

    model = ResNet50ArcFace(num_classes=num_classes, embedding_size=EMBEDDING_SIZE).to(device)
    criterion = make_criterion()

    optimizer = optim.AdamW([
        {"params": model.resnet.parameters(), "lr": 0.0},
        {"params": model.embedding.parameters(), "lr": 1e-3},
        {"params": model.arcface.parameters(), "lr": 1e-3},
    ], weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    print("\n模型信息：")
    print(f"  Backbone：ResNet50 ImageNet 预训练")
    print(f"  输入尺寸：{INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  类别数：{num_classes}")
    print(f"  Embedding size：{EMBEDDING_SIZE}")
    print(f"  Scale：{SCALE}")
    print(f"  Final margin：{FINAL_MARGIN}")
    print(f"  总参数量：{total_params / 1e6:.2f} M")

    best_score = -1.0
    patience_counter = 0
    last_proto_metrics = None
    last_prototypes = None

    print("\n开始训练...")
    print("-" * 76)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        margin, bb_lr, emb_lr, head_lr = configure_train_stage(epoch, model, optimizer)

        train_loss, train_acc_margin, train_acc_plain = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        val_acc = validate_arc_head(model, val_loader, device)

        if epoch % PROTO_EVAL_EVERY == 0 or epoch == 1:
            proto_metrics, prototypes = evaluate_with_prototypes(
                model=model,
                prototype_loader=prototype_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                class_names=class_names,
                device=device,
                epoch=epoch,
            )
            last_proto_metrics = proto_metrics
            last_prototypes = prototypes
        else:
            proto_metrics = last_proto_metrics
            prototypes = last_prototypes

        elapsed = time.time() - epoch_start
        append_csv_log(
            epoch=epoch,
            margin=margin,
            train_loss=train_loss,
            train_acc_margin=train_acc_margin,
            train_acc_plain=train_acc_plain,
            val_acc=val_acc,
            proto_metrics=proto_metrics,
            lrs=(bb_lr, emb_lr, head_lr),
            elapsed=elapsed,
        )

        current_score = proto_metrics["prototype_top1_acc"]
        improved = current_score > best_score

        print("\n" + "=" * 76)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] 总结")
        print(f"  Train Loss          : {train_loss:.4f}")
        print(f"  Train Acc Margin    : {train_acc_margin:.2f}%")
        print(f"  Train Acc Plain     : {train_acc_plain:.2f}%")
        print(f"  Val Acc ArcHead     : {val_acc:.2f}%")
        print(f"  Prototype Top-1     : {proto_metrics['prototype_top1_acc']:.2f}%")
        print(f"  Prototype Top-5     : {proto_metrics['prototype_top5_acc']:.2f}%")
        print(f"  Same Sim Mean       : {proto_metrics['same_class_similarity_mean']:.4f}")
        print(f"  Diff Sim Mean       : {proto_metrics['diff_class_similarity_mean']:.4f}")
        print(f"  Recommend Threshold : {proto_metrics['best_threshold']:.4f}")
        print(f"  Margin              : {margin:.2f}")
        print(f"  LR                  : backbone={bb_lr:.8f}, embedding={emb_lr:.8f}, head={head_lr:.8f}")
        print(f"  Time                : {elapsed:.1f}s")

        if improved:
            best_score = current_score
            patience_counter = 0
            save_best_model(
                model=model,
                class_names=class_names,
                epoch=epoch,
                val_acc=val_acc,
                proto_metrics=proto_metrics,
                prototypes=prototypes,
            )
            print(f"  [保存] 新最佳模型：Prototype Top-1 = {best_score:.2f}%")
            print(f"  模型路径：{MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  [未提升] patience = {patience_counter}/{EARLY_STOP_PATIENCE}")

        print("=" * 76)

        if epoch >= MIN_EPOCHS_BEFORE_EARLY_STOP and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停：epoch >= {MIN_EPOCHS_BEFORE_EARLY_STOP} 且连续 {EARLY_STOP_PATIENCE} 轮未提升。")
            break

    print("\n训练结束！")
    print(f"最佳 Prototype Top-1：{best_score:.2f}%")
    print(f"最佳模型：{MODEL_SAVE_PATH}")
    print(f"类别索引：{CLASS_INDEX_PATH}")
    print(f"训练日志：{CSV_LOG_PATH}")
    print(f"错误样本分析：{WRONG_PRED_CSV_PATH}")
    print("\n建议：")
    print("1. 若 Prototype Top-1 仍低于 30%，先打开 wrong_predictions_r3.csv 检查标签和裁剪质量。")
    print("2. 若 Same Sim Mean 仍小于 Diff Sim Mean，优先清洗数据，而不是继续加训练轮数。")
    print("3. 接入 video_demo.py 时使用 model_state + prototypes + recommended_threshold。")
    print("=" * 76)


if __name__ == "__main__":
    main()
