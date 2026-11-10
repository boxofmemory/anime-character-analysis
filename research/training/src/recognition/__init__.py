"""Recognition training modules."""
from .arcface_model import ArcMarginProduct, ResNet50ArcFace
from .trainer import train_arcface_from_config, validate_arcface_config

__all__ = [
    "ArcMarginProduct",
    "ResNet50ArcFace",
    "train_arcface_from_config",
    "validate_arcface_config",
]
