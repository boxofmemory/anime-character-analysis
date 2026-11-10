"""Detector training modules."""
from .trainer import train_yolo_from_config, validate_yolo_config

__all__ = ["train_yolo_from_config", "validate_yolo_config"]
