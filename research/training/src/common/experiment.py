from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .config_loader import dump_yaml
from .paths import paths


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return slug or "experiment"


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_serialize(child) for child in value]
    return value


@dataclass
class ExperimentRun:
    kind: str
    name: str
    experiment_id: str
    root: Path
    artifacts_dir: Path
    config_snapshot_path: Path
    metrics_path: Path
    summary_path: Path

    def write_config_snapshot(self, config: dict) -> None:
        dump_yaml(self.config_snapshot_path, config)

    def write_metrics(self, payload: dict) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(_serialize(payload), handle, ensure_ascii=False, indent=2)

    def write_summary(self, content: str) -> None:
        self.summary_path.write_text(content, encoding="utf-8")

    def copy_artifact(self, source: Path, target_name: str | None = None) -> Path:
        target = self.artifacts_dir / (target_name or source.name)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return target


def create_experiment_run(kind: str, experiment_name: str, experiment_id_override: str | None = None) -> ExperimentRun:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = experiment_id_override or f"{timestamp}_{_slugify(experiment_name)}"
    root = paths.experiments / kind / experiment_id
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentRun(
        kind=kind,
        name=experiment_name,
        experiment_id=experiment_id,
        root=root,
        artifacts_dir=artifacts_dir,
        config_snapshot_path=root / "config.snapshot.yaml",
        metrics_path=root / "metrics.json",
        summary_path=root / "summary.md",
    )
