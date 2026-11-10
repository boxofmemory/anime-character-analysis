from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRAINING_ROOT.parents[1]


@dataclass(frozen=True)
class TrainingPaths:
    root: Path = TRAINING_ROOT
    project_root: Path = PROJECT_ROOT
    configs: Path = TRAINING_ROOT / "configs"
    src: Path = TRAINING_ROOT / "src"
    scripts: Path = TRAINING_ROOT / "scripts"
    weights: Path = TRAINING_ROOT / "weights"
    outputs: Path = TRAINING_ROOT / "outputs"
    experiments: Path = TRAINING_ROOT / "experiments"
    datasets: Path = TRAINING_ROOT / "datasets"
    legacy: Path = TRAINING_ROOT / "legacy"
    models: Path = PROJECT_ROOT / "models"
    data: Path = PROJECT_ROOT / "data"
    api_root: Path = PROJECT_ROOT / "apps" / "api"

    def ensure(self) -> None:
        for path in (
            self.weights / "detection",
            self.weights / "recognition",
            self.weights / "tracking",
            self.outputs,
            self.experiments,
            self.experiments / "detection",
            self.experiments / "recognition",
            self.experiments / "tracker",
            self.experiments / "library",
            self.experiments / "inference",
            self.datasets,
            self.legacy,
        ):
            path.mkdir(parents=True, exist_ok=True)


paths = TrainingPaths()
paths.ensure()
