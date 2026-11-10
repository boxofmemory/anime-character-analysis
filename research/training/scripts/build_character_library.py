from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config_loader import load_yaml
from src.common.experiment import create_experiment_run
from src.common.paths import paths
from src.library import build_known_database_from_config, validate_library_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build known character library from image folders")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "library" / "known_database.yaml"),
        help="Path to library build config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without building the library",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.validate_only:
        validate_library_config(config)
        print("Character library config validation passed")
        print(f"Config: {Path(args.config).resolve()}")
        return

    run = create_experiment_run("library", str(config.get("experiment_name", "known_database_formal")))
    run.write_config_snapshot(config)
    metrics = build_known_database_from_config(config, run)
    run.write_metrics(metrics)
    run.write_summary(_build_summary(metrics, run))
    print(f"Character library build completed: {run.experiment_id}")
    print(f"Results directory: {run.root}")


def _build_summary(metrics: dict, run) -> str:
    return f"""# Character Library Summary

- Experiment ID: `{metrics['experiment_id']}`
- Dataset: `{metrics['dataset_dir']}`
- Recognition model: `{metrics['recognition_model']}`
- Output JSON: `{metrics['known_database_path']}`
- Character count: `{metrics['character_count']}`
- Filtered classes: `{len(metrics['filtered_classes'])}`
- Config snapshot: `{run.config_snapshot_path}`
"""


if __name__ == "__main__":
    main()
