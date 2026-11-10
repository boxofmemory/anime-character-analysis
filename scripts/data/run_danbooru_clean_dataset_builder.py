from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = PROJECT_ROOT / "research" / "dataset" / "danbooru_clean_dataset_builder.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("danbooru_clean_dataset_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load builder script: {BUILDER_PATH}")
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Formal wrapper for the Danbooru clean dataset builder demo")
    parser.add_argument("mode", choices=["download", "crop", "all", "stats"])
    parser.add_argument("--queries-file", default=str(PROJECT_ROOT / "research" / "dataset" / "正确的熟人库_final.txt"))
    parser.add_argument("--raw-output-dir", default=r"./datasets/anime_face_dataset\danbooru_candidates")
    parser.add_argument("--clean-output-dir", default=r"./datasets/anime_face_dataset\anime_face_dataset_clean_224")
    parser.add_argument("--review-output-dir", default=r"./datasets/anime_face_dataset\anime_face_dataset_review")
    parser.add_argument("--report-dir", default=r"./datasets/anime_face_dataset\anime_face_dataset_reports")
    parser.add_argument("--yolo-model-path", default=str(PROJECT_ROOT / "research" / "dataset" / "yolov8s_anime.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _load_builder_module()
    module.QUERIES_FILE = args.queries_file
    module.RAW_OUTPUT_DIR = args.raw_output_dir
    module.CLEAN_OUTPUT_DIR = args.clean_output_dir
    module.REVIEW_OUTPUT_DIR = args.review_output_dir
    module.REPORT_DIR = args.report_dir
    module.YOLO_MODEL_PATH = args.yolo_model_path

    sys.argv = [str(BUILDER_PATH), args.mode]
    module.main()


if __name__ == "__main__":
    main()
