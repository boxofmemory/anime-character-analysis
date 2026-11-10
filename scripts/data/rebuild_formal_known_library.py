from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[2] / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
settings = importlib.import_module("app.core.settings").settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the formal known character library for a recognition model.")
    parser.add_argument(
        "--dataset-dir",
        default=str(settings.small_dataset_dir),
        help="Character dataset root. Each directory is treated as one known character.",
    )
    parser.add_argument(
        "--recognition-model-path",
        default=str(_default_r4_model_path()),
        help="Recognition model used to extract known-character embeddings.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the selected model profile library after creating a timestamped backup.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_path = Path(args.recognition_model_path).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Recognition model not found: {model_path}")

    library_paths = settings.library_profile_paths(model_path)
    library_paths.ensure_dirs()

    if args.replace:
        _backup_and_clear_profile(library_paths.root_dir, library_paths.profile_key)

    _sync_dataset_to_profile(dataset_dir, library_paths.cropped_dir, library_paths.sources_dir)
    get_extractor = importlib.import_module("app.services.video_pipeline").get_extractor
    build_known_database = importlib.import_module("video_demo").build_known_database

    extractor = get_extractor(str(model_path))
    known_db = build_known_database(
        extractor=extractor,
        dataset_dir=str(library_paths.cropped_dir),
        output_path=str(library_paths.known_db_path),
    )
    _write_manifest(known_db, library_paths)

    print("Formal known library rebuilt")
    print(f"Dataset: {dataset_dir}")
    print(f"Recognition model: {model_path}")
    print(f"Profile key: {library_paths.profile_key}")
    print(f"Known DB: {library_paths.known_db_path}")
    print(f"Character count: {len(known_db)}")


def _default_r4_model_path() -> Path:
    r4_path = settings.models_dir / "production" / "recognition" / "resnet50_anime_arcface_best_r4.pth"
    return r4_path if r4_path.exists() else settings.recognition_model_path


def _backup_and_clear_profile(profile_root: Path, profile_key: str) -> None:
    if not profile_root.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = settings.library_root_dir / "backups" / f"{profile_key}_{timestamp}"
    backup_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(profile_root, backup_root)
    for child in profile_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    profile_root.mkdir(parents=True, exist_ok=True)
    print(f"Backup created: {backup_root}")


def _sync_dataset_to_profile(dataset_dir: Path, cropped_dir: Path, sources_dir: Path) -> None:
    cropped_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    for character_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        images = _list_images(character_dir)
        if not images:
            continue

        target_cropped_dir = cropped_dir / character_dir.name
        target_sources_dir = sources_dir / character_dir.name
        target_cropped_dir.mkdir(parents=True, exist_ok=True)
        target_sources_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(images, start=1):
            suffix = image_path.suffix.lower() or ".jpg"
            filename = f"{index:04d}{suffix}"
            shutil.copy2(image_path, target_cropped_dir / filename)
            shutil.copy2(image_path, target_sources_dir / filename)


def _write_manifest(known_db: dict, library_paths) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    characters = {}
    for name, payload in sorted(known_db.items()):
        folder = name
        cropped_dir = library_paths.cropped_dir / folder
        images = _list_images(cropped_dir)
        portrait_rel_path = None
        if images:
            portrait_target = library_paths.portraits_dir / f"{folder}.jpg"
            portrait_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(images[0], portrait_target)
            portrait_rel_path = portrait_target.relative_to(library_paths.root_dir).as_posix()

        characters[name] = {
            "name": name,
            "folder": folder,
            "created_at": now,
            "updated_at": now,
            "source_types": ["batch_dataset"],
            "sample_count": int(payload.get("count", len(images))),
            "portrait_rel_path": portrait_rel_path,
            "images": [
                {
                    "id": image_path.stem,
                    "cropped_rel_path": image_path.relative_to(library_paths.root_dir).as_posix(),
                    "original_rel_path": (library_paths.sources_dir / folder / image_path.name)
                    .relative_to(library_paths.root_dir)
                    .as_posix(),
                    "quality_score": None,
                    "sharpness": None,
                    "confidence": None,
                    "source_type": "batch_dataset",
                    "created_at": now,
                }
                for image_path in images
            ],
        }

    manifest = {
        "profile_key": library_paths.profile_key,
        "recognition_model_path": str(library_paths.recognition_model_path),
        "characters": characters,
        "updated_at": now,
    }
    library_paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    library_paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


if __name__ == "__main__":
    main()
