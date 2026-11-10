# Training Research

This directory is the training and experiment workspace for this
project. It keeps the original algorithm direction, but replaces scattered
demo-style scripts with stable entrypoints, path-based configs, and
experiment-by-experiment outputs.

## Architecture

- `configs/`: formal configs for detector, tracker, recognition, and character library
- `src/`: reusable training, evaluation, and experiment helpers
- `scripts/`: stable CLI entrypoints
- `weights/`: local research weights used as fine-tuning baselines
- `experiments/`: timestamped experiment outputs
- `datasets/`: dataset mount points
- `legacy/`: archived original scripts from the demo versions

## Recommended Entrypoints

- `python scripts/train_arcface.py --config configs/recognition/arcface_resnet50.yaml`
- `python scripts/train_yolo.py --config configs/detector/yolo_anime_face.yaml`
- `python scripts/evaluate_tracker.py --config configs/tracker/evaluate_bytetrack.yaml`
- `python scripts/build_character_library.py --config configs/library/known_database.yaml`
- `python scripts/run_inference_demo.py --config configs/tracker/inference_demo.yaml`

Each script also supports `--validate-only` so you can check config wiring
before launching a long run.

## Formal Rules

- Keep `legacy/` read-only as the historical reference baseline.
- Use `configs + scripts + experiments` as the main formal workflow.
- Do not overwrite `models/production/` automatically after a training run.
- Promote a model to runtime only after you manually review its experiment
  output and decide it is the new production baseline.
