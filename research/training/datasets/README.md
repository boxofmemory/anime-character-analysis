# Dataset Mount Points

Place your training and evaluation datasets under these directories:

- `arcface_dataset_112x112/`: character folders for ResNet50 + ArcFace training
- `yolo/anime-face/`: YOLO dataset with `images/` and `labels/`
- `tracker_eval/videos/`: short videos for ByteTrack parameter comparison
- `inference_demo/`: sample videos for qualitative inference demos

These folders are intentionally kept inside the formal training workspace so
your future training runs do not depend on scattered absolute paths.
