# Model assets

The runtime expects three kinds of assets. Only the tracker configuration is small enough (and
licensable enough) to live in git; the two network weights are **not** committed.

```
models/
  manifests/model-registry.yaml          asset registry, read by the model showcase API
  production/
    detection/yolov8s_anime.pt           YOLOv8s fine-tuned on anime faces        (not committed)
    recognition/resnet50_anime_arcface_best_r4.pth
                                         ResNet50 + ArcFace face embedder         (not committed)
    tracking/bytetrack_anime.yaml        ByteTrack settings tuned for animation   (committed)
  archive/
    recognition/resnet50_anime_arcface.pth   older baseline, kept for comparison (not committed)
```

## Two-tier layout

- `production/` — what the live service loads. Only change this on purpose.
- `archive/` — historical baselines. Useful when you want to A/B a new model against the old one.

Training runs never write into `production/` automatically. A run produces artefacts under
`research/training/experiments/<timestamp>_<name>/`, and promotion is an explicit action — either
through the training UI or by hand plus a registry edit.

## Getting the weights

Download them from the [latest release](../../releases/latest), or train them yourself (see
[`research/training/README.md`](../research/training/README.md)), or supply your own checkpoints. Usage
terms are in [`NOTICE.md`](../NOTICE.md). The only hard requirements are:

- the detector must emit a single class (`face`),
- the embedder must be L2-normalised, since matching is a plain dot product,
- the embedder output dimension must match your stored character library
  (the API returns a clear error if it does not).

## Registry

`manifests/model-registry.yaml` is the single source of truth for "what is deployed". The
`/api/model-showcase` endpoint renders it, including file size and modification time, so you can see
at a glance whether the registry and the disk agree.
