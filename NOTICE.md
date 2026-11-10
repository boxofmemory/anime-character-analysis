# Notice on data, model weights and third-party content

This repository contains **code and documentation only**. Model weights are published separately as
release assets. The following explains the scope, and the limits, of what is offered here.

## Model weights

The released weights — `yolov8s_anime.pt` (face detector) and `resnet50_anime_arcface_best_r4.pth`
(face embedder) — were trained on publicly obtainable anime face data. They are provided **for
research and educational use only**, and must not be used as part of a commercial product or
service.

## Training data

- The detector was fine-tuned on `anime-face.v5i.yolov8`, a publicly hosted anime face dataset.
- The embedder was trained on an anime character image collection built from public sources,
  including Danbooru via its public API.

The images in those datasets are **not** distributed by this project, in any form. Copyright in
anime frames, character designs and illustrations belongs to the respective animation studios,
licensors and artists. This project claims no rights over that material, and the MIT licence in this
repository does not extend to it.

If you use the dataset-building scripts in `research/dataset`, you are responsible for complying with
the terms of service of every source you download from, and with the rights of the original authors.

## Model behaviour

Both models are **discriminative**. The detector outputs bounding boxes; the embedder outputs a
512-dimensional L2-normalised feature vector. Neither can reconstruct or reproduce a training image,
and neither is intended to. Nothing here generates or redistributes animation frames.

## Screenshots

The images under `docs/assets/` are screenshots of this software running. They include anime frames
and character artwork solely to illustrate what the tool does on real input. That material remains
the property of its rights holders.

## Takedown

If you hold rights over any material referenced here and believe this repository oversteps, please
open an issue on this repository. Files will be removed promptly.
