# Data layout

Runtime data is kept out of the source tree's way: everything the service reads or writes at run
time lives here, and nothing here is committed.

```
data/
  inputs/
    videos/              uploaded videos awaiting or under analysis
    sample_videos/       small clips for manual testing
  libraries/
    known_faces/         default character library (used when no per-model profile exists)
    strangers/           auto-clustered unknown characters
    profiles/
      <model>--<hash>/   one isolated library per recognition model
        known_faces/
        strangers/
  outputs/
    inference/
      annotated/         annotated output videos
      timelines/         timeline JSON + per-run inference reports
    experiments/         exported experiment reports
    training/            exports produced by the training service
```

## Rules

- The service only writes to `inputs`, `libraries` and `outputs/inference`.
- Training artefacts never land in the inference output directory.
- Sample clips stay separate from real uploads so you can clear uploads without losing test material.
- Face libraries are **per recognition model**. Two models produce incompatible embeddings, so
  `profiles/<model>--<hash>/` keeps them apart. Switching models in the UI switches libraries.
