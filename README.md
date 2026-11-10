# Anime Character Analysis

**Video → who appears, when, and for how long.**

An end-to-end pipeline that watches an anime episode, detects every face, keeps the same character
associated across frames, recognises who it is (or learns a new "stranger"), and finally renders a
per-character appearance timeline on a web dashboard.

The interesting part is not any single model — it is the plumbing between them: track-level feature
aggregation, open-set matching with a stranger library, and timeline post-processing that turns noisy
frame-level detections into something a human can actually read.

![Video analysis workbench](docs/assets/video-workbench.png)

*Video analysis workbench: annotated output on the left, per-character appearance timeline below.
Clicking a segment jumps the video to that moment.*

---

## How it works

```mermaid
flowchart LR
    A[Upload video] --> B[YOLOv8s<br/>anime face detection]
    B --> C[ByteTrack<br/>multi-object tracking]
    C --> D[ArcFace ResNet50<br/>per-track embedding]
    D --> E{Match known<br/>library?}
    E -- cosine >= 0.50 --> F[Known character]
    E -- else --> G{Seen stranger<br/>before?}
    G -- cosine > 0.55 --> H[Existing stranger]
    G -- else --> I[New stranger profile]
    F --> J[Timeline builder]
    H --> J
    I --> J
    J --> K[(MySQL)]
    K --> L[Vue 3 dashboard]
```

1. **Detect** — a YOLOv8s model fine-tuned on anime faces runs per frame.
2. **Track** — ByteTrack assigns a track id, so the same face keeps one identity through occlusion
   and fast cuts.
3. **Aggregate** — every track accumulates embeddings; the track mean is what gets matched, not a
   single frame. Tracks shorter than 3 frames are dropped as detector noise.
4. **Recognise** — cosine similarity against the known-character library (threshold 0.50). Unknown
   faces fall through to a second library of *strangers*, clustered at 0.55, so a recurring side
   character gets a stable `Stranger_3` id instead of fifty one-frame ghosts.
5. **Build the timeline** — gaps shorter than 15 frames are merged, segments shorter than 0.5 s are
   discarded, and the result is persisted with the annotated video.
6. **Inspect** — the frontend shows the annotated video, an appearance timeline, a per-character
   share pie, a scene heatmap, and a history of past runs.

A track buffer of 90 frames (~3 s at 30 fps) keeps a character's identity alive across short
disappearances, which is the single biggest quality lever for this kind of footage.

<img src="docs/assets/dashboard.png" width="720" alt="Dashboard with timeline, share pie, ranking and co-occurrence heatmap">

*Dashboard: appearance timeline, share distribution, total screen time ranking, and a character
co-occurrence heatmap (who shares scenes with whom, and for how long).*

---

## Results

Every number below is reproducible from the artefacts in `research/training/experiments/` through
`scripts/data/import_demo_results.py`, which derives `metrics.json` / `summary.md` straight from the
raw `results.csv` and `training_log_r4.csv` logs.

**Hardware:** RTX 3060 Laptop (6 GB), i5-12500H, 15.7 GB RAM.

### Detection — YOLOv8s on anime faces

Dataset `anime-face.v5i.yolov8`: 12,144 train / 1,233 val / 622 test, 640×640 input.

| Metric | Best value | Epoch |
| --- | --- | --- |
| Precision | 0.9806 | 26 |
| Recall | 0.9729 | 39 |
| mAP@0.5 | 0.9881 | 28 |
| mAP@0.5:0.95 | 0.7265 | 52 |

### Tracking — default vs anime-tuned ByteTrack

803-frame evaluation sequence, identical detector in both runs.

| Config | MOTA | IDF1 | ID switches | False negatives |
| --- | --- | --- | --- | --- |
| ByteTrack default | 0.9775 | 0.9449 | 6 | 6 |
| Anime-tuned | **0.9882** | **0.9867** | **2** | **0** |

### Recognition — ResNet50 + ArcFace

130 character classes, 9,737 images, 0.8 stratified split, 512-d embedding.

| Version | Val accuracy | Prototype Top-1 | Prototype Top-5 |
| --- | --- | --- | --- |
| r2 | 13.89% | 11.84% | 26.02% |
| r3 | 28.18% | 28.34% | 54.46% |
| **r4** | **89.45%** | **88.63%** | **95.67%** |

At epoch 42: same-character similarity **0.7302**, different-character **0.3588** — a margin of 0.37.
That margin is what makes the 0.50 / 0.55 thresholds a measurement rather than a guess.

![Model showcase](docs/assets/model-showcase.png)

*Model showcase page: the registry entry for the deployed embedder, with its validation curves,
similarity-margin histogram and hardest-confused character pairs.*

### Timeline smoothing

Same video, same detector, same tracker; only the recognition model changes.

| Run | Recognition model | Segments before | After gap merge | After short filter | Reduction |
| --- | --- | --- | --- | --- | --- |
| video_22 | r4 | 59 | 52 | 47 | **20.3%** |
| video_23 | v2 | 70 | 65 | 59 | 15.7% |

### Open-set clustering

| Video | True strangers | Predicted clusters | Purity | B-Cubed F1 |
| --- | --- | --- | --- | --- |
| 22 (r4) | 18 | 19 | 1.000 | **0.9787** |
| 23 (v2) | 23 | 41 | 1.000 | 0.7437 |

An r4 feature space splits unknowns into roughly the right number of identities; on the weaker v2
space the same mechanism shatters 23 real strangers into 41 clusters. Clustering quality tracks
embedding quality, not clustering logic.

### Ablation — what actually mattered

| Configuration | Top-1 | Top-5 | First-epoch loss |
| --- | --- | --- | --- |
| Full strategy | 86.54 | **95.67** | **3.26** |
| No margin warmup | 86.80 | 95.31 | 12.51 |
| No progressive unfreeze | 85.17 | 95.16 | 2.91 |
| No balanced sampler | 86.80 | 95.62 | 3.51 |

Read honestly: the Top-1 spread (85.17–86.80) sits inside noise for a 9.7 k-image dataset. The one
unambiguous effect is **margin warmup** — drop it and the first epoch loss jumps from 3.26 to 12.51,
because a 0.35 ArcFace margin applied cold destabilises training. The other two changes mainly buy
stability rather than points.

![Detection training curves](docs/assets/training-curves.png)

*YOLOv8s fine-tuning: box, class and DFL losses, precision/recall, mAP@0.5 and mAP@0.5:0.95.*

---

## Features

| Area | What it does |
| --- | --- |
| Analysis | Upload a video, pick a recognition model, follow job progress, get an annotated video back |
| Timeline | Per-character appearance segments, merged and de-noised |
| Analytics | Character count over time, share distribution, appearance heatmap, co-occurrence matrix |
| Character library | Known faces and stranger profiles, rebuild features, delete entries |
| Open-set | Promote a stranger to a known character; historical timelines are corrected too |
| Model workshop | Registry of detection/recognition/tracker assets with metadata and curves |
| Training service | Launch a training job from the UI, stream its log, promote the result to production |
| Single image | Run recognition on one uploaded frame |

---

## Repository layout

```
apps/
  api/            FastAPI service, video pipeline, model registry, training jobs
  web/            Vue 3 + Vite dashboard (Element Plus, ECharts)
research/
  training/       Reusable training code: detector, recognition, tracker evaluation
  dataset/        Dataset construction and label-correction scripts
scripts/          Dev launchers, DB bootstrap, data import helpers
models/           Registry manifests + tracker config (weights ship as release assets)
data/             Documented runtime layout (inputs, libraries, outputs)
docs/             Architecture, API reference, deployment, screenshots
tests/            Service-level tests for the backend
```

---

## Quick start

### Requirements

- Python 3.10+
- Node.js 18+
- MySQL 8 (or any SQLAlchemy-compatible database)
- A CUDA-capable GPU is strongly recommended (CPU inference works but is slow)

### 1. Model weights

Download the two weight files from the [**latest release**](../../releases/latest) and place them
where the registry expects:

```
models/production/detection/yolov8s_anime.pt               (21 MB)
models/production/recognition/resnet50_anime_arcface_best_r4.pth   (94 MB)
models/production/tracking/bytetrack_anime.yaml            (already in the repo)
```

See [`models/README.md`](models/README.md) and [`NOTICE.md`](NOTICE.md) for the usage terms.

### 2. Backend

```bash
cd apps/api
python -m venv .venv && . .venv/Scripts/activate    # Windows
pip install -r requirements.txt

cp ../../.env.example ../../.env                     # then edit DATABASE_URL
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/api/health/ready` — it reports whether the database, the model assets
and the known-character library are all reachable.

### 3. Frontend

```bash
cd apps/web
npm install
npm run dev                                          # http://localhost:5173
```

---

## Configuration

Everything is environment-driven and documented inline in [`.env.example`](.env.example).

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | — | SQLAlchemy DSN, e.g. `mysql+pymysql://user:pass@host:3306/anime_tracker` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `KNOWN_THRESHOLD` | `0.50` | Cosine similarity required to accept a known character |
| `STRANGER_THRESHOLD` | `0.55` | Cosine similarity required to reuse a stranger profile |
| `TRACK_BUFFER` | `90` | Frames a track survives without a detection |
| `MIN_TRACK_FRAMES` | `3` | Minimum detections before a track is trusted |
| `MAX_GAP_FRAMES` | `15` | Timeline gaps merged below this size |
| `MIN_DURATION_SEC` | `0.5` | Segments shorter than this are discarded |
| `DANBOORU_USERNAME` / `DANBOORU_API_KEY` | — | Optional, only for the web-import feature |

---

## API

Full reference in [`docs/api-reference.md`](docs/api-reference.md). The main entry points:

```
POST   /api/upload                        submit a video for analysis
GET    /api/videos/{id}/status            job progress
GET    /api/videos/{id}/timeline          per-character segments
GET    /api/videos/{id}/download          annotated video
GET    /api/dashboard/summary             aggregate analytics
GET    /api/library/characters            known + stranger libraries
POST   /api/characters/rename-stranger    promote/reassign a stranger
GET    /api/model-showcase                model registry view
POST   /api/training/jobs                 start a training run
```

Interactive docs are served at `/docs` by FastAPI.

---

## Data and licensing notes

- **No anime frames, datasets or character images are distributed.** The screenshots under
  `docs/assets/` show the running tool and are illustrative only.
- Model weights are released for **research and educational use**; see [`NOTICE.md`](NOTICE.md).
  The MIT licence covers this code, not any anime work you feed into it.
- The scripts under `research/dataset` fetch images from Danbooru through its public API. They read
  credentials from the environment and ship with none. If you use them, you are responsible for
  complying with Danbooru's terms of service and with the rights of the original artists.

---

## Design notes

A few decisions that were not obvious upfront:

- **Match on track means, not frames.** Single-frame embeddings are jittery near identical designs;
  averaging over a track cut the false-positive rate far more than tuning the threshold did.
- **Two libraries instead of a reject class.** An open-set problem is easier to serve when unknowns
  get their own persisted identities — it also gives the user a natural "promote to known" action,
  which is applied retroactively to already-analysed timelines.
- **Separate thresholds for known and unknown.** 0.50 for accepting a known character, 0.55 for
  merging two unknowns. Over-merging is the worse failure: two strangers sharing one profile is far
  harder to detect than one character split in two.
- **Post-process the timeline, not the detector.** Detector flicker is cheap to clean up downstream,
  and keeping raw detections means the thresholds stay adjustable without re-inference.
- **Paths live in one settings object.** The whole app resolves assets relative to the project root,
  which is what makes the same code run on a laptop and on a training box.

---

## 中文说明

这是一个动漫视频人物分析系统：输入一段动画视频，自动检测人脸、跨帧追踪同一个人物、识别其身份
（已知角色或自动聚类的「陌生人」），并输出「谁在什么时候出现、出现了多久」的时间轴与可视化看板。

识别链路为 YOLOv8s（动漫人脸检测）+ ByteTrack（多目标跟踪）+ ResNet50/ArcFace（特征提取），
FastAPI + MySQL 负责服务与持久化，Vue 3 + ECharts 负责前端展示。检测 mAP@0.5 为 0.988，
识别的 Prototype Top-1 为 88.63%，开放集聚类的 B-Cubed F1 为 0.9787。

仓库只包含**代码、配置与文档**，不包含任何动画素材、数据集或原始图像；模型权重以 Release 附件
形式单独发布，仅限研究与非商业用途（见 `NOTICE.md`）。

---

## License

MIT for the code — see [LICENSE](LICENSE). Model weights and any anime content are excluded, see
[NOTICE.md](NOTICE.md).
