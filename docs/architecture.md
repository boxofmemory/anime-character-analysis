# Architecture

```
                        ┌─────────────────────────────┐
   browser ────────────▶│  apps/web  (Vue 3 + Vite)   │
                        │  Element Plus · ECharts     │
                        └──────────────┬──────────────┘
                                       │ REST / JSON
                        ┌──────────────▼──────────────┐
                        │  apps/api  (FastAPI)        │
                        │  routes → services → repo   │
                        └──┬─────────┬─────────┬──────┘
                           │         │         │
              ┌────────────▼──┐  ┌───▼────┐  ┌─▼──────────────┐
              │ video_pipeline│  │ MySQL  │  │ asset storage  │
              │ YOLO→ByteTrack│  │ SQLAlc │  │ data/ models/  │
              │ →ArcFace      │  └────────┘  └────────────────┘
              └───────────────┘
```

## Layers

**`apps/api/app/api/routes`** — thin HTTP adapters. They validate input, call a service, and map
domain exceptions to status codes. No inference or SQL lives here.

**`apps/api/app/services`** — the actual behaviour: video analysis orchestration, character library
management, dashboard aggregation, model registry rendering, and training job supervision. This is
where the system is interesting; the routes are only plumbing.

**`apps/api/app/repositories`** — persistence seams. Video/character/appearance reads and writes are
kept out of services so they can be tested without a database.

**`apps/api/app/models/entities.py`** — three tables:

| Table | Purpose |
| --- | --- |
| `videos` | one row per uploaded file: filename, status, progress, duration, fps |
| `characters` | known characters and auto-generated strangers, with their type |
| `appearances` | `(video, character, start_time, end_time)` — the timeline itself |

An appearance is a *segment*, not a frame. That is deliberate: the UI never renders frame-level data,
so storing it would only cost space.

**`apps/api/video_demo.py`** — the original standalone inference script, kept as the computational
core (`ResNet50Extractor`, `FaceRecognizer`, `TrackFeatureAggregator`, `process_video`). The service
layer wraps it rather than rewriting it, so the algorithm can still be run from the command line.

**`apps/api/app/core/settings.py`** — every path and threshold in one frozen dataclass. Asset paths
are derived from `PROJECT_ROOT`, never from the current working directory, which is what lets the
same code run from `uvicorn`, from a test, or from a script.

## The analysis pipeline

```
upload → save to data/inputs/videos → row in videos(status=pending)
       → background task:
           for each frame:
               YOLO detect faces
               ByteTrack assign track ids
               crop + embed each face (ArcFace, L2-normalised)
           per track:
               drop if len(detections) < MIN_TRACK_FRAMES
               mean of embeddings → one vector per track
           per track vector:
               max cosine over known library     → known character if ≥ KNOWN_THRESHOLD
               else max cosine over strangers    → existing stranger if > STRANGER_THRESHOLD
               else                              → create Stranger_N
           merge segments separated by < MAX_GAP_FRAMES
           drop segments shorter than MIN_DURATION_SEC
           write appearances, annotated video, timeline JSON
       → status=completed, progress=100
```

Two thresholds instead of one is the core open-set decision. `KNOWN_THRESHOLD` is strict because a
false positive mislabels a character for the whole episode; `STRANGER_THRESHOLD` is looser because
over-merging two strangers is easy to undo later.

## Libraries are per recognition model

`settings.library_profile_paths(model)` resolves the face library to
`data/libraries/profiles/<model-slug>--<sha1(model-path)[:8]>/`.

Embeddings from two different networks live in unrelated vector spaces, so comparing them produces
garbage. Keying the library by the model's path fingerprint makes that impossible by construction:
switch the recognition model in the UI and you transparently get a compatible library, or an empty
one to build. The default profile falls back to the legacy `data/libraries/known_faces` layout when
no per-model profile exists yet.

## Training jobs

`training_service` spawns `research/training/scripts/*` as subprocesses, captures stdout into a log
file, and tracks lifecycle state in memory plus on disk under
`research/training/experiments/<timestamp>_<name>/`.

A finished job is **not** installed automatically. `POST /api/training/jobs/{id}/promote` copies the
selected artefact into `models/production/` and records the decision in
`models/manifests/training_production.json`. Promotion is the only path that mutates production
assets, which keeps the interesting question ("is this model better?") a human decision.

## Frontend

`apps/web` is a small Vue 3 SPA: three pages (dashboard, video analysis, settings) plus an image
recognition view. State stays local to pages — there is no Pinia store, because every screen loads
its own data from the API and nothing needs to be shared. Charts are ECharts wrappers in
`components/charts`.

## Known limitations

- Analysis runs in a background thread inside the API process; a crash loses in-flight jobs. A real
  deployment wants a queue (Celery/RQ) and a separate worker.
- Progress is polled, not pushed. WebSocket/SSE would be smoother on long videos.
- The detector is single-class; grouped shots and background characters are detected but usually
  filtered out by the minimum-duration rule.
- No authentication. It is a single-user local tool by design.
