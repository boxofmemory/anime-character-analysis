# API reference

Base URL: `http://localhost:8000`. Interactive schema at `/docs`.

## Health

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Liveness, returns name and version |
| GET | `/api/health/ready` | Readiness: database, model assets, known library |

## Video analysis

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/upload` | Upload a video (multipart), returns the new `video_id` |
| GET | `/api/analysis/model-options` | Recognition models available for analysis |
| POST | `/api/image-recognition` | Recognise faces in a single uploaded image |
| GET | `/api/videos/{video_id}/status` | Status, progress, error message |
| GET | `/api/videos/{video_id}/timeline` | Per-character appearance segments |
| GET | `/api/videos/{video_id}/download` | Annotated output video |
| PUT | `/api/videos/{video_id}/rename` | Rename a history entry |
| DELETE | `/api/videos/{video_id}` | Delete a video and its appearances |
| GET | `/api/history` | Past analyses, newest first |

## Characters and libraries

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/characters/rename-stranger` | Give a stranger a real name (promotes it to known) |
| POST | `/api/stranger/rename` | Alias of the above, kept for older clients |
| GET | `/api/library/options` | Import options and available recognition models |
| GET | `/api/library/characters` | Known and stranger entries |
| GET | `/api/library/characters/{name}` | One entry with its source images |
| POST | `/api/library/characters/import` | Import images from a local folder |
| POST | `/api/library/characters/import-from-web` | Import from Danbooru (needs credentials) |
| POST | `/api/library/characters/{name}/rebuild` | Recompute the feature vector for one entry |
| DELETE | `/api/library/characters/{name}` | Remove an entry and its assets |
| GET | `/api/library/assets/{path}` | Serve a stored crop or portrait |

## Analytics

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/dashboard/summary` | Totals, character distribution, per-run statistics |

## Models

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/model-showcase` | Registry contents plus on-disk size and timestamps |

## Training

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/training/options` | Trainable targets, configs, dataset availability |
| POST | `/api/training/jobs` | Start a job |
| GET | `/api/training/jobs` | List jobs |
| GET | `/api/training/jobs/{job_id}` | Job detail and state |
| GET | `/api/training/jobs/{job_id}/log` | Accumulated stdout |
| POST | `/api/training/jobs/{job_id}/cancel` | Terminate a running job |
| POST | `/api/training/jobs/{job_id}/promote` | Install artefacts into `models/production` |
| GET | `/api/training/jobs/{job_id}/artifacts/{path}` | Download a produced artefact |
