# Deployment

This is a single-user tool that was built to run on one machine. The notes below describe how to
run it somewhere it stays up.

## Process model

Two processes plus a database:

```
nginx / any static host  ──▶  built frontend (apps/web/dist)
                              proxies /api → uvicorn

uvicorn main:app  (apps/api)  ──▶  MySQL
```

## Backend

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="mysql+pymysql://user:pass@127.0.0.1:3306/anime_tracker"
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Keep `--workers 1`. Analysis jobs live in an in-process thread pool and a second worker would
duplicate them.

The service needs write access to `data/` and `models/`, and it needs the weights in place before
startup — `/api/health/ready` reports exactly what is missing if not.

## Frontend

```bash
cd apps/web
npm ci
npm run build          # emits dist/, serve it with nginx or any static server
```

Point `VITE_API_BASE_URL` at the public API origin before building, or put the frontend and the API
behind the same host so the default relative `/api` path works.

## Database

```sql
CREATE DATABASE anime_tracker CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Tables are created on startup (`Base.metadata.create_all`). There are no migrations yet; if the
schema changes, either add Alembic or recreate the database.

## GPU

The recognition model runs through `onnxruntime-gpu`; the detector goes through `ultralytics`. Both
need a CUDA-capable runtime that matches the installed `torch` build. On CPU everything still works,
just slowly — expect a large video to take tens of minutes instead of minutes.

## Hardening checklist

If this ever faces the internet rather than your laptop:

- [ ] put authentication in front of it — there is none
- [ ] restrict `CORS_ORIGINS` to your real origin instead of `*`
- [ ] move analysis to a real queue with retries and persistence
- [ ] add upload size and duration limits at the proxy
- [ ] run it as a non-root user with write access only to `data/` and `models/`
