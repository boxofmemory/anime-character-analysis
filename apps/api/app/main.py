from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import (
    character_router,
    dashboard_router,
    library_router,
    model_showcase_router,
    training_router,
    video_router,
)
from app.core.database import Base, engine
from app.core.settings import settings, validate_runtime_assets


app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def initialize_runtime() -> None:
    status = {
        "database_initialized": False,
        "assets_valid": False,
        "errors": [],
    }
    try:
        Base.metadata.create_all(bind=engine)
        status["database_initialized"] = True
    except Exception as exc:  # noqa: BLE001 - exposed through readiness endpoint
        status["errors"].append(f"database: {exc}")

    try:
        validate_runtime_assets()
        status["assets_valid"] = True
    except Exception as exc:  # noqa: BLE001 - exposed through readiness endpoint
        status["errors"].append(f"assets: {exc}")

    app.state.startup_status = status


@app.get("/api/health/ready")
def read_ready():
    details = {
        "database": False,
        "assets": False,
        "known_library": False,
    }
    errors = []

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        details["database"] = True
    except Exception as exc:  # noqa: BLE001 - health endpoint must not crash
        errors.append(f"database: {exc}")

    try:
        validate_runtime_assets()
        details["assets"] = True
    except Exception as exc:  # noqa: BLE001 - health endpoint must not crash
        errors.append(f"assets: {exc}")

    library_paths = settings.library_profile_paths(settings.recognition_model_path)
    details["known_library"] = library_paths.known_db_path.exists()
    if not details["known_library"]:
        errors.append(f"known library missing: {library_paths.known_db_path}")

    ready = all(details.values())
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "ready" if ready else "starting",
        "ready": ready,
        "details": details,
        "errors": errors,
        "startup": getattr(app.state, "startup_status", {}),
        "recognition_model": settings.recognition_model_path.name,
        "recognition_library_profile": library_paths.profile_key,
        "known_db_path": str(library_paths.known_db_path),
    }

app.include_router(video_router)
app.include_router(character_router)
app.include_router(dashboard_router)
app.include_router(library_router)
app.include_router(model_showcase_router)
app.include_router(training_router)
