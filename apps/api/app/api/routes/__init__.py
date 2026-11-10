from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.characters import router as character_router
from app.api.routes.library import router as library_router
from app.api.routes.model_showcase import router as model_showcase_router
from app.api.routes.training import router as training_router
from app.api.routes.videos import router as video_router

__all__ = ["video_router", "character_router", "dashboard_router", "library_router", "model_showcase_router", "training_router"]
