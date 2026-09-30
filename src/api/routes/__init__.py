"""FastAPI router package."""
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.api.routes.advanced_predict import router as advanced_predict_router
from src.api.routes.alerts import router as alerts_router
from src.api.routes.transactions import router as transactions_router
from src.api.routes.metrics import router as metrics_router

__all__ = [
    "health_router",
    "predict_router",
    "advanced_predict_router",
    "alerts_router",
    "transactions_router",
    "metrics_router",
]
