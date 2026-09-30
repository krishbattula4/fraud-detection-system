"""FastAPI application factory."""
from fastapi import FastAPI
from src.core.config import get_settings
from src.core.logging import setup_logging
from src.api.routes import (
    health_router,
    predict_router,
    advanced_predict_router,
    alerts_router,
    transactions_router,
    metrics_router,
)


def create_app() -> FastAPI:
    """Construct FastAPI instance and register API routers."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)

    app = FastAPI(
        title="AI Financial Fraud Detection & Risk Intelligence System",
        description="REST API service for financial transaction risk evaluation, alert review, and monitoring.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(advanced_predict_router)
    app.include_router(alerts_router)
    app.include_router(transactions_router)
    app.include_router(metrics_router)

    return app


app = create_app()
