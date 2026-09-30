"""Unit tests for boundary interfaces and FastAPI application factory."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.data.loader import BaseDataLoader
from src.ml.supervised import BaseSupervisedFraudModel
from src.engine.risk_engine import BaseHybridRiskEngine


def test_fastapi_health_endpoint():
    """Verify GET /health returns HTTP 200 and health status."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["service"] == "fraud-detection-system"


def test_fastapi_predict_endpoint_uninitialized():
    """Verify POST /predict returns HTTP 503 when models are not loaded."""
    from src.engine.prediction_service import PredictionService
    from src.ml.registry import ModelRegistry
    from src.api.routes.predict import get_prediction_service

    app = create_app()
    empty_registry = ModelRegistry(model_dir="./models/empty_test_artifacts")
    app.dependency_overrides[get_prediction_service] = lambda: PredictionService(registry=empty_registry)
    client = TestClient(app)
    payload = {
        "transaction": {
            "time": 100.0,
            "amount": 150.0,
            "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 0.0, "v5": 0.0, "v6": 0.0, "v7": 0.0, "v8": 0.0,
            "v9": 0.0, "v10": 0.0, "v11": 0.0, "v12": 0.0, "v13": 0.0, "v14": 0.0, "v15": 0.0, "v16": 0.0,
            "v17": 0.0, "v18": 0.0, "v19": 0.0, "v20": 0.0, "v21": 0.0, "v22": 0.0, "v23": 0.0, "v24": 0.0,
            "v25": 0.0, "v26": 0.0, "v27": 0.0, "v28": 0.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Model service unavailable" in response.json()["detail"] or "Model artifacts not yet loaded" in response.json()["detail"] or "unavailable" in response.json()["detail"]


def test_abstract_interfaces_cannot_be_instantiated():
    """Verify abstract interfaces raise TypeError when instantiated directly."""
    with pytest.raises(TypeError):
        BaseDataLoader()

    with pytest.raises(TypeError):
        BaseSupervisedFraudModel()

    with pytest.raises(TypeError):
        BaseHybridRiskEngine()
