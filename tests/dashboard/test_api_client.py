"""Unit test suite for DashboardAPIClient wrapper."""
import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.dashboard.api_client import DashboardAPIClient


@pytest.fixture
def api_client():
    """Default DashboardAPIClient fixture."""
    return DashboardAPIClient(base_url="http://test-api:8000")


def test_api_client_get_health_success(api_client):
    """Test get_health success path."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy", "database": "connected", "model_artifacts": "available"}

    with patch.object(httpx.Client, "request", return_value=mock_response):
        res = api_client.get_health()
        assert res["success"] is True
        assert res["status_code"] == 200
        assert res["data"]["status"] == "healthy"


def test_api_client_handles_503_missing_artifacts(api_client):
    """Test API client handling 503 missing model artifacts response."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"detail": "Model service unavailable: Required model artifacts for version '1.0.0' are unavailable."}

    with patch.object(httpx.Client, "request", return_value=mock_response):
        res = api_client.predict_transaction({"time": 100.0, "amount": 50.0})
        assert res["success"] is False
        assert res["status_code"] == 503
        assert "Model service unavailable" in res["error"]


def test_api_client_handles_404_not_found(api_client):
    """Test API client handling 404 alert not found response."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"detail": "Alert with ID 'NONEXISTENT' not found."}

    with patch.object(httpx.Client, "request", return_value=mock_response):
        res = api_client.review_alert("NONEXISTENT", "CLOSED")
        assert res["success"] is False
        assert res["status_code"] == 404
        assert "not found" in res["error"]


def test_api_client_handles_connection_failure(api_client):
    """Test API client handling connection failure gracefully without raising uncaught exception."""
    with patch.object(httpx.Client, "request", side_effect=httpx.ConnectError("Connection refused")):
        res = api_client.get_metrics()
        assert res["success"] is False
        assert res["status_code"] == 503
        assert "Unable to connect" in res["error"]


def test_api_client_get_alerts_filtering(api_client):
    """Test get_alerts passes correct query parameters."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []

    with patch.object(httpx.Client, "request", return_value=mock_response) as mock_req:
        res = api_client.get_alerts(status="OPEN", min_risk_score=50.0, limit=10)
        assert res["success"] is True
        mock_req.assert_called_once_with(
            method="GET",
            url="http://test-api:8000/alerts",
            json=None,
            params={"limit": 10, "offset": 0, "status": "OPEN", "min_risk_score": 50.0},
        )
