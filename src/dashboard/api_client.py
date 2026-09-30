"""Centralized HTTP API Client for connecting Streamlit dashboard to FastAPI backend."""
import os
import httpx
import streamlit as st
from typing import Dict, Any, List, Optional
from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("dashboard_api_client")


class DashboardAPIClient:
    """HTTP client wrapper communicating with the FastAPI fraud detection service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 3.0):
        settings = get_settings()
        env_url = os.getenv("API_BASE_URL")
        default_url = f"http://{settings.api_host}:{settings.api_port}" if settings.api_host != "0.0.0.0" else "http://127.0.0.1:8000"
        
        self.base_url = (base_url or env_url or default_url).rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, endpoint: str, json: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method=method, url=url, json=json, params=params)
                
                if response.status_code == 200:
                    return {"success": True, "data": response.json(), "status_code": 200}
                
                # Handle structured error responses (e.g. 503 missing artifacts, 404 not found, 422 validation error)
                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                return {
                    "success": False,
                    "error": error_detail,
                    "status_code": response.status_code,
                }
        except httpx.ConnectError:
            logger.error(f"Failed to connect to API service at '{url}'. API may be offline.")
            return {
                "success": False,
                "error": f"Unable to connect to Fraud API at '{self.base_url}'. Ensure FastAPI service is running.",
                "status_code": 503,
            }
        except Exception as e:
            logger.error(f"HTTP request error ({method} {endpoint}): {str(e)}")
            return {
                "success": False,
                "error": f"API Request Error: {str(e)}",
                "status_code": 500,
            }

    def get_health(self) -> Dict[str, Any]:
        """Fetch system health status."""
        return self._request("GET", "/health")

    def get_metrics(self) -> Dict[str, Any]:
        """Fetch operational performance metrics."""
        return self._request("GET", "/metrics")

    def predict_transaction(self, transaction_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit single transaction for risk prediction."""
        # Ensure payload is wrapped under "transaction" key if not already wrapped
        body = transaction_payload if "transaction" in transaction_payload else {"transaction": transaction_payload}
        return self._request("POST", "/predict", json=body)

    def predict_advanced_transaction(self, advanced_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit multi-entity transaction for advanced IEEE-CIS risk evaluation."""
        return self._request("POST", "/predict/advanced", json=advanced_payload)

    def get_alerts(
        self,
        status: Optional[str] = None,
        min_risk_score: Optional[float] = None,
        risk_level: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Fetch flagged risk alerts with optional filters."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if min_risk_score is not None:
            params["min_risk_score"] = min_risk_score
        if risk_level:
            params["risk_level"] = risk_level
        return self._request("GET", "/alerts", params=params)

    def review_alert(
        self, alert_id: str, new_status: str, notes: Optional[str] = None, analyst_id: str = "ANALYST-UI-01"
    ) -> Dict[str, Any]:
        """Submit analyst investigation action for an alert."""
        body = {
            "status": new_status,
            "notes": notes,
            "analyst_id": analyst_id,
        }
        return self._request("POST", f"/alerts/{alert_id}/review", json=body)

    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Fetch evaluated transaction by system primary key or client transaction ID."""
        return self._request("GET", f"/transactions/{transaction_id}")

    def get_transactions(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch evaluated transaction history."""
        params = {"limit": limit, "offset": offset}
        return self._request("GET", "/transactions", params=params)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_cached_metrics(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Cached wrapper for metrics telemetry API endpoint with 5-second TTL."""
    client = DashboardAPIClient(base_url=base_url)
    return client.get_metrics()


@st.cache_data(ttl=5, show_spinner=False)
def fetch_cached_alerts(base_url: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """Cached wrapper for alerts telemetry API endpoint with 5-second TTL."""
    client = DashboardAPIClient(base_url=base_url)
    return client.get_alerts(limit=limit)

