"""System Health & Infrastructure Diagnostics Page."""
import streamlit as st
from src.dashboard.api_client import DashboardAPIClient
from src.dashboard.styles import apply_custom_theme, render_page_header

st.set_page_config(page_title="System Health — Risk Intelligence", layout="wide", initial_sidebar_state="collapsed")

client = DashboardAPIClient()

def render_health_page():
    theme_mode = st.session_state.get("theme_mode", "dark")
    render_page_header(
        title="🩺 System Health & Infrastructure Diagnostics",
        subtitle="Operational readiness monitoring of REST API backend, database persistence, and ML model artifacts.",
        category="TECHNICAL HEALTH & DIAGNOSTICS"
    )

    with st.spinner("Running infrastructure diagnostics..."):
        health_res = client.get_health()

    if not health_res.get("success"):
        st.error("🔴 **REST API Service: OFFLINE**")
        st.error(f"Error details: `{health_res.get('error')}`")
        st.info("Ensure the FastAPI backend service is running on `http://127.0.0.1:8000`.")
        return

    data = health_res.get("data", {})
    overall_status = data.get("status", "unknown")
    db_status = data.get("database", "unknown")
    models_status = data.get("model_artifacts", "unknown")
    version = data.get("version", "1.0.0")

    # Overall Status Banner
    if overall_status == "healthy":
        st.success("🟢 **System Health: OPTIMAL** — All REST API endpoints, database persistence layer, and ML model artifacts are operational.")
    else:
        st.warning("🟡 **System Health: DEGRADED** — REST API is online, but one or more underlying dependencies require attention.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. REST API Engine")
        st.success("🟢 Service: ONLINE")
        st.write("**Application:** Fraud Detection & Risk Engine")
        st.write(f"**Version:** `{version}`")
        st.write(f"**Endpoint:** `{client.base_url}`")

    with col2:
        st.subheader("2. Database Persistence")
        if db_status == "connected":
            st.success("🟢 Status: CONNECTED")
            st.write("**Engine:** SQLite (SQLAlchemy ORM)")
            st.write("**Tables:** Transactions, Predictions, Alerts, Reviews")
        else:
            st.error("🔴 Status: UNAVAILABLE")
            st.write("Database connection or table verification failed.")

    with col3:
        st.subheader("3. Machine Learning Models")
        if models_status == "available":
            st.success("🟢 Status: READY")
            st.write("Fraud detection & risk models initialized.")
            st.write("Artifact Storage: Operational & Loaded.")
        else:
            st.warning("🟡 Status: UNINITIALIZED")
            st.write("Inference engine models currently loading or unavailable.")

    st.markdown("---")
    st.subheader("4. Live Operational Metrics Payload")
    metrics_res = client.get_metrics()
    if metrics_res.get("success"):
        st.json(metrics_res.get("data", {}))
    else:
        st.caption("Operational metrics endpoint returned no data.")

render_health_page()
