"""Command Center Overview page rendering operational KPIs, risk visualizer, threat ticker, and alert summary."""
import streamlit as st
import plotly.express as px
import pandas as pd
from src.dashboard.api_client import DashboardAPIClient, fetch_cached_metrics, fetch_cached_alerts
from src.dashboard.styles import apply_custom_theme, render_page_header, render_pipeline_banner, render_risk_badge, get_readable_rule_trigger
from src.dashboard.pipeline_3d import render_risk_network_pipeline

st.set_page_config(page_title="Command Center — Fraud Risk Intelligence", layout="wide", initial_sidebar_state="collapsed")

client = DashboardAPIClient()

def render_overview():
    theme_mode = st.session_state.get("theme_mode", "dark")
    render_page_header(
        title="📊 AI FINANCIAL FRAUD & RISK INTELLIGENCE CENTER",
        subtitle="Real-time transaction monitoring, hybrid risk scoring, and multi-model decision signals.",
        category="SYSTEM OVERVIEW & OPERATIONAL COMMAND"
    )

    # Visual decision pipeline banner
    render_pipeline_banner()

    # System Metrics & Health Check (Short TTL cached fetch to prevent stalls)
    metrics_res = fetch_cached_metrics()

    if not metrics_res.get("success"):
        st.error(f"⚠️ **Operational Metrics Unavailable**: {metrics_res.get('error')}")
        st.info("Ensure the FastAPI backend service is running on `http://127.0.0.1:8000`.")
        return

    data = metrics_res.get("data", {})
    total_evals = data.get("total_evaluations", 0)
    total_alerts = data.get("total_alerts_triggered", 0)
    tier_counts = data.get("evaluations_by_tier", {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0})
    suspicious = tier_counts.get("MEDIUM", 0) + tier_counts.get("HIGH", 0) + tier_counts.get("CRITICAL", 0)

    # KPI Metric Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Processed", f"{total_evals:,}")
    k2.metric("Fraud Alerts", f"{total_alerts:,}")
    k3.metric("Suspicious Txns", f"{suspicious:,}")
    k4.metric("Low Risk (0–39)", f"{tier_counts.get('LOW', 0):,}")
    k5.metric("Critical (90–100)", f"{tier_counts.get('CRITICAL', 0):,}")

    st.markdown("---")

    # Threat Ticker & Quick Investigation Action
    st.subheader("🚨 Priority Threat Ticker & Active Alerts")
    alerts_res = fetch_cached_alerts(limit=5)
    if alerts_res.get("success") and alerts_res.get("data"):
        alerts = alerts_res["data"]
        if alerts:
            for alt in alerts:
                alt_id = alt.get("alert_id")
                tx_id = alt.get("transaction_id", "N/A")
                score = float(alt.get("risk_score", 0.0))
                level = alt.get("risk_level", "HIGH")
                status = alt.get("status", "OPEN")

                c_info, c_action = st.columns([4, 1])
                with c_info:
                    badge_html = render_risk_badge(score, level)
                    st.markdown(
                        f"""
                        <div style="padding: 0.6rem 1rem; background: rgba(17, 24, 39, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; margin-bottom: 0.4rem; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <strong>Alert #{alt_id}</strong> | Tx: <code>{tx_id}</code> | Status: <code>{status}</code>
                            </div>
                            <div>{badge_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c_action:
                    if st.button("Investigate ➔", key=f"inv_btn_{alt_id}", use_container_width=True):
                        st.session_state["target_alert_id"] = alt_id
                        st.session_state["_scroll_to_alert_details"] = True
                        st.switch_page("pages/3_Alerts.py")
        else:
            st.info("ℹ️ No active threat alerts in queue.")
    else:
        st.caption("Unable to fetch threat alerts payload.")

    st.markdown("---")

    # Risk Intelligence Visualizer Network
    st.subheader("🌐 Real-Time Risk Intelligence Network Visualizer")
    render_risk_network_pipeline(metrics=data, current_risk_score=18.5 if total_evals == 0 else None, theme_mode=theme_mode)

    st.markdown("---")

    # Risk Distribution Analytics
    st.subheader("📈 Risk Band & Tier Analytics")
    col_chart1, col_chart2 = st.columns(2)

    df_tiers = pd.DataFrame(
        [
            {"Risk Tier": tier, "Evaluations": count}
            for tier, count in tier_counts.items()
        ]
    )

    plotly_tpl = "plotly_white" if theme_mode == "light" else "plotly_dark"
    bg_chart = "rgba(255, 255, 255, 0.8)" if theme_mode == "light" else "rgba(17, 24, 39, 0.6)"

    with col_chart1:
        fig_bar = px.bar(
            df_tiers,
            x="Risk Tier",
            y="Evaluations",
            color="Risk Tier",
            color_discrete_map={
                "LOW": "#10B981",
                "MEDIUM": "#F59E0B",
                "HIGH": "#F97316",
                "CRITICAL": "#EF4444",
            },
            title="Evaluations by Risk Tier",
            template=plotly_tpl,
            text_auto=True
        )
        fig_bar.update_layout(
            paper_bgcolor=bg_chart,
            plot_bgcolor=bg_chart,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        fig_pie = px.pie(
            df_tiers,
            names="Risk Tier",
            values="Evaluations",
            color="Risk Tier",
            color_discrete_map={
                "LOW": "#10B981",
                "MEDIUM": "#F59E0B",
                "HIGH": "#F97316",
                "CRITICAL": "#EF4444",
            },
            title="Risk Tier Proportion",
            hole=0.4,
            template=plotly_tpl
        )
        fig_pie.update_layout(
            paper_bgcolor=bg_chart,
            plot_bgcolor=bg_chart,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

render_overview()
