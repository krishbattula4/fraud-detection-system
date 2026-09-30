"""Fraud Alert Queue & Analyst Investigation Station Page."""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import DashboardAPIClient
from src.dashboard.styles import apply_custom_theme, render_page_header, render_risk_badge, get_readable_rule_trigger, trigger_auto_scroll

st.set_page_config(page_title="Investigation Station — Fraud Risk Intelligence", layout="wide", initial_sidebar_state="collapsed")

client = DashboardAPIClient()

def render_alerts():
    theme_mode = st.session_state.get("theme_mode", "dark")
    render_page_header(
        title="🚨 Fraud Alert Queue & Investigation Station",
        subtitle="Manage flagged suspicious transaction alerts, inspect evidence signals, and record analyst review actions.",
        category="RISK ALERTS & WORKFLOW MANAGEMENT"
    )

    # 1. Queue Filters Bar
    st.subheader("1. Queue Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    status_filter = col_f1.selectbox(
        "Investigation Status Filter", ["All", "OPEN", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"]
    )
    level_filter = col_f2.selectbox("Risk Tier Filter", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    min_score = col_f3.slider("Minimum Risk Score (0–100)", 0.0, 100.0, 0.0)

    status_val = None if status_filter == "All" else status_filter
    level_val = None if level_filter == "All" else level_filter

    with st.spinner("Loading investigation queue..."):
        alerts_res = client.get_alerts(
            status=status_val,
            min_risk_score=min_score if min_score > 0 else None,
            risk_level=level_val,
            limit=100,
        )

    if not alerts_res.get("success"):
        st.error(f"⚠️ **Failed to fetch alert queue**: {alerts_res.get('error')}")
        st.info("Ensure the FastAPI backend service is running on `http://127.0.0.1:8000`.")
        return

    alerts = alerts_res.get("data", [])

    if not alerts:
        st.info("ℹ️ **No risk alerts found matching the selected filter criteria.**")
        st.caption("Submit high-risk transactions via the Live Evaluator tab to populate the alert queue.")
        return

    st.subheader(f"2. Flagged Risk Queue ({len(alerts)} items)")
    df_alerts = pd.DataFrame(alerts)
    st.dataframe(df_alerts, use_container_width=True)

    st.markdown("---")

    # HTML Anchor for Analyst Investigation Console Auto-Scroll
    st.markdown('<div id="alert-details-anchor"></div>', unsafe_allow_html=True)
    if st.session_state.get("_scroll_to_alert_details"):
        trigger_auto_scroll("alert-details-anchor")
        st.session_state["_scroll_to_alert_details"] = False

    # 3. Analyst Investigation Console
    st.subheader("3. Analyst Investigation Console")

    alert_ids = [alt["alert_id"] for alt in alerts]
    
    # Pre-select target alert from session state if available
    target_alert_id = st.session_state.get("target_alert_id")
    default_index = 0
    if target_alert_id and target_alert_id in alert_ids:
        default_index = alert_ids.index(target_alert_id)
        st.info(f"🎯 **Pre-loaded Alert Context**: Investigating Alert `#{target_alert_id}` from Command Center.")

    selected_alert_id = st.selectbox("Select Alert Record to Investigate:", alert_ids, index=default_index)

    if selected_alert_id:
        target = next((a for a in alerts if a["alert_id"] == selected_alert_id), None)
        if target:
            risk_score = float(target.get("risk_score", 0.0))
            risk_level = target.get("risk_level", "HIGH")
            st.markdown(
                f"""
                <div class="glass-card glass-card-accent" style="border-left-color: #EF4444;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">ALERT RECORD #{target['alert_id']}</div>
                            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.2rem;">Transaction ID: {target.get('transaction_id', 'N/A')}</div>
                        </div>
                        <div>
                            {render_risk_badge(risk_score, risk_level)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.write(f"**Alert ID:** `{target['alert_id']}`")
            col_i1.write(f"**Prediction ID:** `{target['prediction_id']}`")
            col_i2.write(f"**Risk Score:** `{target['risk_score']}`")
            col_i2.write(f"**Risk Tier:** `{target['risk_level']}`")
            col_i3.write(f"**Current Status:** `{target['status']}`")

            # Human-Readable Risk Factors & Rule Triggers
            rule_triggers = target.get("rule_triggers", [])
            st.markdown("#### Primary Evidence & Human-Readable Risk Factors")
            if rule_triggers:
                for r in rule_triggers:
                    readable_text = get_readable_rule_trigger(r)
                    st.markdown(f"• 🚩 **{readable_text}** (`{r}`)")
            else:
                st.markdown(f"• ⚡ High hybrid risk score indicator ({risk_score:.1f}/100)")

            st.markdown("---")
            st.markdown("#### Record Analyst Review Decision")
            with st.form("review_form"):
                new_status = st.selectbox(
                    "Updated Review Decision",
                    ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "UNDER_REVIEW", "CLOSED"],
                    help="Select action outcome following cardholder verification"
                )
                analyst_id = st.text_input("Analyst ID / Badge", value="ANALYST-SEC-01")
                notes = st.text_area("Investigation Notes", placeholder="Enter verification notes, phone call confirmation details, block status, etc.")
                submit_review = st.form_submit_button("Submit Analyst Investigation Review")

                if submit_review:
                    review_res = client.review_alert(
                        alert_id=selected_alert_id,
                        new_status=new_status,
                        notes=notes,
                        analyst_id=analyst_id,
                    )
                    if review_res.get("success"):
                        st.success(f"✅ Successfully updated Alert `{selected_alert_id}` status to `{new_status}`!")
                        st.session_state.pop("target_alert_id", None)
                        st.rerun()
                    else:
                        st.error(f"⚠️ **Failed to submit review**: {review_res.get('error')}")

render_alerts()
