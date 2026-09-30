"""Transaction Analysis & Deep Investigation Workstation Page."""
import streamlit as st
import plotly.express as px
import pandas as pd
from src.dashboard.api_client import DashboardAPIClient
from src.dashboard.styles import apply_custom_theme, render_page_header, render_risk_badge, render_risk_gauge, trigger_auto_scroll

st.set_page_config(page_title="Activity Stream — Risk Intelligence", layout="wide", initial_sidebar_state="collapsed")

theme_mode = st.session_state.get("theme_mode", "dark")

client = DashboardAPIClient()

def explain_risk_reasoning(risk_score: float, level: str, xgb_p: float, if_s: float, lof_s: float, exps: list):
    """Generate truthful risk explanation based strictly on model component outputs."""
    st.subheader("💡 Fraud Analyst Risk Diagnosis")
    
    reasons = []
    if xgb_p > 0.5:
        reasons.append(f"Supervised Calibrated XGBoost indicates high fraud probability (`{xgb_p:.4f}`).")
    elif xgb_p > 0.2:
        reasons.append(f"Supervised Calibrated XGBoost indicates elevated fraud probability (`{xgb_p:.4f}`).")
        
    if if_s > 0.6:
        reasons.append(f"Unsupervised Isolation Forest flagged global structural anomaly (`{if_s:.4f}`).")
        
    if lof_s > 0.6:
        reasons.append(f"Unsupervised Local Outlier Factor flagged local density outlier (`{lof_s:.4f}`).")

    if not reasons:
        reasons.append("Multi-model signals remain within baseline normal parameters.")

    explanation_box = f"**Primary Contributing Risk Signals for {level} Tier ({risk_score:.1f}/100):**\n\n"
    for r in reasons:
        explanation_box += f"• {r}\n"

    st.info(explanation_box)

    # Transparency disclosure
    st.markdown(
        """
        <div class="glass-card" style="border-left: 3px solid #4F46E5;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #4F46E5; text-transform: uppercase;">
                🔒 Methodological & Feature Transparency Disclosure
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.4rem;">
                Features <code>V1</code> through <code>V28</code> in the benchmark credit card dataset are anonymized components derived via Principal Component Analysis (PCA) to protect cardholder confidentiality.
                Feature attributions represent exact mathematical feature impacts toward the risk score, and real-world semantic interpretations are only shown when provided in authentic payload fields.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_transactions():
    theme_mode = st.session_state.get("theme_mode", "dark")
    render_page_header(
        title="🔍 Activity Stream & Transaction Workstation",
        subtitle="Inspect risk scores, component signals, SHAP feature attributions, and empirical model rationales.",
        category="TRANSACTION INVESTIGATION WORKSTATION"
    )

    target_tx_id = st.session_state.get("target_transaction_id", "")

    # HTML Anchor for Auto-Scroll to Inspection Details
    st.markdown('<div id="tx-details-anchor"></div>', unsafe_allow_html=True)

    # 1. Search Section
    st.subheader("1. Transaction Record Lookup")
    col_input, col_btn = st.columns([3, 1])
    search_id = col_input.text_input("Enter Transaction ID or Primary Key (e.g. TX-ADV-IEEE-9001 or TX-DEMO-ROW-1):", value=target_tx_id, placeholder="TX-1001")
    search_clicked = col_btn.button("Inspect Transaction", use_container_width=True)
    if search_clicked:
        st.session_state["_scroll_to_tx_details"] = True

    if (search_id and search_clicked) or (search_id and target_tx_id):
        if st.session_state.get("_scroll_to_tx_details"):
            trigger_auto_scroll("tx-details-anchor")
            st.session_state["_scroll_to_tx_details"] = False

        with st.spinner("Loading transaction record..."):
            res = client.get_transaction(search_id.strip())
        if not res.get("success"):
            st.error(f"⚠️ **Transaction Search Failed**: {res.get('error')}")
        else:
            tx = res.get("data", {})
            st.success(f"✅ Record Located: `{tx.get('transaction_id') or tx.get('id')}`")

            pred = tx.get("prediction")
            if not pred:
                st.warning("Transaction record exists in database but has no prediction payload attached.")
            else:
                score = float(pred.get("risk_score", 0.0))
                level = pred.get("risk_level", "UNKNOWN")
                decision = pred.get("decision", "UNKNOWN")
                version = pred.get("model_version", "1.0.0")

                # Main Risk Banner
                st.markdown("---")
                col_g, col_m = st.columns([1, 2])
                with col_g:
                    render_risk_gauge(score, level)
                with col_m:
                    c_dec, c_ver = st.columns(2)
                    c_dec.metric("Automated Decision", decision)
                    c_ver.metric("Model Version", version)
                    st.progress(min(max(score / 100.0, 0.0), 1.0))
                    st.caption("Risk Band Scale: 0–39 LOW | 40–69 MEDIUM | 70–89 HIGH | 90–100 CRITICAL")

                # Component Risk Signals Breakdown
                st.markdown("---")
                st.subheader("2. Multi-Model Signal Breakdown")
                
                comps = pred.get("model_components", {})
                xgb_p = comps.get("xgboost_calibrated_prob", pred.get("xgboost_prob", 0.0))
                if_s = comps.get("isolation_forest_anomaly_score", pred.get("iforest_anomaly", 0.0))
                lof_s = comps.get("lof_anomaly_score", pred.get("lof_anomaly", 0.0))

                df_components = pd.DataFrame(
                    [
                        {"Component": "XGBoost Calibrated Fraud Prob (w=0.8)", "Signal Score": xgb_p},
                        {"Component": "Isolation Forest Anomaly Score (w=0.1)", "Signal Score": if_s},
                        {"Component": "Local Outlier Factor Anomaly Score (w=0.1)", "Signal Score": lof_s},
                    ]
                )

                plotly_tpl = "plotly_white" if theme_mode == "light" else "plotly_dark"
                bg_chart = "rgba(255, 255, 255, 0.8)" if theme_mode == "light" else "rgba(17, 24, 39, 0.6)"

                fig_comp = px.bar(
                    df_components,
                    x="Signal Score",
                    y="Component",
                    orientation="h",
                    range_x=[0.0, 1.0],
                    color="Signal Score",
                    color_continuous_scale="Reds",
                    title="Component Signals Prior to Hybrid Aggregation",
                    template=plotly_tpl
                )
                fig_comp.update_layout(
                    paper_bgcolor=bg_chart,
                    plot_bgcolor=bg_chart,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # SHAP Feature Explanations
                st.markdown("---")
                st.subheader("3. SHAP Feature Importance Attributions")
                explanations = pred.get("explanations", [])
                if explanations:
                    df_shap = pd.DataFrame(explanations)
                    st.dataframe(df_shap, use_container_width=True)
                else:
                    st.caption("SHAP feature attributions not serialized for this specific transaction record.")

                # Risk Rationale Explanation
                st.markdown("---")
                explain_risk_reasoning(score, level, xgb_p, if_s, lof_s, explanations)

    st.markdown("---")
    st.subheader("4. Operational Transaction Activity Stream")
    history_res = client.get_transactions(limit=50)
    if history_res.get("success") and history_res.get("data"):
        tx_list = history_res["data"]
        if tx_list:
            for idx, tx_item in enumerate(tx_list):
                tx_id = tx_item.get("transaction_id") or tx_item.get("id") or f"TX-{idx+1}"
                amount = float(tx_item.get("amount", 0.0))
                time_val = tx_item.get("time") or tx_item.get("timestamp") or tx_item.get("created_at") or "N/A"
                if isinstance(time_val, float):
                    time_str = f"{time_val:.1f}s"
                else:
                    time_str = str(time_val)

                pred_data = tx_item.get("prediction") or {}
                if isinstance(pred_data, dict) and pred_data:
                    risk_score = float(pred_data.get("risk_score", 0.0))
                    risk_level = pred_data.get("risk_level", "LOW")
                    decision = pred_data.get("decision", "APPROVE")
                else:
                    risk_score = float(tx_item.get("risk_score", 0.0))
                    risk_level = tx_item.get("risk_level", "LOW")
                    decision = tx_item.get("decision", "APPROVE")

                badge_html = render_risk_badge(risk_score, risk_level)

                col_card, col_action = st.columns([4, 1])
                with col_card:
                    st.markdown(
                        f"""
                        <div style="padding: 0.75rem 1.25rem; background: rgba(17, 24, 39, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; margin-bottom: 0.4rem; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <strong style="font-size: 1.05rem;">{tx_id}</strong>
                                <span style="margin-left: 1rem; color: #94A3B8; font-size: 0.9rem;">Amount: <strong>${amount:,.2f}</strong></span>
                                <span style="margin-left: 1rem; color: #94A3B8; font-size: 0.9rem;">Time: <code>{time_str}</code></span>
                                <span style="margin-left: 1rem; color: #94A3B8; font-size: 0.9rem;">Action: <code>{decision}</code></span>
                            </div>
                            <div>{badge_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_action:
                    if st.button("Inspect Details ➔", key=f"tx_stream_btn_{idx}_{tx_id}", use_container_width=True):
                        st.session_state["target_transaction_id"] = tx_id
                        st.session_state["_scroll_to_tx_details"] = True
                        st.rerun()
        else:
            st.info("ℹ️ No transactions evaluated yet. Use the Live Evaluator tab to execute predictions.")
    else:
        st.info("ℹ️ No transaction history available.")

render_transactions()
