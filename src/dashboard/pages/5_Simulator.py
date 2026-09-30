"""Live Transaction Console & Risk Simulator Page supporting Benchmark & Advanced Risk Intelligence."""
import streamlit as st
import pandas as pd
from src.dashboard.api_client import DashboardAPIClient
from src.dashboard.data_loader import load_cached_dataset_sample
from src.dashboard.styles import apply_custom_theme, render_page_header, render_risk_badge, render_risk_gauge, get_readable_rule_trigger

st.set_page_config(page_title="Live Evaluator — Risk Intelligence", layout="wide", initial_sidebar_state="collapsed")

theme_mode = st.session_state.get("theme_mode", "dark")

client = DashboardAPIClient()

def render_baseline_prediction_result(res: dict):
    """Render baseline prediction response with graceful error handling and risk score dial."""
    if not res.get("success"):
        status_code = res.get("status_code")
        error_msg = res.get("error", "")

        if status_code == 503:
            st.warning("⚠️ **Inference Service Unavailable**: Fraud detection models are offline or API backend is unavailable.")
            st.caption("Ensure the backend API service is running on `http://127.0.0.1:8000`.")
        else:
            st.error(f"⚠️ **Prediction Request Failed**: {error_msg}")
        return

    data = res.get("data", {})
    st.success("✅ Transaction Evaluated via Baseline Multi-Model Pipeline!")

    st.markdown("---")
    score = float(data.get("risk_score", 0.0))
    level = data.get("risk_level", "N/A")
    decision = data.get("decision", "N/A")

    col_gauge, col_details = st.columns([1, 2])
    with col_gauge:
        render_risk_gauge(score, level)

    with col_details:
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Tier Classification", level)
        c2.metric("Automated Decision Action", decision)
        c3.metric("Engine Model Version", data.get("model_version", "1.0.0"))

        st.progress(min(max(score / 100.0, 0.0), 1.0))
        st.caption("Decision Scale: 0–39 LOW (APPROVE) | 40–69 MEDIUM (FLAG) | 70–89 HIGH (FLAG) | 90–100 CRITICAL (DECLINE)")

    # Component breakdown
    comps = data.get("model_components", {})
    st.markdown("---")
    st.subheader("Model Risk Component Signals")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("XGBoost Calibrated Fraud Prob", f"{comps.get('xgboost_calibrated_prob', 0.0):.4f}")
    mc2.metric("Isolation Forest Anomaly Score", f"{comps.get('isolation_forest_anomaly_score', 0.0):.4f}")
    mc3.metric("Local Outlier Factor Anomaly Score", f"{comps.get('lof_anomaly_score', 0.0):.4f}")

    # Feature Attributions
    exps = data.get("explanations", [])
    if exps:
        st.markdown("---")
        st.subheader("Top Feature Attributions")
        st.dataframe(pd.DataFrame(exps), use_container_width=True)
        st.info("ℹ️ Features V1–V28 are anonymized components from the standard benchmark dataset.")

def render_advanced_prediction_result(res: dict):
    """Render Advanced IEEE-CIS prediction response with evidence breakdown."""
    if not res.get("success"):
        status_code = res.get("status_code")
        error_msg = res.get("error", "")

        if status_code == 503:
            st.warning("⚠️ **Advanced Model Service Unavailable**: Authentic IEEE-CIS models are offline or API backend is unavailable.")
            st.caption("Ensure the backend API service is running on `http://127.0.0.1:8000`.")
        else:
            st.error(f"⚠️ **Advanced Prediction Request Failed**: {error_msg}")
        return

    data = res.get("data", {})
    st.success("✅ Multi-Entity Transaction Evaluated via Advanced Risk Intelligence Engine!")

    st.markdown("---")
    st.subheader("1. Core Risk Assessment Result")
    score = float(data.get("risk_score", 0.0))
    level = data.get("risk_level", "N/A")
    decision = data.get("decision", "N/A")
    version = data.get("model_version", "v2.0.0-authentic-ieee")

    col_gauge, col_details = st.columns([1, 2])
    with col_gauge:
        render_risk_gauge(score, level)

    with col_details:
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Tier Classification", level)
        c2.metric("Automated Decision Action", decision)
        c3.metric("Advanced Model Pipeline", version)

        st.progress(min(max(score / 100.0, 0.0), 1.0))
        st.caption("Risk Score 0–100 represents a calibrated decision-ranking score.")

    # Component breakdown
    comps = data.get("model_components", {})
    st.markdown("---")
    st.subheader("2. Multi-Model Signal Breakdown")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Calibrated XGBoost Fraud Probability", f"{comps.get('supervised_xgboost_prob', 0.0):.4f}")
    mc2.metric("Isolation Forest Anomaly Contribution", f"{comps.get('isolation_forest_anomaly_score', 0.0):.4f}")
    mc3.metric("Behavioral / Entity Velocity Risk Contribution", f"{comps.get('behavioral_velocity_score', 0.0):.4f}")

    # Evidence-Based Risk Factors & Rule Triggers
    rule_triggers = data.get("rule_triggers", [])
    explanations = data.get("explanations", [])

    st.markdown("---")
    st.subheader("3. Evidence-Based Risk Factor Explanations")
    if rule_triggers:
        st.markdown("**Triggered Risk Factors:**")
        for r in rule_triggers:
            readable_text = get_readable_rule_trigger(r)
            st.markdown(f"• 🚩 **{readable_text}** (`{r}`)")

    if explanations:
        st.dataframe(pd.DataFrame(explanations), use_container_width=True)
    else:
        st.info("🟢 No elevated risk factors detected for this transaction vector.")

def render_simulator_page():
    theme_mode = st.session_state.get("theme_mode", "dark")
    render_page_header(
        title="⚡ Live Transaction Console & Risk Simulator",
        subtitle="Execute real-time credit-card transaction risk evaluations via benchmark dataset sampling or advanced multi-entity scoring.",
        category="LIVE TRANSACTION SIMULATION & MONITORING"
    )

    # Interactive Scenario Preset Buttons
    st.subheader("🎯 Quick Risk Scenario Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_chosen = None

    with col_p1:
        if st.button("🟢 Legitimate E-Commerce Txn", use_container_width=True):
            preset_chosen = "legit"
    with col_p2:
        if st.button("🟡 Elevated Velocity Preset", use_container_width=True):
            preset_chosen = "velocity"
    with col_p3:
        if st.button("🔴 High-Risk Multi-Device Anomaly", use_container_width=True):
            preset_chosen = "critical"

    st.markdown("---")

    tab_demo, tab_advanced, tab_ieee = st.tabs([
        "💳 Benchmark Dataset Mode (Baseline)",
        "⚡ Custom PCA Vector Mode (Baseline)",
        "🛡️ Advanced Multi-Entity Risk Intelligence",
    ])

    # ---------------------------------------------------------
    # Mode A: DEMO / BENCHMARK DATASET TRANSACTION MODE
    # ---------------------------------------------------------
    with tab_demo:
        st.subheader("1. Benchmark Dataset Transaction Selector")
        st.caption("Select a real historical transaction record to preview and evaluate.")

        df_sample = load_cached_dataset_sample()

        if df_sample.empty:
            st.warning("⚠️ Benchmark dataset sample is not accessible. Use Advanced API Mode below.")
        else:
            options = []
            for idx, row in df_sample.iterrows():
                is_fraud = int(row.get("Class", 0)) == 1
                label_tag = "🔴 [HISTORICAL FRAUD]" if is_fraud else "🟢 [HISTORICAL LEGITIMATE]"
                options.append(f"Row #{idx + 1} | {label_tag} | Time: {row['Time']:.0f}s | Amount: ${row['Amount']:.2f}")

            selected_idx = st.selectbox("Choose Transaction Record to Evaluate:", range(len(options)), format_func=lambda i: options[i])
            selected_row = df_sample.iloc[selected_idx]

            st.markdown("---")
            st.markdown("#### Transaction Feature Preview")
            
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Transaction Time", f"{selected_row['Time']:.1f} s")
            p_col2.metric("Transaction Amount", f"${selected_row['Amount']:.2f}")
            
            is_fraud_val = int(selected_row.get("Class", 0))
            ground_truth_label = "Class 1 (Historical Fraud)" if is_fraud_val == 1 else "Class 0 (Historical Legitimate)"
            p_col3.metric("Dataset Historical Target (Context)", ground_truth_label)

            st.caption("🔒 **Note:** The `Class` attribute above is historical ground truth context ONLY. It is **NOT** included in the live prediction payload.")

            with st.expander("Inspect Raw Anonymized PCA Features (V1–V28)", expanded=False):
                v_cols_display = {f"V{i}": selected_row[f"V{i}"] for i in range(1, 29)}
                st.json(v_cols_display)

            if st.button("🚀 Analyze Baseline Transaction Risk", use_container_width=True):
                with st.spinner("Evaluating baseline risk pipeline..."):
                    payload = {
                        "transaction_id": f"TX-DEMO-ROW-{selected_idx + 1}",
                        "time": float(selected_row["Time"]),
                        "amount": float(selected_row["Amount"]),
                        **{f"v{i}": float(selected_row[f"V{i}"]) for i in range(1, 29)}
                    }

                    res = client.predict_transaction(payload)
                    render_baseline_prediction_result(res)

    # ---------------------------------------------------------
    # Mode B: BASELINE CUSTOM VECTOR MODE
    # ---------------------------------------------------------
    with tab_advanced:
        st.subheader("2. Baseline Technical Testing Input Form")
        st.caption("Directly construct a custom baseline feature payload to test model response parameters.")
        
        st.info("ℹ️ Features **V1–V28** are anonymized components. Values typically range between -5.0 and +5.0.")

        with st.form("baseline_tx_form"):
            col_t, col_a, col_id = st.columns(3)
            time_val = col_t.number_input("Time (seconds elapsed)", min_value=0.0, value=100.0, step=1.0)
            amount_val = col_a.number_input("Amount ($)", min_value=0.0, value=150.0, step=10.0)
            tx_id_val = col_id.text_input("Custom Transaction ID", value="TX-BASE-CUSTOM-001")

            st.markdown("##### Anonymized Feature Matrix (V1 – V28)")
            v_dict = {}
            v_cols = st.columns(4)
            for idx in range(1, 29):
                col_idx = (idx - 1) % 4
                v_dict[f"v{idx}"] = v_cols[col_idx].number_input(f"V{idx}", value=0.0, step=0.1, format="%.4f")

            submit_base = st.form_submit_button("Submit Baseline Risk Evaluation")

            if submit_base:
                with st.spinner("Evaluating baseline risk pipeline..."):
                    payload = {
                        "transaction_id": tx_id_val,
                        "time": float(time_val),
                        "amount": float(amount_val),
                        **v_dict,
                    }
                    res = client.predict_transaction(payload)
                    render_baseline_prediction_result(res)

    # ---------------------------------------------------------
    # Mode C: ADVANCED MULTI-ENTITY RISK INTELLIGENCE (IEEE-CIS)
    # ---------------------------------------------------------
    with tab_ieee:
        st.subheader("3. Advanced Multi-Entity Risk Intelligence Console")
        st.caption("Evaluate multi-entity transactions using authentic IEEE-CIS models, behavioral velocity, and device signals.")

        # Preset default overrides
        default_amt = 450.0
        default_c1 = 3.0
        default_c2 = 6.0
        default_device = "mobile"
        default_tx_id = "TX-ADV-IEEE-9001"

        if preset_chosen == "legit":
            default_amt = 35.50
            default_c1 = 1.0
            default_c2 = 1.0
            default_device = "desktop"
            default_tx_id = "TX-PRESET-LEGIT-001"
        elif preset_chosen == "velocity":
            default_amt = 1200.0
            default_c1 = 25.0
            default_c2 = 40.0
            default_device = "mobile"
            default_tx_id = "TX-PRESET-VELOCITY-002"
        elif preset_chosen == "critical":
            default_amt = 8500.0
            default_c1 = 90.0
            default_c2 = 120.0
            default_device = "mobile"
            default_tx_id = "TX-PRESET-CRITICAL-003"

        with st.form("advanced_ieee_form"):
            ac1, ac2, ac3 = st.columns(3)
            adv_tx_id = ac1.text_input("Transaction ID", value=default_tx_id)
            adv_time = ac2.number_input("Timestamp (Elapsed Seconds / TransactionDT)", min_value=0.0, value=86450.0, step=100.0)
            adv_amt = ac3.number_input("Transaction Amount ($)", min_value=0.0, value=default_amt, step=25.0)

            st.markdown("##### Entity & Channel Attributes")
            ec1, ec2, ec3 = st.columns(3)
            card_id = ec1.text_input("Card Account Identifier (card_id)", value="card_1001")
            billing_region = ec2.text_input("Billing Region (addr1)", value="addr_315")
            purchaser_domain = ec3.text_input("Purchaser Email Domain (P_emaildomain)", value="gmail.com")

            ec4, ec5, ec6 = st.columns(3)
            recipient_domain = ec4.text_input("Recipient Email Domain (R_emaildomain)", value="yahoo.com")
            device_type = ec5.selectbox("Device Channel Type", options=["mobile", "desktop", "tablet"], index=0 if default_device=="mobile" else 1)
            device_info = ec6.text_input("Device Details (DeviceInfo)", value="iOS/Safari")

            st.markdown("##### Velocity & Delta Signals (C1, C2, D1)")
            vc1, vc2, vc3 = st.columns(3)
            c1_val = vc1.number_input("Transaction Count C1", min_value=0.0, value=default_c1, step=1.0)
            c2_val = vc2.number_input("Transaction Count C2", min_value=0.0, value=default_c2, step=1.0)
            d1_val = vc3.number_input("Timedelta Days D1", min_value=0.0, value=12.0, step=1.0)

            submit_ieee = st.form_submit_button("🛡️ Execute Advanced Multi-Entity Risk Evaluation")

            if submit_ieee or preset_chosen:
                with st.spinner("Executing authentic IEEE-CIS advanced inference pipeline..."):
                    adv_payload = {
                        "transaction_id": adv_tx_id,
                        "timestamp": float(adv_time),
                        "amount": float(adv_amt),
                        "card_id": card_id,
                        "billing_region": billing_region,
                        "purchaser_email_domain": purchaser_domain,
                        "recipient_email_domain": recipient_domain,
                        "device_type": device_type,
                        "device_info": device_info,
                        "counting_features": {"C1": float(c1_val), "C2": float(c2_val)},
                        "timedelta_features": {"D1": float(d1_val)},
                    }
                    res_adv = client.predict_advanced_transaction(adv_payload)
                    render_advanced_prediction_result(res_adv)

render_simulator_page()
