import time
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, Optional

RULE_TRIGGER_MAP = {
    "RULE_SUPERVISED_HIGH_PROBABILITY": "High supervised fraud probability detected by XGBoost",
    "RULE_SUPERVISED_ELEVATED_PROBABILITY": "Elevated supervised fraud score detected",
    "RULE_UNSUPERVISED_ANOMALY": "Structural anomaly flagged by unsupervised models",
    "RULE_UNSUPERVISED_ISOLATION_FOREST": "Global structural outlier (Isolation Forest)",
    "RULE_UNSUPERVISED_LOF": "Local density outlier (Local Outlier Factor)",
    "RULE_HIGH_VELOCITY_1H": "High 1-hour transaction velocity",
    "RULE_HIGH_VELOCITY_24H": "High 24-hour transaction velocity",
    "RULE_NEW_DEVICE_DETECTED": "Unrecognized device channel",
    "RULE_HIGH_AMOUNT_OUTLIER": "High-value transaction amount outlier",
    "RULE_ACCOUNT_AGE_DELTA": "Recent account or card activation delta",
    "RULE_GEO_MISMATCH": "Billing region mismatch detected",
}

def init_session_state():
    """Initialize default session state keys safely across all dashboard pages."""
    try:
        if "reset" in st.query_params:
            st.query_params.clear()
            st.session_state["intro_shown"] = False
            st.session_state.pop("intro_start_time", None)
            st.session_state.pop("_applied_theme_mode", None)
    except Exception:
        pass

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"
    if "intro_shown" not in st.session_state:
        st.session_state["intro_shown"] = False

def render_intro_screen(theme_mode: str = "dark"):
    """Render a one-time dimensional FRAUDLENS startup logo reveal (~4.0s branded timeline)."""
    init_session_state()
    if st.session_state.get("intro_shown", False):
        return

    now = time.time()
    if "intro_start_time" not in st.session_state:
        st.session_state["intro_start_time"] = now

    elapsed = now - st.session_state["intro_start_time"]
    if elapsed > 4.2:
        st.session_state["intro_shown"] = True
        st.rerun()
        return

    is_light = theme_mode == "light"

    bg_color = "#F8FAFC" if is_light else "#080C14"
    title_color = "#0F172A" if is_light else "#F8FAFC"
    tagline_color = "#475569" if is_light else "#94A3B8"

    if is_light:
        depth_shadow = "-1px 1px 0 #E2E8F0, -2px 2px 0 #CBD5E1, -3px 3px 0 #94A3B8, -4px 4px 0 #64748B, -5px 5px 0 #475569, -6px 6px 0 #334155, -7px 7px 0 #1E293B, -8px 8px 0 #0F172A, -12px 16px 24px rgba(15, 23, 42, 0.25)"
    else:
        depth_shadow = "-1px 1px 0 #475569, -2px 2px 0 #334155, -3px 3px 0 #283445, -4px 4px 0 #222D3D, -5px 5px 0 #1D2736, -6px 6px 0 #18202E, -7px 7px 0 #131A26, -8px 8px 0 #0F141F, -9px 9px 0 #0B0F17, -10px 10px 0 #080B12, -14px 18px 28px rgba(0, 0, 0, 0.85)"

    intro_html = f"""<div id="fraudlens-intro-overlay">
<div class="fraudlens-dimensional-container">
<div class="fraudlens-dimensional-title">FRAUDLENS</div>
<p class="fraudlens-intro-tagline">Detect. Analyze. Protect.</p>
</div>
</div>
<style>
#fraudlens-intro-overlay {{
position: fixed;
top: 0;
left: 0;
width: 100vw;
height: 100vh;
background: {bg_color};
z-index: 999999;
display: flex;
align-items: center;
justify-content: center;
overflow: hidden;
perspective: 600px !important;
transform-style: preserve-3d !important;
animation: fraudlensOverlayFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) 3.4s forwards;
pointer-events: none !important;
}}
.fraudlens-dimensional-container {{
text-align: center;
display: flex;
flex-direction: column;
align-items: center;
justify-content: center;
padding: 2rem;
perspective: 600px !important;
transform-style: preserve-3d !important;
animation: fraudlensContainerSettle 3.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}
.fraudlens-dimensional-title {{
font-family: 'Inter', -apple-system, sans-serif !important;
font-size: 3.8rem !important;
font-weight: 900 !important;
letter-spacing: 0.16em !important;
color: {title_color} !important;
-webkit-text-fill-color: {title_color} !important;
text-transform: uppercase;
margin: 0 !important;
padding: 0 !important;
line-height: 1.15 !important;
background: none !important;
text-shadow: {depth_shadow} !important;
transform-style: preserve-3d !important;
display: block !important;
opacity: 0;
animation: fraudlensTitleReveal 1.15s cubic-bezier(0.16, 1, 0.3, 1) 0.05s forwards;
}}
.fraudlens-intro-tagline {{
font-family: 'Inter', -apple-system, sans-serif !important;
font-size: 0.95rem !important;
font-weight: 600 !important;
letter-spacing: 0.08em !important;
color: {tagline_color} !important;
margin-top: 0.75rem !important;
margin-bottom: 0 !important;
opacity: 0;
animation: fraudlensTaglineReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) 1.2s forwards;
}}
@keyframes fraudlensTitleReveal {{
0% {{
opacity: 0;
transform: perspective(600px) rotateX(32deg) rotateY(-22deg) translateZ(-140px) scale(0.72);
}}
60% {{
opacity: 1;
transform: perspective(600px) rotateX(8deg) rotateY(-5deg) translateZ(-20px) scale(0.96);
}}
100% {{
opacity: 1;
transform: perspective(600px) rotateX(0deg) rotateY(0deg) translateZ(0px) scale(1.0);
}}
}}
@keyframes fraudlensTaglineReveal {{
0% {{
opacity: 0;
transform: translateY(6px);
}}
100% {{
opacity: 0.9;
transform: translateY(0);
}}
}}
@keyframes fraudlensContainerSettle {{
0% {{
opacity: 1;
transform: translateY(0);
}}
90% {{
opacity: 1;
transform: translateY(0);
}}
100% {{
opacity: 0;
transform: translateY(-8px) scale(0.98);
}}
}}
@keyframes fraudlensOverlayFade {{
0% {{
opacity: 1;
top: 0;
left: 0;
visibility: visible;
pointer-events: auto;
}}
99% {{
opacity: 0;
top: 0;
left: 0;
visibility: visible;
pointer-events: none;
}}
100% {{
opacity: 0;
top: -9999px;
left: -9999px;
width: 0;
height: 0;
visibility: hidden;
pointer-events: none;
}}
}}
@media (max-width: 640px) {{
.fraudlens-dimensional-title {{
font-size: 2.2rem !important;
letter-spacing: 0.08em !important;
}}
.fraudlens-intro-tagline {{
font-size: 0.82rem !important;
}}
}}
</style>"""
    st.markdown(intro_html, unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function() {
            function cleanup() {
                try {
                    var doc = window.parent.document;
                    var nodes = doc.querySelectorAll('#fraudlens-intro-overlay');
                    for (var i = 0; i < nodes.length; i++) {
                        var el = nodes[i];
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        el.style.opacity = '0';
                        el.style.pointerEvents = 'none';
                        el.style.zIndex = '-999999';
                        if (el.parentNode) {
                            el.parentNode.removeChild(el);
                        }
                    }
                } catch(e) {}
            }
            setTimeout(function() {
                cleanup();
                setInterval(cleanup, 300);
            }, 3400);
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def get_readable_rule_trigger(rule_code: str) -> str:
    """Translate raw backend rule trigger strings into human-readable presentation text."""
    if not rule_code:
        return "Standard risk indicator"
    clean_code = str(rule_code).strip()
    if clean_code in RULE_TRIGGER_MAP:
        return RULE_TRIGGER_MAP[clean_code]
    # Fallback formatting for unmapped rules
    readable = clean_code.replace("RULE_", "").replace("_", " ").title()
    return readable

def trigger_auto_scroll(anchor_id: str):
    """Inject a lightweight one-shot client-side scroll script targeting anchor_id."""
    scroll_js = f"""
    <script>
    (function() {{
        try {{
            var el = document.getElementById('{anchor_id}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }} catch(e) {{}}
    }})();
    </script>
    """
    st.markdown(scroll_js, unsafe_allow_html=True)

def apply_custom_theme(theme_mode: str = "dark", page_key: str = "default"):
    """Inject application-wide CSS for Dark or Light theme and full-width layout without left sidebar."""
    init_session_state()

    # Always ensure session_state theme_mode takes precedence if present
    if "theme_mode" in st.session_state and st.session_state["theme_mode"]:
        theme_mode = st.session_state["theme_mode"]

    is_light = theme_mode == "light"

    bg_dark = "#F8FAFC" if is_light else "#080C14"
    bg_card = "#FFFFFF" if is_light else "#111726"
    border_glass = "rgba(0, 0, 0, 0.08)" if is_light else "rgba(255, 255, 255, 0.07)"
    border_glow = "rgba(0, 0, 0, 0.12)" if is_light else "rgba(255, 255, 255, 0.15)"
    text_primary = "#0F172A" if is_light else "#F8FAFC"
    text_secondary = "#475569" if is_light else "#94A3B8"
    text_muted = "#64748B" if is_light else "#64748B"
    header_gradient = "linear-gradient(135deg, #0F172A 0%, #334155 100%)" if is_light else "linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%)"
    pipeline_bg = "rgba(241, 245, 249, 0.9)" if is_light else "rgba(17, 23, 39, 0.8)"

    # Page-context aware theme key prevents unneeded CSS duplication on reruns while ensuring newly mounted workspace receives theme CSS
    applied_key = f"{page_key}_{theme_mode}"
    if st.session_state.get("_applied_theme_key") != applied_key:
        st.session_state["_applied_theme_key"] = applied_key
        st.markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;600&display=swap');

            /* Complete Removal & Suppression of Left Sidebar, Chevron Arrow, Header Chrome */
            header[data-testid="stHeader"],
            [data-testid="stHeader"],
            section[data-testid="stSidebar"],
            div[data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarExpandButton"],
            button[data-testid="stSidebarCollapseButton"],
            button[data-testid="stSidebarExpandButton"],
            button[aria-label="Toggle sidebar"],
            button[aria-label="Expand sidebar"],
            button[aria-label="Open sidebar"],
            button[title="Expand sidebar"],
            [data-testid="stSidebarNav"],
            [data-testid="stSidebarNavItems"],
            [data-testid="stSidebarNavSeparator"],
            [data-testid="stHeaderChevron"],
            [data-testid="stDecoration"],
            [data-testid="stToolbar"],
            [data-testid="stStatusWidget"],
            #MainMenu, footer {{
                display: none !important;
                visibility: hidden !important;
                width: 0 !important;
                height: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                opacity: 0 !important;
                pointer-events: none !important;
                overflow: hidden !important;
            }}

            /* Restrained FRAUDLENS Custom Neutral Loading Indicator */
            [data-testid="stSpinner"], .stSpinner {{
                background: {"rgba(255, 255, 255, 0.85)" if is_light else "rgba(17, 23, 38, 0.85)"} !important;
                border: 1px solid {"rgba(0, 0, 0, 0.08)" if is_light else "rgba(255, 255, 255, 0.08)"} !important;
                border-radius: 8px !important;
                padding: 0.5rem 0.9rem !important;
                margin: 0.5rem 0 !important;
                box-shadow: {"0 2px 8px rgba(0,0,0,0.04)" if is_light else "0 4px 14px rgba(0,0,0,0.25)"} !important;
                display: inline-flex !important;
                align-items: center !important;
                gap: 0.6rem !important;
            }}

            [data-testid="stSpinner"] > div:first-child,
            .stSpinner > div:first-child,
            .stSpinnerIcon {{
                width: 14px !important;
                height: 14px !important;
                border: 2px solid {"rgba(71, 85, 105, 0.3)" if is_light else "rgba(148, 163, 184, 0.25)"} !important;
                border-top-color: {"#475569" if is_light else "#94A3B8"} !important;
                border-radius: 50% !important;
                animation: fraudlensRestrainedSpin 0.75s linear infinite !important;
                background: transparent !important;
            }}

            @keyframes fraudlensRestrainedSpin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            [data-testid="stSpinner"] p,
            .stSpinner p,
            .stSpinnerMessage {{
                color: {"#334155" if is_light else "#CBD5E1"} !important;
                font-family: 'Inter', -apple-system, sans-serif !important;
                font-size: 0.84rem !important;
                font-weight: 600 !important;
                letter-spacing: 0.01em !important;
                margin: 0 !important;
            }}

            /* Full Viewport Container Layout */
            .stApp {{
                background: {bg_dark} !important;
                font-family: 'Inter', -apple-system, sans-serif !important;
                color: {text_primary} !important;
            }}

            .main .block-container {{
                max-width: 1400px !important;
                padding-top: 1.25rem !important;
                padding-bottom: 2rem !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }}

            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Inter', sans-serif !important;
                color: {text_primary} !important;
                font-weight: 700 !important;
                letter-spacing: -0.025em !important;
            }}
            
            h1 {{
                font-size: 1.85rem !important;
                line-height: 1.25 !important;
                background: {header_gradient};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .stCaption, p {{
                color: {text_secondary} !important;
            }}

            .glass-card, .surface-panel {{
                background: {bg_card};
                border: 1px solid {border_glass};
                border-radius: 10px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: {"0 2px 8px rgba(0, 0, 0, 0.04)" if is_light else "0 4px 16px rgba(0, 0, 0, 0.3)"};
                transition: all 0.2s ease-in-out;
            }}
            
            .glass-card:hover, .surface-panel:hover {{
                border-color: {border_glow};
            }}

            .glass-card-accent {{
                border-left: 4px solid #4F46E5;
            }}

            div[data-testid="stMetric"] {{
                background: {bg_card} !important;
                border: 1px solid {border_glass} !important;
                border-radius: 8px !important;
                padding: 0.9rem 1.1rem !important;
                box-shadow: {"0 2px 6px rgba(0, 0, 0, 0.03)" if is_light else "0 4px 12px rgba(0, 0, 0, 0.25)"} !important;
            }}
            
            div[data-testid="stMetric"] label {{
                color: {text_secondary} !important;
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.04em !important;
            }}
            
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
                color: {text_primary} !important;
                font-family: 'Fira Code', monospace !important;
                font-weight: 700 !important;
                font-size: 1.6rem !important;
            }}

            .risk-badge {{
                display: inline-flex;
                align-items: center;
                padding: 0.3rem 0.65rem;
                border-radius: 6px;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .risk-low {{
                background: rgba(16, 185, 129, 0.12);
                color: {"#047857" if is_light else "#34D399"};
                border: 1px solid rgba(16, 185, 129, 0.35);
            }}

            .risk-medium {{
                background: rgba(245, 158, 11, 0.12);
                color: {"#B45309" if is_light else "#FBBF24"};
                border: 1px solid rgba(245, 158, 11, 0.35);
            }}

            .risk-high {{
                background: rgba(249, 115, 22, 0.12);
                color: {"#C2410C" if is_light else "#FB923C"};
                border: 1px solid rgba(249, 115, 22, 0.35);
            }}

            .risk-critical {{
                background: rgba(239, 68, 68, 0.15);
                color: {"#B91C1C" if is_light else "#F87171"};
                border: 1px solid rgba(239, 68, 68, 0.4);
            }}

            .pipeline-flow {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: {pipeline_bg};
                border: 1px solid {border_glass};
                border-radius: 10px;
                padding: 0.85rem 1.25rem;
                margin-bottom: 1.25rem;
            }}

            .pipeline-step {{
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }}

            .pipeline-step-title {{
                font-size: 0.7rem;
                color: {text_muted};
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 600;
            }}

            .pipeline-step-val {{
                font-size: 0.9rem;
                color: {text_primary};
                font-weight: 700;
                margin-top: 0.15rem;
            }}

            .pipeline-arrow {{
                color: #4F46E5;
                font-size: 1.1rem;
                font-weight: bold;
                opacity: 0.7;
            }}

            code, pre {{
                font-family: 'Fira Code', monospace !important;
                background: {"#F1F5F9" if is_light else "rgba(15, 23, 42, 0.9)"} !important;
                color: {text_primary} !important;
                border: 1px solid {border_glass} !important;
                border-radius: 6px !important;
            }}

            /* General Action Button Styling */
            .stButton>button {{
                background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                padding: 0.45rem 1rem;
                box-shadow: 0 2px 10px rgba(79, 70, 229, 0.25);
                transition: all 0.2s ease;
            }}

            .stButton>button:hover {{
                box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
                transform: translateY(-1px);
            }}

            /* Top Navigation Product Tabs Styling - Restrained, Clean & Compact */
            div[data-testid="stHorizontalBlock"]:first-of-type button {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                color: {"#475569" if is_light else "#94A3B8"} !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                padding: 0.35rem 0.5rem !important;
                white-space: nowrap !important;
                border-bottom: 2px solid transparent !important;
                border-radius: 0 !important;
                transform: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:first-of-type button:hover {{
                color: {"#0F172A" if is_light else "#F8FAFC"} !important;
                background: {"rgba(0,0,0,0.03)" if is_light else "rgba(255,255,255,0.05)"} !important;
                box-shadow: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:first-of-type button[kind="primary"],
            div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stBaseButton-primary"] {{
                color: {"#4F46E5" if is_light else "#818CF8"} !important;
                font-weight: 700 !important;
                border-bottom: 2px solid #6366F1 !important;
                background: transparent !important;
                box-shadow: none !important;
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid {border_glass} !important;
                border-radius: 8px !important;
            }}

            /* Custom Risk Dial Score Gauge CSS */
            .gauge-container {{
                text-align: center;
                padding: 1.1rem;
                border-radius: 12px;
                background: {bg_card};
                border: 1px solid {border_glass};
            }}
            .gauge-score {{
                font-family: 'Fira Code', monospace;
                font-size: 2.75rem;
                font-weight: 800;
                line-height: 1;
                margin: 0.4rem 0;
            }}

            /* Product-Style Top Navigation Bar Styling */
            .top-nav-bar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.6rem 1rem;
                background: {bg_card};
                border: 1px solid {border_glass};
                border-radius: 10px;
                margin-bottom: 1.25rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    render_intro_screen(theme_mode)

def render_top_navigation(active_workspace: str = "Overview"):
    """Render top workspace navigation bar with concise product labels and active underline indicator."""
    init_session_state()
    theme_mode = st.session_state.get("theme_mode", "dark")

    # Product label mapping matching Stage 8 directive
    workspaces = [
        ("Overview", "pages/1_Overview.py"),
        ("Investigate", "pages/3_Alerts.py"),
        ("Risk Check", "pages/5_Simulator.py"),
        ("Transactions", "pages/2_Transactions.py"),
        ("Intelligence", "pages/4_Model_Performance.py"),
        ("Status", "pages/6_System_Health.py"),
    ]

    col_brand, col_nav, col_controls = st.columns([2.0, 7.0, 1.5])

    with col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.5rem; height: 100%; padding-top: 0.2rem;">
                <span style="font-size: 1.3rem;">🛡️</span>
                <span style="font-weight: 800; font-size: 0.95rem; letter-spacing: -0.02em; white-space: nowrap;">FRAUD INTELLIGENCE</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_nav:
        sub_cols = st.columns(len(workspaces))
        for idx, (name, page_path) in enumerate(workspaces):
            is_active = (name == active_workspace)
            label = f"• {name}" if is_active else name
            with sub_cols[idx]:
                if st.button(
                    label,
                    key=f"top_nav_btn_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.switch_page(page_path)

    with col_controls:
        c_status, c_theme = st.columns([1.0, 1.0])
        with c_status:
            st.markdown(
                """
                <div style="font-size: 0.72rem; font-weight: 700; color: #10B981; padding-top: 0.45rem; white-space: nowrap;">
                    🟢 Ready
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_theme:
            btn_label = "☀️" if theme_mode == "dark" else "🌙"
            if st.button(btn_label, key="header_theme_toggle", help="Toggle Dark/Light theme"):
                st.session_state["theme_mode"] = "light" if theme_mode == "dark" else "dark"
                st.session_state.pop("_applied_theme_mode", None)
                st.session_state.pop("_applied_theme_key", None)
                st.rerun()

    st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str, category: str = "RISK INTELLIGENCE"):
    """Render standardized page header with full-width navigation header shell."""
    init_session_state()
    theme_mode = st.session_state.get("theme_mode", "dark")
    
    # Map title to active workspace name for top nav highlight
    active_ws = "Overview"
    if "Alert" in title or "Investigation" in title or "Queue" in title:
        active_ws = "Investigate"
    elif "Simulator" in title or "Live" in title or "Console" in title or "Evaluator" in title:
        active_ws = "Risk Check"
    elif "Transaction" in title or "Activity" in title or "Stream" in title:
        active_ws = "Transactions"
    elif "Model" in title or "Intelligence" in title or "Benchmark" in title:
        active_ws = "Intelligence"
    elif "Health" in title or "Diagnostics" in title or "Status" in title:
        active_ws = "Status"

    apply_custom_theme(theme_mode, page_key=active_ws)

    render_top_navigation(active_workspace=active_ws)

    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #4F46E5; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.25rem;">
                ⚡ {category}
            </div>
            <h1 style="margin: 0; padding: 0;">{title}</h1>
            <div style="font-size: 0.92rem; margin-top: 0.25rem; color: #94A3B8;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_pipeline_banner():
    """Render prominent visual pipeline flow banner."""
    st.markdown(
        """
        <div class="pipeline-flow">
            <div class="pipeline-step">
                <span class="pipeline-step-title">1. INGESTION</span>
                <span class="pipeline-step-val">💳 TRANSACTION</span>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <span class="pipeline-step-title">2. MULTI-MODEL AI</span>
                <span class="pipeline-step-val">🧠 XGB + IF + LOF</span>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <span class="pipeline-step-title">3. HYBRID SCORE</span>
                <span class="pipeline-step-val">⚖️ RISK 0–100</span>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <span class="pipeline-step-title">4. AUTOMATED ACTION</span>
                <span class="pipeline-step-val">🛡️ DECISION</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_risk_badge(score: float, level: str) -> str:
    """Return formatted HTML risk badge."""
    lvl = (level or "LOW").upper()
    css_class = "risk-low"
    if lvl == "MEDIUM":
        css_class = "risk-medium"
    elif lvl == "HIGH":
        css_class = "risk-high"
    elif lvl == "CRITICAL":
        css_class = "risk-critical"

    return f'<span class="risk-badge {css_class}">{lvl} ({score:.1f})</span>'

def render_risk_gauge(score: float, level: str):
    """Render styled score dial visual card."""
    lvl = (level or "LOW").upper()
    score_color = "#10B981"
    if lvl == "MEDIUM":
        score_color = "#F59E0B"
    elif lvl == "HIGH":
        score_color = "#F97316"
    elif lvl == "CRITICAL":
        score_color = "#EF4444"

    st.markdown(
        f"""
        <div class="gauge-container">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; color: #94A3B8; text-transform: uppercase;">
                HYBRID RISK SCORE
            </div>
            <div class="gauge-score" style="color: {score_color};">
                {score:.1f}
            </div>
            <div>
                {render_risk_badge(score, level)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
