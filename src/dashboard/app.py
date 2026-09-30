import importlib
import streamlit as st
from src.dashboard.styles import apply_custom_theme, init_session_state

st.set_page_config(
    page_title="AI Financial Fraud Detection & Risk Intelligence Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State Safely
init_session_state()

# Render Command Center Overview directly on home entrypoint
overview_mod = importlib.import_module("src.dashboard.pages.1_Overview")
overview_mod.render_overview()



