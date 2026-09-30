"""Cached dataset loading utilities for Streamlit dashboard demo transactions."""
import os
import pandas as pd
import streamlit as st
from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("dashboard_data_loader")

@st.cache_data(max_entries=1, ttl=3600)
def load_cached_dataset_sample(num_legit: int = 50, num_fraud: int = 25) -> pd.DataFrame:
    """Load a lightweight representative sample of creditcard.csv for demo selection.
    
    Caches the sample in Streamlit memory so creditcard.csv is not re-read on every rerun.
    """
    settings = get_settings()
    dataset_path = settings.raw_dataset_path

    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset file '{dataset_path}' not found. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        # Read a subset of rows to extract representative legit & fraud samples quickly
        df_chunk = pd.read_csv(dataset_path, nrows=10000)
        
        frauds = df_chunk[df_chunk["Class"] == 1].head(num_fraud)
        legit = df_chunk[df_chunk["Class"] == 0].head(num_legit)
        
        combined = pd.concat([legit, frauds]).reset_index(drop=True)
        return combined
    except Exception as e:
        logger.error(f"Error loading dataset sample: {e}")
        return pd.DataFrame()
