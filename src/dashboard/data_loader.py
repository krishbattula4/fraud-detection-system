"""Benchmark dataset loading utilities for the Streamlit dashboard."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("dashboard_data_loader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark_samples"
    / "baseline_benchmark_samples.json"
)


@st.cache_data(max_entries=1, ttl=3600)
def load_cached_dataset_sample(
    num_legit: int = 50,
    num_fraud: int = 25,
) -> pd.DataFrame:
    """Load the packaged benchmark sample, falling back to the local raw dataset."""

    required_columns = ["Time", "Amount", "Class"] + [
        f"V{i}" for i in range(1, 29)
    ]

    # Deployment-safe packaged benchmark sample.
    if PACKAGED_SAMPLE_PATH.exists():
        try:
            with PACKAGED_SAMPLE_PATH.open("r", encoding="utf-8") as file:
                records = json.load(file)

            df = pd.DataFrame(records)

            missing = [
                column for column in required_columns
                if column not in df.columns
            ]

            if missing:
                logger.error(
                    "Packaged benchmark sample is missing columns: %s",
                    missing,
                )
                return pd.DataFrame()

            frauds = df[df["Class"] == 1].head(num_fraud)
            legit = df[df["Class"] == 0].head(num_legit)

            combined = pd.concat(
                [legit, frauds],
                ignore_index=True,
            )

            logger.info(
                "Loaded packaged benchmark sample: %d records "
                "(%d legitimate, %d fraud).",
                len(combined),
                len(legit),
                len(frauds),
            )

            return combined[required_columns]

        except Exception as exc:
            logger.error(
                "Error loading packaged benchmark sample: %s",
                exc,
            )

    # Local development fallback using the full raw dataset.
    settings = get_settings()
    dataset_path = settings.raw_dataset_path

    if not os.path.exists(dataset_path):
        logger.warning(
            "Neither packaged benchmark sample nor raw dataset was found."
        )
        return pd.DataFrame()

    try:
        df_chunk = pd.read_csv(dataset_path, nrows=10000)

        missing = [
            column for column in required_columns
            if column not in df_chunk.columns
        ]

        if missing:
            logger.error(
                "Raw dataset is missing columns: %s",
                missing,
            )
            return pd.DataFrame()

        frauds = df_chunk[df_chunk["Class"] == 1].head(num_fraud)
        legit = df_chunk[df_chunk["Class"] == 0].head(num_legit)

        combined = pd.concat(
            [legit, frauds],
            ignore_index=True,
        )

        return combined[required_columns]

    except Exception as exc:
        logger.error("Error loading raw dataset sample: %s", exc)
        return pd.DataFrame()
