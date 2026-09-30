"""Artifact manager for dataset profiles, split metadata, and preprocessor states."""
import os
import json
from typing import Dict, Any, Optional
from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("data_artifacts")


class DatasetArtifactManager:
    """Manager for persisting and retrieving dataset metadata and preprocessing state."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or get_settings().data_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_split_metadata(self, metadata: Dict[str, Any], filename: str = "split_metadata.json") -> str:
        """Save train/val/test split metadata JSON."""
        target_path = os.path.join(self.output_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Split metadata saved to '{target_path}'.")
        return target_path

    def save_dataset_profile(self, profile: Dict[str, Any], filename: str = "dataset_profile.json") -> str:
        """Save statistical EDA profile JSON."""
        target_path = os.path.join(self.output_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Dataset profile saved to '{target_path}'.")
        return target_path

    def load_metadata(self, filename: str) -> Dict[str, Any]:
        """Load JSON metadata file."""
        target_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Metadata file not found at '{target_path}'.")
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
