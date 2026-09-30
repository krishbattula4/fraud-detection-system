"""Model registry for artifact serialization, versioning, and disk persistence."""
import os
import json
from typing import Any, Optional, Dict, Tuple
import joblib

from src.ml.base import ModelArtifactManager, ModelMetadata
from src.core.config import get_settings
from src.core.exceptions import ModelNotFoundError, ValidationError
from src.core.logging import get_logger

logger = get_logger("model_registry")


class ModelRegistry(ModelArtifactManager):
    """Concrete model artifact manager for serializing and deserializing models with version metadata."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or get_settings().model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_artifact(
        self,
        model_object: Any,
        artifact_name: str,
        version: str = "1.0.0",
        metadata: Optional[ModelMetadata] = None
    ) -> str:
        """Serialize model artifact and accompanying JSON metadata to storage directory."""
        if model_object is None:
            raise ValidationError("Cannot save None model object.")

        filename = f"{artifact_name}_v{version}.joblib"
        target_path = os.path.join(self.model_dir, filename)
        
        meta_filename = f"{artifact_name}_v{version}_metadata.json"
        meta_path = os.path.join(self.model_dir, meta_filename)

        # Build default metadata if not supplied
        if metadata is None:
            metadata = ModelMetadata(
                model_name=artifact_name,
                version=version,
                algorithm=type(model_object).__name__,
            )

        # Save model joblib artifact
        joblib.dump(model_object, target_path)
        logger.info(f"Saved model artifact '{artifact_name}' (v{version}) to '{target_path}'.")

        # Save JSON metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        logger.info(f"Saved artifact metadata to '{meta_path}'.")

        return target_path

    def load_artifact(self, artifact_name: str, version: Optional[str] = None) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Load serialized model artifact and metadata from storage directory."""
        version_tag = f"_v{version}" if version else ""
        
        # Locate latest matching file if version not specified
        if not version:
            matching_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith(f"{artifact_name}_v") and f.endswith(".joblib")
            ]
            if not matching_files:
                raise ModelNotFoundError(f"No artifact found for '{artifact_name}' in directory '{self.model_dir}'.")
            matching_files.sort(reverse=True)
            filename = matching_files[0]
        else:
            filename = f"{artifact_name}_v{version}.joblib"

        target_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(target_path):
            raise ModelNotFoundError(f"Model artifact file not found at '{target_path}'.")

        model_object = joblib.load(target_path)
        
        # Load metadata JSON if exists
        meta_filename = filename.replace(".joblib", "_metadata.json")
        meta_path = os.path.join(self.model_dir, meta_filename)
        metadata_dict = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

        logger.info(f"Successfully loaded artifact '{filename}' from '{target_path}'.")
        return model_object, metadata_dict
