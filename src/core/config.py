"""Centralized application configuration using Pydantic settings."""
import os
from functools import lru_cache
from typing import Optional
from pydantic import Field

def _default_dataset_path() -> str:
    """Locate creditcard.csv in repository root or data directory."""
    if os.path.exists("creditcard.csv"):
        return "creditcard.csv"
    if os.path.exists("./data/raw/creditcard.csv"):
        return "./data/raw/creditcard.csv"
    return "creditcard.csv"

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        """Application settings loaded from environment variables or .env file."""
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
        
        environment: str = Field(default="development", description="Execution environment")
        log_level: str = Field(default="INFO", description="Logging output level")
        database_url: str = Field(default="sqlite:///./fraud_system.db", description="Database connection URL")
        api_host: str = Field(default="0.0.0.0", description="API host IP")
        api_port: int = Field(default=8000, description="API listen port")
        secret_key: str = Field(default="dev-secret-key-32bytes-minimum-length-change-in-prod", description="API Secret key")
        model_dir: str = Field(default="./models/artifacts", description="Model artifacts storage directory")
        data_dir: str = Field(default="./data", description="Data storage directory")
        raw_dataset_path: str = Field(default_factory=_default_dataset_path, description="Path to raw dataset CSV file")

except ImportError:
    from pydantic import BaseModel
    
    class Settings(BaseModel):
        environment: str = "development"
        log_level: str = "INFO"
        database_url: str = "sqlite:///./fraud_system.db"
        api_host: str = "0.0.0.0"
        api_port: int = 8000
        secret_key: str = "dev-secret-key-32bytes-minimum-length-change-in-prod"
        model_dir: str = "./models/artifacts"
        data_dir: str = "./data"
        raw_dataset_path: str = _default_dataset_path()


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
