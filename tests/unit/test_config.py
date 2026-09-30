"""Unit tests for configuration loading."""
from src.core.config import Settings, get_settings


def test_default_settings_values():
    """Verify default values of Settings."""
    settings = Settings()
    assert settings.environment == "development"
    assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
    assert settings.api_port > 0
    assert "sqlite" in settings.database_url


def test_get_settings_singleton():
    """Verify get_settings returns a cached instance."""
    settings_1 = get_settings()
    settings_2 = get_settings()
    assert settings_1 is settings_2
