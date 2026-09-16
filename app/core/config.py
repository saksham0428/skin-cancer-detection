"""
app/core/config.py — Application configuration via environment variables.

All environment-specific settings are read from the environment (or a .env file).
Never hardcode production values. Defaults are safe for local development.

Usage:
    from app.core.config import settings
    print(settings.model_path)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings — read from environment variables or .env file.

    Precedence (highest to lowest):
        1. Actual environment variables
        2. .env file in the project root
        3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    app_name: str = "Skin Lesion Classification API"
    app_version: str = "1.0.0"
    app_env: str = "development"   # "development" | "production"
    debug: bool = False

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    # Model type: "resnet18" or "simple_cnn"
    model_type: str = "resnet18"
    # Path to the .pth checkpoint file. Relative to project root.
    model_checkpoint: str = "models/skin_lesion_resnet18.pth"

    @property
    def model_path(self) -> Path:
        """Resolve the checkpoint path relative to the project root."""
        # This file is at app/core/config.py → project root is 2 levels up
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / self.model_checkpoint

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    # Comma-separated list of allowed origins for CORS.
    # Example: "http://localhost:3000,https://myapp.com"
    # Use "*" ONLY during development and document it clearly.
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # -----------------------------------------------------------------------
    # File upload limits
    # -----------------------------------------------------------------------
    max_upload_size_mb: float = 10.0  # megabytes

    @property
    def max_upload_size_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    # -----------------------------------------------------------------------
    # API
    # -----------------------------------------------------------------------
    api_prefix: str = "/api/v1"


# ---------------------------------------------------------------------------
# Singleton — import this from anywhere
# ---------------------------------------------------------------------------
settings = Settings()
