import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from liver_calculator.paths import MODELS_DIR


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelArtifactConfig:
    name: str
    model_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class AppSettings:
    title: str
    host: str
    port: int
    reload: bool


@lru_cache(maxsize=1)
def get_model_config() -> ModelArtifactConfig:
    return ModelArtifactConfig(
        name=os.getenv("MODEL_NAME", "meta_logistic"),
        model_path=_env_path("MODEL_PATH", MODELS_DIR / "lr_calibrated_triage_model.joblib"),
        metadata_path=_env_path(
            "MODEL_METADATA_PATH",
            MODELS_DIR / "metadata" / "lr_calibrated_triage_meta.json",
        ),
    )


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings(
        title=os.getenv("APP_TITLE", "Liver Meta Calculator"),
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=_env_bool("APP_RELOAD", False),
    )
