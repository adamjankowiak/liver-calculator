from dataclasses import dataclass
from pathlib import Path

from liver_calculator.paths import MODELS_DIR


@dataclass(frozen=True)
class ModelArtifactConfig:
    name: str
    model_path: Path
    metadata_path: Path


DEFAULT_MODEL = ModelArtifactConfig(
    name="meta_logistic",
    model_path=MODELS_DIR / "lr_calibrated_triage_model.joblib",
    metadata_path=MODELS_DIR / "metadata" / "lr_calibrated_triage_meta.json",
)
