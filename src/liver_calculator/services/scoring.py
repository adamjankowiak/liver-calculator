import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from liver_calculator.config import get_model_config
from liver_calculator.schemas import PatientFeatures


@dataclass(frozen=True)
class LoadedModelBundle:
    model: Any
    meta: dict[str, Any]


@lru_cache(maxsize=1)
def load_model_bundle(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LoadedModelBundle:
    model_config = get_model_config()
    resolved_model_path = Path(model_path) if model_path else model_config.model_path
    resolved_metadata_path = Path(metadata_path) if metadata_path else model_config.metadata_path

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {resolved_model_path}")
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {resolved_metadata_path}")

    model = joblib.load(resolved_model_path)
    meta = load_model_metadata(resolved_metadata_path)
    return LoadedModelBundle(model=model, meta=meta)


@lru_cache(maxsize=4)
def load_model_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    model_config = get_model_config()
    resolved_metadata_path = Path(metadata_path) if metadata_path else model_config.metadata_path

    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {resolved_metadata_path}")

    with resolved_metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_feature_values(payload: PatientFeatures) -> dict[str, float]:
    return {
        "AGE": payload.age,
        "PLT": payload.platelets_k_per_ul,
        "AST": payload.ast_u_l,
        "ALT": payload.alt_u_l,
        "ALBUMIN": payload.albumin_g_l,
        "FIB4": payload.fib4,
        "APRI": payload.apri,
        "RITIS": payload.ritis,
        "NAFLD": payload.nafld,
    }


def _resolve_feature_value(feature_name: str, payload: PatientFeatures) -> float:
    normalized = re.sub(r"[^A-Z0-9]+", "", feature_name.upper())
    values = _canonical_feature_values(payload)

    if normalized.startswith("AGE"):
        return values["AGE"]
    if normalized.startswith("PLT"):
        return values["PLT"]
    if normalized.startswith("AST"):
        return values["AST"]
    if normalized.startswith("ALT"):
        return values["ALT"]
    if normalized.startswith("ALBUMIN"):
        return values["ALBUMIN"]
    if normalized.startswith("FIB4"):
        return values["FIB4"]
    if normalized.startswith("APRI"):
        return values["APRI"]
    if normalized.startswith("RITIS"):
        return values["RITIS"]
    if normalized.startswith("NAFLD"):
        return values["NAFLD"]

    raise KeyError(f"Unsupported feature name in metadata: {feature_name}")


def build_feature_frame(payload: PatientFeatures, feature_cols: list[str]) -> pd.DataFrame:
    row = {feature_name: _resolve_feature_value(feature_name, payload) for feature_name in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def triage_from_probability(probability_positive: float, t_out: float, t_in: float) -> str:
    if probability_positive < t_out:
        return "OUT"
    if probability_positive >= t_in:
        return "IN"
    return "GREY"


def score_patient(
    payload: PatientFeatures,
    bundle: LoadedModelBundle | None = None,
) -> dict[str, Any]:
    loaded_bundle = bundle or load_model_bundle()
    meta = loaded_bundle.meta
    model_config = get_model_config()

    feature_cols = meta["feature_cols"]
    frame = build_feature_frame(payload, feature_cols)
    probabilities = loaded_bundle.model.predict_proba(frame)[0]

    positive_label = meta["POS_LABEL"]
    negative_label = meta["NEG_LABEL"]
    positive_index = list(loaded_bundle.model.classes_).index(positive_label)
    probability_positive = float(probabilities[positive_index])
    triage_zone = triage_from_probability(probability_positive, meta["t_out"], meta["t_in"])

    return {
        "model_name": meta.get("model_name", model_config.name),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "probability_positive": probability_positive,
        "triage_zone": triage_zone,
        "threshold_out": float(meta["t_out"]),
        "threshold_in": float(meta["t_in"]),
    }


def get_model_summary(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    model_config = get_model_config()
    meta = metadata or load_model_metadata()

    return {
        "model_name": meta.get("model_name", model_config.name),
        "model_path": str(model_config.model_path),
        "metadata_path": str(model_config.metadata_path),
        "positive_label": meta["POS_LABEL"],
        "negative_label": meta["NEG_LABEL"],
        "threshold_out": float(meta["t_out"]),
        "threshold_in": float(meta["t_in"]),
        "feature_count": len(meta["feature_cols"]),
        "feature_cols": meta["feature_cols"],
    }
