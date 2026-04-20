from fastapi import APIRouter, HTTPException

from liver_calculator.config import get_model_config
from liver_calculator.schemas import PatientFeatures, PredictionResponse
from liver_calculator.services.scoring import get_model_summary, load_model_metadata, score_patient


router = APIRouter(tags=["inference"])


@router.get("/health")
def read_health() -> dict[str, bool | str]:
    model_config = get_model_config()
    artifact_present = model_config.model_path.exists()
    metadata_present = model_config.metadata_path.exists()

    metadata_valid = False
    if metadata_present:
        try:
            load_model_metadata()
            metadata_valid = True
        except (FileNotFoundError, ValueError, KeyError, OSError, TypeError):
            metadata_valid = False

    return {
        "status": "ok" if artifact_present and metadata_valid else "degraded",
        "model_artifact_present": artifact_present,
        "metadata_present": metadata_present,
        "metadata_valid": metadata_valid,
        "ready": artifact_present and metadata_valid,
    }


@router.get("/model-info")
def read_model_info() -> dict[str, object]:
    try:
        return get_model_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures) -> PredictionResponse:
    try:
        prediction = score_patient(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(**prediction)
