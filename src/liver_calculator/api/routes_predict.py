from fastapi import APIRouter, HTTPException

from liver_calculator.config import get_model_config
from liver_calculator.schemas import PatientFeatures, PredictionResponse
from liver_calculator.services.scoring import get_model_summary, load_model_bundle, score_patient


router = APIRouter(tags=["inference"])

MODEL_LOAD_ERRORS = (
    FileNotFoundError,
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
)


@router.get("/health")
def read_health() -> dict[str, bool | str]:
    model_config = get_model_config()
    artifact_present = model_config.model_path.exists()
    metadata_present = model_config.metadata_path.exists()

    model_loadable = False
    if artifact_present and metadata_present:
        try:
            load_model_bundle()
            model_loadable = True
        except MODEL_LOAD_ERRORS:
            model_loadable = False

    return {
        "status": "ok" if model_loadable else "degraded",
        "model_artifact_present": artifact_present,
        "metadata_present": metadata_present,
        "model_loadable": model_loadable,
        "ready": model_loadable,
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
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(**prediction)
