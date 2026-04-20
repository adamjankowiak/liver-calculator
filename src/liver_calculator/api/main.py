from fastapi import FastAPI, HTTPException

from liver_calculator.config import DEFAULT_MODEL
from liver_calculator.schemas import PatientFeatures, PredictionResponse
from liver_calculator.services.scoring import score_patient


app = FastAPI(
    title="Liver Meta Calculator API",
    version="0.1.0",
    description="Inference API for the liver fibrosis meta-calculator.",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Liver Meta Calculator API"}


@app.get("/health")
def read_health() -> dict[str, bool]:
    return {
        "model_artifact_present": DEFAULT_MODEL.model_path.exists(),
        "metadata_present": DEFAULT_MODEL.metadata_path.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures) -> PredictionResponse:
    try:
        prediction = score_patient(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**prediction)
