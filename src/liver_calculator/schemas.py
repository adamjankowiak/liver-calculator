from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=0)
    platelets_k_per_ul: float = Field(..., gt=0)
    ast_u_l: float = Field(..., ge=0)
    alt_u_l: float = Field(..., ge=0)
    albumin_g_l: float = Field(..., gt=0)
    fib4: float
    apri: float
    ritis: float
    nafld: float


class PredictionResponse(BaseModel):
    model_name: str
    positive_label: str
    negative_label: str
    probability_positive: float
    triage_zone: Literal["OUT", "IN", "GREY"]
    threshold_out: float
    threshold_in: float
