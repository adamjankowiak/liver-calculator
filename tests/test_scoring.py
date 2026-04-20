import numpy as np

from liver_calculator.schemas import PatientFeatures
from liver_calculator.services.scoring import (
    LoadedModelBundle,
    build_feature_frame,
    score_patient,
    triage_from_probability,
)


class DummyModel:
    classes_ = ["1-2", "3-4"]

    def predict_proba(self, frame):
        assert list(frame.columns) == [
            "Age",
            "PLT [k/Âµl]",
            "AST [U/l]",
            "ALT [U/l]",
            "Albumin [g/l]",
            "FIB4",
            "APRI",
            "RITIS",
            "NAFLD",
        ]
        return np.array([[0.08, 0.92]])


def make_payload() -> PatientFeatures:
    return PatientFeatures(
        age=55,
        platelets_k_per_ul=180,
        ast_u_l=42,
        alt_u_l=35,
        albumin_g_l=39,
        fib4=2.1,
        apri=0.9,
        ritis=1.2,
        nafld=-0.3,
    )


def test_build_feature_frame_uses_metadata_order():
    payload = make_payload()
    feature_cols = [
        "Age",
        "PLT [k/Âµl]",
        "AST [U/l]",
        "ALT [U/l]",
        "Albumin [g/l]",
        "FIB4",
        "APRI",
        "RITIS",
        "NAFLD",
    ]

    frame = build_feature_frame(payload, feature_cols)

    assert list(frame.columns) == feature_cols
    assert frame.iloc[0]["Age"] == 55
    assert frame.iloc[0]["NAFLD"] == -0.3


def test_triage_from_probability():
    assert triage_from_probability(0.05, 0.10, 0.80) == "OUT"
    assert triage_from_probability(0.90, 0.10, 0.80) == "IN"
    assert triage_from_probability(0.40, 0.10, 0.80) == "GREY"


def test_score_patient_returns_expected_shape():
    bundle = LoadedModelBundle(
        model=DummyModel(),
        meta={
            "feature_cols": [
                "Age",
                "PLT [k/Âµl]",
                "AST [U/l]",
                "ALT [U/l]",
                "Albumin [g/l]",
                "FIB4",
                "APRI",
                "RITIS",
                "NAFLD",
            ],
            "POS_LABEL": "3-4",
            "NEG_LABEL": "1-2",
            "t_out": 0.10,
            "t_in": 0.80,
        },
    )

    result = score_patient(make_payload(), bundle=bundle)

    assert result["positive_label"] == "3-4"
    assert result["triage_zone"] == "IN"
    assert result["probability_positive"] == 0.92
