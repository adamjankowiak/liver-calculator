from fastapi.testclient import TestClient

from liver_calculator.api.main import app
from liver_calculator.api import routes_predict


client = TestClient(app)


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(
        routes_predict,
        "score_patient",
        lambda payload: {
            "model_name": "meta_logistic",
            "positive_label": "3-4",
            "negative_label": "1-2",
            "probability_positive": 0.87,
            "triage_zone": "IN",
            "threshold_out": 0.15,
            "threshold_in": 0.80,
        },
    )

    response = client.post(
        "/predict",
        json={
            "age": 55,
            "platelets_k_per_ul": 180,
            "ast_u_l": 42,
            "alt_u_l": 35,
            "albumin_g_l": 39,
            "fib4": 2.1,
            "apri": 0.9,
            "ritis": 1.2,
            "nafld": -0.3,
        },
    )

    assert response.status_code == 200
    assert response.json()["triage_zone"] == "IN"


def test_model_info_endpoint(monkeypatch):
    monkeypatch.setattr(
        routes_predict,
        "get_model_summary",
        lambda: {
            "model_name": "meta_logistic",
            "model_path": "models/model.joblib",
            "metadata_path": "models/metadata/model.json",
            "positive_label": "3-4",
            "negative_label": "1-2",
            "threshold_out": 0.10,
            "threshold_in": 0.80,
            "feature_count": 9,
            "feature_cols": ["Age"],
        },
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["feature_count"] == 9


def test_health_reports_degraded_when_model_cannot_load(monkeypatch):
    monkeypatch.setattr(
        routes_predict,
        "load_model_bundle",
        lambda: (_ for _ in ()).throw(AttributeError("incompatible sklearn artifact")),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ready"] is False
