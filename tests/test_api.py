from fastapi.testclient import TestClient

from liver_calculator.api import main as api_main


client = TestClient(api_main.app)


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(
        api_main,
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
