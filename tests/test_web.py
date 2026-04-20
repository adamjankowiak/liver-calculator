from fastapi.testclient import TestClient

from liver_calculator.api.main import app
from liver_calculator.web import routes as web_routes


client = TestClient(app)


def test_index_page_renders(monkeypatch):
    monkeypatch.setattr(web_routes, "is_model_ready", lambda: True)
    monkeypatch.setattr(
        web_routes,
        "get_model_summary",
        lambda: {
            "model_name": "meta_logistic",
            "feature_count": 9,
            "threshold_out": 0.10,
            "threshold_in": 0.80,
        },
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "Liver Meta Calculator" in response.text
    assert "Run Prediction" in response.text
