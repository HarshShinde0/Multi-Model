from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_endpoint() -> None:
    response = client.post("/generate", json={"prompt": "a glass house in a forest"})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["generated_image"]
    assert body["clip_analysis"]["global_concepts"]
