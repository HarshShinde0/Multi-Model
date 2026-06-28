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
    assert body["clip_analysis"]["concepts"]
    assert body["clip_analysis"]["global_concepts"]
    assert body["clip_analysis"]["confidence_scores"]
    assert "basic_segmentation" in body
    assert "segmentation" in body


def _get_test_image_base64() -> str:
    import base64
    import io
    from PIL import Image
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_segment_endpoint() -> None:
    img_b64 = _get_test_image_base64()
    response = client.post(
        "/segment",
        json={
            "image_base64": img_b64,
            "points_per_side": 8,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.9,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert "segmentation" in body
    assert "masks" in body["segmentation"]
    assert "overlay_image" in body["segmentation"]
    # Check that polygon returned has coords
    assert len(body["segmentation"]["polygons"]) > 0


def test_probe_endpoint() -> None:
    img_b64 = _get_test_image_base64()
    response = client.post(
        "/probe",
        json={
            "image_base64": img_b64,
            "concepts": ["a red square", "a circle", "sky", "grass"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert "global_scores" in body
    assert "regional_scores" in body
    assert "a red square" in body["global_scores"]


def test_visualize_endpoint() -> None:
    img_b64 = _get_test_image_base64()
    for viz_type in ["overlay", "contour", "bbox", "isolate"]:
        response = client.post(
            "/visualize",
            json={
                "image_base64": img_b64,
                "visualization_type": viz_type,
                "alpha": 0.5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["request_id"]
        assert body["visualization_image"]


def test_generate_validation_error() -> None:
    # prompt is too short (min_length=1)
    response = client.post("/generate", json={"prompt": "", "image_size": 512})
    assert response.status_code == 422


def test_segment_validation_error() -> None:
    # image_base64 is missing
    response = client.post("/segment", json={"points_per_side": 8})
    assert response.status_code == 422


def test_probe_validation_error() -> None:
    # concepts must have at least 1 item
    img_b64 = _get_test_image_base64()
    response = client.post("/probe", json={"image_base64": img_b64, "concepts": []})
    assert response.status_code == 422


def test_visualize_validation_error() -> None:
    # alpha must be <= 1.0 (ge=0.0, le=1.0)
    img_b64 = _get_test_image_base64()
    response = client.post(
        "/visualize",
        json={
            "image_base64": img_b64,
            "visualization_type": "overlay",
            "alpha": 1.5,
        },
    )
    assert response.status_code == 422
