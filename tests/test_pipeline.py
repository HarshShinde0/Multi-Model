from __future__ import annotations

from PIL import Image

from services.image_utils import image_to_base64
from services.pipeline import Pipeline


def test_generate_returns_expected_payload() -> None:
    pipeline = Pipeline()
    response = pipeline.generate("a lighthouse at dusk", 256)

    assert response.request_id
    assert response.generated_image
    assert response.clip_analysis.global_concepts
    assert response.basic_segmentation.masks
    assert response.basic_segmentation.polygons


def test_analyze_accepts_base64_image() -> None:
    pipeline = Pipeline()
    image = Image.new("RGB", (128, 128), color="navy")

    response = pipeline.analyze(image_to_base64(image), prompt="navy square")

    assert response.request_id
    assert response.clip_analysis.global_concepts
    assert response.segmentation.masks
