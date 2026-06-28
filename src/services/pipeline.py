from __future__ import annotations

import uuid
from dataclasses import dataclass

from PIL import Image

from core.config import Settings
from schemas import AnalysisResponse, ClipAnalysis, ImageResponse, SegmentResponse, SegmentationResult, SegmentedRegion
from services.analysis import ClipAnalyzer
from services.generation import ImageGenerator
from services.image_utils import base64_to_image, image_to_base64
from services.segmentation import Segmenter


@dataclass(slots=True)
class PipelineServices:
    generator: ImageGenerator
    analyzer: ClipAnalyzer
    segmenter: Segmenter


class Pipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.services = PipelineServices(
            generator=ImageGenerator(),
            analyzer=ClipAnalyzer(),
            segmenter=Segmenter(),
        )

    def generate(self, prompt: str, image_size: int | None = None) -> ImageResponse:
        size = image_size or self.settings.default_image_size
        generation = self.services.generator.generate(prompt=prompt, size=size)
        segmentation = self.services.segmenter.segment(generation.image)
        clip_result = self.services.analyzer.analyze(generation.image, prompt=prompt, regions=segmentation.segmented_regions)
        return ImageResponse(
            request_id=self._request_id(),
            generated_image=image_to_base64(generation.image),
            clip_analysis=self._clip_schema(clip_result),
            basic_segmentation=self._segmentation_schema(segmentation, generation.image),
        )

    def analyze(self, image_base64: str, prompt: str | None = None) -> AnalysisResponse:
        image = base64_to_image(image_base64)
        segmentation = self.services.segmenter.segment(image)
        clip_result = self.services.analyzer.analyze(image, prompt=prompt, regions=segmentation.segmented_regions)
        return AnalysisResponse(
            request_id=self._request_id(),
            clip_analysis=self._clip_schema(clip_result),
            segmentation=self._segmentation_schema(segmentation, image),
        )

    def segment(self, image_base64: str) -> SegmentResponse:
        image = base64_to_image(image_base64)
        segmentation = self.services.segmenter.segment(image)
        return SegmentResponse(
            request_id=self._request_id(),
            segmentation=self._segmentation_schema(segmentation, image),
        )

    def _clip_schema(self, result) -> ClipAnalysis:
        return ClipAnalysis(
            global_concepts=result.global_concepts,
            confidence_scores=result.confidence_scores,
            summary=result.summary,
            regional_analysis=result.regional_analysis,
        )

    def _segmentation_schema(self, result, image: Image.Image) -> SegmentationResult:
        regions = [
            SegmentedRegion(
                region_id=region["region_id"],
                polygon=region["polygon"],
                image_base64=region["image_base64"],
                clip_analysis=region.get("clip_analysis", {}),
            )
            for region in result.segmented_regions
        ]
        return SegmentationResult(masks=result.masks, polygons=result.polygons, segmented_regions=regions)

    def _request_id(self) -> str:
        return str(uuid.uuid4())
