from __future__ import annotations

import uuid
from dataclasses import dataclass

from PIL import Image

from core.config import Settings
from schemas import (
    AnalysisResponse,
    ClipAnalysis,
    ImageResponse,
    ProbeRegionResult,
    ProbeResponse,
    SegmentResponse,
    SegmentationResult,
    SegmentedRegion,
    VisualizeResponse,
)
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
        schema_seg = self._segmentation_schema(segmentation, generation.image)
        return ImageResponse(
            request_id=self._request_id(),
            generated_image=image_to_base64(generation.image),
            clip_analysis=self._clip_schema(clip_result),
            basic_segmentation=schema_seg,
            segmentation=schema_seg,
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

    def segment(
        self,
        image_base64: str,
        points_per_side: int | None = None,
        pred_iou_thresh: float | None = None,
        stability_score_thresh: float | None = None,
        crop_n_layers: int | None = None,
        box_nms_thresh: float | None = None,
    ) -> SegmentResponse:
        image = base64_to_image(image_base64)
        segmentation = self.services.segmenter.segment(
            image,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
        )
        return SegmentResponse(
            request_id=self._request_id(),
            segmentation=self._segmentation_schema(segmentation, image),
        )

    def probe(self, image_base64: str, concepts: list[str]) -> ProbeResponse:
        image = base64_to_image(image_base64)
        segmentation = self.services.segmenter.segment(image)
        
        # Calculate global scores
        global_scores = self.services.analyzer.probe_concepts(image, concepts)
        
        # Calculate regional scores on individual cropped region images
        regional_scores = []
        for region in segmentation.segmented_regions:
            try:
                region_img = base64_to_image(region["image_base64"])
                scores = self.services.analyzer.probe_concepts(region_img, concepts)
            except Exception:
                scores = {concept: 0.0 for concept in concepts}
                
            regional_scores.append(
                ProbeRegionResult(
                    region_id=region["region_id"],
                    polygon=region["polygon"],
                    scores=scores,
                )
            )
            
        return ProbeResponse(
            request_id=self._request_id(),
            global_scores=global_scores,
            regional_scores=regional_scores,
        )

    def visualize(
        self,
        image_base64: str,
        visualization_type: str = "overlay",
        alpha: float = 0.4,
        line_thickness: int = 2,
    ) -> VisualizeResponse:
        image = base64_to_image(image_base64)
        visualized_img = self.services.segmenter.visualize(
            image,
            visualization_type=visualization_type,
            alpha=alpha,
            line_thickness=line_thickness,
        )
        return VisualizeResponse(
            request_id=self._request_id(),
            visualization_image=image_to_base64(visualized_img),
        )

    def _clip_schema(self, result) -> ClipAnalysis:
        return ClipAnalysis(
            concepts=result.global_concepts,
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
                bbox=region["bbox"],
                image_base64=region["image_base64"],
                image=region["image_base64"],
                clip_analysis=region.get("clip_analysis", {}),
            )
            for region in result.segmented_regions
        ]
        return SegmentationResult(
            masks=result.masks,
            polygons=result.polygons,
            segmented_regions=regions,
            overlay_image=result.overlay_image,
        )

    def _request_id(self) -> str:
        return str(uuid.uuid4())
