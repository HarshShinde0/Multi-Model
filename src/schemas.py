from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    image_size: int | None = Field(default=None, ge=64, le=1024)


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    prompt: str | None = Field(default=None, max_length=500)


class SegmentRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    points_per_side: int | None = Field(default=None, ge=4, le=64)
    pred_iou_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    stability_score_thresh: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_n_layers: int | None = Field(default=None, ge=0, le=4)
    box_nms_thresh: float | None = Field(default=None, ge=0.0, le=1.0)


class ConceptScore(BaseModel):
    concept: str
    confidence: float


class SegmentedRegion(BaseModel):
    region_id: str
    polygon: list[list[int]]
    bbox: list[int]
    image_base64: str
    image: str
    clip_analysis: dict[str, Any] = Field(default_factory=dict)


class ClipAnalysis(BaseModel):
    concepts: list[str]
    global_concepts: list[str]
    confidence_scores: dict[str, float]
    summary: str | None = None
    regional_analysis: list[dict[str, Any]] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    masks: list[str]
    polygons: list[list[list[int]]]
    segmented_regions: list[SegmentedRegion] = Field(default_factory=list)
    overlay_image: str | None = None


class ImageResponse(BaseModel):
    request_id: str
    generated_image: str
    clip_analysis: ClipAnalysis
    basic_segmentation: SegmentationResult
    segmentation: SegmentationResult


class AnalysisResponse(BaseModel):
    request_id: str
    clip_analysis: ClipAnalysis
    segmentation: SegmentationResult


class SegmentResponse(BaseModel):
    request_id: str
    segmentation: SegmentationResult


class ProbeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    concepts: list[str] = Field(min_items=1)


class ProbeRegionResult(BaseModel):
    region_id: str
    polygon: list[list[int]]
    scores: dict[str, float]


class ProbeResponse(BaseModel):
    request_id: str
    global_scores: dict[str, float]
    regional_scores: list[ProbeRegionResult]


class VisualizeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    visualization_type: str = Field(default="overlay")
    alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    line_thickness: int = Field(default=2, ge=1, le=10)


class VisualizeResponse(BaseModel):
    request_id: str
    visualization_image: str


class ErrorResponse(BaseModel):
    detail: str
