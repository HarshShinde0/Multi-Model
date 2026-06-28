from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    image_size: int | None = Field(default=None, ge=64, le=1024)


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    prompt: str | None = Field(default=None, max_length=500)


class SegmentRequest(AnalyzeRequest):
    pass


class ConceptScore(BaseModel):
    concept: str
    confidence: float


class SegmentedRegion(BaseModel):
    region_id: str
    polygon: list[list[int]]
    image_base64: str
    clip_analysis: dict[str, Any] = Field(default_factory=dict)


class ClipAnalysis(BaseModel):
    global_concepts: list[str]
    confidence_scores: dict[str, float]
    summary: str | None = None
    regional_analysis: list[dict[str, Any]] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    masks: list[str]
    polygons: list[list[list[int]]]
    segmented_regions: list[SegmentedRegion] = Field(default_factory=list)


class ImageResponse(BaseModel):
    request_id: str
    generated_image: str
    clip_analysis: ClipAnalysis
    basic_segmentation: SegmentationResult


class AnalysisResponse(BaseModel):
    request_id: str
    clip_analysis: ClipAnalysis
    segmentation: SegmentationResult


class SegmentResponse(BaseModel):
    request_id: str
    segmentation: SegmentationResult


class ErrorResponse(BaseModel):
    detail: str
