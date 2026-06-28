from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration


@dataclass(slots=True)
class AnalysisResult:
    global_concepts: list[str]
    confidence_scores: dict[str, float]
    regional_analysis: list[dict[str, Any]]
    summary: str | None = None


class ClipAnalyzer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", blip_model: str = "Salesforce/blip-image-captioning-base"):
        """Initialize CLIP and BLIP models for image analysis."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load BLIP for summary/caption generation
        self.blip_processor = BlipProcessor.from_pretrained(blip_model)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model).to(self.device)

    def analyze(self, image: Image.Image, prompt: str | None = None, regions: Sequence[dict[str, Any]] | None = None) -> AnalysisResult:
        """Analyze image using CLIP and generate summary using BLIP."""
        default_concepts = [
            "a photo", "a clear image", "a dark image", "a bright image",
            "a colorful scene", "a detailed object", "a landscape", "a portrait"
        ]
        
        prompt_concepts = self._prompt_concepts(prompt)
        all_concepts = self._merge_concepts(default_concepts, prompt_concepts)
        
        confidence_scores = self._calculate_batch_scores(image, all_concepts)
        
        # Get top 3 concepts
        top_concepts = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        global_concepts = [c for c, _ in top_concepts[:3]]
        
        # Ensure prompt concepts are included in global concepts
        for pc in prompt_concepts:
            if pc not in global_concepts:
                global_concepts.append(pc)
                
        regional_analysis = self._regional_analysis(regions or [], image, all_concepts)
        
        # Generate image summary
        summary = self._generate_summary(image)
        
        return AnalysisResult(global_concepts, confidence_scores, regional_analysis, summary)

    def _generate_summary(self, image: Image.Image) -> str:
        """Generate a short summary description of the image."""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=100, min_length=20)
            text = self.blip_processor.decode(out[0], skip_special_tokens=True).capitalize()
            if not text.endswith("."):
                text += "."
            return text
        except Exception as e:
            return "Summary generation failed."

    def _prompt_concepts(self, prompt: str | None) -> list[str]:
        if not prompt:
            return []
        # If user provides comma separated classes, split by comma
        if "," in prompt:
            return [p.strip() for p in prompt.split(",") if p.strip()]
            
        words = [word.strip(".,;:!?()[]{}\"'").lower() for word in prompt.split()]
        useful = [word for word in words if len(word) > 2]
        return useful[:5]

    def _calculate_batch_scores(self, image: Image.Image, texts: list[str]) -> dict[str, float]:
        try:
            with torch.no_grad():
                inputs = self.processor(
                    text=texts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                
            return {text: round(float(prob), 3) for text, prob in zip(texts, probs)}
        except Exception:
            return {text: round(1.0 / len(texts), 3) for text in texts}

    def _merge_concepts(self, *groups: Sequence[str]) -> list[str]:
        ordered: list[str] = []
        for group in groups:
            for item in group:
                if item not in ordered:
                    ordered.append(item)
        return ordered

    def _regional_analysis(self, regions: Sequence[dict[str, Any]], image: Image.Image, all_concepts: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, region in enumerate(regions):
            confidence_scores = self._calculate_batch_scores(image, all_concepts)
            top_concepts = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            region_concepts = [c for c, _ in top_concepts[:3]]
            
            results.append(
                {
                    "region_id": region.get("region_id", f"region-{index + 1}"),
                    "concepts": region_concepts,
                    "confidence_scores": confidence_scores,
                }
            )
        return results
