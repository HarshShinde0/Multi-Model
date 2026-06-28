from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


@dataclass(slots=True)
class GenerationResult:
    image: Image.Image
    model_name: str


class ImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)
        self.pipeline.enable_attention_slicing()

    def generate(self, prompt: str, size: int = 512) -> GenerationResult:
        """Generate image using Stable Diffusion.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (must be multiple of 8, default 512)
            
        Returns:
            GenerationResult with generated image and model name
        """
        # Ensure size is multiple of 8
        size = (size // 8) * 8 if size >= 512 else 512
        
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
        
        return GenerationResult(
            image=result.images[0],
            model_name="stable-diffusion-v1.5",
        )
