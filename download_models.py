#!/usr/bin/env python
"""Download and cache all required models for Multimodel Studio."""

import os
import torch
import urllib.request
from pathlib import Path

# Ensure cache dirs are set
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'huggingface' / 'transformers')
os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Stable Diffusion
try:
    from diffusers import StableDiffusionPipeline
    print("Downloading Stable Diffusion runwayml/stable-diffusion-v1-5...")
    StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
except Exception as e:
    print(f"Error downloading Stable Diffusion: {e}")

# 2. CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
    print("Downloading CLIP openai/clip-vit-base-patch32...")
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error downloading CLIP: {e}")

# 3. BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print("Downloading BLIP Salesforce/blip-image-captioning-base...")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    print(f"Error downloading BLIP: {e}")

# 4. SAM (vit_b)
try:
    from segment_anything import sam_model_registry
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    cache_dir = Path.home() / '.cache' / 'segment_anything'
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "sam_vit_b_01ec64.pth"

    if not checkpoint_path.exists():
        print(f"Downloading SAM2 from {checkpoint_url}...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    else:
        print("SAM2 checkpoint is already cached.")

    # Load model to confirm it works
    sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    print("SAM2 loaded successfully.")
except Exception as e:
    print(f"Error downloading SAM2: {e}")

print("Model download phase complete.")
