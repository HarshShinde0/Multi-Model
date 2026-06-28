from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from services.image_utils import image_to_base64


@dataclass(slots=True)
class SegmentationResultData:
    masks: list[str]
    polygons: list[list[list[int]]]
    segmented_regions: list[dict[str, Any]]


class Segmenter:
    def __init__(self, model_size: str = "base"):
        """Initialize Segment Anything Model 2.
        
        Args:
            model_size: Size of the model - "base" (default), "large", or "huge"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SAM2 model registry
        model_name_map = {
            "base": ("vit_b", "sam_vit_b_01ec64.pth"),
            "large": ("vit_l", "sam_vit_l_0b3195.pth"),
            "huge": ("vit_h", "sam_vit_h_6b3a56.pth"),
        }
        
        registry_key, model_name = model_name_map.get(model_size, model_name_map["base"])
        
        import os
        from pathlib import Path
        import urllib.request
        
        cache_dir = Path.home() / '.cache' / 'segment_anything'
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / model_name
        
        if not checkpoint_path.exists():
            print(f"Downloading SAM checkpoint to {checkpoint_path}...")
            urllib.request.urlretrieve(f"https://dl.fbaipublicfiles.com/segment_anything/{model_name}", checkpoint_path)
            
        self.sam = sam_model_registry[registry_key](checkpoint=str(checkpoint_path))
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def segment(self, image: Image.Image) -> SegmentationResultData:
        """Segment image using Segment Anything Model 2.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            SegmentationResultData with masks, polygons, and regions
        """
        # Convert PIL image to numpy array
        image_array = np.array(image.convert("RGB"))
        
        # Generate masks using SAM2
        masks = self.mask_generator.generate(image_array)
        
        mask_strs: list[str] = []
        polygons: list[list[list[int]]] = []
        regions: list[dict[str, Any]] = []
        
        height, width = image_array.shape[:2]
        
        for idx, mask_data in enumerate(masks):
            # Get the segmentation mask
            seg_mask = mask_data["segmentation"].astype(np.uint8) * 255
            
            # Convert mask to PIL image for encoding
            mask_image = Image.fromarray(seg_mask, mode="L")
            mask_strs.append(image_to_base64(mask_image.convert("RGB")))
            
            # Get bounding box polygon
            bbox = mask_data["bbox"]
            x, y, w, h = bbox
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            polygons.append(polygon)
            
            # Extract region image
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            if x2 > x1 and y2 > y1:
                region_image = image.crop((x1, y1, x2, y2))
                regions.append(
                    {
                        "region_id": f"region-{idx + 1}",
                        "polygon": polygon,
                        "image_base64": image_to_base64(region_image),
                        "label": f"segment-{idx + 1}",
                        "area": mask_data.get("area", 0),
                        "predicted_iou": float(mask_data.get("predicted_iou", 0)),
                    }
                )
        
        if not mask_strs:
            empty = Image.new("RGB", image.size, color="black")
            mask_strs.append(image_to_base64(empty))
            polygons.append([[0, 0], [width, 0], [width, height], [0, height]])
        
        return SegmentationResultData(masks=mask_strs, polygons=polygons, segmented_regions=regions)
