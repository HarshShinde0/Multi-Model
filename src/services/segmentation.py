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
    overlay_image: str | None = None


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

    def segment(
        self,
        image: Image.Image,
        points_per_side: int | None = None,
        pred_iou_thresh: float | None = None,
        stability_score_thresh: float | None = None,
        crop_n_layers: int | None = None,
        box_nms_thresh: float | None = None,
    ) -> SegmentationResultData:
        """Segment image using Segment Anything Model 2.
        
        Args:
            image: PIL Image to segment
            points_per_side: Advanced segmentation parameters override
            pred_iou_thresh: Advanced segmentation parameters override
            stability_score_thresh: Advanced segmentation parameters override
            crop_n_layers: Advanced segmentation parameters override
            box_nms_thresh: Advanced segmentation parameters override
            
        Returns:
            SegmentationResultData with masks, polygons, and regions
        """
        import cv2
        # Convert PIL image to numpy array
        image_array = np.array(image.convert("RGB"))
        
        # Instantiate dynamic generator if custom parameters are provided
        if any(v is not None for v in [points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers, box_nms_thresh]):
            kwargs = {}
            if points_per_side is not None:
                kwargs["points_per_side"] = points_per_side
            if pred_iou_thresh is not None:
                kwargs["pred_iou_thresh"] = pred_iou_thresh
            if stability_score_thresh is not None:
                kwargs["stability_score_thresh"] = stability_score_thresh
            if crop_n_layers is not None:
                kwargs["crop_n_layers"] = crop_n_layers
            if box_nms_thresh is not None:
                kwargs["box_nms_thresh"] = box_nms_thresh
            mask_generator = SamAutomaticMaskGenerator(self.sam, **kwargs)
        else:
            mask_generator = self.mask_generator

        # Generate masks using SAM2
        masks = mask_generator.generate(image_array)
        
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
            
            # Get actual contour polygon coordinates
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_polygon = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                contour_polygon = approx.reshape(-1, 2).tolist()
            
            # Fallback to bounding box polygon if no contour found
            bbox = mask_data["bbox"]
            x, y, w, h = bbox
            bbox_polygon = [[int(x), int(y)], [int(x + w), int(y)], [int(x + w), int(y + h)], [int(x), int(y + h)]]
            
            if not contour_polygon:
                contour_polygon = bbox_polygon
                
            polygons.append(contour_polygon)
            
            # Extract region image
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            if x2 > x1 and y2 > y1:
                region_image = image.crop((x1, y1, x2, y2))
                regions.append(
                    {
                        "region_id": f"region-{idx + 1}",
                        "polygon": contour_polygon,
                        "bbox": [int(x), int(y), int(w), int(h)],
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
            overlay_base64 = image_to_base64(image)
        else:
            # Generate default overlay
            overlay_img = self.visualize_masks(image, masks, visualization_type="overlay")
            overlay_base64 = image_to_base64(overlay_img)
        
        return SegmentationResultData(
            masks=mask_strs,
            polygons=polygons,
            segmented_regions=regions,
            overlay_image=overlay_base64
        )

    def visualize_masks(
        self,
        image: Image.Image,
        masks: list[dict[str, Any]],
        visualization_type: str = "overlay",
        alpha: float = 0.4,
        line_thickness: int = 2,
    ) -> Image.Image:
        """Create visual representation of SAM2 masks."""
        import cv2
        
        image_array = np.array(image.convert("RGB"))
        if not masks:
            return image
            
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        if visualization_type == "overlay":
            overlay = image_array.copy()
            for idx, mask_data in enumerate(masks):
                mask = mask_data["segmentation"]
                color = colors[idx].tolist()
                overlay[mask] = color
                
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_array, contours, -1, color, line_thickness)
                
            blended = cv2.addWeighted(overlay, alpha, image_array, 1 - alpha, 0)
            return Image.fromarray(blended)
            
        elif visualization_type == "contour":
            output = image_array.copy()
            for idx, mask_data in enumerate(masks):
                mask = mask_data["segmentation"]
                color = colors[idx].tolist()
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, color, line_thickness)
            return Image.fromarray(output)
            
        elif visualization_type == "bbox":
            output = image_array.copy()
            for idx, mask_data in enumerate(masks):
                color = colors[idx].tolist()
                x, y, w, h = mask_data["bbox"]
                cv2.rectangle(output, (int(x), int(y)), (int(x + w), int(y + h)), color, line_thickness)
                cv2.putText(output, f"segment-{idx+1}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return Image.fromarray(output)
            
        elif visualization_type == "isolate":
            output = np.zeros_like(image_array)
            for idx, mask_data in enumerate(masks):
                mask = mask_data["segmentation"]
                output[mask] = image_array[mask]
            return Image.fromarray(output)
            
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

    def visualize(
        self,
        image: Image.Image,
        visualization_type: str = "overlay",
        alpha: float = 0.4,
        line_thickness: int = 2,
    ) -> Image.Image:
        """Helper to generate masks and visualize them."""
        image_array = np.array(image.convert("RGB"))
        masks = self.mask_generator.generate(image_array)
        return self.visualize_masks(image, masks, visualization_type, alpha, line_thickness)
