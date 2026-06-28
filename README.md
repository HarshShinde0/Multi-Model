# Text-to-Image Generation with Multi-Model Analysis

Generate images from text descriptions using Stable Diffusion, analyze them using CLIP and BLIP, and perform instance segmentation using Meta's Segment Anything Model 2 (SAM2).

## Hardware Setup & Environment Installation

This project supports both **GPU (CUDA)** and **CPU** execution configurations.

### 1. Conda Environment Setup

Ensure Conda is installed, and create/activate the workspace environment:
```bash
conda create -n deeplearning python=3.10 -y
conda activate deeplearning
```

### 2. Dependency Installation

#### Option A: GPU (CUDA-accelerated) Setup (Recommended)
This configuration leverages NVIDIA CUDA acceleration for high-performance inference.
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CPU-Only Setup
This configuration runs all inference workloads on the CPU (suitable for testing or systems without NVIDIA GPUs).
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Model Weight Downloads
Run the download script to pre-fetch and cache model weights (Stable Diffusion v1.5, SAM2, CLIP, and BLIP):
```bash
python download_models.py
```

---

## Running the Application

### 1. Run via Uvicorn Server Locally
Start the FastAPI application:
```bash
cd src
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Then visit `http://127.0.0.1:8000/` in your browser to launch the **Multimodel Studio UI**.

### 2. Run via Docker
To run inside a containerized setup, a basic Docker configuration is provided.
```bash
docker build -t tti-multimodel-analysis .
docker run -p 8000:8000 tti-multimodel-analysis
```

---

## API Documentation

### 1. Generate Image (`POST /generate`)
Generates an image from a prompt, performs CLIP/BLIP classification, and extracts SAM2 segmentation masks. Supports both Basic and Extended formats.

* **Request Payload:**
```json
{
  "prompt": "a red kite flying over a coastal cliff",
  "image_size": 512
}
```
* **Response Details:** Follows the schema in [example_generate_response.json](examples/example_generate_response.json) (populates both basic `concepts`/`basic_segmentation` and extended `global_concepts`/`segmentation`).

### 2. Analyze Image (`POST /analyze`)
Analyzes an uploaded image with CLIP classification and BLIP captions.

* **Request Payload:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "prompt": "coastal, cliff"
}
```
* **Response Details:** Follows the schema in [example_clip_response.json](examples/example_clip_response.json).

### 3. Segment Image (`POST /segment`)
Performs SAM2 segmentation. Supports dynamic parameter controls for segmentation thresholds.

* **Request Payload:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "points_per_side": 32,
  "pred_iou_thresh": 0.88,
  "stability_score_thresh": 0.95,
  "crop_n_layers": 0
}
```
* **Response Details:** Follows the schema in [example_segment_response.json](examples/example_segment_response.json) (includes detailed boundary contour polygons, crop boxes, and region image clips).

### 4. Concept Probing (`POST /probe`)
Runs CLIP concept probing relative to custom-defined concepts across the global image and each segmented sub-region.

* **Request Payload:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "concepts": ["red square", "circle", "sky", "grass"]
}
```
* **Response Details:** Follows the schema in [example_probe_response.json](examples/example_probe_response.json).

### 5. Advanced Visualization (`POST /visualize`)
Generates visual representation overlays for segmentation boundaries.

* **Request Payload:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "visualization_type": "overlay",
  "alpha": 0.4,
  "line_thickness": 2
}
```
* **Response Details:** Follows the schema in [example_visualize_response.json](examples/example_visualize_response.json).

---

## Model Configurations

- **Image Generation:** `runwayml/stable-diffusion-v1-5` (configured dynamically for half-precision `float16` when running on CUDA-capable GPUs).
- **Text & Image Classification:** `openai/clip-vit-base-patch32` (runs zero-shot similarity scores).
- **Caption Summary:** `Salesforce/blip-image-captioning-base`.
- **Instance Segmentation:** `sam2.1_hiera_base_plus` (uses OpenCV contours to reconstruct detailed boundary coordinate polygons).
