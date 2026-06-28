from __future__ import annotations

import pytest
from PIL import Image

from services.generation import ImageGenerator
from services.segmentation import Segmenter
from services.analysis import ClipAnalyzer
from services.image_utils import image_to_base64, base64_to_image


class TestImageGenerator:
    """Test Stable Diffusion image generation."""
    
    @pytest.fixture
    def generator(self):
        """Initialize ImageGenerator with Stable Diffusion."""
        return ImageGenerator()
    
    def test_generate_creates_image(self, generator):
        """Test that generate() creates a valid PIL Image."""
        result = generator.generate(prompt="a serene landscape", size=512)
        
        assert result.image is not None
        assert isinstance(result.image, Image.Image)
        assert result.image.mode == "RGB"
        assert result.image.size == (512, 512)
    
    def test_generate_returns_correct_model_name(self, generator):
        """Test that the model name is correctly reported."""
        result = generator.generate(prompt="a red apple", size=512)
        
        assert result.model_name == "stable-diffusion-v1.5"
    
    def test_generate_with_different_sizes(self, generator):
        """Test generation with different sizes."""
        sizes = [256, 512]
        
        for size in sizes:
            result = generator.generate(prompt="test", size=size)
            expected_size = ((size // 8) * 8) if size >= 512 else 512
            assert result.image.size == (expected_size, expected_size)
    
    def test_generate_with_custom_prompt(self, generator):
        """Test that generation works with custom prompts."""
        prompts = [
            "a futuristic city",
            "a cat wearing glasses",
            "abstract art in blue and gold"
        ]
        
        for prompt in prompts:
            result = generator.generate(prompt=prompt, size=512)
            assert result.image is not None
            assert isinstance(result.image, Image.Image)


class TestSegmenter:
    """Test Segment Anything Model 2 segmentation."""
    
    @pytest.fixture
    def segmenter(self):
        """Initialize Segmenter with SAM2."""
        return Segmenter(model_size="base")
    
    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (256, 256), color="red")
    
    def test_segment_returns_segmentation_data(self, segmenter, test_image):
        """Test that segment() returns valid SegmentationResultData."""
        result = segmenter.segment(test_image)
        
        assert result.masks is not None
        assert len(result.masks) > 0
        assert result.polygons is not None
        assert len(result.polygons) > 0
        assert result.segmented_regions is not None
    
    def test_segment_masks_are_base64_encoded(self, segmenter, test_image):
        """Test that masks are properly base64 encoded."""
        result = segmenter.segment(test_image)
        
        # Base64 strings contain valid characters
        for mask in result.masks:
            assert isinstance(mask, str)
            assert len(mask) > 0
    
    def test_segment_polygons_have_coordinates(self, segmenter, test_image):
        """Test that polygons contain valid coordinate lists."""
        result = segmenter.segment(test_image)
        
        for polygon in result.polygons:
            assert isinstance(polygon, list)
            assert len(polygon) >= 3  # At least 3 points for a polygon
            for point in polygon:
                assert isinstance(point, list)
                assert len(point) == 2  # [x, y]
    
    def test_segment_regions_have_required_fields(self, segmenter, test_image):
        """Test that segmented regions contain required fields."""
        result = segmenter.segment(test_image)
        
        for region in result.segmented_regions:
            assert "region_id" in region
            assert "polygon" in region
            assert "image_base64" in region
            assert "label" in region
            assert "area" in region
            assert "predicted_iou" in region
    
    def test_segment_with_different_image_sizes(self, segmenter):
        """Test segmentation with different image sizes."""
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for width, height in sizes:
            test_image = Image.new("RGB", (width, height), color="blue")
            result = segmenter.segment(test_image)
            assert len(result.masks) > 0


class TestClipAnalyzer:
    """Test CLIP image analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Initialize ClipAnalyzer with CLIP."""
        return ClipAnalyzer(model_name="openai/clip-vit-base-patch32")
    
    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (224, 224), color="green")
    
    def test_analyze_returns_analysis_result(self, analyzer, test_image):
        """Test that analyze() returns valid AnalysisResult."""
        result = analyzer.analyze(test_image)
        
        assert result.global_concepts is not None
        assert isinstance(result.global_concepts, list)
        assert len(result.global_concepts) > 0
        assert result.confidence_scores is not None
        assert isinstance(result.confidence_scores, dict)
        assert result.regional_analysis is not None
        assert isinstance(result.regional_analysis, list)
    
    def test_analyze_with_prompt(self, analyzer, test_image):
        """Test analyze with a text prompt."""
        result = analyzer.analyze(test_image, prompt="a green forest")
        
        assert result.global_concepts is not None
        assert len(result.global_concepts) > 0
        assert result.confidence_scores is not None
    
    def test_analyze_confidence_scores_are_valid(self, analyzer, test_image):
        """Test that confidence scores are between 0 and 1."""
        result = analyzer.analyze(test_image)
        
        for concept, score in result.confidence_scores.items():
            assert isinstance(concept, str)
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0
    
    def test_analyze_with_regions(self, analyzer, test_image):
        """Test analyze with image regions."""
        regions = [
            {
                "region_id": "region-1",
                "label": "top-left",
                "polygon": [[0, 0], [112, 0], [112, 112], [0, 112]]
            },
            {
                "region_id": "region-2",
                "label": "bottom-right",
                "polygon": [[112, 112], [224, 112], [224, 224], [112, 224]]
            }
        ]
        
        result = analyzer.analyze(test_image, regions=regions)
        
        assert len(result.regional_analysis) > 0
        for region_result in result.regional_analysis:
            assert "region_id" in region_result
            assert "concepts" in region_result
            assert "confidence_scores" in region_result
    
    def test_analyze_without_regions(self, analyzer, test_image):
        """Test analyze without regions (should use None default)."""
        result = analyzer.analyze(test_image, regions=None)
        
        assert isinstance(result.regional_analysis, list)


class TestServiceIntegration:
    """Integration tests for all three services working together."""
    
    def test_pipeline_services_together(self):
        """Test that all three services can work together."""
        from services.pipeline import Pipeline
        
        pipeline = Pipeline()
        
        # Generate image with Stable Diffusion
        gen_result = pipeline.generate("a beautiful sunset", 256)
        assert gen_result.generated_image is not None
        
        # Analyze with CLIP and SAM2
        assert gen_result.clip_analysis.global_concepts is not None
        assert gen_result.basic_segmentation.masks is not None
    
    def test_analyze_generated_image(self):
        """Test analyzing an image through the full pipeline."""
        from services.pipeline import Pipeline
        
        pipeline = Pipeline()
        
        # Create a simple image
        image = Image.new("RGB", (256, 256), color="purple")
        image_base64 = image_to_base64(image)
        
        # Analyze with CLIP and SAM2
        analysis = pipeline.analyze(image_base64, prompt="purple square")
        assert analysis.clip_analysis.global_concepts is not None
        assert analysis.segmentation.masks is not None
    
    def test_segment_and_analyze(self):
        """Test segmentation followed by analysis."""
        from services.pipeline import Pipeline
        
        pipeline = Pipeline()
        image = Image.new("RGB", (256, 256), color="yellow")
        
        # Segment image
        segmentation = pipeline.services.segmenter.segment(image)
        assert len(segmentation.masks) > 0
        
        # Analyze image
        analysis = pipeline.services.analyzer.analyze(image)
        assert len(analysis.global_concepts) > 0
