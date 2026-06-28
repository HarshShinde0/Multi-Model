from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from core.config import get_settings
from schemas import AnalyzeRequest, GenerateRequest, SegmentRequest
from services.pipeline import Pipeline

router = APIRouter()
pipeline = Pipeline(get_settings())
settings = get_settings()


UI_HTML = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Multimodel Studio</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Space+Mono&display=swap" rel="stylesheet">
        <style>
            :root {
                color-scheme: dark;
                --bg: #090a0f;
                --panel: #111218;
                --panel-header: #171821;
                --input-bg: #07080b;
                --text: #ffffff;
                --muted: #8a8f98;
                --cyan: #1bdae6;
                --magenta: #ff007a;
                --purple: #9b51e0;
                --border: rgba(255, 255, 255, 0.05);
                --border-active: rgba(27, 218, 230, 0.4);
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(27, 218, 230, 0.08), transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(155, 81, 224, 0.08), transparent 45%);
                background-attachment: fixed;
                letter-spacing: -0.01em;
            }
            
            /* Header Styling matching PennyLane */
            header {
                border-bottom: 1px solid var(--border);
                background: rgba(9, 10, 15, 0.8);
                backdrop-filter: blur(12px);
                position: sticky;
                top: 0;
                z-index: 100;
            }
            .header-inner {
                max-width: 1200px;
                margin: 0 auto;
                padding: 16px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .logo {
                font-weight: 700;
                font-size: 1.1rem;
                letter-spacing: 0.15em;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .logo-slashes {
                color: var(--cyan);
            }
            .nav-links {
                display: flex;
                gap: 24px;
            }
            .nav-link {
                color: var(--muted);
                text-decoration: none;
                font-size: 0.9rem;
                font-weight: 500;
                transition: color 0.2s;
            }
            .nav-link:hover {
                color: var(--cyan);
            }
            .nav-link.active {
                color: #fff;
                border-bottom: 1px solid var(--cyan);
            }

            .shell {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 24px 64px;
            }
            
            .hero {
                margin-bottom: 40px;
            }
            .eyebrow {
                color: var(--magenta);
                font-family: 'Space Mono', monospace;
                text-transform: uppercase;
                letter-spacing: 0.15em;
                font-size: 0.8rem;
                margin-bottom: 12px;
                display: inline-block;
            }
            h1 {
                margin: 0 0 16px;
                font-size: clamp(2rem, 5vw, 3rem);
                font-weight: 700;
                line-height: 1.1;
                background: linear-gradient(135deg, #fff 40%, var(--cyan) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p {
                margin: 0;
                color: var(--muted);
                line-height: 1.6;
                font-size: 1.05rem;
                max-width: 80ch;
            }
            
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }
            @media (max-width: 960px) {
                .grid { grid-template-columns: 1fr; }
            }
            
            /* Sharp containers like PennyLane */
            .card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 24px;
                display: flex;
                flex-direction: column;
                position: relative;
                overflow: hidden;
            }
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 2px;
                background: linear-gradient(90deg, var(--cyan), var(--purple));
                opacity: 0.8;
            }
            .card h2 {
                margin: 0 0 20px;
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            
            label {
                display: block;
                margin: 16px 0 8px;
                color: var(--muted);
                font-size: 0.85rem;
                text-transform: uppercase;
                font-family: 'Space Mono', monospace;
                letter-spacing: 0.05em;
            }
            
            input[type="text"], textarea, input[type="number"], input[type="file"] {
                width: 100%;
                border-radius: 6px;
                border: 1px solid var(--border);
                background: var(--input-bg);
                color: var(--text);
                padding: 12px 14px;
                font-family: inherit;
                font-size: 0.95rem;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            input[type="text"]:focus, textarea:focus, input[type="number"]:focus, input[type="file"]:focus {
                outline: none;
                border-color: var(--cyan);
                box-shadow: 0 0 0 2px rgba(27, 218, 230, 0.15);
            }
            textarea { min-height: 100px; resize: vertical; }
            
            .row {
                display: grid;
                grid-template-columns: 1fr 180px;
                gap: 16px;
            }
            
            .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 20px;
            }
            
            /* Professional solid color buttons */
            button {
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: inherit;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                color: #090a0f;
                background: var(--cyan);
                transition: background-color 0.2s, transform 0.1s, opacity 0.2s;
            }
            button:hover {
                background: #15b7c2;
                opacity: 0.95;
            }
            button:active {
                transform: translateY(1px);
            }
            button.secondary {
                background: transparent;
                color: var(--text);
                border: 1px solid var(--border);
            }
            button.secondary:hover {
                background: rgba(255, 255, 255, 0.03);
                border-color: var(--cyan);
            }
            
            .preview {
                display: grid;
                gap: 24px;
            }
            
            /* Sharp frames with solid backgrounds */
            .image-frame {
                min-height: 360px;
                border-radius: 6px;
                border: 1px solid var(--border);
                background: #06070a;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                position: relative;
            }
            .image-frame img {
                width: 100%;
                height: auto;
                display: block;
            }
            
            #downloadBtn {
                position: absolute;
                top: 16px;
                right: 16px;
                background: rgba(9, 10, 15, 0.85);
                backdrop-filter: blur(8px);
                color: var(--text);
                font-size: 0.8rem;
                padding: 8px 14px;
                border-radius: 4px;
                border: 1px solid var(--border);
                display: none;
                z-index: 10;
            }
            #downloadBtn:hover {
                border-color: var(--cyan);
                background: rgba(9, 10, 15, 0.95);
            }
            
            /* Code styling matching space mono */
            pre {
                margin: 0;
                white-space: pre-wrap;
                word-break: break-word;
                font-family: 'Space Mono', monospace;
                font-size: 0.85rem;
                background: #06070a;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 16px;
                max-height: 360px;
                overflow: auto;
                color: #c5c9db;
            }
            
            .hint {
                font-size: 0.85rem;
                color: var(--muted);
                font-family: 'Space Grotesk', sans-serif;
            }
            
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                gap: 12px;
                margin-top: 16px;
            }
            .gallery img {
                width: 100%;
                border-radius: 4px;
                border: 1px solid var(--border);
                cursor: pointer;
                transition: border-color 0.2s, transform 0.2s;
            }
            .gallery img:hover {
                border-color: var(--cyan);
                transform: scale(1.02);
            }
            
            .gallery-item {
                position: relative;
            }
            .gallery-item span {
                position: absolute;
                bottom: 6px;
                left: 6px;
                background: rgba(9, 10, 15, 0.85);
                backdrop-filter: blur(4px);
                font-size: 0.65rem;
                font-family: 'Space Mono', monospace;
                padding: 3px 6px;
                border-radius: 3px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            #fancyOutput {
                display: none;
                background: #06070a;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 20px;
            }
            
            .score-bar {
                background: var(--panel-header);
                border-radius: 2px;
                height: 6px;
                margin-top: 6px;
                overflow: hidden;
            }
            .score-fill {
                background: linear-gradient(90deg, var(--cyan), var(--purple));
                height: 100%;
            }
            
            .score-row {
                margin-bottom: 14px;
                font-size: 0.9rem;
            }
            .score-row span {
                font-weight: 500;
            }
            .score-label {
                font-family: 'Space Mono', monospace;
                font-size: 0.8rem;
            }
            
            #maskPreviewBox {
                display: none;
                margin-top: 16px;
                background: #06070a;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 16px;
            }
            
            h3 {
                font-family: 'Space Mono', monospace;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin: 0 0 12px;
                color: var(--muted);
            }
        </style>
    </head>
    <body>
        <header>
            <div class="header-inner">
                <div class="logo">
                    <span class="logo-slashes">//</span> MULTIMODEL STUDIO
                </div>
                <div class="nav-links">
                    <a href="#" class="nav-link active">Studio</a>
                    <a href="https://github.com/HarshShinde0/Multi-Model" target="_blank" class="nav-link">GitHub</a>
                </div>
            </div>
        </header>

        <div class="shell">
            <div class="hero">
                <div class="eyebrow">Interactive ML Sandbox</div>
                <h1>Text-to-Image Generation & Analysis.</h1>
                <p>
                    A high-performance workspace combining Stable Diffusion, CLIP, BLIP, and SAM2.
                    Generate visual content, classify concepts, summarize objects, and extract regional masks instantly.
                </p>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>Parameters & Inputs</h2>
                    
                    <label for="prompt">Generation Prompt / Probe Classes</label>
                    <textarea id="prompt">a red kite flying over a coastal cliff</textarea>
                    
                    <div class="row">
                        <div>
                            <label for="imageSize">Output Dimensions</label>
                            <input id="imageSize" type="number" min="64" max="1024" step="64" value="512" />
                        </div>
                        <div>
                            <label>&nbsp;</label>
                            <button id="generateBtn" type="button" style="width: 100%">Run Generation</button>
                        </div>
                    </div>

                    <h2 style="margin-top: 40px;">Analysis & Segmentation</h2>
                    <label for="imageInput">Source Image (Upload)</label>
                    <input id="imageInput" type="file" accept="image/*" />

                    <div class="actions">
                        <button id="analyzeBtn" class="secondary" type="button">Run CLIP/BLIP Analysis</button>
                        <button id="segmentBtn" class="secondary" type="button">Run SAM2 Segmentation</button>
                    </div>
                </div>

                <div class="preview">
                    <div class="card">
                        <h2>Output Canvas</h2>
                        <div class="image-frame">
                            <button id="downloadBtn" type="button">Download Asset</button>
                            <img id="previewImage" alt="Generated preview" style="display:none;" />
                            <span id="previewEmpty" class="hint">No active render. Trigger generation or upload an image.</span>
                        </div>
                        
                        <div id="maskPreviewBox">
                            <h3>Segmentation Masks (SAM2)</h3>
                            <div id="maskGallery" class="gallery"></div>
                        </div>
                        
                        <div id="gallery" class="gallery"></div>
                    </div>
                    
                    <div class="card">
                        <h2>Inspection Console</h2>
                        <pre id="output">Ready for input.</pre>
                        <div id="fancyOutput"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const output = document.getElementById('output');
            const fancyOutput = document.getElementById('fancyOutput');
            const previewImage = document.getElementById('previewImage');
            const previewEmpty = document.getElementById('previewEmpty');
            const imageInput = document.getElementById('imageInput');
            const downloadBtn = document.getElementById('downloadBtn');
            const gallery = document.getElementById('gallery');
            const maskPreviewBox = document.getElementById('maskPreviewBox');
            const maskGallery = document.getElementById('maskGallery');
            
            let currentImageBase64 = null;

            function showOutput(value) {
                fancyOutput.style.display = 'none';
                output.style.display = 'block';
                output.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
            }

            function showAnalyzeOutput(data) {
                output.style.display = 'none';
                fancyOutput.style.display = 'block';
                
                let html = `<div style="margin-bottom: 20px;">
                    <div class="score-label" style="color: var(--muted); margin-bottom: 4px;">BEST MATCH</div>
                    <span style="font-size: 1.4rem; font-weight: 700; color: var(--cyan);">${data.best_match}</span>
                </div>`;
                
                if (data.summary) {
                    html += `<div style="margin-bottom: 24px;">
                        <div class="score-label" style="color: var(--muted); margin-bottom: 6px;">NATURAL LANGUAGE SUMMARY</div>
                        <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: #fff;">${data.summary}</p>
                    </div>`;
                }
                
                html += `<div class="score-label" style="color: var(--muted); margin-bottom: 12px;">CONCEPT CONFIDENCE SCORES</div>`;
                
                const sorted = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
                for (const [label, score] of sorted) {
                    const pct = (score * 100).toFixed(1);
                    html += `
                    <div class="score-row">
                        <div style="display:flex; justify-content:space-between">
                            <span style="font-family: 'Space Mono', monospace; font-size: 0.85rem;">${label}</span>
                            <span style="color: var(--cyan); font-family: 'Space Mono', monospace; font-size: 0.85rem;">${pct}%</span>
                        </div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${pct}%"></div>
                        </div>
                    </div>`;
                }
                fancyOutput.innerHTML = html;
            }

            function showImage(base64) {
                currentImageBase64 = base64;
                previewImage.src = `data:image/png;base64,${base64}`;
                previewImage.style.display = 'block';
                previewEmpty.style.display = 'none';
                downloadBtn.style.display = 'block';
            }
            
            function renderGallery(regions) {
                gallery.innerHTML = '';
                if (!regions || regions.length === 0) return;
                
                regions.forEach((region, i) => {
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${region.image_base64}`;
                    img.title = `Click to view region ${i+1}`;
                    img.onclick = () => showImage(region.image_base64);
                    
                    const span = document.createElement('span');
                    span.textContent = `Region ${i+1}`;
                    
                    div.appendChild(img);
                    div.appendChild(span);
                    gallery.appendChild(div);
                });
            }
            
            function renderMasks(masks) {
                maskGallery.innerHTML = '';
                if (!masks || masks.length === 0) {
                    maskPreviewBox.style.display = 'none';
                    return;
                }
                maskPreviewBox.style.display = 'block';
                
                masks.forEach((maskBase64, i) => {
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${maskBase64}`;
                    img.title = `Click to view mask ${i+1}`;
                    img.onclick = () => showImage(maskBase64);
                    
                    const span = document.createElement('span');
                    span.textContent = `Mask ${i+1}`;
                    
                    div.appendChild(img);
                    div.appendChild(span);
                    maskGallery.appendChild(div);
                });
            }

            downloadBtn.addEventListener('click', () => {
                if (!currentImageBase64) return;
                const a = document.createElement('a');
                a.href = `data:image/png;base64,${currentImageBase64}`;
                a.download = 'image.png';
                a.click();
            });

            async function fileToBase64(file) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const result = String(reader.result || '');
                        resolve(result.split(',')[1] || '');
                    };
                    reader.onerror = () => reject(reader.error);
                    reader.readAsDataURL(file);
                });
            }

            document.getElementById('generateBtn').addEventListener('click', async () => {
                showOutput('Generating image via Stable Diffusion...');
                gallery.innerHTML = '';
                maskPreviewBox.style.display = 'none';
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: document.getElementById('prompt').value,
                        image_size: Number(document.getElementById('imageSize').value),
                    }),
                });
                const body = await response.json();
                showOutput(body);
                if (body.generated_image) {
                    showImage(body.generated_image);
                }
            });

            document.getElementById('analyzeBtn').addEventListener('click', async () => {
                const file = imageInput.files && imageInput.files[0];
                if (!file) {
                    showOutput('Choose a source image first.');
                    return;
                }
                showOutput('Running CLIP classifications & BLIP captions...');
                gallery.innerHTML = '';
                maskPreviewBox.style.display = 'none';
                const image_base64 = await fileToBase64(file);
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_base64, prompt: document.getElementById('prompt').value }),
                });
                const body = await response.json();
                
                if (body.clip_analysis && body.clip_analysis.confidence_scores) {
                    const data = {
                        best_match: body.clip_analysis.global_concepts[0] || 'Unknown',
                        scores: body.clip_analysis.confidence_scores,
                        summary: body.clip_analysis.summary
                    };
                    showAnalyzeOutput(data);
                } else {
                    showOutput(body);
                }
                showImage(image_base64);
            });

            document.getElementById('segmentBtn').addEventListener('click', async () => {
                const file = imageInput.files && imageInput.files[0];
                if (!file) {
                    showOutput('Choose a source image first.');
                    return;
                }
                showOutput('Extracting segments via SAM2...');
                gallery.innerHTML = '';
                const image_base64 = await fileToBase64(file);
                const response = await fetch('/segment', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_base64 }),
                });
                const body = await response.json();
                
                const displayBody = JSON.parse(JSON.stringify(body));
                
                if (displayBody.segmentation) {
                    if (displayBody.segmentation.masks) {
                        displayBody.segmentation.masks = displayBody.segmentation.masks.map(m => '<base64...>');
                    }
                    if (displayBody.segmentation.segmented_regions) {
                        displayBody.segmentation.segmented_regions.forEach(r => r.image_base64 = '<base64...>');
                    }
                }
                
                showOutput(displayBody);
                showImage(image_base64);
                
                if (body.segmentation && body.segmentation.masks) {
                    renderMasks(body.segmentation.masks);
                }
                
                if (body.segmentation && body.segmentation.segmented_regions) {
                    renderGallery(body.segmentation.segmented_regions);
                }
            });
        </script>
    </body>
</html>
"""


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@router.post("/generate")
async def generate(request: GenerateRequest):
    return await asyncio.wait_for(
        run_in_threadpool(pipeline.generate, request.prompt, request.image_size),
        timeout=settings.api_timeout_seconds,
    )


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        return await asyncio.wait_for(
            run_in_threadpool(pipeline.analyze, request.image_base64, request.prompt),
            timeout=settings.api_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/segment")
async def segment(request: SegmentRequest):
    try:
        return await asyncio.wait_for(
            run_in_threadpool(pipeline.segment, request.image_base64),
            timeout=settings.api_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
