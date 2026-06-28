from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from core.config import get_settings
from schemas import AnalyzeRequest, GenerateRequest, ProbeRequest, SegmentRequest, VisualizeRequest
from services.pipeline import Pipeline

logger = logging.getLogger("api")
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

                    <details style="margin-top: 16px; border: 1px solid var(--border); padding: 12px; border-radius: 6px;">
                        <summary style="cursor: pointer; font-family: 'Space Mono', monospace; font-size: 0.8rem; color: var(--muted);">Advanced SAM2 Settings</summary>
                        <div style="margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <div>
                                <label for="pointsPerSide">Points Per Side</label>
                                <input id="pointsPerSide" type="number" min="4" max="64" value="32" style="padding: 6px; font-size: 0.85rem;" />
                            </div>
                            <div>
                                <label for="predIouThresh">IoU Thresh</label>
                                <input id="predIouThresh" type="number" min="0" max="1" step="0.05" value="0.88" style="padding: 6px; font-size: 0.85rem;" />
                            </div>
                            <div>
                                <label for="stabilityThresh">Stability Thresh</label>
                                <input id="stabilityThresh" type="number" min="0" max="1" step="0.05" value="0.95" style="padding: 6px; font-size: 0.85rem;" />
                            </div>
                            <div>
                                <label for="cropLayers">Crop Layers</label>
                                <input id="cropLayers" type="number" min="0" max="4" value="0" style="padding: 6px; font-size: 0.85rem;" />
                            </div>
                        </div>
                    </details>

                    <h2 style="margin-top: 30px;">Visualization & Probing</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                        <div>
                            <label for="vizType">Viz Type</label>
                            <select id="vizType" style="width: 100%; border-radius: 6px; border: 1px solid var(--border); background: var(--input-bg); color: var(--text); padding: 12px 14px; font-family: inherit; font-size: 0.95rem;">
                                <option value="overlay">Overlay + Contours</option>
                                <option value="contour">Contours Only</option>
                                <option value="bbox">Bounding Boxes</option>
                                <option value="isolate">Isolate Segments</option>
                            </select>
                        </div>
                        <div>
                            <label for="vizAlpha">Opacity (Alpha)</label>
                            <input id="vizAlpha" type="number" min="0" max="1" step="0.1" value="0.4" style="padding: 12px 14px;" />
                        </div>
                    </div>
                    <button id="vizBtn" class="secondary" type="button" style="width: 100%; margin-bottom: 20px;">Generate Visualization</button>

                    <label for="probeConcepts">Concept Probing (Comma Separated)</label>
                    <textarea id="probeConcepts" style="min-height: 60px; margin-bottom: 12px;" placeholder="e.g. grass, sky, bird, kite, rock"></textarea>
                    <button id="probeBtn" class="secondary" type="button" style="width: 100%;">Probe Custom Concepts</button>
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
                fancyOutput.replaceChildren(); // Safe clear
                
                // Best match section
                const bmDiv = document.createElement('div');
                bmDiv.style.marginBottom = '20px';
                
                const bmLbl = document.createElement('div');
                bmLbl.className = 'score-label';
                bmLbl.style.color = 'var(--muted)';
                bmLbl.style.marginBottom = '4px';
                bmLbl.textContent = 'BEST MATCH';
                bmDiv.appendChild(bmLbl);
                
                const bmVal = document.createElement('span');
                bmVal.style.fontSize = '1.4rem';
                bmVal.style.fontWeight = '700';
                bmVal.style.color = 'var(--cyan)';
                bmVal.textContent = data.best_match;
                bmDiv.appendChild(bmVal);
                
                fancyOutput.appendChild(bmDiv);
                
                // Summary section
                if (data.summary) {
                    const sumDiv = document.createElement('div');
                    sumDiv.style.marginBottom = '24px';
                    
                    const sumLbl = document.createElement('div');
                    sumLbl.className = 'score-label';
                    sumLbl.style.color = 'var(--muted)';
                    sumLbl.style.marginBottom = '6px';
                    sumLbl.textContent = 'NATURAL LANGUAGE SUMMARY';
                    sumDiv.appendChild(sumLbl);
                    
                    const sumVal = document.createElement('p');
                    sumVal.style.margin = '0';
                    sumVal.style.fontSize = '0.95rem';
                    sumVal.style.lineHeight = '1.5';
                    sumVal.style.color = '#fff';
                    sumVal.textContent = data.summary;
                    sumDiv.appendChild(sumVal);
                    
                    fancyOutput.appendChild(sumDiv);
                }
                
                const scoresLbl = document.createElement('div');
                scoresLbl.className = 'score-label';
                scoresLbl.style.color = 'var(--muted)';
                scoresLbl.style.marginBottom = '12px';
                scoresLbl.textContent = 'CONCEPT CONFIDENCE SCORES';
                fancyOutput.appendChild(scoresLbl);
                
                const sorted = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
                for (const [label, score] of sorted) {
                    const pct = (score * 100).toFixed(1);
                    
                    const row = document.createElement('div');
                    row.className = 'score-row';
                    
                    const info = document.createElement('div');
                    info.style.display = 'flex';
                    info.style.justifyContent = 'space-between';
                    
                    const lblSpan = document.createElement('span');
                    lblSpan.style.fontFamily = "'Space Mono', monospace";
                    lblSpan.style.fontSize = '0.85rem';
                    lblSpan.textContent = label;
                    
                    const pctSpan = document.createElement('span');
                    pctSpan.style.color = 'var(--cyan)';
                    pctSpan.style.fontFamily = "'Space Mono', monospace";
                    pctSpan.style.fontSize = '0.85rem';
                    pctSpan.textContent = pct + '%';
                    
                    info.appendChild(lblSpan);
                    info.appendChild(pctSpan);
                    row.appendChild(info);
                    
                    const bar = document.createElement('div');
                    bar.className = 'score-bar';
                    
                    const fill = document.createElement('div');
                    fill.className = 'score-fill';
                    fill.style.width = pct + '%';
                    
                    bar.appendChild(fill);
                    row.appendChild(bar);
                    
                    fancyOutput.appendChild(row);
                }
            }

            function showImage(base64) {
                currentImageBase64 = base64;
                previewImage.src = `data:image/png;base64,${base64}`;
                previewImage.style.display = 'block';
                previewEmpty.style.display = 'none';
                downloadBtn.style.display = 'block';
            }
            
            function renderGallery(regions) {
                gallery.replaceChildren();
                if (!regions || regions.length === 0) return;
                
                regions.forEach((region, i) => {
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${region.image_base64}`;
                    img.title = `Click to view region ${i+1}`;
                    img.onclick = () => {
                        showImage(region.image_base64);
                        if (region.clip_analysis && region.clip_analysis.confidence_scores) {
                            const data = {
                                best_match: (region.clip_analysis.global_concepts && region.clip_analysis.global_concepts[0]) || 'Unknown',
                                scores: region.clip_analysis.confidence_scores,
                                summary: `Region ${i+1} bounding box: ${JSON.stringify(region.bbox)}.\nPolygon vertices: ${region.polygon.length}`
                            };
                            showAnalyzeOutput(data);
                        }
                    };
                    
                    const span = document.createElement('span');
                    span.textContent = `Region ${i+1}`;
                    
                    div.appendChild(img);
                    div.appendChild(span);
                    gallery.appendChild(div);
                });
            }
            
            function renderMasks(masks) {
                maskGallery.replaceChildren();
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
                gallery.replaceChildren();
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
                gallery.replaceChildren();
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
                showOutput('Extracting segments via SAM2 with settings...');
                gallery.replaceChildren();
                const image_base64 = await fileToBase64(file);
                
                const points_per_side = document.getElementById('pointsPerSide').value ? Number(document.getElementById('pointsPerSide').value) : null;
                const pred_iou_thresh = document.getElementById('predIouThresh').value ? Number(document.getElementById('predIouThresh').value) : null;
                const stability_score_thresh = document.getElementById('stabilityThresh').value ? Number(document.getElementById('stabilityThresh').value) : null;
                const crop_n_layers = document.getElementById('cropLayers').value ? Number(document.getElementById('cropLayers').value) : null;
                
                const response = await fetch('/segment', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_base64,
                        points_per_side,
                        pred_iou_thresh,
                        stability_score_thresh,
                        crop_n_layers
                    }),
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
                    if (displayBody.segmentation.overlay_image) {
                        displayBody.segmentation.overlay_image = '<base64...>';
                    }
                }
                
                showOutput(displayBody);
                
                if (body.segmentation && body.segmentation.overlay_image) {
                    showImage(body.segmentation.overlay_image);
                } else {
                    showImage(image_base64);
                }
                
                if (body.segmentation && body.segmentation.masks) {
                    renderMasks(body.segmentation.masks);
                }
                
                if (body.segmentation && body.segmentation.segmented_regions) {
                    renderGallery(body.segmentation.segmented_regions);
                }
            });

            document.getElementById('vizBtn').addEventListener('click', async () => {
                const file = imageInput.files && imageInput.files[0];
                if (!file && !currentImageBase64) {
                    showOutput('Choose a source image or run generation first.');
                    return;
                }
                showOutput('Generating advanced visualization overlay...');
                
                let image_base64 = currentImageBase64;
                if (file) {
                    image_base64 = await fileToBase64(file);
                }
                
                const visualization_type = document.getElementById('vizType').value;
                const alpha = Number(document.getElementById('vizAlpha').value);
                
                const response = await fetch('/visualize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_base64,
                        visualization_type,
                        alpha,
                        line_thickness: 2
                    }),
                });
                const body = await response.json();
                showOutput(body);
                if (body.visualization_image) {
                    showImage(body.visualization_image);
                }
            });

            document.getElementById('probeBtn').addEventListener('click', async () => {
                const file = imageInput.files && imageInput.files[0];
                if (!file && !currentImageBase64) {
                    showOutput('Choose a source image or run generation first.');
                    return;
                }
                const conceptsText = document.getElementById('probeConcepts').value.trim();
                if (!conceptsText) {
                    showOutput('Please enter some concepts to probe.');
                    return;
                }
                
                const concepts = conceptsText.split(',').map(c => c.trim()).filter(c => c.length > 0);
                if (concepts.length === 0) {
                    showOutput('Please enter at least one valid concept.');
                    return;
                }
                
                showOutput('Running concept probing on global image and individual regions...');
                
                let image_base64 = currentImageBase64;
                if (file) {
                    image_base64 = await fileToBase64(file);
                }
                
                const response = await fetch('/probe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_base64, concepts }),
                });
                const body = await response.json();
                
                // Show raw JSON output first
                showOutput(body);
                
                // Render a beautiful interactive table for probing results
                if (body.global_scores && body.regional_scores) {
                    output.style.display = 'none';
                    fancyOutput.style.display = 'block';
                    fancyOutput.replaceChildren();
                    
                    const h3 = document.createElement('h3');
                    h3.textContent = 'Concept Probing Results';
                    fancyOutput.appendChild(h3);
                    
                    const table = document.createElement('table');
                    table.style.width = '100%';
                    table.style.borderCollapse = 'collapse';
                    table.style.marginTop = '12px';
                    table.style.fontSize = '0.85rem';
                    table.style.fontFamily = "'Space Mono', monospace";
                    
                    const header = document.createElement('tr');
                    header.style.borderBottom = '1px solid var(--border)';
                    header.style.textAlign = 'left';
                    
                    const thConcept = document.createElement('th');
                    thConcept.style.padding = '8px';
                    thConcept.textContent = 'Concept';
                    header.appendChild(thConcept);
                    
                    const thGlobal = document.createElement('th');
                    thGlobal.style.padding = '8px';
                    thGlobal.textContent = 'Global';
                    header.appendChild(thGlobal);
                    
                    body.regional_scores.forEach(reg => {
                        const thReg = document.createElement('th');
                        thReg.style.padding = '8px';
                        thReg.textContent = reg.region_id;
                        header.appendChild(thReg);
                    });
                    table.appendChild(header);
                    
                    concepts.forEach(concept => {
                        const tr = document.createElement('tr');
                        tr.style.borderBottom = '1px solid rgba(255,255,255,0.02)';
                        
                        const tdConcept = document.createElement('td');
                        tdConcept.style.padding = '8px';
                        tdConcept.textContent = concept;
                        tr.appendChild(tdConcept);
                        
                        const tdGlobal = document.createElement('td');
                        tdGlobal.style.padding = '8px';
                        tdGlobal.style.color = 'var(--cyan)';
                        const gScore = body.global_scores[concept] !== undefined ? (body.global_scores[concept] * 100).toFixed(1) + '%' : '0.0%';
                        tdGlobal.textContent = gScore;
                        tr.appendChild(tdGlobal);
                        
                        body.regional_scores.forEach(reg => {
                            const tdReg = document.createElement('td');
                            tdReg.style.padding = '8px';
                            const rScore = reg.scores[concept] !== undefined ? (reg.scores[concept] * 100).toFixed(1) + '%' : '0.0%';
                            tdReg.textContent = rScore;
                            tr.appendChild(tdReg);
                        });
                        table.appendChild(tr);
                    });
                    
                    fancyOutput.appendChild(table);
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
    logger.info(f"Endpoint /generate triggered with prompt: '{request.prompt}', size: {request.image_size}")
    try:
        res = await asyncio.wait_for(
            run_in_threadpool(pipeline.generate, request.prompt, request.image_size),
            timeout=settings.api_timeout_seconds,
        )
        logger.info("Generation and basic analysis successfully finished")
        return res
    except asyncio.TimeoutError as exc:
        logger.error("Generation request timed out")
        raise HTTPException(status_code=504, detail="Request timed out") from exc
    except Exception as exc:
        logger.error(f"Generation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    logger.info("Endpoint /analyze triggered")
    try:
        res = await asyncio.wait_for(
            run_in_threadpool(pipeline.analyze, request.image_base64, request.prompt),
            timeout=settings.api_timeout_seconds,
        )
        logger.info("CLIP analysis and basic segmentation successfully finished")
        return res
    except asyncio.TimeoutError as exc:
        logger.error("Analysis request timed out")
        raise HTTPException(status_code=504, detail="Request timed out") from exc
    except ValueError as exc:
        logger.error(f"Validation/parsing error during analysis: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/segment")
async def segment(request: SegmentRequest):
    logger.info(f"Endpoint /segment triggered with settings: points_per_side={request.points_per_side}")
    try:
        res = await asyncio.wait_for(
            run_in_threadpool(
                pipeline.segment,
                image_base64=request.image_base64,
                points_per_side=request.points_per_side,
                pred_iou_thresh=request.pred_iou_thresh,
                stability_score_thresh=request.stability_score_thresh,
                crop_n_layers=request.crop_n_layers,
                box_nms_thresh=request.box_nms_thresh,
            ),
            timeout=settings.api_timeout_seconds,
        )
        logger.info("Segmentation successfully finished")
        return res
    except asyncio.TimeoutError as exc:
        logger.error("Segmentation request timed out")
        raise HTTPException(status_code=504, detail="Request timed out") from exc
    except ValueError as exc:
        logger.error(f"Validation/parsing error during segmentation: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Segmentation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/probe")
async def probe(request: ProbeRequest):
    logger.info(f"Endpoint /probe triggered for concepts: {request.concepts}")
    try:
        res = await asyncio.wait_for(
            run_in_threadpool(pipeline.probe, request.image_base64, request.concepts),
            timeout=settings.api_timeout_seconds,
        )
        logger.info("Concept probing successfully finished")
        return res
    except asyncio.TimeoutError as exc:
        logger.error("Probing request timed out")
        raise HTTPException(status_code=504, detail="Request timed out") from exc
    except ValueError as exc:
        logger.error(f"Validation/parsing error during concept probing: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Concept probing error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/visualize")
async def visualize(request: VisualizeRequest):
    logger.info(f"Endpoint /visualize triggered with type: {request.visualization_type}")
    try:
        res = await asyncio.wait_for(
            run_in_threadpool(
                pipeline.visualize,
                request.image_base64,
                request.visualization_type,
                request.alpha,
                request.line_thickness,
            ),
            timeout=settings.api_timeout_seconds,
        )
        logger.info("Visualization successfully finished")
        return res
    except asyncio.TimeoutError as exc:
        logger.error("Visualization request timed out")
        raise HTTPException(status_code=504, detail="Request timed out") from exc
    except ValueError as exc:
        logger.error(f"Validation/parsing error during visualization: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Visualization error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
