// DOM Elements
const fetchMastBtn = document.getElementById('fetch-mast-btn');
const runHpcBtn = document.getElementById('run-hpc-btn');
const loadDbBtn = document.getElementById('load-db-btn');
const targetInput = document.getElementById('target-input');
const mastStatus = document.getElementById('mast-status');
const hpcTelemetry = document.getElementById('hpc-telemetry');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const canvas = document.getElementById('deep-field-canvas');
const ctx = canvas.getContext('2d');
const aiFeed = document.getElementById('ai-feed');
const dbFeed = document.getElementById('db-feed');

// State
let currentImage = null;
let currentSources = [];
let currentScale = 1;
let currentTargetName = "";
let currentTargetRa = 0;
let currentTargetDec = 0;

// Resize canvas to fit container
function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawImageAndSources();
}
window.addEventListener('resize', resizeCanvas);

function showLoading(msg) {
    loadingText.innerText = msg;
    loadingOverlay.classList.remove('hidden');
}
function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// 1. Fetch from MAST
fetchMastBtn.addEventListener('click', async () => {
    const target = targetInput.value.trim();
    if (!target) return;
    
    showLoading("UPLINK TO MAST API...");
    mastStatus.innerText = `Querying STScI for ${target}...`;
    
    try {
        const res = await fetch('/api/mast/fetch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ target_name: target })
        });
        
        if (!res.ok) throw new Error("Target not found in JWST Archive");
        const data = await res.json();
        
        currentTargetName = data.target;
        currentTargetRa = data.resolved_ra;
        currentTargetDec = data.resolved_dec;
        
        const obs = data.best_observation;
        mastStatus.innerHTML = `
            <strong>TARGET ACQUIRED:</strong> ${target}<br>
            <strong>RA/DEC:</strong> ${data.resolved_ra.toFixed(4)}, ${data.resolved_dec.toFixed(4)}<br>
            <strong>OBS ID:</strong> ${obs.obs_id}<br>
            <strong>EXPOSURE:</strong> ${(obs.exposure_time / 3600).toFixed(1)} hrs
        `;
        
        // Load the image onto canvas
        let imgUrl = obs.jpeg_url;
        if (imgUrl.startsWith("mast:")) {
            imgUrl = imgUrl.replace("mast:", "https://mast.stsci.edu/api/v0/download/file/");
        }
        
        showLoading("DOWNLOADING HIGH-RES DATA...");
        
        currentImage = new Image();
        currentImage.crossOrigin = "Anonymous";
        currentImage.onload = () => {
            resizeCanvas();
            hideLoading();
            runHpcBtn.disabled = false;
        };
        // For hackathon fallback, if MAST jpeg fails due to CORS, use synthetic /api/deepfield
        currentImage.onerror = async () => {
            mastStatus.innerHTML += "<br><span class='gold'>MAST image blocked by browser CORS. Falling back to synthetic HPC payload.</span>";
            const backupRes = await fetch('/api/deepfield');
            const backupData = await backupRes.json();
            currentImage.src = backupData.mosaic_b64;
        };
        currentImage.src = imgUrl;
        
    } catch (e) {
        mastStatus.innerHTML = `<span class="alert-red">ERROR: ${e.message}</span>`;
        hideLoading();
    }
});

// 2. Run HPC Extraction
runHpcBtn.addEventListener('click', async () => {
    showLoading("INITIALIZING PARALLEL HPC CORE...");
    
    try {
        // Trigger run
        const runRes = await fetch('/api/hpc/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ threshold_sigma: 3.2, cutout_size: 64 })
        });
        const runData = await runRes.json();
        
        // Get sources
        showLoading("EXTRACTING SOURCE CATALOG...");
        const srcRes = await fetch('/api/sources');
        const srcData = await srcRes.json();
        
        currentSources = srcData.sources;
        
        // Update telemetry
        hpcTelemetry.classList.remove('hidden');
        document.getElementById('tel-sources').innerText = runData.sources_count;
        document.getElementById('tel-workers').innerText = runData.telemetry.num_workers;
        document.getElementById('tel-time').innerText = runData.telemetry.execution_time_ms.toFixed(0) + 'ms';
        
        drawImageAndSources();
        hideLoading();
        
    } catch (e) {
        console.error(e);
        hideLoading();
    }
});

function drawImageAndSources() {
    if (!currentImage) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scale to fit
    const scaleX = canvas.width / currentImage.width;
    const scaleY = canvas.height / currentImage.height;
    currentScale = Math.min(scaleX, scaleY);
    
    const x = (canvas.width - currentImage.width * currentScale) / 2;
    const y = (canvas.height - currentImage.height * currentScale) / 2;
    
    ctx.drawImage(currentImage, x, y, currentImage.width * currentScale, currentImage.height * currentScale);
    
    // Draw bounding boxes
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.5;
    
    currentSources.forEach(src => {
        const [bx, by, bw, bh] = src.bbox;
        const drawX = x + (bx * currentScale);
        const drawY = y + (by * currentScale);
        const drawW = bw * currentScale;
        const drawH = bh * currentScale;
        
        ctx.strokeRect(drawX, drawY, drawW, drawH);
    });
}

// 3. Canvas Click -> Gemini AI
canvas.addEventListener('click', async (e) => {
    if (!currentImage || currentSources.length === 0) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const x = (canvas.width - currentImage.width * currentScale) / 2;
    const y = (canvas.height - currentImage.height * currentScale) / 2;
    
    // Find clicked source
    const imgX = (mouseX - x) / currentScale;
    const imgY = (mouseY - y) / currentScale;
    
    const clickedSource = currentSources.find(src => {
        const [bx, by, bw, bh] = src.bbox;
        return imgX >= bx && imgX <= bx + bw && imgY >= by && imgY <= by + bh;
    });
    
    if (clickedSource) {
        aiFeed.innerHTML = `<div class="empty-state">
            <div class="spinner" style="width:30px;height:30px;margin:0 auto;"></div>
            <br>Gemini 2.0 Analyzing Source ${clickedSource.id}...
        </div>`;
        
        try {
            const res = await fetch(`/api/gemini/analyze/${clickedSource.id}`, { method: 'POST' });
            const data = await res.json();
            
            const ai = data.analysis;
            
            // Format report
            aiFeed.innerHTML = `
                <div class="ai-card" id="current-discovery">
                    <h4>ID: ${clickedSource.id}</h4>
                    <p><strong style="color:var(--neon-blue)">CLASS:</strong> ${ai.classification}</p>
                    <p><strong style="color:var(--jwst-gold)">CONFIDENCE:</strong> ${ai.confidence}%</p>
                    <p class="mt-2">${ai.reasoning}</p>
                    <button class="save-btn" onclick='saveDiscovery(${JSON.stringify(ai).replace(/'/g, "\\'")})'>SAVE TO DATABASE</button>
                </div>
            `;
            
        } catch (e) {
            aiFeed.innerHTML = `<div class="empty-state" style="color:var(--alert-red)">Analysis failed: ${e.message}</div>`;
        }
    }
});

// 4. Database Operations
window.saveDiscovery = async function(aiData) {
    try {
        const payload = {
            target_name: currentTargetName || "Unknown Deep Field",
            ra: currentTargetRa || 0,
            dec: currentTargetDec || 0,
            classification: aiData.classification,
            report_text: aiData.reasoning,
            image_url: ""
        };
        
        const res = await fetch('/api/discoveries', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        if (res.ok) {
            alert("Discovery Saved to SQLite Database!");
            loadDbBtn.click();
        }
    } catch (e) {
        alert("Failed to save: " + e.message);
    }
}

loadDbBtn.addEventListener('click', async () => {
    dbFeed.innerHTML = `<div class="empty-state">Loading Archive...</div>`;
    
    try {
        const res = await fetch('/api/discoveries');
        const data = await res.json();
        
        if (data.length === 0) {
            dbFeed.innerHTML = `<div class="empty-state">No discoveries logged yet.</div>`;
            return;
        }
        
        dbFeed.innerHTML = '';
        data.forEach(d => {
            const date = new Date(d.created_at).toLocaleString();
            dbFeed.innerHTML += `
                <div class="db-card">
                    <h4>${d.target_name}</h4>
                    <p style="font-size: 0.75rem; color:#64748b;">${date}</p>
                    <p class="mt-2"><strong style="color:#4ade80">${d.classification}</strong></p>
                    <p>${d.report_text.substring(0, 80)}...</p>
                </div>
            `;
        });
    } catch (e) {
        dbFeed.innerHTML = `<div class="empty-state" style="color:var(--alert-red)">Failed to load DB.</div>`;
    }
});

// Init
resizeCanvas();
