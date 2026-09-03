/**
 * CosmoLens HPC: Interactive Mission Control & Gemini AI Astronomical Viewer
 */

// Application State
const state = {
  sources: [],
  selectedSourceId: null,
  currentAnalysis: null,
  deepFieldImage: null,
  header: {},
  telemetry: {},
  filter: 'ALL',
  showBoxes: true,
  showGrid: true,
  
  // Canvas Viewport transform
  scale: 1.0,
  panX: 0,
  panY: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0
};

// DOM References
const canvas = document.getElementById('deepFieldCanvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvasContainer');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) lucide.createIcons();
  setupCanvas();
  setupEventListeners();
  await loadInitialData();
});

function setupCanvas() {
  const resize = () => {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawScene();
  };
  window.addEventListener('resize', resize);
  resize();
}

function setupEventListeners() {
  // Canvas Mouse Navigation
  canvas.addEventListener('mousedown', (e) => {
    state.isDragging = true;
    state.dragStartX = e.clientX - state.panX;
    state.dragStartY = e.clientY - state.panY;
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isDragging) {
      state.panX = e.clientX - state.dragStartX;
      state.panY = e.clientY - state.dragStartY;
      drawScene();
    }
    updateCoordinatesHud(e);
  });

  window.addEventListener('mouseup', () => {
    state.isDragging = false;
  });

  // Canvas Wheel Zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    zoomAtPoint(e.clientX, e.clientY, zoomFactor);
  }, { passive: false });

  // Canvas Click Detection
  canvas.addEventListener('click', (e) => {
    if (Math.abs(e.clientX - (state.dragStartX + state.panX)) > 4 ||
        Math.abs(e.clientY - (state.dragStartY + state.panY)) > 4) {
      return; // Dragged, not clicked
    }
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert screen coordinates to deep-field image coordinates
    const imgX = (mouseX - state.panX) / state.scale;
    const imgY = (mouseY - state.panY) / state.scale;

    // Find closest source within detection radius
    let closest = null;
    let minDist = 25 / state.scale; // 25 px radius

    for (const s of state.sources) {
      const dist = Math.sqrt((s.x - imgX) ** 2 + (s.y - imgY) ** 2);
      if (dist < minDist) {
        minDist = dist;
        closest = s;
      }
    }

    if (closest) {
      selectSource(closest.id);
    }
  });

  // Viewport Control Buttons
  document.getElementById('zoomInBtn').addEventListener('click', () => {
    zoomAtPoint(canvas.width / 2, canvas.height / 2, 1.25);
  });
  document.getElementById('zoomOutBtn').addEventListener('click', () => {
    zoomAtPoint(canvas.width / 2, canvas.height / 2, 0.8);
  });
  document.getElementById('resetViewBtn').addEventListener('click', resetView);

  const toggleBoxesBtn = document.getElementById('toggleBoxesBtn');
  toggleBoxesBtn.addEventListener('click', () => {
    state.showBoxes = !state.showBoxes;
    toggleBoxesBtn.classList.toggle('active', state.showBoxes);
    drawScene();
  });

  const toggleGridBtn = document.getElementById('toggleGridBtn');
  toggleGridBtn.addEventListener('click', () => {
    state.showGrid = !state.showGrid;
    toggleGridBtn.classList.toggle('active', state.showGrid);
    drawScene();
  });

  // Filter Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.filter = chip.dataset.filter;
      renderCatalog();
      drawScene();
    });
  });

  // Action Buttons
  document.getElementById('runHpcBtn').addEventListener('click', runHpcPipeline);
  document.getElementById('batchClassifyBtn').addEventListener('click', runBatchClassify);
  document.getElementById('searchBtn').addEventListener('click', executeSemanticSearch);
  document.getElementById('semanticSearchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') executeSemanticSearch();
  });

  // Modals
  const keyModal = document.getElementById('keyModal');
  document.getElementById('configKeyBtn').addEventListener('click', () => keyModal.classList.add('active'));
  document.getElementById('closeKeyModal').addEventListener('click', () => keyModal.classList.remove('active'));
  document.getElementById('saveKeyBtn').addEventListener('click', saveGeminiKey);

  const reportModal = document.getElementById('reportModal');
  document.getElementById('closeReportModal').addEventListener('click', () => reportModal.classList.remove('active'));
  document.getElementById('copyMemoBtn').addEventListener('click', () => {
    const text = document.getElementById('reportContent').innerText;
    navigator.clipboard.writeText(text);
    alert('Discovery Memo copied to clipboard!');
  });
}

function zoomAtPoint(clientX, clientY, factor) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = clientX - rect.left;
  const mouseY = clientY - rect.top;

  const newScale = Math.min(Math.max(state.scale * factor, 0.25), 10.0);
  state.panX = mouseX - (mouseX - state.panX) * (newScale / state.scale);
  state.panY = mouseY - (mouseY - state.panY) * (newScale / state.scale);
  state.scale = newScale;

  document.getElementById('hudZoom').textContent = `Zoom: ${Math.round(state.scale * 100)}%`;
  drawScene();
}

function resetView() {
  if (!state.deepFieldImage) return;
  const scaleX = (canvas.width * 0.95) / state.deepFieldImage.width;
  const scaleY = (canvas.height * 0.95) / state.deepFieldImage.height;
  state.scale = Math.min(scaleX, scaleY, 1.0);
  state.panX = (canvas.width - state.deepFieldImage.width * state.scale) / 2;
  state.panY = (canvas.height - state.deepFieldImage.height * state.scale) / 2;
  document.getElementById('hudZoom').textContent = `Zoom: ${Math.round(state.scale * 100)}%`;
  drawScene();
}

// Coordinate Tracking HUD
function updateCoordinatesHud(e) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const imgX = (mouseX - state.panX) / state.scale;
  const imgY = (mouseY - state.panY) / state.scale;

  const raCenter = state.header.RA_DEG || 110.8208;
  const decCenter = state.header.DEC_DEG || -73.4542;
  const pixscale = state.header.PIXSCALE || 0.031;

  const dx_arcsec = (imgX - 600) * pixscale;
  const dy_arcsec = (imgY - 600) * pixscale;

  const ra = raCenter - (dx_arcsec / (3600.0 * Math.cos(decCenter * Math.PI / 180.0)));
  const dec = decCenter + (dy_arcsec / 3600.0);

  const raH = Math.floor(ra / 15);
  const raM = Math.floor((ra / 15 - raH) * 60);
  const raS = ((ra / 15 - raH) * 60 - raM) * 60;

  const decSign = dec >= 0 ? '+' : '-';
  const decD = Math.floor(Math.abs(dec));
  const decM = Math.floor((Math.abs(dec) - decD) * 60);
  const decS = ((Math.abs(dec) - decD) * 60 - decM) * 60;

  const coordStr = `RA: ${raH.toString().padStart(2, '0')}h${raM.toString().padStart(2, '0')}m${raS.toFixed(1).padStart(4, '0')}s  •  Dec: ${decSign}${decD.toString().padStart(2, '0')}°${decM.toString().padStart(2, '0')}'${decS.toFixed(1).padStart(4, '0')}"`;
  document.getElementById('hudCoords').textContent = coordStr;
}

// Data Fetching
async function loadInitialData() {
  try {
    // 1. Fetch system status
    const statusRes = await fetch('/api/status');
    const status = await statusRes.json();
    updateStatusHud(status);

    // 2. Fetch deepfield mosaic
    const dfRes = await fetch('/api/deepfield');
    const dfData = await dfRes.json();
    state.header = dfData.header;

    const img = new Image();
    img.onload = () => {
      state.deepFieldImage = img;
      resetView();
    };
    img.src = dfData.mosaic_b64;

    // 3. Fetch sources
    const srcRes = await fetch('/api/sources');
    const srcData = await srcRes.json();
    state.sources = srcData.sources;

    updateCounts();
    renderCatalog();
    drawScene();

    // Auto-select first lens candidate
    const firstLens = state.sources.find(s => s.morphology.is_lens_candidate);
    if (firstLens) selectSource(firstLens.id);
  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

function updateStatusHud(status) {
  document.getElementById('tCores').textContent = `${status.active_hpc_workers} CPU`;
  if (status.telemetry && status.telemetry.throughput_mpix_per_sec) {
    updateTelemetryView(status.telemetry);
  }
  const keyStatus = document.getElementById('keyStatusText');
  if (status.gemini_live_active) {
    keyStatus.textContent = 'Gemini Live Active';
    keyStatus.style.color = 'var(--color-green)';
  } else {
    keyStatus.textContent = 'Gemini Simulation';
    keyStatus.style.color = 'var(--color-cyan)';
  }
}

function updateTelemetryView(t) {
  document.getElementById('tThroughput').textContent = `${t.throughput_mpix_per_sec} Mpix/s`;
  document.getElementById('tLatency').textContent = `${t.execution_time_ms} ms`;
  document.getElementById('tExtractionRate').textContent = `${t.detection_rate_objects_per_sec} obj/s`;
  document.getElementById('tMem').textContent = `${t.memory_usage_mb} MB`;
  const pct = Math.min((t.memory_usage_mb / 2048) * 100, 100);
  document.getElementById('tMemBar').style.width = `${pct}%`;
}

function updateCounts() {
  let countLens = 0, countMerger = 0, countHighz = 0, countOther = 0;
  for (const s of state.sources) {
    const m = s.morphology;
    if (m.is_lens_candidate) countLens++;
    else if (m.is_merger_candidate) countMerger++;
    else if (s.f444_f090_ratio > 2.0) countHighz++;
    else countOther++;
  }
  document.getElementById('countAll').textContent = state.sources.length;
  document.getElementById('countLens').textContent = countLens;
  document.getElementById('countMerger').textContent = countMerger;
  document.getElementById('countHighz').textContent = countHighz;
  document.getElementById('countOther').textContent = countOther;
  document.getElementById('catalogCount').textContent = `${state.sources.length} Objects`;
}

// Catalog List Rendering
function renderCatalog() {
  const listEl = document.getElementById('catalogList');
  listEl.innerHTML = '';

  const filtered = state.sources.filter(s => {
    if (state.filter === 'ALL') return true;
    if (state.filter === 'LENS') return s.morphology.is_lens_candidate;
    if (state.filter === 'MERGER') return s.morphology.is_merger_candidate;
    if (state.filter === 'HIGHZ') return s.f444_f090_ratio > 2.0;
    if (state.filter === 'OTHER') return !s.morphology.is_lens_candidate && !s.morphology.is_merger_candidate && s.f444_f090_ratio <= 2.0;
    return true;
  });

  if (filtered.length === 0) {
    listEl.innerHTML = `<div class="empty-state"><p>No objects in this category.</p></div>`;
    return;
  }

  filtered.forEach(s => {
    const item = document.createElement('div');
    item.className = `catalog-item ${s.id === state.selectedSourceId ? 'selected' : ''}`;
    item.onclick = () => selectSource(s.id);

    let tagClass = 'tag-other';
    let tagText = 'Cluster Member';
    if (s.morphology.is_lens_candidate) {
      tagClass = 'tag-lens';
      tagText = 'Einstein Arc';
    } else if (s.morphology.is_merger_candidate) {
      tagClass = 'tag-merger';
      tagText = 'Tidal Merger';
    } else if (s.f444_f090_ratio > 2.0) {
      tagClass = 'tag-highz';
      tagText = 'High-z Dropout';
    }

    item.innerHTML = `
      <div class="catalog-info">
        <div class="catalog-top">
          <span class="c-id">${s.id}</span>
          <span class="c-tag ${tagClass}">${tagText}</span>
        </div>
        <div class="catalog-coords">${s.ra_str} &bull; ${s.dec_str}</div>
        <div class="catalog-stats">
          <span>SNR: ${s.snr}</span>
          <span>&epsilon;: ${s.morphology.ellipticity}</span>
          <span>F444/F090: ${s.f444_f090_ratio}</span>
        </div>
      </div>
    `;
    listEl.appendChild(item);
  });
}

// Source Selection & Inspector
async function selectSource(sourceId) {
  state.selectedSourceId = sourceId;
  renderCatalog();
  drawScene();

  const badge = document.getElementById('inspectId');
  badge.textContent = sourceId;

  const body = document.getElementById('inspectorBody');
  body.innerHTML = `
    <div class="empty-state">
      <i data-lucide="loader-2" class="spin"></i>
      <p>Loading candidate cutout & running Gemini inference...</p>
    </div>
  `;
  if (window.lucide) lucide.createIcons();

  try {
    // 1. Fetch full details (includes thumbnail)
    const detailRes = await fetch(`/api/source/${sourceId}`);
    const source = await detailRes.json();

    // 2. Fetch Gemini AI analysis
    const aiRes = await fetch(`/api/gemini/analyze/${sourceId}`, { method: 'POST' });
    const aiData = await aiRes.json();
    const analysis = aiData.analysis;
    state.currentAnalysis = analysis;

    renderInspectorView(source, analysis);

    // Smoothly pan canvas to center on the selected object
    panToSource(source.x, source.y);
  } catch (err) {
    console.error('Failed to inspect source:', err);
    body.innerHTML = `<div class="empty-state"><p>Error loading inspection data.</p></div>`;
  }
}

function panToSource(targetX, targetY) {
  const targetScale = Math.max(state.scale, 2.0);
  state.scale = targetScale;
  state.panX = canvas.width / 2 - targetX * targetScale;
  state.panY = canvas.height / 2 - targetY * targetScale;
  document.getElementById('hudZoom').textContent = `Zoom: ${Math.round(state.scale * 100)}%`;
  drawScene();
}

function renderInspectorView(source, analysis) {
  const body = document.getElementById('inspectorBody');
  const m = source.morphology;

  let bannerClass = 'banner-other';
  if (analysis.classification === 'EINSTEIN_RING_OR_ARC') bannerClass = 'banner-lens';
  else if (analysis.classification === 'INTERACTING_MERGER') bannerClass = 'banner-merger';
  else if (analysis.classification === 'HIGH_REDSHIFT_CANDIDATE') bannerClass = 'banner-highz';

  const instHtml = (analysis.recommended_instruments || [])
    .map(i => `<li>${i}</li>`).join('');

  body.innerHTML = `
    <!-- Cutout Image Card -->
    <div class="cutout-wrapper">
      <img class="cutout-img-large" src="${source.thumbnail_b64}" alt="${source.id}">
      <div class="cutout-meta">
        <span>Cutout: 64x64 px</span>
        <span>RGB (F444W, F200W, F090W)</span>
      </div>
    </div>

    <!-- Gemini AI Scientific Card -->
    <div class="gemini-card">
      <div class="gemini-header">
        <div class="gemini-title">
          <i data-lucide="sparkles"></i>
          <span>Gemini 2.0 Scientific Analysis</span>
        </div>
        <span class="badge badge-target">${Math.round(analysis.confidence * 100)}% Match</span>
      </div>

      <div class="classification-banner ${bannerClass}">
        ${analysis.classification.replace(/_/g, ' ')}
      </div>

      <div class="ai-summary">
        ${analysis.summary}
      </div>

      <div class="ai-stats-grid">
        <div class="ai-stat">
          <span class="ai-stat-label">Estimated Redshift</span>
          <span class="ai-stat-val text-cyan">${analysis.estimated_redshift}</span>
        </div>
        <div class="ai-stat">
          <span class="ai-stat-label">Magnification</span>
          <span class="ai-stat-val text-gold">${analysis.magnification_factor}</span>
        </div>
        <div class="ai-stat">
          <span class="ai-stat-label">Shear Angle</span>
          <span class="ai-stat-val">${analysis.deflection_shear_angle_deg}°</span>
        </div>
        <div class="ai-stat">
          <span class="ai-stat-label">Follow-up Priority</span>
          <span class="ai-stat-val text-green">${analysis.astrophysical_interest_score} / 10</span>
        </div>
      </div>

      <div class="ai-detail">
        ${analysis.physical_interpretation}
      </div>

      <button id="openReportBtn" class="btn btn-primary btn-sm" style="margin-top: 4px;">
        <i data-lucide="file-text"></i>
        <span>Generate ApJ Discovery Memo</span>
      </button>
    </div>

    <!-- Quantitative Morphology Specs -->
    <div class="metric-box">
      <div class="metric-title">
        <i data-lucide="sliders"></i>
        <span>HPC Morphology Parameters</span>
      </div>

      <div class="metric-row"><span class="m-key">Coordinates (RA / Dec)</span><span class="m-val">${source.ra_str}, ${source.dec_str}</span></div>
      <div class="metric-row"><span class="m-key">Signal-to-Noise Ratio (SNR)</span><span class="m-val text-amber">${source.snr}</span></div>
      <div class="metric-row"><span class="m-key">Infrared Color (F444W / F090W)</span><span class="m-val text-cyan">${source.f444_f090_ratio}</span></div>
      <div class="metric-row"><span class="m-key">Ellipticity (&epsilon; = 1 - b/a)</span><span class="m-val">${m.ellipticity}</span></div>
      <div class="metric-row"><span class="m-key">Curvature Index</span><span class="m-val text-gold">${m.curvature_score}</span></div>
      <div class="metric-row"><span class="m-key">Gini Coefficient (G)</span><span class="m-val">${m.gini}</span></div>
      <div class="metric-row"><span class="m-key">Moment of Light (M20)</span><span class="m-val">${m.m20}</span></div>
      <div class="metric-row"><span class="m-key">Concentration Index (C)</span><span class="m-val">${m.concentration}</span></div>
      <div class="metric-row"><span class="m-key">Rotational Asymmetry (A)</span><span class="m-val">${m.asymmetry}</span></div>
      <div class="metric-row"><span class="m-key">Cluster Core Distance</span><span class="m-val">${m.dist_to_cluster_core_px} px</span></div>
    </div>
  `;

  if (window.lucide) lucide.createIcons();

  document.getElementById('openReportBtn').addEventListener('click', () => {
    openDiscoveryReport(source.id);
  });
}

// Discovery Report Modal
async function openDiscoveryReport(sourceId) {
  const modal = document.getElementById('reportModal');
  const title = document.getElementById('reportModalTitle');
  const content = document.getElementById('reportContent');

  title.textContent = `ApJ Discovery Memorandum — ${sourceId}`;
  content.textContent = 'Synthesizing observational memorandum with LaTeX mathematical formulations...';
  modal.classList.add('active');

  try {
    const res = await fetch(`/api/gemini/report/${sourceId}`);
    const data = await res.json();
    content.textContent = data.memo_markdown;
  } catch (err) {
    content.textContent = 'Failed to generate discovery memo.';
  }
}

// Canvas Drawing Engine
function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.scale, state.scale);

  // 1. Draw Deep Field Mosaic
  if (state.deepFieldImage) {
    ctx.drawImage(state.deepFieldImage, 0, 0);
  }

  // 2. Draw Celestial Grid lines
  if (state.showGrid) {
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.lineWidth = 1 / state.scale;
    for (let x = 0; x <= 1200; x += 150) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 1200);
      ctx.stroke();
    }
    for (let y = 0; y <= 1200; y += 150) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(1200, y);
      ctx.stroke();
    }
  }

  // 3. Draw Cluster Core Crosshair
  ctx.strokeStyle = 'rgba(251, 191, 36, 0.3)';
  ctx.lineWidth = 1 / state.scale;
  ctx.beginPath();
  ctx.arc(600, 600, 135, 0, Math.PI * 2); // Approximate Einstein radius
  ctx.stroke();

  // 4. Draw Detection Bounding Boxes and Indicators
  if (state.showBoxes && state.sources) {
    for (const s of state.sources) {
      const isSelected = s.id === state.selectedSourceId;
      const m = s.morphology;

      let strokeColor = '#3b82f6'; // Blue
      if (m.is_lens_candidate) strokeColor = '#fbbf24'; // Gold
      else if (m.is_merger_candidate) strokeColor = '#ec4899'; // Magenta
      else if (s.f444_f090_ratio > 2.0) strokeColor = '#06b6d4'; // Cyan
      else if (s.snr > 60 && s.total_flux > 2500) strokeColor = '#ef4444'; // Red

      // Filter check
      if (state.filter === 'LENS' && !m.is_lens_candidate) continue;
      if (state.filter === 'MERGER' && !m.is_merger_candidate) continue;
      if (state.filter === 'HIGHZ' && s.f444_f090_ratio <= 2.0) continue;

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = (isSelected ? 2.5 : 1.2) / state.scale;

      // Draw bounding box
      if (s.bbox) {
        const [x1, y1, x2, y2] = s.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      // Draw special target reticle on selected source
      if (isSelected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.0 / state.scale;
        ctx.beginPath();
        ctx.arc(s.x, s.y, 22, 0, Math.PI * 2);
        ctx.stroke();

        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(s.x - 28, s.y); ctx.lineTo(s.x + 28, s.y);
        ctx.moveTo(s.x, s.y - 28); ctx.lineTo(s.x, s.y + 28);
        ctx.stroke();

        // Label
        ctx.fillStyle = '#ffffff';
        ctx.font = `${Math.max(11 / state.scale, 8)}px 'JetBrains Mono'`;
        ctx.fillText(s.id, s.x + 26, s.y - 12);
      }
    }
  }

  ctx.restore();
}

// Pipeline Execution
async function runHpcPipeline() {
  const btn = document.getElementById('runHpcBtn');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i><span>Processing...</span>`;
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/hpc/run', { method: 'POST' });
    const data = await res.json();
    
    updateTelemetryView(data.telemetry);
    await loadInitialData();
  } catch (err) {
    alert('HPC Pipeline run failed: ' + err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="cpu"></i><span>Run HPC Pipeline</span>`;
    if (window.lucide) lucide.createIcons();
  }
}

// Batch Classify
async function runBatchClassify() {
  const btn = document.getElementById('batchClassifyBtn');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i><span>Classifying...</span>`;
  if (window.lucide) lucide.createIcons();

  try {
    const res = await fetch('/api/gemini/batch-analyze', { method: 'POST' });
    const data = await res.json();
    alert(`Gemini successfully classified ${data.analyzed_count} candidate galaxies!`);
    if (state.selectedSourceId) selectSource(state.selectedSourceId);
  } catch (err) {
    alert('Batch classification failed: ' + err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="bot"></i><span>Gemini Classify</span>`;
    if (window.lucide) lucide.createIcons();
  }
}

// Semantic Sky Search
async function executeSemanticSearch() {
  const input = document.getElementById('semanticSearchInput');
  const query = input.value.trim();
  if (!query) return;

  const btn = document.getElementById('searchBtn');
  btn.textContent = 'Searching...';

  try {
    const res = await fetch('/api/gemini/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();

    if (data.results && data.results.length > 0) {
      const topMatch = data.results[0];
      selectSource(topMatch.source_id);
      alert(`Semantic Search Result:\n${data.explanation}`);
    } else {
      alert(`No matches found for: "${query}"`);
    }
  } catch (err) {
    alert('Semantic search error: ' + err);
  } finally {
    btn.textContent = 'Query Sky';
  }
}

// Save Gemini Key
async function saveGeminiKey() {
  const input = document.getElementById('geminiApiKeyInput');
  const key = input.value.trim();
  if (!key) return;

  try {
    const res = await fetch('/api/settings/key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key })
    });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('keyModal').classList.remove('active');
      const keyStatus = document.getElementById('keyStatusText');
      keyStatus.textContent = 'Gemini Live Active';
      keyStatus.style.color = 'var(--color-green)';
      alert('Gemini API key saved! Live multimodal classification is now active.');
      if (state.selectedSourceId) selectSource(state.selectedSourceId);
    }
  } catch (err) {
    alert('Failed to save API key: ' + err);
  }
}
