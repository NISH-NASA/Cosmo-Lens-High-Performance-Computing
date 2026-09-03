# 🌌 CosmoLens HPC: Autonomous JWST Deep-Field Processing & Gravitational Lens Discovery Engine

> **Built for MLH Hackathon with Google Gemini 2.0 API**  
> *Pairing High-Performance Computing (HPC) with Gemini Multimodal AI to detect Einstein rings, gravitationally lensed arcs, and primeval galaxies in James Webb Space Telescope observations.*

---

## 🚀 Overview

The **James Webb Space Telescope (JWST)** produces unprecedented gigapixel deep-field infrared observations of massive galaxy clusters like **SMACS J0723.3-7327**. These images contain hundreds of thousands of celestial objects and rare cosmological phenomena:
- **Strong Gravitational Lenses (Einstein Rings & Arcs)**: Distant background galaxies magnified and sheared into curves by the cluster's intense gravitational well.
- **Interacting Galaxy Mergers**: Colliding galaxies with tidal bridges, disrupted morphologies, and active starburst knots.
- **Ultra High-Redshift Candidates ($z > 7$)**: Primeval galaxies formed within the first few hundred million years of cosmic dawn.

Standard manual inspection and classical astronomy pipelines struggle with the sheer scale and subtle morphological distortions in multi-band JWST infrared data.

**CosmoLens HPC** bridges this gap:
1. **HPC Multi-Core Image Processing Engine**: Ingests multi-band NIRCam infrared arrays (F090W, F200W, F444W), applies non-linear Lupton Asinh dynamic range compression, calculates 2D background sigma-clipping, extracts sources at over **400+ objects/sec**, and computes quantitative non-parametric morphology metrics (Gini, $M_{20}$, Concentration, Ellipticity, Tangential Shear).
2. **Gemini 2.0 Multimodal AI Core**: Inspects candidate cutouts alongside physical metrics to classify phenomena, estimate gravitational magnification factors and redshifts, and reason about cluster deflection caustics.
3. **Interactive Mission Observatory**: An interactive deep-sky canvas console with smooth pan/zoom, live celestial coordinate tracking (RA/Dec), candidate inspection drawers, natural language semantic sky-search, and automated **ApJ Discovery Memorandum** export.

---

## 🏛️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │       Raw JWST NIRCam Deep Field       │
                       │  (SMACS 0723 / Webb's First Deep Field) │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          HPC PROCESSING CORE (Python / Multi-Core)                          │
│  • Parallel multi-channel FITS / HDR array ingestion                                        │
│  • High-throughput background noise modeling (2D Sigma-Clipping)                            │
│  • Fast source extraction & centroid segmentation (vectorized NumPy / SciPy)                │
│  • Quantitative morphology: Gini (G), M20, Concentration (C), Asymmetry (A), Ellipticity    │
│  • Curvilinear tangential shear detection (Gravitational Arc Candidates)                    │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ Cutouts + Photometric & Morphology stats
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            GEMINI 2.0 MULTIMODAL REASONING CORE                             │
│  • Visual inspection of candidate cutouts & morphology profiles                             │
│  • Scientific classification: Strong Lens, Galaxy Merger, High-z Candidate, Stellar Artifact │
│  • Physical inference: Deflection geometry, estimated redshift, magnification factor        │
│  • Autonomous Discovery Memo: Formal ApJ-style research letter with LaTeX formulas          │
│  • Natural language semantic sky-search: "Find distorted arcs near the cluster core"        │
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             INTERACTIVE MISSION DASHBOARD                                   │
│  • High-res deep-field canvas viewer with pan, zoom, and dynamic celestial coordinate grid  │
│  • Real-time classification overlays (Gold: Einstein Rings, Magenta: Mergers, Cyan: High-z) │
│  • HPC Telemetry Hud: Throughput (Mpix/s), CPU cores active, extraction rate, RAM usage     │
│  • Galaxy Inspector: High-res cutouts, morphology radar specs, multi-band color ratio       │
│  • Gemini Co-Pilot Chat & One-Click ApJ Discovery Report Export                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Requirements
- Python 3.10+
- Installed packages: `fastapi`, `uvicorn`, `pillow`, `astropy`, `numpy`, `scipy`, `google-genai`

### 2. Run the Observatory Server
```bash
python3 main.py
```
Or with custom port:
```bash
PORT=8000 python3 main.py
```

Open your browser to:
👉 **`http://localhost:8000`**

### 3. (Optional) Configure Gemini API Key
You can pass your Gemini API key in the environment:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
python3 main.py
```
*Or simply click the **"API Key"** button in the web header to enter it directly into the running application! If no key is supplied, CosmoLens runs in calibrated scientific simulation mode with full morphological inference.*

---

## 🔬 Hackathon Pitch & Demo Walkthrough

When presenting to judges, follow this 4-step sequence:

1. **The Problem & Scale**:
   - Show the JWST Deep Field mosaic on the main viewport. Explain that JWST images are gigapixels in size with hundreds of thousands of galaxies, making manual discovery of rare Einstein rings and mergers nearly impossible.
2. **HPC Execution & Live Telemetry**:
   - Click **"Run HPC Pipeline"**. Point to the left sidebar showing real-time hardware telemetry:
     - Multi-core CPU utilization
     - Throughput: **~1.0+ Mpix/s**
     - Detection rate: **400+ objects/sec**
     - Sub-second background noise modeling and source segmentation.
3. **Gemini 2.0 Multimodal Reasoning**:
   - Click on an Einstein Arc candidate (e.g. `JWST-CL-0026`).
   - The canvas smoothly zooms and centers on the arc.
   - Point to the right-hand **Galaxy Inspector**: Show the quantitative morphology metrics (Gini, $M_{20}$, Ellipticity $\epsilon = 0.72$, Curvature Score $0.65$).
   - Show the **Gemini 2.0 Scientific Analysis**: Explain how Gemini combines visual inspection with physical parameters to identify the caustic crossing, estimate redshift ($z \sim 1.43$), and calculate gravitational magnification ($18\times - 25\times$).
4. **Natural Language Semantic Sky-Search**:
   - In the top search bar, type:
     ```
     find distorted arcs near the cluster core
     ```
   - Click **"Query Sky"**. The engine parses the semantic intent, filters the catalog, and focuses directly onto the top gravitational lens!
5. **One-Click ApJ Discovery Memo**:
   - Click **"Generate ApJ Discovery Memo"**.
   - Show the academic-quality Research Note with LaTeX formulas, observational coordinates, magnification model, and recommended follow-up instruments (JWST NIRSpec IFU, ALMA Band 7).

---

## 🧪 Running the Test Suite

```bash
python3 -m unittest discover -s tests
```
All 12 automated unit and integration tests verify:
- Astronomical contrast stretching algorithms (Asinh, Log, ZScale)
- Non-parametric morphology calculations (Gini, $M_{20}$, Concentration, Asymmetry, Ellipticity, Tangential Shear)
- Multi-core parallel HPC execution
- Gemini AI multimodal agent classification and fallback simulation
- Semantic natural language sky query engine
- FastAPI REST endpoints
