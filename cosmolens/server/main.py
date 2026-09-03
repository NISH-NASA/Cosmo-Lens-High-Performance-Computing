"""
FastAPI Server for CosmoLens HPC: Astronomical Processing & AI Discovery Engine.
"""

import os
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from cosmolens.hpc.parallel_engine import get_hpc_engine
from cosmolens.ai.gemini_agent import get_gemini_agent
from cosmolens.ai.discovery_report import generate_apj_discovery_memo
from cosmolens.ai.natural_search import search_sky_catalog
from cosmolens.hpc.mast_fetcher import search_and_fetch_target

app = FastAPI(
    title="CosmoLens HPC",
    description="High-Performance JWST Deep-Field Processing & Gravitational Lens Discovery Engine",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class HPCRunRequest(BaseModel):
    threshold_sigma: float = 3.2
    cutout_size: int = 64
    grid_size: int = 1200


class MastFetchRequest(BaseModel):
    target_name: str


class GeminiKeyRequest(BaseModel):
    api_key: str


class NaturalSearchRequest(BaseModel):
    query: str


class BatchAnalyzeRequest(BaseModel):
    max_count: int = 15


@app.get("/api/status")
def get_system_status():
    """Returns system status, active CPU cores, Gemini status, and data state."""
    engine = get_hpc_engine()
    agent = get_gemini_agent()
    
    return {
        "status": "online",
        "cpu_cores": engine.cpu_count,
        "active_hpc_workers": engine.num_workers,
        "gemini_live_active": agent.is_live_api_active(),
        "gemini_model": agent.model_name,
        "has_data": len(engine.current_sources) > 0,
        "sources_count": len(engine.current_sources),
        "telemetry": engine.last_telemetry
    }


@app.post("/api/mast/fetch")
def fetch_from_mast(req: MastFetchRequest):
    """
    Queries the official STScI MAST Archive for the target name,
    resolves coordinates, and fetches the best JWST NIRCam observation.
    """
    result = search_and_fetch_target(req.target_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/hpc/run")
def trigger_hpc_run(req: HPCRunRequest = None):
    """Executes parallel HPC extraction pipeline on JWST deep-field dataset."""
    engine = get_hpc_engine()
    sigma = req.threshold_sigma if req else 3.2
    cutout_sz = req.cutout_size if req else 64
    
    res = engine.run_pipeline(
        threshold_sigma=sigma,
        cutout_size=cutout_sz
    )
    return {
        "status": "success",
        "telemetry": res["telemetry"],
        "sources_count": res["sources_count"],
        "header": res["header"]
    }


@app.get("/api/deepfield")
def get_deepfield_data():
    """Returns the rendered mosaic image and observation metadata."""
    engine = get_hpc_engine()
    if not engine.current_mosaic_b64:
        # Run pipeline if not already executed
        engine.run_pipeline()
        
    return {
        "mosaic_b64": engine.current_mosaic_b64,
        "header": engine.current_header,
        "sources_count": len(engine.current_sources)
    }


@app.get("/api/sources")
def list_sources():
    """Returns the catalog of detected sources."""
    engine = get_hpc_engine()
    if not engine.current_sources:
        engine.run_pipeline()

    # Return lightweight metadata (thumbnails excluded for fast catalog display)
    catalog = []
    for s in engine.current_sources:
        catalog.append({
            "id": s["id"],
            "x": s["x"],
            "y": s["y"],
            "ra_str": s["ra_str"],
            "dec_str": s["dec_str"],
            "snr": s["snr"],
            "total_flux": s["total_flux"],
            "f444_f090_ratio": s["f444_f090_ratio"],
            "morphology": s["morphology"],
            "bbox": s["bbox"]
        })
    return {"count": len(catalog), "sources": catalog}


@app.get("/api/source/{source_id}")
def get_source_detail(source_id: str):
    """Returns detailed photometric, morphological, and thumbnail data for a specific source."""
    engine = get_hpc_engine()
    for s in engine.current_sources:
        if s["id"] == source_id:
            return s
    raise HTTPException(status_code=404, detail="Source not found")


@app.post("/api/gemini/analyze/{source_id}")
def analyze_single_source(source_id: str):
    """Invokes Gemini Multimodal AI to analyze and classify a candidate galaxy cutout."""
    engine = get_hpc_engine()
    target = None
    for s in engine.current_sources:
        if s["id"] == source_id:
            target = s
            break
            
    if not target:
        raise HTTPException(status_code=404, detail="Source not found")

    # Decode thumbnail bytes for live Gemini API if needed
    raw_bytes = None
    if "thumbnail_b64" in target and target["thumbnail_b64"]:
        b64_data = target["thumbnail_b64"].split(",")[-1]
        raw_bytes = base64.b64decode(b64_data)

    agent = get_gemini_agent()
    analysis = agent.analyze_source(target, raw_png_bytes=raw_bytes)
    return {
        "source_id": source_id,
        "analysis": analysis
    }


@app.post("/api/gemini/batch-analyze")
def batch_analyze_candidates(req: BatchAnalyzeRequest = None):
    """Analyzes top gravitational lens and merger candidates in one go."""
    engine = get_hpc_engine()
    if not engine.current_sources:
        engine.run_pipeline()

    max_c = req.max_count if req else 15
    agent = get_gemini_agent()

    # Prioritize candidate lenses and mergers
    candidates = []
    for s in engine.current_sources:
        m = s.get("morphology", {})
        if m.get("is_lens_candidate") or m.get("is_merger_candidate") or s.get("f444_f090_ratio", 1.0) > 2.0:
            candidates.append(s)

    # If few, add other high SNR sources
    if len(candidates) < max_c:
        for s in engine.current_sources:
            if s not in candidates:
                candidates.append(s)
            if len(candidates) >= max_c:
                break

    candidates = candidates[:max_c]
    results = []

    for s in candidates:
        raw_bytes = None
        if "thumbnail_b64" in s:
            b64_data = s["thumbnail_b64"].split(",")[-1]
            raw_bytes = base64.b64decode(b64_data)
        analysis = agent.analyze_source(s, raw_png_bytes=raw_bytes)
        results.append({
            "source_id": s["id"],
            "x": s["x"],
            "y": s["y"],
            "thumbnail_b64": s.get("thumbnail_b64"),
            "analysis": analysis
        })

    return {
        "analyzed_count": len(results),
        "results": results
    }


@app.post("/api/gemini/search")
def search_catalog_semantic(req: NaturalSearchRequest):
    """Executes natural language semantic query across extracted celestial catalog."""
    engine = get_hpc_engine()
    if not engine.current_sources:
        engine.run_pipeline()

    agent = get_gemini_agent()
    results = search_sky_catalog(req.query, engine.current_sources, agent)
    return results


@app.get("/api/gemini/report/{source_id}")
def get_discovery_report(source_id: str):
    """Generates an academic ApJ-style discovery report for the target object."""
    engine = get_hpc_engine()
    target = None
    for s in engine.current_sources:
        if s["id"] == source_id:
            target = s
            break
            
    if not target:
        raise HTTPException(status_code=404, detail="Source not found")

    raw_bytes = None
    if "thumbnail_b64" in target:
        b64_data = target["thumbnail_b64"].split(",")[-1]
        raw_bytes = base64.b64decode(b64_data)

    agent = get_gemini_agent()
    analysis = agent.analyze_source(target, raw_png_bytes=raw_bytes)
    memo = generate_apj_discovery_memo(target, analysis, engine.current_header)

    return {
        "source_id": source_id,
        "classification": analysis.get("classification"),
        "memo_markdown": memo
    }


@app.post("/api/settings/key")
def update_gemini_key(req: GeminiKeyRequest):
    """Updates the Gemini API key at runtime without restarting the server."""
    agent = get_gemini_agent()
    agent.set_api_key(req.api_key.strip())
    return {
        "status": "success",
        "live_active": agent.is_live_api_active(),
        "model": agent.model_name
    }


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>CosmoLens HPC Server Running</h1><p>Static directory not found.</p>")
