"""
Natural Language Semantic Sky-Search Engine for CosmoLens HPC.
Translates astronomer natural language queries into celestial filtering criteria and visual focus targets.
"""

import re
from typing import List, Dict, Any


def search_sky_catalog(
    query: str,
    sources: List[Dict[str, Any]],
    gemini_agent = None
) -> Dict[str, Any]:
    """
    Evaluates natural language query against detected sources and returns ranked matches.
    """
    q_lower = query.lower()
    matches = []
    
    # Check key concepts
    wants_lenses = any(w in q_lower for w in ["lens", "einstein", "arc", "ring", "curved", "shear", "distorted", "magnif"])
    wants_mergers = any(w in q_lower for w in ["merger", "collid", "tidal", "tail", "interact", "double", "dual"])
    wants_highz = any(w in q_lower for w in ["high-z", "high redshift", "dropout", "red", "distant", "primeval", "early universe"])
    wants_core = any(w in q_lower for w in ["core", "center", "bcg", "central", "cluster"])
    wants_stars = any(w in q_lower for w in ["star", "spike", "artifact", "foreground", "diffraction"])

    for s in sources:
        morph = s.get("morphology", {})
        score = 0.0
        reasons = []

        # Lenses / Arcs
        if wants_lenses:
            if morph.get("is_lens_candidate", False):
                score += 50.0
                reasons.append("High tangential shear & arc curvature")
            if morph.get("ellipticity", 0) > 0.45:
                score += morph.get("ellipticity", 0) * 30.0
            if morph.get("dist_to_cluster_core_px", 999) < 250:
                score += 15.0

        # Mergers
        if wants_mergers:
            if morph.get("is_merger_candidate", False):
                score += 50.0
                reasons.append("High M20 moment & rotational asymmetry")
            if morph.get("asymmetry", 0) > 0.25:
                score += morph.get("asymmetry", 0) * 40.0

        # High Redshift
        if wants_highz:
            ratio = s.get("f444_f090_ratio", 1.0)
            if ratio > 1.8:
                score += min(ratio * 25.0, 75.0)
                reasons.append(f"Extreme infrared color ratio F444W/F090W = {ratio:.2f}")

        # Core
        if wants_core:
            dist = morph.get("dist_to_cluster_core_px", 999)
            if dist < 120:
                score += max(0, (120 - dist) * 0.8)
                reasons.append(f"Located near cluster centroid (d = {dist:.1f}px)")

        # Stars
        if wants_stars:
            if s.get("snr", 0) > 60 and s.get("total_flux", 0) > 2500:
                score += 60.0
                reasons.append("Foreground stellar brightness and diffraction profile")

        # General high SNR / brightness fallback
        if not (wants_lenses or wants_mergers or wants_highz or wants_core or wants_stars):
            # General search
            score = float(s.get("snr", 0)) + float(morph.get("ellipticity", 0)) * 20.0
            reasons.append(f"Matched by SNR ({s.get('snr')}) and morphology prominence")

        if score > 15.0:
            matches.append({
                "source_id": s.get("id"),
                "x": s.get("x"),
                "y": s.get("y"),
                "ra_str": s.get("ra_str"),
                "dec_str": s.get("dec_str"),
                "snr": s.get("snr"),
                "relevance_score": round(score, 1),
                "match_rationale": "; ".join(reasons) if reasons else "Morphological match",
                "thumbnail_b64": s.get("thumbnail_b64")
            })

    # Sort descending by relevance score
    matches.sort(key=lambda m: m["relevance_score"], reverse=True)
    top_matches = matches[:10]

    explanation = f"Found {len(matches)} celestial candidate(s) matching your query: '{query}'."
    if top_matches:
        top_id = top_matches[0]["source_id"]
        explanation += f" Top target is {top_id} ({top_matches[0]['match_rationale']})."

    return {
        "query": query,
        "total_matches": len(matches),
        "explanation": explanation,
        "results": top_matches
    }
