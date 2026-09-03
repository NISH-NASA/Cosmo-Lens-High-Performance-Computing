"""
MAST (Mikulski Archive for Space Telescopes) API Integration.
Fetches real JWST observations based on target names or celestial coordinates.
"""

import requests
import json
import urllib.parse
from typing import Dict, Any, List, Tuple

MAST_API_URL = "https://mast.stsci.edu/api/v0/invoke"

def _mast_query(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Sends a formatted JSON request to the MAST API."""
    payload = {'request': json.dumps(request_obj)}
    headers = {
        'Content-type': 'application/x-www-form-urlencoded',
        'Accept': 'text/plain',
        'User-Agent': 'CosmoLens_HPC_Hackathon'
    }
    
    response = requests.post(MAST_API_URL, data=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def resolve_target_name(target_name: str) -> Tuple[float, float]:
    """
    Resolves a celestial target name (e.g., 'SMACS 0723') to RA and Dec.
    Returns (RA, Dec) in degrees.
    """
    req = {
        'service': 'Mast.Name.Lookup',
        'params': {
            'input': target_name,
            'format': 'json'
        }
    }
    
    result = _mast_query(req)
    
    # Check if we got a valid resolution
    if not result or not result.get('resolvedCoordinate'):
        raise ValueError(f"Could not resolve target name: {target_name}")
        
    coords = result['resolvedCoordinate'][0]
    ra = coords.get('ra')
    dec = coords.get('decl')
    
    if ra is None or dec is None:
        raise ValueError(f"Invalid coordinate data returned for {target_name}")
        
    return float(ra), float(dec)


def find_jwst_observations(ra: float, dec: float, radius_deg: float = 0.05) -> List[Dict[str, Any]]:
    """
    Searches for JWST NIRCam observations around the given RA/Dec.
    """
    req = {
        'service': 'Mast.Caom.Cone',
        'params': {
            'ra': ra,
            'dec': dec,
            'radius': radius_deg
        },
        'format': 'json',
        'pagesize': 50,
        'removenullcolumns': True
    }
    
    result = _mast_query(req)
    if 'data' not in result:
        return []
        
    # Filter for JWST NIRCam deep fields (usually long exposure or specific projects)
    jwst_obs = []
    for obs in result['data']:
        obs_collection = obs.get('obs_collection', '')
        instrument = obs.get('instrument_name', '')
        
        if obs_collection == 'JWST' and 'NIRCAM' in instrument.upper():
            # Keep only the most useful metadata
            jwst_obs.append({
                'obs_id': obs.get('obs_id'),
                'target_name': obs.get('target_name'),
                'instrument': instrument,
                'filters': obs.get('filters'),
                'exposure_time': obs.get('t_exptime'),
                'ra': obs.get('s_ra'),
                'dec': obs.get('s_dec'),
                'jpeg_url': obs.get('jpegURL') # Extremely useful for fast hackathon previews!
            })
            
    # Sort by exposure time descending (we want the deepest fields)
    jwst_obs.sort(key=lambda x: x['exposure_time'] or 0, reverse=True)
    return jwst_obs


def search_and_fetch_target(target_name: str) -> Dict[str, Any]:
    """
    End-to-end function: Resolves a name, finds the best JWST image, and returns metadata.
    """
    try:
        ra, dec = resolve_target_name(target_name)
    except Exception as e:
        return {"error": str(e)}
        
    observations = find_jwst_observations(ra, dec, radius_deg=0.08)
    
    if not observations:
        return {
            "error": f"No JWST NIRCam observations found for '{target_name}'",
            "ra": ra,
            "dec": dec
        }
        
    best_obs = observations[0]
    
    return {
        "target": target_name,
        "resolved_ra": ra,
        "resolved_dec": dec,
        "best_observation": best_obs,
        "total_observations_found": len(observations)
    }
