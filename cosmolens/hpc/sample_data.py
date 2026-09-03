"""
JWST SMACS 0723 Deep Field Benchmark Data Generator.
Generates an accurate, calibrated multi-band infrared scientific array representing Webb's First Deep Field,
complete with massive cluster cores, gravitational Einstein arcs, interacting galaxy mergers,
high-redshift dropouts, and JWST NIRCam hexagonal diffraction spikes.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.ndimage import gaussian_filter


def generate_benchmark_deepfield(
    width: int = 1200,
    height: int = 1200,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Synthesizes a realistic 3-band JWST NIRCam deep field (R=F444W, G=F200W, B=F090W).
    Returns (r_band, g_band, b_band, ground_truth_sources, wcs_header).
    """
    np.random.seed(random_seed)
    
    # 1. Realistic sky background noise (readout noise + zodiacal infrared background)
    r_band = np.random.normal(loc=12.0, scale=1.4, size=(height, width)).astype(np.float32)
    g_band = np.random.normal(loc=10.0, scale=1.2, size=(height, width)).astype(np.float32)
    b_band = np.random.normal(loc=8.0, scale=1.0, size=(height, width)).astype(np.float32)

    # Smooth correlated noise (simulates drizzled pixel sampling)
    r_band += gaussian_filter(np.random.normal(0, 0.8, (height, width)), sigma=1.2)
    g_band += gaussian_filter(np.random.normal(0, 0.7, (height, width)), sigma=1.2)
    b_band += gaussian_filter(np.random.normal(0, 0.6, (height, width)), sigma=1.2)

    cx_core, cy_core = width // 2, height // 2
    sources = []

    # Helper to draw sersic/elliptical galaxy profile
    y_grid, x_grid = np.indices((height, width))

    def add_sersic(
        cx: float, cy: float, flux: float, r_eff: float, n_sersic: float,
        q: float, theta_deg: float, color_ratio: Tuple[float, float, float]
    ):
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = x_grid - cx
        dy = y_grid - cy
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        r_ell = np.sqrt(x_rot**2 + (y_rot / max(q, 0.1))**2)
        
        # Sersic profile approximation: I(r) = I0 * exp(-b_n * (r/r_eff)**(1/n))
        bn = 2.0 * n_sersic - 0.324
        profile = flux * np.exp(-bn * (np.maximum(r_ell, 0.0) / max(r_eff, 0.5))**(1.0 / n_sersic))
        
        r_band[...] += (profile * color_ratio[0]).astype(np.float32)
        g_band[...] += (profile * color_ratio[1]).astype(np.float32)
        b_band[...] += (profile * color_ratio[2]).astype(np.float32)

    # 2. Add Brightest Cluster Galaxy (BCG) - SMACS J0723.3-7327 Massive Elliptical Core
    add_sersic(
        cx=cx_core, cy=cy_core, flux=1800.0, r_eff=32.0, n_sersic=4.0,
        q=0.82, theta_deg=35.0, color_ratio=(1.3, 1.1, 0.7)
    )
    sources.append({
        "name": "SMACS 0723 BCG (Cluster Core)",
        "cx": cx_core, "cy": cy_core, "type": "CLUSTER_CORE",
        "redshift": 0.39, "description": "Massive central brightest cluster galaxy producing cluster lensing potential."
    })

    # Add diffuse intracluster light (ICL) halo around core
    r_core_dist = np.sqrt((x_grid - cx_core)**2 + (y_grid - cy_core)**2)
    icl_halo = 45.0 * np.exp(-r_core_dist / 140.0)
    r_band += icl_halo * 1.2
    g_band += icl_halo * 1.0
    b_band += icl_halo * 0.6

    # 3. Add Iconic Gravitational Lenses (Einstein Arcs)
    # Arc 1: The famous Sparkler Arc / Distorted crescent (tangential arc)
    def add_curved_arc(
        cx: float, cy: float, radius: float, arc_angle_deg: float, span_deg: float,
        thickness: float, flux: float, color_ratio: Tuple[float, float, float]
    ):
        theta_rad = np.radians(arc_angle_deg)
        # Angle from center
        dx = x_grid - cx
        dy = y_grid - cy
        dist = np.sqrt(dx**2 + dy**2)
        angles = np.degrees(np.arctan2(dy, dx)) % 360.0
        
        arc_center_mod = arc_angle_deg % 360.0
        angle_diff = np.abs((angles - arc_center_mod + 180) % 360 - 180)
        
        radial_profile = np.exp(-0.5 * ((dist - radius) / max(thickness, 1.0))**2)
        angular_profile = np.exp(-0.5 * (angle_diff / (span_deg / 2.0))**4) # flattened top
        arc = flux * radial_profile * angular_profile

        # Add star-forming clumpy knots along the arc
        for knot_off in [-15, 0, 12, 22]:
            knot_angle = np.radians(arc_center_mod + knot_off)
            kx = cx + radius * np.cos(knot_angle)
            ky = cy + radius * np.sin(knot_angle)
            kdist = np.sqrt((x_grid - kx)**2 + (y_grid - ky)**2)
            arc += (flux * 1.5) * np.exp(-0.5 * (kdist / 2.5)**2)

        r_band[...] += (arc * color_ratio[0]).astype(np.float32)
        g_band[...] += (arc * color_ratio[1]).astype(np.float32)
        b_band[...] += (arc * color_ratio[2]).astype(np.float32)

    # Primary Einstein Arc (East flank)
    add_curved_arc(
        cx=cx_core, cy=cy_core, radius=135.0, arc_angle_deg=25.0, span_deg=55.0,
        thickness=3.2, flux=340.0, color_ratio=(1.4, 1.0, 0.5)
    )
    # Approximate centroid for East Arc
    arc1_x = cx_core + 135.0 * np.cos(np.radians(25.0))
    arc1_y = cy_core + 135.0 * np.sin(np.radians(25.0))
    sources.append({
        "name": "Einstein Arc Alpha (The Sparkler)",
        "cx": round(arc1_x, 1), "cy": round(arc1_y, 1), "type": "EINSTEIN_RING_OR_ARC",
        "redshift": 1.427, "description": "Highly magnified background galaxy stretched into prominent tangential arc with resolved star clusters."
    })

    # Secondary Mirrored Arc (South-West flank)
    add_curved_arc(
        cx=cx_core, cy=cy_core, radius=142.0, arc_angle_deg=205.0, span_deg=45.0,
        thickness=2.8, flux=260.0, color_ratio=(1.35, 0.95, 0.52)
    )
    arc2_x = cx_core + 142.0 * np.cos(np.radians(205.0))
    arc2_y = cy_core + 142.0 * np.sin(np.radians(205.0))
    sources.append({
        "name": "Einstein Arc Beta (Counter-Image)",
        "cx": round(arc2_x, 1), "cy": round(arc2_y, 1), "type": "EINSTEIN_RING_OR_ARC",
        "redshift": 1.427, "description": "Multiple lensed counter-image demonstrating gravitational lensing parities."
    })

    # Tertiary Slender Arc (North flank)
    add_curved_arc(
        cx=cx_core, cy=cy_core, radius=110.0, arc_angle_deg=105.0, span_deg=35.0,
        thickness=2.2, flux=210.0, color_ratio=(1.5, 1.1, 0.4)
    )
    arc3_x = cx_core + 110.0 * np.cos(np.radians(105.0))
    arc3_y = cy_core + 110.0 * np.sin(np.radians(105.0))
    sources.append({
        "name": "Gravitational Arc Gamma",
        "cx": round(arc3_x, 1), "cy": round(arc3_y, 1), "type": "EINSTEIN_RING_OR_ARC",
        "redshift": 2.18, "description": "Thin elongated caustic-crossing arc magnified > 25x."
    })

    # 4. Add Colliding Galaxy Merger Pair with Tidal Disruption Tails
    merger_x, merger_y = cx_core - 280, cy_core + 220
    # Core 1
    add_sersic(merger_x - 14, merger_y - 8, flux=420.0, r_eff=9.0, n_sersic=2.2, q=0.55, theta_deg=70.0, color_ratio=(1.2, 1.0, 0.9))
    # Core 2
    add_sersic(merger_x + 12, merger_y + 10, flux=380.0, r_eff=8.0, n_sersic=1.8, q=0.62, theta_deg=-45.0, color_ratio=(1.1, 1.0, 1.0))
    # Tidal bridge and tail
    dx_m = x_grid - merger_x
    dy_m = y_grid - merger_y
    tail1 = 90.0 * np.exp(-0.5 * ((dy_m - 0.7 * dx_m - 20)**2 / 16.0)) * np.exp(-0.5 * (dx_m / 40.0)**2)
    tail2 = 75.0 * np.exp(-0.5 * ((dy_m + 0.5 * dx_m + 15)**2 / 14.0)) * np.exp(-0.5 * (dx_m / 45.0)**2)
    r_band += tail1 * 1.1 + tail2 * 1.0
    g_band += tail1 * 0.9 + tail2 * 0.9
    b_band += tail1 * 0.8 + tail2 * 0.7
    sources.append({
        "name": "Tidal Merger Pair NGC-7327M",
        "cx": merger_x, "cy": merger_y, "type": "INTERACTING_MERGER",
        "redshift": 0.52, "description": "Major galaxy merger showing dual nuclei, tidal interaction bridge, and active starburst plumes."
    })

    # 5. Add Ultra High-Redshift Dropout Galaxy (Visible only in F444W, faint in F200W, absent in F090W)
    highz_x, highz_y = cx_core + 290, cy_core - 260
    add_sersic(highz_x, highz_y, flux=190.0, r_eff=3.0, n_sersic=1.2, q=0.75, theta_deg=10.0, color_ratio=(2.8, 0.4, 0.02))
    sources.append({
        "name": "Candidate High-z Galaxy (GL-z9.8)",
        "cx": highz_x, "cy": highz_y, "type": "HIGH_REDSHIFT_CANDIDATE",
        "redshift": 9.85, "description": "Lyman-break dropout candidate with extreme red infrared color index ((F444W - F090W) > 3.5 mag)."
    })

    # 6. Add Iconic JWST Hexagonal Diffraction Spike Artifact (Bright foreground Milky Way star)
    star_x, star_y = cx_core + 350, cy_core + 280
    add_sersic(star_x, star_y, flux=3500.0, r_eff=4.0, n_sersic=1.0, q=1.0, theta_deg=0.0, color_ratio=(1.0, 1.0, 1.0))
    # 6 primary spikes (at 0, 60, 120, 180, 240, 300 deg)
    for spike_deg in [0, 60, 120]:
        rad = np.radians(spike_deg)
        cos_s, sin_s = np.cos(rad), np.sin(rad)
        dx_s = x_grid - star_x
        dy_s = y_grid - star_y
        dist_along = dx_s * cos_s + dy_s * sin_s
        dist_perp = np.abs(-dx_s * sin_s + dy_s * cos_s)
        spike = 400.0 * np.exp(-dist_perp / 1.1) * np.exp(-np.abs(dist_along) / 220.0)
        r_band += spike * 1.0
        g_band += spike * 1.0
        b_band += spike * 1.0
    sources.append({
        "name": "Milky Way Foreground Star (Diffraction Artifact)",
        "cx": star_x, "cy": star_y, "type": "DIFFRACTION_SPIKE_ARTIFACT",
        "redshift": 0.0, "description": "Bright Galactic star exhibiting characteristic JWST 6-fold hexagonal beryllium mirror diffraction spikes."
    })

    # 7. Add population of 60 background & cluster galaxies scattered throughout field
    for i in range(60):
        # Cluster concentration towards center
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.exponential(scale=240.0) + 40.0
        gx = int(cx_core + dist * np.cos(angle))
        gy = int(cy_core + dist * np.sin(angle))
        if 40 <= gx < width - 40 and 40 <= gy < height - 40:
            flux = float(np.random.uniform(40.0, 380.0))
            reff = float(np.random.uniform(2.0, 11.0))
            sersic_n = float(np.random.choice([1.0, 2.5, 4.0]))
            q = float(np.random.uniform(0.35, 0.95))
            theta = float(np.random.uniform(0, 180))
            
            # Distance-dependent color
            r_c = float(np.random.uniform(0.9, 1.6))
            g_c = float(np.random.uniform(0.8, 1.2))
            b_c = float(np.random.uniform(0.5, 1.1))
            add_sersic(gx, gy, flux, reff, sersic_n, q, theta, (r_c, g_c, b_c))
            
            if i % 10 == 0:
                sources.append({
                    "name": f"Field Galaxy #{i+1:02d}",
                    "cx": gx, "cy": gy, "type": "COMPACT_ELLIPTICAL_OR_CLUSTER",
                    "redshift": round(float(np.random.uniform(0.4, 1.8)), 2),
                    "description": "Standard quiescent cluster or field member galaxy."
                })

    wcs_header = {
        "TELESCOP": "JWST",
        "INSTRUME": "NIRCAM",
        "OBJECT": "SMACS J0723.3-7327",
        "RA_DEG": 110.8208,
        "DEC_DEG": -73.4542,
        "PIXSCALE": 0.031, # arcseconds per pixel
        "FILTERS": ["F444W (R)", "F200W (G)", "F090W (B)"],
        "EXPTIME": 45000.0, # seconds
        "DIMENSIONS": [width, height]
    }

    return r_band, g_band, b_band, sources, wcs_header
