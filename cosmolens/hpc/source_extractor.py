"""
Astronomical Source Extraction and Segmentation Engine.
Vectorized peak detection, centroid refinement, aperture photometry,
and multi-spectral cutout extraction.
"""

import numpy as np
from scipy.ndimage import label, find_objects, maximum_filter, gaussian_filter
from typing import List, Dict, Any, Tuple
from cosmolens.hpc.morphology import extract_morphology_profile
from cosmolens.hpc.fits_processor import array_to_base64_png, asinh_stretch


def estimate_background_2d(data: np.ndarray, box_size: int = 64) -> Tuple[np.ndarray, float]:
    """
    Robust 2D background estimation with spatial mesh and sigma clipping.
    """
    h, w = data.shape
    # Fast global statistics
    med = float(np.median(data))
    # Robust standard deviation via MAD (Median Absolute Deviation)
    mad = float(np.median(np.abs(data - med)))
    std = max(1.4826 * mad, 1e-4)

    # Coarse grid estimation for non-uniform sky gradients
    grid_h = max(h // box_size, 2)
    grid_w = max(w // box_size, 2)
    bkg_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    dy = h / grid_h
    dx = w / grid_w

    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = int(i * dy), int((i + 1) * dy)
            x1, x2 = int(j * dx), int((j + 1) * dx)
            patch = data[y1:y2, x1:x2]
            # Clip outliers (stars/galaxies)
            valid = patch[np.abs(patch - med) < 3.0 * std]
            bkg_grid[i, j] = np.median(valid) if len(valid) > 10 else med

    # Upsample background grid to full image resolution
    from scipy.ndimage import zoom
    bkg_map = zoom(bkg_grid, (h / grid_h, w / grid_w), order=1)[:h, :w]
    return bkg_map, std


def detect_peaks_2d(
    data_sub: np.ndarray,
    bkg_std: float,
    threshold_sigma: float = 3.5,
    min_distance: int = 12
) -> List[Tuple[int, int]]:
    """
    High-performance local maximum peak detector.
    """
    threshold = threshold_sigma * bkg_std
    # Apply light Gaussian filter to suppress noise spikes
    smoothed = gaussian_filter(data_sub, sigma=1.5)
    
    # Neighborhood maximum filter
    local_max = maximum_filter(smoothed, size=min_distance) == smoothed
    detected_mask = local_max & (smoothed > threshold)
    
    y_coords, x_coords = np.where(detected_mask)
    return list(zip(x_coords, y_coords))


def extract_sources(
    rgb_data: np.ndarray,
    wcs_header: Dict[str, Any],
    threshold_sigma: float = 3.2,
    cutout_size: int = 64,
    max_sources: int = 120
) -> List[Dict[str, Any]]:
    """
    Scans the multi-band JWST mosaic, detects celestial objects, extracts cutouts,
    computes physical photometry & morphology parameters, and maps to celestial coordinates.
    """
    height, width, channels = rgb_data.shape
    # Luminance channel for detection
    detection_img = 0.299 * rgb_data[..., 0] + 0.587 * rgb_data[..., 1] + 0.114 * rgb_data[..., 2]

    # Background estimation
    bkg_map, bkg_std = estimate_background_2d(detection_img)
    data_sub = np.maximum(detection_img - bkg_map, 0.0)

    # Detect candidate peaks
    raw_peaks = detect_peaks_2d(data_sub, bkg_std, threshold_sigma=threshold_sigma, min_distance=14)
    
    # Sort candidate peaks by brightness
    peaks_sorted = sorted(raw_peaks, key=lambda pt: data_sub[pt[1], pt[0]], reverse=True)
    if len(peaks_sorted) > max_sources:
        peaks_sorted = peaks_sorted[:max_sources]

    sources = []
    cluster_center = (width / 2.0, height / 2.0)
    ra_center = wcs_header.get("RA_DEG", 110.8208)
    dec_center = wcs_header.get("DEC_DEG", -73.4542)
    pixscale = wcs_header.get("PIXSCALE", 0.031) # arcsec / pixel

    half_box = cutout_size // 2

    for idx, (px, py) in enumerate(peaks_sorted):
        # Bounds check for cutout extraction
        x1 = max(0, px - half_box)
        x2 = min(width, px + half_box)
        y1 = max(0, py - half_box)
        y2 = min(height, py + half_box)

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            continue

        cutout_rgb = rgb_data[y1:y2, x1:x2]
        cutout_det = data_sub[y1:y2, x1:x2]

        # Centroid refinement within cutout
        cy_loc, cx_loc = py - y1, px - x1
        w_cutout = np.maximum(cutout_det, 0.0)
        tot_flux = np.sum(w_cutout)
        if tot_flux > 0:
            y_indices, x_indices = np.indices(cutout_det.shape)
            cx_refined = float(np.sum(x_indices * w_cutout) / tot_flux)
            cy_refined = float(np.sum(y_indices * w_cutout) / tot_flux)
        else:
            cx_refined, cy_refined = float(cx_loc), float(cy_loc)

        cx_global = x1 + cx_refined
        cy_global = y1 + cy_refined

        # Photometry in individual NIRCam bands
        r_cut = cutout_rgb[..., 0]
        g_cut = cutout_rgb[..., 1]
        b_cut = cutout_rgb[..., 2]
        
        flux_r = float(np.sum(np.maximum(r_cut, 0.0)))
        flux_g = float(np.sum(np.maximum(g_cut, 0.0)))
        flux_b = float(np.sum(np.maximum(b_cut, 0.0)))
        total_flux = float(tot_flux)
        
        peak_val = float(detection_img[py, px])
        snr = float(peak_val / max(bkg_std, 1e-4))

        # Color indices (flux ratios)
        # Red / Short ratio: High ratio signifies high-redshift or dusty obscured system
        f444_f090_ratio = float(round(flux_r / max(flux_b, 1.0), 3))
        f200_f090_ratio = float(round(flux_g / max(flux_b, 1.0), 3))

        # Extract morphology
        morphology = extract_morphology_profile(
            cutout_rgb,
            cx_refined,
            cy_refined,
            cx_global,
            cy_global,
            cluster_center=cluster_center
        )

        # Convert pixel to celestial coordinates (WCS tangent projection)
        dx_arcsec = (cx_global - cluster_center[0]) * pixscale
        dy_arcsec = (cy_global - cluster_center[1]) * pixscale
        
        # RA increases to the left (East)
        ra_obj = ra_center - (dx_arcsec / (3600.0 * np.cos(np.radians(dec_center))))
        dec_obj = dec_center + (dy_arcsec / 3600.0)

        # Prepare base64 thumbnail for instant UI display
        # Contrast stretched for optimal visualization
        cutout_display = np.copy(cutout_rgb)
        for c in range(3):
            cutout_display[..., c] = asinh_stretch(cutout_rgb[..., c], Q=8.0) * 255.0
        thumbnail_b64 = array_to_base64_png(cutout_display.astype(np.uint8))

        source_record = {
            "id": f"JWST-CL-{idx+1:04d}",
            "x": round(cx_global, 2),
            "y": round(cy_global, 2),
            "ra_deg": round(ra_obj, 6),
            "dec_deg": round(dec_obj, 6),
            "ra_str": format_ra(ra_obj),
            "dec_str": format_dec(dec_obj),
            "snr": round(snr, 1),
            "total_flux": round(total_flux, 1),
            "peak_flux": round(peak_val, 1),
            "f444_f090_ratio": f444_f090_ratio,
            "f200_f090_ratio": f200_f090_ratio,
            "morphology": morphology,
            "thumbnail_b64": thumbnail_b64,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        }
        sources.append(source_record)

    return sources


def format_ra(ra_deg: float) -> str:
    """Format RA degrees into HH:MM:SS.ss."""
    ra_deg = ra_deg % 360.0
    hours = ra_deg / 15.0
    h = int(hours)
    minutes = (hours - h) * 60.0
    m = int(minutes)
    s = (minutes - m) * 60.0
    return f"{h:02d}h{m:02d}m{s:05.2f}s"


def format_dec(dec_deg: float) -> str:
    """Format Dec degrees into ±DD:MM:SS.s."""
    sign = "+" if dec_deg >= 0 else "-"
    d_abs = abs(dec_deg)
    d = int(d_abs)
    minutes = (d_abs - d) * 60.0
    m = int(minutes)
    s = (minutes - m) * 60.0
    return f"{sign}{d:02d}°{m:02d}'{s:04.1f}\""
