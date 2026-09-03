"""
Vectorized Galaxy Morphology Statistics for Extragalactic Astronomy & Lens Detection.
Computes non-parametric indicators: Gini, M20, Concentration (C), Asymmetry (A), Ellipticity,
and Curvilinear Tangential Shear metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_gini(image_cutout: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Computes Gini coefficient (Lotz et al. 2004).
    Measures distribution of light among pixels: 0 = uniform, 1 = concentrated in single pixel.
    """
    if mask is not None:
        pixels = image_cutout[mask]
    else:
        pixels = image_cutout.flatten()
    
    # Positive fluxes only
    pixels = pixels[pixels > 0]
    n = len(pixels)
    if n < 4:
        return 0.0
    
    sorted_pixels = np.sort(np.abs(pixels))
    mean_flux = np.mean(sorted_pixels)
    if mean_flux == 0:
        return 0.0

    rank = np.arange(1, n + 1)
    gini = np.sum((2 * rank - n - 1) * sorted_pixels) / (mean_flux * n * (n - 1))
    return float(np.clip(gini, 0.0, 1.0))


def compute_m20(image_cutout: np.ndarray, cx: float, cy: float, mask: np.ndarray = None) -> float:
    """
    Computes M20 moment of light (Lotz et al. 2004).
    Normalized second-order moment of the brightest 20% of the galaxy flux.
    Mergers / double nuclei have M20 > -1.1. Normal ellipticals have M20 < -1.8.
    """
    h, w = image_cutout.shape
    y_coords, x_coords = np.indices((h, w))
    
    if mask is not None:
        valid = mask & (image_cutout > 0)
    else:
        valid = (image_cutout > 0)
    
    if np.count_nonzero(valid) < 5:
        return -1.5

    f = image_cutout[valid]
    x = x_coords[valid]
    y = y_coords[valid]

    # Spatial distances squared from centroid
    r2 = (x - cx)**2 + (y - cy)**2
    m_pixel = f * r2
    m_tot = np.sum(m_pixel)
    if m_tot <= 0:
        return -2.0

    # Sort pixels by flux descending
    sort_idx = np.argsort(f)[::-1]
    sorted_f = f[sort_idx]
    sorted_m = m_pixel[sort_idx]

    cum_f = np.cumsum(sorted_f)
    tot_f = cum_f[-1]
    if tot_f <= 0:
        return -2.0

    # Find brightest 20%
    idx_20 = np.searchsorted(cum_f, 0.20 * tot_f)
    m_20_sum = np.sum(sorted_m[:idx_20 + 1])
    
    val = max(m_20_sum / m_tot, 1e-10)
    return float(np.log10(val))


def compute_concentration(image_cutout: np.ndarray, cx: float, cy: float) -> Tuple[float, float, float]:
    """
    Computes Concentration index C = 5 * log10(r80 / r20).
    Also returns r20 and r80 in pixel units.
    """
    h, w = image_cutout.shape
    y_coords, x_coords = np.indices((h, w))
    r = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    
    positive = (image_cutout > 0)
    if np.count_nonzero(positive) < 5:
        return 2.5, 2.0, 5.0

    r_flat = r[positive]
    f_flat = image_cutout[positive]
    
    sort_idx = np.argsort(r_flat)
    sorted_r = r_flat[sort_idx]
    sorted_f = f_flat[sort_idx]
    
    cum_f = np.cumsum(sorted_f)
    tot_f = cum_f[-1]
    if tot_f <= 0:
        return 2.5, 2.0, 5.0

    idx_20 = np.searchsorted(cum_f, 0.20 * tot_f)
    idx_80 = np.searchsorted(cum_f, 0.80 * tot_f)
    
    r20 = max(float(sorted_r[min(idx_20, len(sorted_r) - 1)]), 0.5)
    r80 = max(float(sorted_r[min(idx_80, len(sorted_r) - 1)]), r20 + 0.5)
    
    c_index = 5.0 * np.log10(r80 / r20)
    return float(np.clip(c_index, 1.0, 5.5)), r20, r80


def compute_asymmetry(image_cutout: np.ndarray, cx: float, cy: float) -> float:
    """
    Computes rotational asymmetry A = sum|I_0 - I_180| / (2 * sum|I_0|).
    """
    h, w = image_cutout.shape
    # Center cutout symmetrically around (cx, cy)
    r_max = int(min(cx, w - 1 - cx, cy, h - 1 - cy))
    if r_max < 3:
        return 0.1
    
    x_min, x_max = int(round(cx)) - r_max, int(round(cx)) + r_max + 1
    y_min, y_max = int(round(cy)) - r_max, int(round(cy)) + r_max + 1
    
    sub = image_cutout[y_min:y_max, x_min:x_max]
    rot = np.rot90(sub, 2)
    
    denom = 2.0 * np.sum(np.abs(sub))
    if denom <= 0:
        return 0.0
    
    diff = np.sum(np.abs(sub - rot))
    asym = diff / denom
    return float(np.clip(asym, 0.0, 1.0))


def compute_shape_moments(image_cutout: np.ndarray, cx: float, cy: float) -> Dict[str, float]:
    """
    Second-order spatial moments for semi-major axis (a), semi-minor axis (b),
    ellipticity (1 - b/a), and position angle (theta).
    """
    h, w = image_cutout.shape
    y_coords, x_coords = np.indices((h, w))
    
    weights = np.maximum(image_cutout, 0.0)
    tot_weight = np.sum(weights)
    if tot_weight <= 0:
        return {"a": 2.0, "b": 2.0, "ellipticity": 0.0, "theta_deg": 0.0}

    dx = x_coords - cx
    dy = y_coords - cy

    mu_xx = np.sum(weights * dx * dx) / tot_weight
    mu_yy = np.sum(weights * dy * dy) / tot_weight
    mu_xy = np.sum(weights * dx * dy) / tot_weight

    # Eigenvalues of moment matrix
    delta = np.sqrt(max((mu_xx - mu_yy)**2 + 4 * mu_xy**2, 0.0))
    lambda1 = max(0.5 * (mu_xx + mu_yy + delta), 1e-4)
    lambda2 = max(0.5 * (mu_xx + mu_yy - delta), 1e-4)

    a = 2.0 * np.sqrt(lambda1)
    b = 2.0 * np.sqrt(lambda2)
    ellipticity = 1.0 - (b / a)
    theta_rad = 0.5 * np.arctan2(2 * mu_xy, mu_xx - mu_yy)
    theta_deg = float(np.degrees(theta_rad))

    return {
        "semi_major": float(a),
        "semi_minor": float(b),
        "ellipticity": float(np.clip(ellipticity, 0.0, 0.99)),
        "theta_deg": theta_deg
    }


def compute_arc_shear_metric(
    image_cutout: np.ndarray,
    ellipticity: float,
    cx_global: float,
    cy_global: float,
    cluster_center: Tuple[float, float] = (500.0, 500.0)
) -> Dict[str, Any]:
    """
    Calculates gravitational lensing shear orientation relative to cluster center.
    In strong gravitational lensing, background galaxies are sheared tangentially to the cluster core.
    Tangential alignment: |theta_lens - theta_cluster_perpendicular| ~ 0.
    """
    # Angle from cluster core to object
    dx_cluster = cx_global - cluster_center[0]
    dy_cluster = cy_global - cluster_center[1]
    dist_to_core = np.sqrt(dx_cluster**2 + dy_cluster**2)
    radial_angle_deg = np.degrees(np.arctan2(dy_cluster, dx_cluster))
    tangential_angle_deg = (radial_angle_deg + 90.0) % 180.0

    # Arc curvature detection: measure circular variance / bent contour
    # Higher thresholded contour bend indicates an Einstein arc
    thresh = np.percentile(image_cutout, 75)
    binary = image_cutout > thresh
    if np.count_nonzero(binary) > 10:
        y_pts, x_pts = np.where(binary)
        # Compute contour curvature via polynomial fit (y vs x or rotated)
        cov = np.cov(x_pts, y_pts)
        evals = np.real(np.linalg.eigvalsh(cov))
        aspect = float(max(evals) / max(min(evals), 1e-5))
        curvature_score = float(np.clip((aspect - 1.0) / 10.0 * ellipticity, 0.0, 1.0))
    else:
        curvature_score = 0.1

    # Lens probability heuristic score based on morphology
    lens_indicator = 0.45 * ellipticity + 0.35 * curvature_score + (0.2 if dist_to_core < 450 else 0.0)
    
    return {
        "distance_to_cluster_center_px": float(dist_to_core),
        "tangential_predicted_deg": float(tangential_angle_deg),
        "curvature_score": float(np.clip(curvature_score, 0.0, 1.0)),
        "lens_geometric_score": float(np.clip(lens_indicator, 0.0, 1.0))
    }


def extract_morphology_profile(
    cutout: np.ndarray,
    cx_local: float,
    cy_local: float,
    cx_global: float,
    cy_global: float,
    cluster_center: Tuple[float, float] = (500.0, 500.0)
) -> Dict[str, Any]:
    """
    Comprehensive morphology extraction combining non-parametric indicators and lensing shear metrics.
    """
    # Use central band or 2D luminance
    if cutout.ndim == 3:
        # Weighted luminance
        data_2d = 0.299 * cutout[..., 0] + 0.587 * cutout[..., 1] + 0.114 * cutout[..., 2]
    else:
        data_2d = cutout.astype(np.float32)

    # Clean background
    bkg = np.median(data_2d)
    data_sub = np.maximum(data_2d - bkg, 0.0)

    moments = compute_shape_moments(data_sub, cx_local, cy_local)
    gini = compute_gini(data_sub)
    m20 = compute_m20(data_sub, cx_local, cy_local)
    c_index, r20, r80 = compute_concentration(data_sub, cx_local, cy_local)
    asymmetry = compute_asymmetry(data_sub, cx_local, cy_local)
    shear = compute_arc_shear_metric(
        data_sub,
        moments["ellipticity"],
        cx_global,
        cy_global,
        cluster_center
    )

    return {
        "gini": round(gini, 4),
        "m20": round(m20, 4),
        "concentration": round(c_index, 4),
        "asymmetry": round(asymmetry, 4),
        "ellipticity": round(moments["ellipticity"], 4),
        "semi_major_px": round(moments["semi_major"], 2),
        "semi_minor_px": round(moments["semi_minor"], 2),
        "position_angle_deg": round(moments["theta_deg"], 1),
        "curvature_score": round(shear["curvature_score"], 4),
        "lens_geometric_score": round(shear["lens_geometric_score"], 4),
        "dist_to_cluster_core_px": round(shear["distance_to_cluster_center_px"], 1),
        "is_merger_candidate": bool((m20 > -1.2 and gini < 0.55) or asymmetry > 0.40),
        "is_lens_candidate": bool(moments["ellipticity"] > 0.55 and shear["lens_geometric_score"] > 0.45)
    }
