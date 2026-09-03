"""
FITS and Multi-band Infrared Image Processor for JWST Deep Field Observations.
Implements astronomical contrast stretching (Asinh, Log, ZScale) and Lupton RGB composite synthesis.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any

try:
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, AsinhStretch
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def zscale_clip(data: np.ndarray, nsamples: int = 1000, contrast: float = 0.25) -> Tuple[float, float]:
    """
    Robust IRAF-style zscale algorithm to find optimal display bounds.
    Falls back to percentile clipping if astropy is not available.
    """
    if ASTROPY_AVAILABLE:
        try:
            interval = ZScaleInterval(nsamples=nsamples, contrast=contrast)
            vmin, vmax = interval.get_limits(data)
            return float(vmin), float(vmax)
        except Exception:
            pass

    # High-performance numpy fallback
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite_data, 1.0))
    vmax = float(np.percentile(finite_data, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-5
    return vmin, vmax


def asinh_stretch(data: np.ndarray, Q: float = 8.0, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    Astronomical Asinh stretch (Lupton et al. 2004) to reveal faint diffuse outer structures
    without saturating high-surface-brightness galaxy cores or diffraction spikes.
    """
    if vmin is None or vmax is None:
        zmin, zmax = zscale_clip(data)
        vmin = zmin if vmin is None else vmin
        vmax = zmax if vmax is None else vmax

    # Normalize between 0 and 1
    denom = max(vmax - vmin, 1e-8)
    norm = np.clip((data - vmin) / denom, 0.0, None)
    
    # Asinh non-linear transformation
    stretched = np.arcsinh(Q * norm) / np.arcsinh(Q)
    return np.clip(stretched, 0.0, 1.0)


def log_stretch(data: np.ndarray, a: float = 1000.0, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Logarithmic contrast stretching."""
    if vmin is None or vmax is None:
        vmin, vmax = zscale_clip(data)
    denom = max(vmax - vmin, 1e-8)
    norm = np.clip((data - vmin) / denom, 0.0, 1.0)
    stretched = np.log(a * norm + 1.0) / np.log(a + 1.0)
    return np.clip(stretched, 0.0, 1.0)


def create_rgb_composite(
    r_band: np.ndarray,
    g_band: np.ndarray,
    b_band: np.ndarray,
    stretch: str = "asinh",
    Q: float = 8.0
) -> np.ndarray:
    """
    Creates an astronomical 3-color RGB composite array from JWST multi-band infrared channels.
    Typically:
      R = F356W / F444W (long-wavelength infrared, red-shifted dust & high-z)
      G = F200W / F277W (mid-wavelength infrared)
      B = F090W / F150W (short-wavelength infrared, younger stellar populations)
    """
    channels = [r_band, g_band, b_band]
    norm_channels = []
    
    for ch in channels:
        if stretch == "asinh":
            stretched = asinh_stretch(ch, Q=Q)
        elif stretch == "log":
            stretched = log_stretch(ch)
        else:
            vmin, vmax = zscale_clip(ch)
            denom = max(vmax - vmin, 1e-8)
            stretched = np.clip((ch - vmin) / denom, 0.0, 1.0)
        norm_channels.append((stretched * 255).astype(np.uint8))

    rgb = np.stack(norm_channels, axis=-1)
    return rgb


def array_to_png_bytes(img_array: np.ndarray) -> bytes:
    """Convert numpy array (2D grayscale or 3D RGB) to PNG bytes."""
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            vmin, vmax = np.percentile(img_array, [1, 99.5])
            denom = max(vmax - vmin, 1e-5)
            img_array = (np.clip((img_array - vmin) / denom, 0, 1) * 255).astype(np.uint8)

    if img_array.ndim == 2:
        image = Image.fromarray(img_array, mode="L")
    elif img_array.ndim == 3:
        if img_array.shape[2] == 3:
            image = Image.fromarray(img_array, mode="RGB")
        elif img_array.shape[2] == 4:
            image = Image.fromarray(img_array, mode="RGBA")
        else:
            raise ValueError(f"Unsupported channel count: {img_array.shape[2]}")
    else:
        raise ValueError(f"Unsupported array dimensions: {img_array.ndim}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def array_to_base64_png(img_array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded data URL string for web rendering."""
    png_bytes = array_to_png_bytes(img_array)
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def load_fits_file(filepath: str, ext: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a FITS file and returns (data_array, header_dict)."""
    if not ASTROPY_AVAILABLE:
        raise RuntimeError("astropy is required to load .fits files directly.")
    
    with fits.open(filepath) as hdul:
        # Find first HDU with valid data
        target_hdu = hdul[ext]
        if target_hdu.data is None:
            for h in hdul:
                if h.data is not None:
                    target_hdu = h
                    break
        
        data = np.nan_to_num(target_hdu.data.astype(np.float32))
        header = dict(target_hdu.header)
        return data, header
