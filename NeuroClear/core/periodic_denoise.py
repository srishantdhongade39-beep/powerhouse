"""
periodic_denoise.py
---------------------
Responsible for removing periodic/structured scanner artifact noise
identified in the frequency domain via adaptive notch filtering
(Gaussian or Butterworth notch reject filters).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def create_notch_filter(
    image_shape: Tuple[int, int],
    noise_info: Dict[str, Any],
    notch_radius: float = 6.0,
    notch_order: int = 2,
    filter_type: str = "gaussian",
) -> np.ndarray:
    """
    Construct a frequency-domain notch reject filter mask H(u, v) in [0, 1]
    designed to suppress the periodic noise components identified by
    noise_analysis.detect_periodic_noise().

    Supports Gaussian notch reject (smooth, zero ringing) and Butterworth notch reject.

    Args:
        image_shape: (height, width) of the target image.
        noise_info: Output dict from detect_periodic_noise() containing 'peaks'.
        notch_radius: D0 radius parameter controlling attenuation bandwidth around peaks.
        notch_order: Order of the Butterworth filter (if filter_type='butterworth').
        filter_type: 'gaussian' (default) or 'butterworth'.

    Returns:
        2D numpy array mask of shape image_shape with values in [0, 1].
    """
    h, w = image_shape
    cr, cc = h // 2, w // 2

    peaks = noise_info.get("peaks", [])
    if not peaks:
        return np.ones((h, w), dtype=np.float64)

    y_grid, x_grid = np.ogrid[:h, :w]
    d0 = max(1.0, float(notch_radius))
    notch_mask = np.ones((h, w), dtype=np.float64)

    for peak in peaks:
        u = peak.get("u", 0)
        v = peak.get("v", 0)

        # Peak location in shifted frequency coordinates
        p1_r = cr + u
        p1_c = cc + v

        # Conjugate symmetric location
        p2_r = cr - u
        p2_c = cc - v

        d1_sq = (y_grid - p1_r) ** 2 + (x_grid - p1_c) ** 2
        d2_sq = (y_grid - p2_r) ** 2 + (x_grid - p2_c) ** 2

        if filter_type.lower() == "butterworth":
            n = max(1, notch_order)
            d1 = np.sqrt(d1_sq)
            d2 = np.sqrt(d2_sq)
            # Standard Butterworth notch formulation
            term1 = 1.0 / (1.0 + (d0 / np.maximum(d1, 1e-6)) ** (2 * n))
            term2 = 1.0 / (1.0 + (d0 / np.maximum(d2, 1e-6)) ** (2 * n))
            current_notch = term1 * term2
        else:
            # Gaussian notch reject: H(u,v) = (1 - exp(-D1^2/(2*D0^2))) * (1 - exp(-D2^2/(2*D0^2)))
            h1 = 1.0 - np.exp(-d1_sq / (2.0 * (d0 ** 2)))
            h2 = 1.0 - np.exp(-d2_sq / (2.0 * (d0 ** 2)))
            current_notch = h1 * h2

        notch_mask *= current_notch

    return np.clip(notch_mask, 0.0, 1.0)


def remove_periodic_noise(
    image: np.ndarray,
    noise_info: Optional[Dict[str, Any]] = None,
    notch_radius: float = 6.0,
    filter_type: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply frequency-domain notch filtering to remove periodic scanner noise
    from a CT image while preserving overall HU calibration and spatial detail.

    Args:
        image: 2D numpy array (single slice).
        noise_info: Optional pre-computed periodic noise info (from detect_periodic_noise()).
            If None, computed automatically.
        notch_radius: Notch filter radius D0 (bandwidth).
        filter_type: 'gaussian' or 'butterworth'.

    Returns:
        Tuple of:
            - denoised_image: 2D numpy array (float32), same shape and HU scale.
            - filter_mask: 2D numpy array of the frequency notch mask applied.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    if noise_info is None:
        from core.noise_analysis import detect_periodic_noise
        noise_info = detect_periodic_noise(img)

    peaks = noise_info.get("peaks", [])
    notch_mask = create_notch_filter(
        (h, w),
        noise_info,
        notch_radius=notch_radius,
        filter_type=filter_type
    )

    if not peaks:
        return img.astype(np.float32), notch_mask

    # Forward 2D FFT
    fft_val = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_val)

    # Apply notch reject mask
    filtered_shifted = fft_shifted * notch_mask

    # Inverse 2D FFT
    filtered_fft = np.fft.ifftshift(filtered_shifted)
    denoised_spatial = np.real(np.fft.ifft2(filtered_fft))

    return denoised_spatial.astype(np.float32), notch_mask
