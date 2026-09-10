"""
poisson_denoise.py
--------------------
Responsible for reducing Poisson (quantum) noise in CT images while
preserving anatomical structures (edges, tissue boundaries, gray/white
matter contrast).

Implements edge-preserving classical algorithms:
- Non-Local Means (NLM) via scikit-image
- Bilateral Filtering via OpenCV
- Total Variation (TV) Chambolle via scikit-image
- Multi-scale Wavelet Denoising via scikit-image
- Optional Anscombe Variance-Stabilizing Transform for Poisson statistics
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from scipy import ndimage
from skimage.restoration import (
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
)


def _robust_estimate_sigma(image: np.ndarray) -> float:
    """
    Robust edge-insensitive noise standard deviation estimator (Immerkaer / Donoho).
    Convolves with high-frequency 3x3 Laplacian operator:
      [ 1, -2,  1]
      [-2,  4, -2]
      [ 1, -2,  1]
    and computes Median Absolute Deviation (MAD) / (0.6745 * 6.0).
    Captures high-frequency Poisson/quantum fluctuations while median rejects edge boundaries.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2 or img.shape[0] < 5 or img.shape[1] < 5:
        return 0.03
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    filtered = ndimage.convolve(img, kernel, mode="reflect")
    mad = float(np.median(np.abs(filtered)))
    sigma_est = float(mad / (0.6745 * 6.0))
    return max(0.008, sigma_est)


def _anscombe_transform(x: np.ndarray) -> np.ndarray:
    """Forward Anscombe transform: converts Poisson counts into unit-variance Gaussian noise."""
    return 2.0 * np.sqrt(np.maximum(0.0, x) + (3.0 / 8.0))


def _inverse_anscombe_transform(y: np.ndarray) -> np.ndarray:
    """Asymptotically unbiased inverse Anscombe transform."""
    return (y / 2.0) ** 2 - (3.0 / 8.0)


def denoise_poisson(
    image: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Reduce Poisson/quantum noise while strictly preserving anatomical edges.

    Args:
        image: 2D numpy array (single slice, typically in HU or windowed space).
        params: Optional configuration dictionary:
            - 'method': 'nlm' (default), 'bilateral', 'tv', or 'wavelet'
            - 'strength': float factor (0.1 to 3.0, default 1.0)
            - 'use_anscombe': bool (default False)

    Returns:
        Denoised 2D numpy array with same shape and intensity scale as input.
    """
    img = np.asarray(image, dtype=np.float32)
    p = params or {}
    method = str(p.get("method", "nlm")).lower()
    strength = max(0.05, float(p.get("strength", 1.0)))
    use_anscombe = bool(p.get("use_anscombe", False))

    orig_min = float(np.min(img))
    orig_max = float(np.max(img))
    orig_range = orig_max - orig_min

    if orig_range < 1e-6:
        return img.copy()

    # Normalize to [0.0, 1.0] for stable numerical processing across classical CV filters
    norm_img = (img - orig_min) / orig_range

    # Apply Anscombe variance-stabilizing transform if requested
    if use_anscombe:
        # Scale to photon count range (~100 to 1000 counts typical for CT detectors)
        count_scale = 500.0
        photon_counts = norm_img * count_scale
        transformed = _anscombe_transform(photon_counts)
        t_min = float(np.min(transformed))
        t_max = float(np.max(transformed))
        t_range = max(1e-6, t_max - t_min)
        filter_input = ((transformed - t_min) / t_range).astype(np.float32)
    else:
        filter_input = norm_img.astype(np.float32)

    # Estimate noise standard deviation in normalized space via robust Laplacian MAD
    sigma_est = _robust_estimate_sigma(filter_input)

    # Dispatch to edge-preserving filter
    if method == "bilateral":
        # OpenCV bilateralFilter expects float32
        # d: diameter of pixel neighborhood calibrated to preserve fine bone trabeculae and sharp interfaces
        d = int(np.clip(3 + 2 * int(strength * 2), 3, 9))
        sigma_color = float(np.clip(1.30 * strength * sigma_est, 0.035, 0.35))
        sigma_space = float(np.clip(3.0 * strength, 2.0, 9.0))
        denoised_norm = cv2.bilateralFilter(
            filter_input,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=cv2.BORDER_REFLECT,
        )

    elif method == "tv":
        # Total Variation Chambolle denoising
        tv_weight = float(np.clip(0.18 * strength * sigma_est, 0.01, 0.22))
        denoised_norm = denoise_tv_chambolle(
            filter_input,
            weight=tv_weight,
            max_num_iter=100,
        )

    elif method == "wavelet":
        # Wavelet thresholding (BayesShrink) with fallback if PyWavelets is unavailable
        try:
            denoised_norm = denoise_wavelet(
                filter_input,
                method="BayesShrink",
                mode="soft",
                wavelet="db4",
                rescale_sigma=True,
            )
        except (ImportError, ModuleNotFoundError):
            # Fallback to edge-preserving bilateral filter when PyWavelets is not installed
            d = int(np.clip(3 + 2 * int(strength * 2), 3, 9))
            sigma_color = float(np.clip(1.30 * strength * sigma_est, 0.035, 0.35))
            sigma_space = float(np.clip(3.0 * strength, 2.0, 9.0))
            denoised_norm = cv2.bilateralFilter(
                filter_input,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space,
                borderType=cv2.BORDER_REFLECT,
            )

    else:
        # Default: Non-Local Means (NLM)
        # Calibrated with robust Immerkaer sigma estimation for high-fidelity CT texture retention
        h_param = max(0.015, 0.95 * strength * sigma_est)
        denoised_norm = denoise_nl_means(
            filter_input,
            h=h_param,
            fast_mode=True,
            patch_size=3,
            patch_distance=5,
        )

    # Classical Multi-Scale Detail Preservation & Natural Texture Retention
    detail_boost = max(1.0, float(p.get("detail_boost", 1.0)))
    texture_blend = float(np.clip(p.get("texture_blend", 0.10), 0.0, 0.35))

    # Extract high-frequency micro-texture residual layer
    texture_residual = filter_input - denoised_norm

    if detail_boost > 1.001:
        # Apply soft coring threshold calibrated to noise sigma to isolate anatomical structures from photon fluctuations
        coring_tau = float(np.clip(0.65 * strength * sigma_est, 0.005, 0.08))
        detail_clean = np.sign(texture_residual) * np.maximum(0.0, np.abs(texture_residual) - coring_tau)
        # Boost true anatomical micro-structures (sulci, gyri, trabeculae, cortices)
        denoised_norm = np.clip(denoised_norm + (detail_boost - 1.0) * detail_clean, 0.0, 1.0)

    # Blend subtle natural texture to preserve realistic clinical CT parenchyma appearance (prevents plastic/waxy artifact)
    if texture_blend > 0.001:
        denoised_norm = np.clip(denoised_norm + (texture_blend * texture_residual), 0.0, 1.0)

    # Invert Anscombe transform if applied
    if use_anscombe:
        restored_anscombe = (denoised_norm * t_range) + t_min
        untransformed = _inverse_anscombe_transform(restored_anscombe)
        denoised_norm = np.clip(untransformed / count_scale, 0.0, 1.0)

    # Rescale back to original input HU / intensity scale
    denoised_final = (denoised_norm * orig_range) + orig_min
    return denoised_final.astype(np.float32)
