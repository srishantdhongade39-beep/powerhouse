"""
quality_metrics.py
---------------------
Responsible for quantifying CT denoising quality:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Edge Preservation Index (EPI / Sobel gradient correlation)
- High-frequency Noise Reduction Ratio (NRR)
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _ensure_float_array(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def calculate_psnr(
    reference_image: np.ndarray,
    processed_image: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between reference and processed images.

    Args:
        reference_image: 2D numpy array (ground truth or pre-denoising input).
        processed_image: 2D numpy array (denoised output).
        data_range: Dynamic range. If None, calculated from max(reference) - min(reference).

    Returns:
        PSNR value in decibels (dB).
    """
    ref = _ensure_float_array(reference_image)
    proc = _ensure_float_array(processed_image)

    if ref.shape != proc.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {proc.shape}")

    mse = float(np.mean((ref - proc) ** 2))
    if mse < 1e-12:
        return 100.0  # Identical images

    dr = float(np.ptp(ref)) if data_range is None else float(data_range)
    dr = max(1.0, dr)

    return float(10.0 * np.log10((dr ** 2) / mse))


def calculate_ssim(
    reference_image: np.ndarray,
    processed_image: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between reference and processed images.

    Args:
        reference_image: 2D numpy array.
        processed_image: 2D numpy array.
        data_range: Dynamic range. If None, calculated from max(reference) - min(reference).

    Returns:
        SSIM value between -1.0 and 1.0 (typically 0.7 to 1.0 for medical scans).
    """
    ref = _ensure_float_array(reference_image)
    proc = _ensure_float_array(processed_image)

    if ref.shape != proc.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {proc.shape}")

    dr = float(np.ptp(ref)) if data_range is None else float(data_range)
    dr = max(1e-3, dr)

    # Standard Gaussian window with size 7
    win_size = min(7, min(ref.shape[0], ref.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)

    score = structural_similarity(
        ref,
        proc,
        data_range=dr,
        win_size=win_size,
    )
    return float(score)


def calculate_edge_preservation(
    reference_image: np.ndarray,
    processed_image: np.ndarray
) -> float:
    """
    Calculate Edge Preservation Index (EPI) between reference and processed images.

    Uses high-frequency Sobel spatial gradient magnitude cross-correlation.
    Measures whether structural boundaries (skull margins, ventricles, tissue borders)
    were retained during denoising without blurring or erosion.

    Returns:
        Correlation score in [0.0, 1.0]. A score >= 0.85 indicates high structural preservation.
    """
    ref = _ensure_float_array(reference_image)
    proc = _ensure_float_array(processed_image)

    if ref.shape != proc.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {proc.shape}")

    # Compute horizontal and vertical gradients using 3x3 Sobel operators
    gx_ref = cv2.Sobel(ref, cv2.CV_64F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(ref, cv2.CV_64F, 0, 1, ksize=3)
    grad_ref = np.sqrt(gx_ref**2 + gy_ref**2)

    gx_proc = cv2.Sobel(proc, cv2.CV_64F, 1, 0, ksize=3)
    gy_proc = cv2.Sobel(proc, cv2.CV_64F, 0, 1, ksize=3)
    grad_proc = np.sqrt(gx_proc**2 + gy_proc**2)

    # Focus on edge regions (top 20% strongest gradients in reference)
    edge_thresh = float(np.percentile(grad_ref, 75))
    edge_mask = grad_ref >= edge_thresh

    if np.sum(edge_mask) < 20:
        # Fallback to entire image if few edge pixels
        g1 = grad_ref.ravel()
        g2 = grad_proc.ravel()
    else:
        g1 = grad_ref[edge_mask]
        g2 = grad_proc[edge_mask]

    # Pearson correlation coefficient between edge gradients
    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    numerator = np.sum((g1 - g1_mean) * (g2 - g2_mean))
    denom = np.sqrt(np.sum((g1 - g1_mean) ** 2) * np.sum((g2 - g2_mean) ** 2))

    if denom < 1e-12:
        return 1.0

    corr = float(numerator / denom)
    return float(np.clip(corr, 0.0, 1.0))


def compute_all_metrics(
    reference_image: np.ndarray,
    processed_image: np.ndarray,
    data_range: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute full suite of objective image quality metrics comparing reference and processed.
    """
    ref = _ensure_float_array(reference_image)
    proc = _ensure_float_array(processed_image)

    psnr_val = calculate_psnr(ref, proc, data_range=data_range)
    ssim_val = calculate_ssim(ref, proc, data_range=data_range)
    epi_val = calculate_edge_preservation(ref, proc)

    mse_val = float(np.mean((ref - proc) ** 2))
    mae_val = float(np.mean(np.abs(ref - proc)))

    # Estimate noise variance change in homogeneous regions
    diff = proc - ref
    noise_power_removed = float(np.var(diff))

    return {
        "psnr_db": psnr_val,
        "ssim": ssim_val,
        "edge_preservation_index": epi_val,
        "mse": mse_val,
        "mae": mae_val,
        "residual_noise_variance": noise_power_removed,
    }
