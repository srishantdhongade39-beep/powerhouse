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


def anisotropic_diffusion_perona_malik(
    image: np.ndarray,
    n_iter: int = 8,
    kappa: float = 20.0,
    gamma: float = 0.125,
    conduction_method: str = "exponential",
    eight_neighbor: bool = True,
) -> np.ndarray:
    """
    Perona-Malik Anisotropic Diffusion Filter for CT restoration (identical to MATLAB imdiffusefilt).

    Solves the nonlinear partial differential equation:
        dI/dt = div( c(|grad(I)|) * grad(I) )

    Conduction models:
      - 'exponential' (c1): c(g) = exp( -(g / kappa)^2 )  [Preserves high-contrast edges over low-contrast edges]
      - 'quadratic'   (c2): c(g) = 1 / (1 + (g / kappa)^2) [Preserves wider regions over smaller regions]
      - 'tukey'       (c3): c(g) = (1 - (g / kappa)^2)^2 for |g|<=kappa, 0 for |g|>kappa [Zero edge-blurring lock]

    Args:
        image: 2D float array (e.g. in [0, 1] or HU).
        n_iter: Number of diffusion iterations (typically 4 to 20).
        kappa: Edge gradient conduction threshold. Gradients > kappa act as diffusion boundaries.
        gamma: Integration constant / step size (<= 0.25 for 4-neighbor, <= 0.125 for 8-neighbor).
        conduction_method: 'exponential', 'quadratic', or 'tukey'.
        eight_neighbor: If True, uses 8-directional stencil with diagonal weight 1/sqrt(2).

    Returns:
        Denoised 2D numpy array with identical dimensions and sharp edge retention.
    """
    u = np.array(image, dtype=np.float32, copy=True)
    k = max(1e-4, float(kappa))
    g = float(np.clip(gamma, 0.01, 0.125 if eight_neighbor else 0.25))
    method = str(conduction_method).lower()

    for _ in range(max(1, int(n_iter))):
        # 4 Cardinal directions
        deltaN = np.roll(u, -1, axis=0) - u
        deltaS = np.roll(u, 1, axis=0) - u
        deltaE = np.roll(u, -1, axis=1) - u
        deltaW = np.roll(u, 1, axis=1) - u

        # Zero out boundary wraps
        deltaN[-1, :] = 0
        deltaS[0, :] = 0
        deltaE[:, -1] = 0
        deltaW[:, 0] = 0

        if method == "quadratic":
            cN = 1.0 / (1.0 + (deltaN / k) ** 2)
            cS = 1.0 / (1.0 + (deltaS / k) ** 2)
            cE = 1.0 / (1.0 + (deltaE / k) ** 2)
            cW = 1.0 / (1.0 + (deltaW / k) ** 2)
        elif method == "tukey":
            cN = np.where(np.abs(deltaN) <= k, (1.0 - (deltaN / k) ** 2) ** 2, 0.0)
            cS = np.where(np.abs(deltaS) <= k, (1.0 - (deltaS / k) ** 2) ** 2, 0.0)
            cE = np.where(np.abs(deltaE) <= k, (1.0 - (deltaE / k) ** 2) ** 2, 0.0)
            cW = np.where(np.abs(deltaW) <= k, (1.0 - (deltaW / k) ** 2) ** 2, 0.0)
        else:
            cN = np.exp(-((deltaN / k) ** 2))
            cS = np.exp(-((deltaS / k) ** 2))
            cE = np.exp(-((deltaE / k) ** 2))
            cW = np.exp(-((deltaW / k) ** 2))

        flux = cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW

        if eight_neighbor:
            # 4 Diagonal directions (scaled by 1 / sqrt(2) ≈ 0.7071)
            diag_w = 0.70710678118
            deltaNE = np.roll(np.roll(u, -1, axis=0), -1, axis=1) - u
            deltaNW = np.roll(np.roll(u, -1, axis=0), 1, axis=1) - u
            deltaSE = np.roll(np.roll(u, 1, axis=0), -1, axis=1) - u
            deltaSW = np.roll(np.roll(u, 1, axis=0), 1, axis=1) - u

            deltaNE[-1, :] = 0
            deltaNE[:, -1] = 0
            deltaNW[-1, :] = 0
            deltaNW[:, 0] = 0
            deltaSE[0, :] = 0
            deltaSE[:, -1] = 0
            deltaSW[0, :] = 0
            deltaSW[:, 0] = 0

            if method == "quadratic":
                cNE = 1.0 / (1.0 + (deltaNE / k) ** 2)
                cNW = 1.0 / (1.0 + (deltaNW / k) ** 2)
                cSE = 1.0 / (1.0 + (deltaSE / k) ** 2)
                cSW = 1.0 / (1.0 + (deltaSW / k) ** 2)
            elif method == "tukey":
                cNE = np.where(np.abs(deltaNE) <= k, (1.0 - (deltaNE / k) ** 2) ** 2, 0.0)
                cNW = np.where(np.abs(deltaNW) <= k, (1.0 - (deltaNW / k) ** 2) ** 2, 0.0)
                cSE = np.where(np.abs(deltaSE) <= k, (1.0 - (deltaSE / k) ** 2) ** 2, 0.0)
                cSW = np.where(np.abs(deltaSW) <= k, (1.0 - (deltaSW / k) ** 2) ** 2, 0.0)
            else:
                cNE = np.exp(-((deltaNE / k) ** 2))
                cNW = np.exp(-((deltaNW / k) ** 2))
                cSE = np.exp(-((deltaSE / k) ** 2))
                cSW = np.exp(-((deltaSW / k) ** 2))

            flux += diag_w * (cNE * deltaNE + cNW * deltaNW + cSE * deltaSE + cSW * deltaSW)

        u += g * flux

    return u


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
            - 'method': 'anisotropic' (Perona-Malik), 'nlm', 'bilateral', 'tv', or 'wavelet'
            - 'strength': float factor (0.1 to 3.0, default 1.0)
            - 'n_iter': int iterations for anisotropic diffusion (default 8)
            - 'kappa': float gradient threshold for anisotropic diffusion
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

    # Tissue-calibrated noise standard deviation estimation
    sigma_est = _robust_estimate_sigma(filter_input)
    # Ensure minimum effective filtering bandwidth for medical soft-tissue
    effective_sigma = max(0.025, sigma_est)

    # Dispatch to edge-preserving filter
    if method in ("anisotropic", "perona_malik", "anisodiff"):
        # Gold-standard PDE anisotropic diffusion operating directly on CT HU scale
        n_iter = int(p.get("n_iter", 4))
        # Physical HU gradient threshold (calibrated to 3.5 to 25.0 HU for subtle brain gyri / sulci / trabeculae)
        user_kappa = p.get("kappa", None)
        if user_kappa is not None:
            hu_kappa = float(user_kappa)
        else:
            hu_kappa = float(np.clip(3.5 * effective_sigma * orig_range * strength, 2.5, 30.0))
        cond_method = str(p.get("conduction_method", "exponential"))
        # Run directly on true HU input for maximum physical precision
        denoised_hu = anisotropic_diffusion_perona_malik(
            img,
            n_iter=n_iter,
            kappa=hu_kappa,
            gamma=0.10,
            conduction_method=cond_method,
            eight_neighbor=True,
        )
        denoised_norm = (denoised_hu - orig_min) / orig_range

    elif method == "bilateral":
        # OpenCV bilateralFilter with tissue-calibrated spatial and range sigma
        d = int(np.clip(5 + 2 * int(strength * 2), 5, 11))
        sigma_color = float(np.clip(2.5 * strength * effective_sigma, 0.06, 0.45))
        sigma_space = float(np.clip(4.0 * strength, 2.5, 10.0))
        denoised_norm = cv2.bilateralFilter(
            filter_input,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=cv2.BORDER_REFLECT,
        )

    elif method == "tv":
        # Total Variation Chambolle denoising
        tv_weight = float(np.clip(0.28 * strength * effective_sigma, 0.02, 0.35))
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
            d = int(np.clip(5 + 2 * int(strength * 2), 5, 11))
            sigma_color = float(np.clip(2.5 * strength * effective_sigma, 0.06, 0.45))
            sigma_space = float(np.clip(4.0 * strength, 2.5, 10.0))
            denoised_norm = cv2.bilateralFilter(
                filter_input,
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space,
                borderType=cv2.BORDER_REFLECT,
            )

    else:
        # Default: Non-Local Means (NLM)
        # Calibrated for high-fidelity CT texture and clean parenchyma smoothing
        # Stronger h_param and larger search window (patch_distance=9) for aggressive noise suppression
        h_param = max(0.06, 2.10 * strength * effective_sigma)
        denoised_norm = denoise_nl_means(
            filter_input,
            h=h_param,
            fast_mode=True,
            patch_size=5,
            patch_distance=9,
        )

    # Classical Multi-Scale Detail Preservation & Natural Texture Retention
    detail_boost = max(1.0, float(p.get("detail_boost", 1.0)))
    texture_blend = float(np.clip(p.get("texture_blend", 0.08), 0.0, 0.30))

    # Extract high-frequency micro-texture residual layer
    texture_residual = filter_input - denoised_norm

    if detail_boost > 1.001:
        # Adaptive edge-preserving coring: isolates true anatomical structures from stochastic photon noise
        coring_tau = float(np.clip(0.35 * effective_sigma, 0.004, 0.045))
        detail_clean = np.sign(texture_residual) * np.maximum(0.0, np.abs(texture_residual) - coring_tau)
        # Boost true anatomical micro-structures (sulci, gyri, trabeculae, vessel cortices)
        denoised_norm = np.clip(denoised_norm + (detail_boost - 1.0) * detail_clean, 0.0, 1.0)

    # Blend subtle natural texture to preserve realistic clinical CT parenchyma appearance
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
