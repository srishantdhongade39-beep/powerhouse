"""
noise_analysis.py
------------------
Responsible for characterizing noise present in a CT image, including
general noise statistics, periodic/structured scanner artifacts
(detected via frequency-domain FFT peak analysis), and Poisson/quantum
noise estimation.
"""

from typing import Any, Dict, List
import numpy as np
from scipy import ndimage


def detect_periodic_noise(
    image: np.ndarray,
    threshold_factor: float = 2.8,
    min_distance: int = 4,
    dc_exclude_radius: int = 8,
) -> Dict[str, Any]:
    """
    Detect periodic/structured scanner noise using 2D FFT frequency-domain peak analysis.

    Real-space periodic patterns (e.g. scanner ring, stripe, grid artifacts) manifest
    as anomalous narrow spikes/peaks in the Fourier magnitude spectrum away from DC.

    Args:
        image: 2D numpy array (single slice).
        threshold_factor: Multiplier of standard deviation above median background
            to classify an FFT spike as an artifact.
        min_distance: Minimum pixel distance between adjacent detected peaks.
        dc_exclude_radius: Central radius (in frequency pixels) around DC (0,0) to ignore,
            preserving natural low-frequency anatomical energy.

    Returns:
        Dictionary containing:
            - 'detected': True if periodic peaks are found
            - 'peak_count': Number of unique peak pairs found
            - 'peaks': List of detected peak dicts (coordinates, frequencies, magnitude)
            - 'log_magnitude': 2D log-magnitude spectrum array for visualization
            - 'diff_spectrum': Highpass-filtered spectrum highlighting spikes
            - 'center': (center_row, center_col)
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    cr, cc = h // 2, w // 2

    # 2D Fast Fourier Transform
    fft_shifted = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(fft_shifted)
    log_mag = np.log1p(magnitude)

    # Estimate smooth natural anatomical background spectrum using median filter
    bg = ndimage.median_filter(log_mag, size=7)
    diff = log_mag - bg

    # Create mask excluding central DC region and border edges
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_idx - cr) ** 2 + (x_idx - cc) ** 2)

    valid_mask = (dist_from_center > dc_exclude_radius) & (dist_from_center < min(cr, cc) - 4)
    # Avoid border edge artifacts
    valid_mask[:4, :] = False
    valid_mask[-4:, :] = False
    valid_mask[:, :4] = False
    valid_mask[:, -4:] = False

    # Exclude crosshair axes (width 3) where rectangular frame edge leakage and text lines concentrate
    valid_mask[max(0, cr - 1):min(h, cr + 2), :] = False
    valid_mask[:, max(0, cc - 1):min(w, cc + 2)] = False

    # Identify local maxima of the frequency difference spectrum
    local_max = (ndimage.maximum_filter(diff, size=min_distance) == diff) & (ndimage.maximum_filter(log_mag, size=min_distance) == log_mag)
    maxima_mask = local_max & valid_mask

    if not np.any(maxima_mask):
        return {
            "detected": False,
            "peak_count": 0,
            "peaks": [],
            "log_magnitude": log_mag,
            "diff_spectrum": diff,
            "center": (cr, cc),
        }

    maxima_diff = diff[maxima_mask]
    median_val = np.median(maxima_diff)
    std_val = np.std(maxima_diff)
    thresh = median_val + (threshold_factor * std_val)

    peak_candidates = maxima_mask & (diff >= thresh)
    candidate_rows, candidate_cols = np.where(peak_candidates)

    peaks: List[Dict[str, Any]] = []
    visited = np.zeros((h, w), dtype=bool)

    # Sort candidates by diff prominence descending
    candidate_prominence = diff[candidate_rows, candidate_cols]
    sort_order = np.argsort(-candidate_prominence)

    for idx in sort_order:
        if len(peaks) >= 6:  # Cap at max 6 harmonic pairs to protect anatomical bandwidth
            break

        r = int(candidate_rows[idx])
        c = int(candidate_cols[idx])

        if visited[r, c]:
            continue

        # Offset from DC center
        u = r - cr
        v = c - cc
        radial_dist = float(np.sqrt(u**2 + v**2))

        # Conjugate symmetric position in shifted FFT: (cr - u, cc - v)
        conj_r = cr - u
        conj_c = cc - v

        # A genuine scanner periodic artifact exhibits conjugate Fourier symmetry
        if not (0 <= conj_r < h and 0 <= conj_c < w and diff[conj_r, conj_c] >= thresh * 0.50):
            continue

        peak_info = {
            "row": r,
            "col": c,
            "u": int(u),
            "v": int(v),
            "radial_dist": radial_dist,
            "freq_normalized": float(radial_dist / max(cr, cc)),
            "magnitude": float(log_mag[r, c]),
            "prominence": float(diff[r, c]),
            "conjugate": {
                "row": int(conj_r),
                "col": int(conj_c),
            },
        }
        peaks.append(peak_info)

        # Mark neighborhood as visited
        rr_min = max(0, r - min_distance // 2)
        rr_max = min(h, r + min_distance // 2 + 1)
        cc_min = max(0, c - min_distance // 2)
        cc_max = min(w, c + min_distance // 2 + 1)
        visited[rr_min:rr_max, cc_min:cc_max] = True

        if 0 <= conj_r < h and 0 <= conj_c < w:
            cr_min = max(0, conj_r - min_distance // 2)
            cr_max = min(h, conj_r + min_distance // 2 + 1)
            conj_c_min = max(0, conj_c - min_distance // 2)
            conj_c_max = min(w, conj_c + min_distance // 2 + 1)
            visited[cr_min:cr_max, conj_c_min:conj_c_max] = True

    return {
        "detected": len(peaks) > 0,
        "peak_count": len(peaks),
        "peaks": peaks,
        "log_magnitude": log_mag,
        "diff_spectrum": diff,
        "threshold": float(thresh),
        "center": (cr, cc),
    }


def estimate_poisson_noise(image: np.ndarray) -> Dict[str, Any]:
    """
    Estimate Poisson (quantum) noise level in a CT image.

    Uses a robust Laplacian-based high-frequency noise variance estimator
    combined with homogeneous-tissue patch standard deviation analysis.
    In CT, quantum noise variance is signal-dependent and follows Poisson statistics
    during photon detection before log-reconstruction.

    Args:
        image: 2D numpy array (single CT slice).

    Returns:
        Dictionary with estimated sigma, variance, SNR estimate, and severity.
    """
    img = np.asarray(image, dtype=np.float64)

    # 1. Robust pseudo-Laplacian noise estimator (Immerkaer / Donoho formulation)
    # Mask kernel:
    #  [ 1, -2,  1]
    #  [-2,  4, -2]
    #  [ 1, -2,  1]
    # Normalizing factor for white Gaussian/Poisson high-frequency residual: sqrt(pi/2) / (6 * (W-2)*(H-2))
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    filtered = ndimage.convolve(img, kernel, mode="reflect")
    # Median Absolute Deviation (MAD) of the high-frequency residual
    mad = float(np.median(np.abs(filtered)))
    # For Gaussian distribution: sigma = MAD / (0.6745 * sqrt(sum(kernel^2)))
    # sum(kernel^2) = 1+4+1 + 4+16+4 + 1+4+1 = 36 -> sqrt(36) = 6
    sigma_laplacian = float(mad / (0.6745 * 6.0))

    # 2. Homogeneous-region patch estimation:
    # Divide image into 16x16 patches, calculate standard deviation in each,
    # take lower 15th percentile of non-zero patches (homogeneous brain parenchyma or background).
    h, w = img.shape
    patch_size = 16
    stds = []
    for r in range(0, h - patch_size, patch_size):
        for c in range(0, w - patch_size, patch_size):
            patch = img[r : r + patch_size, c : c + patch_size]
            s = float(np.std(patch))
            if s > 1e-4:
                stds.append(s)

    if stds:
        sigma_patch = float(np.percentile(stds, 15))
    else:
        sigma_patch = sigma_laplacian

    # Blended robust sigma estimate
    sigma_est = float(0.5 * (sigma_laplacian + sigma_patch))
    variance_est = float(sigma_est ** 2)

    # Signal-to-noise ratio estimate (dynamic range / sigma)
    data_range = float(np.ptp(img))
    if sigma_est > 1e-6 and data_range > 0:
        snr_db = float(20.0 * np.log10(data_range / sigma_est))
    else:
        snr_db = 0.0

    # Qualitative classification
    if snr_db > 35.0:
        level = "Low Noise (High Quality)"
    elif snr_db > 25.0:
        level = "Moderate Noise"
    else:
        level = "High Quantum Noise (Low-Dose CT Profile)"

    return {
        "estimated_sigma": sigma_est,
        "estimated_variance": variance_est,
        "snr_db": snr_db,
        "noise_level": level,
        "sigma_laplacian": sigma_laplacian,
        "sigma_homogeneous_patch": sigma_patch,
    }


def analyze_noise(image: np.ndarray) -> Dict[str, Any]:
    """
    Run comprehensive noise characterization pass on a CT slice.

    Args:
        image: 2D numpy array (HU or windowed display array).

    Returns:
        Dictionary summarizing detected noise types, periodic peak count,
        Poisson quantum noise parameters, dynamic range, and overall SNR.
    """
    img = np.asarray(image, dtype=np.float32)

    periodic_info = detect_periodic_noise(img)
    poisson_info = estimate_poisson_noise(img)

    return {
        "mean_intensity": float(np.mean(img)),
        "std_intensity": float(np.std(img)),
        "min_intensity": float(np.min(img)),
        "max_intensity": float(np.max(img)),
        "dynamic_range": float(np.ptp(img)),
        "periodic": periodic_info,
        "poisson": poisson_info,
        "periodic_noise_detected": periodic_info["detected"],
        "periodic_peak_count": periodic_info["peak_count"],
        "estimated_noise_sigma": poisson_info["estimated_sigma"],
        "estimated_snr_db": poisson_info["snr_db"],
    }
