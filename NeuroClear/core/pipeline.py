"""
pipeline.py
-------------
Responsible for orchestrating the end-to-end NeuroClear CT denoising flow:
1. DICOM Ingestion & HU conversion
2. Pre-denoise Noise Characterization (FFT peak detection & Poisson variance)
3. Periodic Scanner Noise Removal (adaptive notch filtering)
4. Poisson/Quantum Noise Reduction (edge-preserving spatial filtering)
5. Post-denoise Quality Metrics (PSNR, SSIM, Edge Preservation Index)
6. Windowed display preparation
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pydicom

from core.dicom_loader import (
    WINDOW_PRESETS,
    apply_window,
    convert_to_hounsfield_units,
    get_dicom_metadata,
    load_dicom,
)
from core.noise_analysis import analyze_noise
from core.periodic_denoise import remove_periodic_noise
from core.poisson_denoise import denoise_poisson
from core.quality_metrics import (
    calculate_edge_preservation,
    calculate_psnr,
    calculate_ssim,
    compute_all_metrics,
)


def run_neuroclear_pipeline(
    dicom_input: Union[str, pydicom.Dataset, np.ndarray, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full NeuroClear processing pipeline on a DICOM input or HU array.

    Args:
        dicom_input: Path to DICOM file, open BytesIO buffer, pydicom.Dataset,
            or calibrated HU 2D numpy array.
        options: Optional pipeline configuration dictionary:
            - 'skip_periodic': bool (default False)
            - 'skip_poisson': bool (default False)
            - 'notch_radius': float (default 6.0)
            - 'notch_filter_type': str ('gaussian' or 'butterworth')
            - 'poisson_method': str ('nlm', 'bilateral', 'tv', 'wavelet')
            - 'poisson_strength': float (default 1.0)
            - 'use_anscombe': bool (default False)
            - 'window_center': float (default 40.0)
            - 'window_width': float (default 80.0)
            - 'ground_truth': Optional 2D numpy array (clean reference for synthetic demos)

    Returns:
        Comprehensive dictionary of all pipeline intermediate arrays, metadata,
        noise analyses, and quality metrics.
    """
    opts = options or {}
    skip_periodic = bool(opts.get("skip_periodic", False))
    skip_poisson = bool(opts.get("skip_poisson", False))
    notch_radius = float(opts.get("notch_radius", 6.0))
    notch_type = str(opts.get("notch_filter_type", "gaussian"))
    poisson_method = str(opts.get("poisson_method", "nlm"))
    poisson_strength = float(opts.get("poisson_strength", 1.0))
    use_anscombe = bool(opts.get("use_anscombe", False))
    w_center = float(opts.get("window_center", 40.0))
    w_width = float(opts.get("window_width", 80.0))
    ground_truth = opts.get("ground_truth", None)

    # 1. Ingestion & HU conversion
    metadata: Dict[str, Any] = {}
    if isinstance(dicom_input, np.ndarray):
        hu_original = dicom_input.astype(np.float32)
        metadata = {
            "patient_id": "In-Memory Array",
            "modality": "CT",
            "series_description": "Direct HU Input",
            "rows": hu_original.shape[0],
            "columns": hu_original.shape[1],
            "rescale_slope": 1.0,
            "rescale_intercept": 0.0,
        }
    elif isinstance(dicom_input, pydicom.Dataset):
        ds = dicom_input
        metadata = get_dicom_metadata(ds)
        hu_original = convert_to_hounsfield_units(ds)
    else:
        # File path or stream
        ds = load_dicom(dicom_input)
        metadata = get_dicom_metadata(ds)
        hu_original = convert_to_hounsfield_units(ds)

    # 2. Initial Noise Analysis
    initial_noise = analyze_noise(hu_original)

    # 3. Periodic Noise Removal (Notch Filtering)
    if not skip_periodic:
        hu_periodic, notch_mask = remove_periodic_noise(
            hu_original,
            noise_info=initial_noise["periodic"],
            notch_radius=notch_radius,
            filter_type=notch_type,
        )
    else:
        hu_periodic = hu_original.copy()
        notch_mask = np.ones(hu_original.shape, dtype=np.float32)

    # 4. Poisson / Quantum Noise Reduction (Edge-Preserving Filtering)
    if not skip_poisson:
        detail_boost = float(opts.get("detail_boost", 1.0))
        poisson_params = {
            "method": poisson_method,
            "strength": poisson_strength,
            "use_anscombe": use_anscombe,
            "detail_boost": detail_boost,
        }
        hu_denoised = denoise_poisson(hu_periodic, params=poisson_params)
    else:
        hu_denoised = hu_periodic.copy()

    # 5. Post-denoise Noise Analysis
    post_noise = analyze_noise(hu_denoised)

    # 6. Quality Metrics
    difference_map = hu_original - hu_denoised

    metrics = compute_all_metrics(hu_original, hu_denoised)
    metrics["edge_preservation"] = metrics["edge_preservation_index"]

    # If clean ground truth is available (e.g. from synthetic phantom), compute improvement metrics
    gt_metrics: Optional[Dict[str, Any]] = None
    if ground_truth is not None and ground_truth.shape == hu_original.shape:
        gt = np.asarray(ground_truth, dtype=np.float64)
        noisy_gt = compute_all_metrics(gt, hu_original)
        denoised_gt = compute_all_metrics(gt, hu_denoised)

        gt_metrics = {
            "input_psnr_db": noisy_gt["psnr_db"],
            "output_psnr_db": denoised_gt["psnr_db"],
            "psnr_improvement_db": denoised_gt["psnr_db"] - noisy_gt["psnr_db"],
            "input_ssim": noisy_gt["ssim"],
            "output_ssim": denoised_gt["ssim"],
            "ssim_improvement": denoised_gt["ssim"] - noisy_gt["ssim"],
            "input_edge_preservation": noisy_gt["edge_preservation_index"],
            "output_edge_preservation": denoised_gt["edge_preservation_index"],
        }

    # 7. Windowed Display Arrays
    display_original = apply_window(hu_original, w_center, w_width, as_uint8=True)
    display_periodic = apply_window(hu_periodic, w_center, w_width, as_uint8=True)
    display_denoised = apply_window(hu_denoised, w_center, w_width, as_uint8=True)

    return {
        "metadata": metadata,
        "hu_original": hu_original,
        "hu_periodic": hu_periodic,
        "hu_denoised": hu_denoised,
        "display_original": display_original,
        "display_periodic": display_periodic,
        "display_denoised": display_denoised,
        "difference_map": difference_map,
        "notch_mask": notch_mask,
        "initial_noise": initial_noise,
        "post_noise": post_noise,
        "metrics": metrics,
        "ground_truth_metrics": gt_metrics,
        "options_applied": {
            "skip_periodic": skip_periodic,
            "skip_poisson": skip_poisson,
            "notch_radius": notch_radius,
            "notch_type": notch_type,
            "poisson_method": poisson_method,
            "poisson_strength": poisson_strength,
            "use_anscombe": use_anscombe,
            "window_center": w_center,
            "window_width": w_width,
        },
    }
