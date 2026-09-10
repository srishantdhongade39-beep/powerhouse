"""
pipeline.py
-------------
Responsible for orchestrating the end-to-end NeuroClear CT denoising flow:
1. DICOM Ingestion & HU conversion (Immutable original preservation)
2. Pre-denoise Noise Characterization (FFT peak detection & Poisson variance)
3. Periodic Scanner Noise Removal (adaptive notch filtering)
4. Poisson/Quantum Noise Reduction (edge-preserving spatial filtering)
5. Safety & Quality Output Validation (IEC 62304 / ISO 14971 informed gate)
6. Post-denoise Quality Metrics (PSNR, SSIM, Edge Preservation Index)
7. Algorithm Decision Trace & Audit Trail Generation
8. Windowed display preparation
"""

import time
from typing import Any, Dict, Optional, Union
import numpy as np
import pydicom

from core.dicom_loader import (
    apply_window,
    convert_to_hounsfield_units,
    get_dicom_metadata,
    load_dicom,
)
from core.noise_analysis import analyze_noise
from core.output_validation import (
    generate_algorithm_decision_trace,
    validate_pipeline_output,
)
from core.periodic_denoise import remove_periodic_noise
from core.poisson_denoise import denoise_poisson
from core.quality_metrics import compute_all_metrics


def run_neuroclear_pipeline(
    dicom_input: Union[str, pydicom.Dataset, np.ndarray, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full NeuroClear processing pipeline on a DICOM input or HU array
    with formal output validation and decision traceability.
    """
    start_time = time.perf_counter()
    opts = options or {}
    skip_periodic = bool(opts.get("skip_periodic", False))
    skip_poisson = bool(opts.get("skip_poisson", False))
    notch_radius = float(opts.get("notch_radius", 6.0))
    notch_type = str(opts.get("notch_filter_type", "gaussian"))
    poisson_method = str(opts.get("poisson_method", "nlm"))
    poisson_strength = float(opts.get("poisson_strength", 1.0))
    use_anscombe = bool(opts.get("use_anscombe", False))
    detail_boost = float(opts.get("detail_boost", 1.0))
    w_center = float(opts.get("window_center", 40.0))
    w_width = float(opts.get("window_width", 80.0))
    ground_truth = opts.get("ground_truth", None)

    # 1. Ingestion & HU conversion (Original preservation)
    metadata: Dict[str, Any] = {}
    if isinstance(dicom_input, np.ndarray):
        hu_original = np.array(dicom_input, dtype=np.float32, copy=True)
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
        poisson_params = {
            "method": poisson_method,
            "strength": poisson_strength,
            "use_anscombe": use_anscombe,
            "detail_boost": detail_boost,
        }
        hu_denoised_raw = denoise_poisson(hu_periodic, params=poisson_params)
    else:
        hu_denoised_raw = hu_periodic.copy()

    # 5. Output Validation Gate (IEC 62304 / ISO 14971 Safety Check)
    validation = validate_pipeline_output(
        input_hu=hu_original,
        output_hu=hu_denoised_raw,
        min_edge_preservation=0.45,
        max_mean_shift_hu=50.0,
    )
    hu_denoised = validation["safe_output_hu"]

    # 6. Post-denoise Noise Analysis & Metrics
    post_noise = analyze_noise(hu_denoised)
    difference_map = hu_original - hu_denoised

    metrics = compute_all_metrics(hu_original, hu_denoised)
    metrics["edge_preservation"] = validation["edge_preservation"]

    # Ground Truth Comparison (Strictly for synthetic benchmark where clean ground truth exists)
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

    elapsed_time = time.perf_counter() - start_time

    # 7. Algorithmic Decision Trail & Audit Trail Generation
    options_applied = {
        "skip_periodic": skip_periodic,
        "skip_poisson": skip_poisson,
        "notch_radius": notch_radius,
        "notch_filter_type": notch_type,
        "poisson_method": poisson_method,
        "poisson_strength": poisson_strength,
        "use_anscombe": use_anscombe,
        "detail_boost": detail_boost,
        "window_center": w_center,
        "window_width": w_width,
    }
    decision_trace = generate_algorithm_decision_trace(
        noise_analysis=initial_noise,
        options_applied=options_applied,
        validation_result=validation,
        execution_time_seconds=elapsed_time,
    )

    # 8. Windowed Display Arrays (Display-only copies, original float32 HU preserved)
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
        "validation": validation,
        "decision_trace": decision_trace,
        "options_applied": options_applied,
        "execution_time_seconds": elapsed_time,
    }

