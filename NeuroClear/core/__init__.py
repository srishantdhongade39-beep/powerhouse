"""
NeuroClear core package.

Contains the DICOM I/O, noise analysis, periodic notch filtering,
Poisson denoising, quality metrics, and end-to-end pipeline.
"""

from core.dicom_loader import (
    WINDOW_PRESETS,
    apply_window,
    convert_to_hounsfield_units,
    get_dicom_metadata,
    load_dicom,
    load_dicom_files_list,
    load_dicom_series,
    sort_dicom_slices,
)
from core.noise_analysis import (
    analyze_noise,
    detect_periodic_noise,
    estimate_poisson_noise,
)
from core.periodic_denoise import (
    create_notch_filter,
    remove_periodic_noise,
)
from core.pipeline import run_neuroclear_pipeline
from core.poisson_denoise import denoise_poisson
from core.quality_metrics import (
    calculate_edge_preservation,
    calculate_psnr,
    calculate_ssim,
    compute_all_metrics,
)
from core.synthetic_data import (
    create_synthetic_dicom_dataset,
    generate_brain_ct_phantom,
    generate_brain_ct_volume,
)

__all__ = [
    "load_dicom",
    "load_dicom_series",
    "load_dicom_files_list",
    "sort_dicom_slices",
    "convert_to_hounsfield_units",
    "apply_window",
    "WINDOW_PRESETS",
    "get_dicom_metadata",
    "analyze_noise",
    "detect_periodic_noise",
    "estimate_poisson_noise",
    "create_notch_filter",
    "remove_periodic_noise",
    "denoise_poisson",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_edge_preservation",
    "compute_all_metrics",
    "run_neuroclear_pipeline",
    "generate_brain_ct_phantom",
    "generate_brain_ct_volume",
    "create_synthetic_dicom_dataset",
]
