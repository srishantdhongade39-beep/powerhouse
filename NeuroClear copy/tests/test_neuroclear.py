"""
test_neuroclear.py
-------------------
Comprehensive automated test suite for NeuroClear (SEC086).
Tests DICOM loading, HU conversion, windowing presets, noise analysis (FFT peak detection),
periodic notch filtering, Poisson edge-preserving denoising, quality metrics, and full pipeline.
"""

import numpy as np
import pytest

from core.dicom_loader import (
    WINDOW_PRESETS,
    apply_window,
    convert_to_hounsfield_units,
    get_dicom_metadata,
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
)


@pytest.fixture
def synthetic_slice():
    noisy, clean, info = generate_brain_ct_phantom(
        size=128,
        add_periodic_artifact=True,
        periodic_amplitude=30.0,
        add_poisson_noise=True,
        random_seed=42,
    )
    return noisy, clean, info


def test_synthetic_data_generation(synthetic_slice):
    noisy, clean, info = synthetic_slice
    assert noisy.shape == (128, 128)
    assert clean.shape == (128, 128)
    assert info["periodic_injected"] is True
    assert info["poisson_injected"] is True
    assert np.max(clean) >= 900.0


def test_dicom_io_and_metadata(synthetic_slice):
    noisy, _, _ = synthetic_slice
    ds = create_synthetic_dicom_dataset(noisy, patient_id="TEST_001")

    meta = get_dicom_metadata(ds)
    assert meta["patient_id"] == "TEST_001"
    assert meta["modality"] == "CT"
    assert meta["rows"] == 128
    assert meta["columns"] == 128
    assert meta["rescale_slope"] == 1.0
    assert meta["rescale_intercept"] == -1024.0

    hu = convert_to_hounsfield_units(ds)
    assert hu.shape == (128, 128)
    np.testing.assert_allclose(hu, noisy, atol=2.0)


def test_apply_window(synthetic_slice):
    _, clean, _ = synthetic_slice
    win_uint8 = apply_window(clean, window_center=40.0, window_width=80.0, as_uint8=True)
    assert win_uint8.dtype == np.uint8
    assert win_uint8.shape == clean.shape
    assert np.min(win_uint8) == 0
    assert np.max(win_uint8) == 255

    win_float = apply_window(clean, window_center=40.0, window_width=80.0, as_uint8=False)
    assert win_float.dtype == np.float32
    assert 0.0 <= np.min(win_float) <= 1.0
    assert 0.0 <= np.max(win_float) <= 1.0


def test_periodic_noise_detection(synthetic_slice):
    noisy, clean, info = synthetic_slice
    analysis = detect_periodic_noise(noisy, threshold_factor=2.5, dc_exclude_radius=8)

    assert analysis["detected"] is True
    assert analysis["peak_count"] >= 1
    assert "peaks" in analysis
    first_peak = analysis["peaks"][0]
    assert "u" in first_peak and "v" in first_peak
    assert "magnitude" in first_peak


def test_periodic_noise_removal(synthetic_slice):
    noisy, clean, info = synthetic_slice
    denoised_periodic, mask = remove_periodic_noise(noisy, notch_radius=5.0)

    assert denoised_periodic.shape == noisy.shape
    assert mask.shape == noisy.shape
    assert 0.0 <= np.min(mask) <= 1.0
    assert np.var(denoised_periodic) < np.var(noisy)


def test_poisson_noise_estimation(synthetic_slice):
    noisy, clean, _ = synthetic_slice
    est = estimate_poisson_noise(noisy)

    assert est["estimated_sigma"] > 0.0
    assert est["snr_db"] > 0.0
    assert "noise_level" in est


@pytest.mark.parametrize("method", ["nlm", "bilateral", "tv", "wavelet"])
def test_poisson_denoise_methods(synthetic_slice, method):
    noisy, clean, _ = synthetic_slice
    denoised = denoise_poisson(noisy, params={"method": method, "strength": 0.8})

    assert denoised.shape == noisy.shape
    assert not np.isnan(denoised).any()
    assert not np.isinf(denoised).any()


def test_anscombe_transform_poisson(synthetic_slice):
    noisy, _, _ = synthetic_slice
    denoised = denoise_poisson(noisy, params={"method": "bilateral", "strength": 0.8, "use_anscombe": True})
    assert denoised.shape == noisy.shape
    assert not np.isnan(denoised).any()


def test_quality_metrics(synthetic_slice):
    _, clean, _ = synthetic_slice
    psnr_perfect = calculate_psnr(clean, clean)
    assert psnr_perfect >= 99.0

    ssim_perfect = calculate_ssim(clean, clean)
    assert np.isclose(ssim_perfect, 1.0, atol=1e-3)

    epi_perfect = calculate_edge_preservation(clean, clean)
    assert np.isclose(epi_perfect, 1.0, atol=1e-2)

    noisy = clean + np.random.normal(0, 10.0, size=clean.shape)
    metrics = compute_all_metrics(clean, noisy)
    assert 10.0 < metrics["psnr_db"] < 60.0
    assert 0.0 < metrics["ssim"] < 1.0
    assert 0.5 < metrics["edge_preservation_index"] <= 1.0


def test_full_pipeline_orchestration(synthetic_slice):
    noisy, clean, _ = synthetic_slice
    ds = create_synthetic_dicom_dataset(noisy)

    results = run_neuroclear_pipeline(
        ds,
        options={
            "skip_periodic": False,
            "skip_poisson": False,
            "notch_radius": 5.0,
            "poisson_method": "bilateral",
            "poisson_strength": 1.0,
            "ground_truth": clean,
        },
    )

    assert "hu_original" in results
    assert "hu_denoised" in results
    assert "difference_map" in results
    assert "metrics" in results
    assert "ground_truth_metrics" in results

    assert results["metrics"]["edge_preservation"] > 0.75

    gt_m = results["ground_truth_metrics"]
    assert gt_m["output_psnr_db"] > gt_m["input_psnr_db"]
    assert gt_m["output_ssim"] >= gt_m["input_ssim"]
