# NeuroClear — Requirements Traceability & Risk Management Matrix
**Document Version:** 0.1.0  
**Standards Context:** IEC 62304 (Software Lifecycle), ISO 14971 (Risk Management), IEC 62366-1 (Usability), IEC 60601-1 (Safety Context)  
**Notice:** *This documentation and software are standards-informed research prototype specifications. Not for clinical diagnostic use.*

---

## 1. IEC 62304 System Requirements Traceability Matrix

| Requirement ID | Description | Source Module / Component | Verification / Test Case | Lifecycle Status |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-001** | DICOM Ingestion & Parsing: Parse single/multi-frame `.dcm` files, extract patient metadata, photometric interpretation, slice thickness, and rescale slope/intercept. | [`core/dicom_loader.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/dicom_loader.py) | `tests/test_neuroclear.py::test_dicom_loader` | Verified |
| **SYS-002** | Hounsfield Unit (HU) Calibration: Apply `RescaleSlope * raw_value + RescaleIntercept` to convert raw attenuation numbers into calibrated float32 HU space. | [`core/dicom_loader.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/dicom_loader.py) | `tests/test_neuroclear.py::test_hu_calibration` | Verified |
| **SYS-003** | Adaptive Bilateral Filtering: Preserve sharp anatomical edges and skull-brain boundaries while smoothing Gaussian noise based on local HU variance. | [`core/bilateral.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/bilateral.py) | `tests/test_neuroclear.py::test_bilateral_filter` | Verified |
| **SYS-004** | Dual-Tree Complex Wavelet Transform (DTCWT): Multi-scale directional sub-band thresholding with shift-invariance to eliminate high-frequency Poisson noise. | [`core/wavelet.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/wavelet.py) | `tests/test_neuroclear.py::test_wavelet_denoising` | Verified |
| **SYS-005** | Total Variation Regularization: Split-Bregman / Chambolle gradient minimization to preserve piecewise constant tissue regions without introducing ringing artifacts. | [`core/total_variation.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/total_variation.py) | `tests/test_neuroclear.py::test_tv_denoising` | Verified |
| **SYS-006** | FFT High-Frequency Attenuation: Dynamic Butterworth/Gaussian spatial-frequency filtering with DC-preservation to prevent mean attenuation shift. | [`core/fft_filter.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/fft_filter.py) | `tests/test_neuroclear.py::test_fft_filter` | Verified |
| **SYS-007** | Pipeline Output Validation Gate: Automated verification verifying array dimensionality, NaN/Inf absence, physiological HU bounds [-1024, 3071], and edge preservation index ($\rho \ge 0.45$) with safe fallback to original input on failure. | [`core/output_validation.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/output_validation.py) | `tests/test_neuroclear.py::test_output_validation_gate` | Verified |
| **SYS-008** | Usability & Traceability HUD: Standardized labeling (`ORIGINAL CT`, `NEUROCLEAR PROCESSED CT`, `REMOVED SIGNAL / DIFFERENCE MAP`), side-by-side synchronized viewport, and algorithm decision trace telemetry. | [`visualization/ct_viewer.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/visualization/ct_viewer.py), [`visualization/difference_map.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/visualization/difference_map.py) | `tests/test_neuroclear.py::test_visualization_labels` | Verified |
| **SYS-009** | Reference-Independent Quality Metrics: Contrast-to-Noise Ratio (CNR), Edge Preservation Index ($\rho_\nabla$), and Natural Image Quality Evaluator (NIQE) for non-reference clinical scans. | [`core/metrics.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/metrics.py) | `tests/test_neuroclear.py::test_metrics_calculation` | Verified |
| **SYS-010** | Safe Fallback & Error Containment: Graceful handling of corrupted files, zero-division, and out-of-range parameters with immutable retention of original DICOM input. | [`core/pipeline.py`](file:///c:/Users/anand/Desktop/Neuroclear/NeuroClear/core/pipeline.py) | `tests/test_neuroclear.py::test_safe_fallback_mechanism` | Verified |

---

## 2. ISO 14971 Medical Device Risk Management Analysis

| Risk ID | Hazard / Potential Failure Mode | Potential Clinical / Usability Impact | Severity (1-5) | Probability (1-5) | Initial Risk Level | Risk Mitigation & Software Safety Controls | Post-Mitigation Risk | Verification Reference |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **R-001** | Corrupted or truncated DICOM byte stream during file upload. | Crash or unhandled exception preventing study review. | 2 | 3 | Medium | Validate DICOM magic preamble (`DICM`), structured try-catch exception handling, display descriptive user alert. | **Low** | `SYS-001`, `test_dicom_loader` |
| **R-002** | Missing or non-standard `RescaleSlope` / `RescaleIntercept` tags. | Incorrect Hounsfield Units, leading to misinterpretation of tissue densities. | 4 | 2 | High | Safe default fallback (`slope=1.0`, `intercept=0.0` or `-1024.0` with explicit telemetry warning logged in decision trace). | **Low** | `SYS-002`, `test_hu_calibration` |
| **R-003** | Excessive spatial smoothing attenuating micro-calcifications or subtle subdural hematomas. | Critical pathological features blurred or obscured. | 5 | 2 | High | Constrained range bounds on bilateral $\sigma_{spatial}$ and $\sigma_{range}$; strict edge preservation index threshold ($\rho_\nabla \ge 0.45$). | **Low** | `SYS-003`, `test_output_validation_gate` |
| **R-004** | DTCWT high-frequency thresholding removing faint hemorrhage margins. | Misinterpretation of lesion boundaries. | 4 | 2 | High | Directional selective thresholding with sub-band energy preservation; user-adjustable blending ratio. | **Low** | `SYS-004`, `test_wavelet_denoising` |
| **R-005** | Total Variation over-regularization creating artificial "staircasing" artifacts. | False appearance of sharp anatomical contours or false positive infarct margins. | 3 | 3 | High | Bounded TV regularization weight ($\lambda \le 0.15$) and iterative Split-Bregman stopping criteria. | **Low** | `SYS-005`, `test_tv_denoising` |
| **R-006** | FFT frequency filter cutting low frequencies or shifting DC component. | Global mean HU baseline shift altering tissue classification. | 4 | 2 | High | DC component hard-pinned; maximum allowable global mean drift constrained to $< 5.0$ HU in validation gate. | **Low** | `SYS-006`, `test_fft_filter` |
| **R-007** | Numerical overflow, underflow, NaN, or Inf generation during floating point calculations. | Corrupted output slice display, rendering visual noise or blank screen. | 3 | 2 | Medium | Float32 range clamping and automated NaN/Inf checks in `validate_pipeline_output()` before UI rendering. | **Low** | `SYS-007`, `test_output_validation_gate` |
| **R-008** | Inadvertent modification or overwrite of original DICOM HU voxel buffer in memory. | Permanent loss of authentic raw scan data in multi-step workflows. | 5 | 1 | High | Enforce immutable `np.copy(original_hu)` clone at pipeline entrypoint; original slice permanently accessible in side-by-side view. | **Low** | `SYS-008`, `test_safe_fallback_mechanism` |
| **R-009** | Misinterpretation of PSNR/SSIM on clinical scans where no pristine ground truth exists. | User relies on invalid statistical metrics for clinical assessment. | 3 | 3 | High | Clearly display `PSNR: N/A (Reference scan unavailable)` and `SSIM: N/A` for clinical scans, emphasizing reference-free CNR and $\rho_\nabla$. | **Low** | `SYS-009`, `test_metrics_calculation` |
| **R-010** | Misinterpretation of the difference map as a diagnostic pathology overlay. | Clinician mistakes noise residual for anatomical lesion or bleeding. | 4 | 2 | High | Mandatory standard label `REMOVED SIGNAL / DIFFERENCE MAP` with explicit advisory disclaimer that the map represents algorithm residuals only. | **Low** | `SYS-008`, `test_visualization_labels` |
| **R-011** | User inadvertently evaluating processed image without visual comparison against original raw CT. | Loss of diagnostic context during review. | 4 | 2 | High | Default side-by-side synchronized dual-viewport with clear persistent `ORIGINAL CT` and `NEUROCLEAR PROCESSED CT` badge overlays. | **Low** | `SYS-008`, `test_visualization_labels` |
| **R-012** | Complete pipeline failure or unhandled numerical exception during batch processing. | Workstation freeze, potential data loss during urgent review. | 3 | 2 | Medium | Output validation fallback mechanism automatically emits unchanged original slice with critical safety notification and trace reason. | **Low** | `SYS-010`, `test_safe_fallback_mechanism` |

---

## 3. IEC 60601-1 Safety Context Reference Statement

> **Safety Context Note:**  
> NeuroClear is a standalone software application operating on off-the-shelf commercial computing hardware. It does not interface directly with physical CT scanner electronics, high-voltage generators, gantry rotation controllers, or patient-contacting medical sensors.  
> IEC 60601-1 physical and electrical safety specifications (e.g., electrical leakage currents, patient isolation barriers, mechanical enclosure ratings) are the responsibility of the primary diagnostic imaging modality manufacturer. NeuroClear strictly adheres to medical software data integrity, numerical safety, and non-destructive post-processing principles.

---

## 4. Software Version & Change Traceability

- **Software Version:** `v0.1.0`
- **Release Stage:** Standards-Informed Academic & Engineering Prototype
- **Repository Commit Baseline:** Git controlled with full automated test suite coverage (`pytest`).
