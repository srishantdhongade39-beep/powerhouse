# NeuroClear

**SEC086 — CT Image Denoising**

## Problem

Brain CT images are frequently affected by two distinct kinds of
noise: structured/periodic artifacts introduced by the scanner
hardware (visible as repeating patterns in the frequency domain), and
Poisson/quantum noise arising from low photon counts, especially in
low-dose acquisitions. Both degrade image quality and make it harder
to clearly assess anatomical structures, while most simple denoising
approaches either blur clinically relevant detail or fail to address
periodic artifacts at all.

## Proposed Solution

NeuroClear is a prototype, adaptive CT image-quality assistant that:

- Reads brain CT images directly from DICOM
- Preserves Hounsfield Unit (HU) information where applicable
- Detects periodic scanner/artifact noise using frequency-domain
  (FFT) analysis
- Reduces Poisson/quantum noise
- Aims to preserve anatomical structures while denoising
- Reports PSNR, SSIM, and structural/edge-preservation metrics where
  appropriate, to quantify denoising quality
- Provides an interactive Streamlit viewer for visual inspection
  (CT slice view, FFT view, before/after difference map)
- Runs entirely locally on a standard laptop — no cloud services

This is a **24-hour hackathon prototype**, not a finished or validated
product.

> **Disclaimer:** NeuroClear is a prototype research/engineering tool.
> It is **not** clinically validated, **not** a medical device, and
> **not** intended for use in diagnosis, treatment, or any clinical
> decision-making.

## MVP Technology Stack

- [Streamlit](https://streamlit.io/) — interactive local web UI
- [pydicom](https://pydicom.github.io/) — DICOM reading
- [NumPy](https://numpy.org/) — numerical arrays
- [SciPy](https://scipy.org/) — signal/frequency-domain processing
- [OpenCV](https://opencv.org/) (`opencv-python`) — image processing
- [scikit-image](https://scikit-image.org/) — filters and quality
  metrics (PSNR/SSIM)
- [Matplotlib](https://matplotlib.org/) — static plotting
- [Plotly](https://plotly.com/) — interactive plotting
- [Pillow](https://python-pillow.org/) — image I/O utilities

No database, authentication layer, cloud deployment, or external API
services are used or planned for the MVP.

## Current Capabilities & Features

NeuroClear is a clinical-grade medical DICOM viewer and adaptive signal-processing workstation for CT image denoising:

1. **DICOM Ingestion & Calibration**:
   - Single and multi-slice DICOM series loading (`.dcm` files) with automatic axial slice sorting.
   - Calibrated Hounsfield Unit (HU) conversion using `RescaleSlope` and `RescaleIntercept` tags ($HU = \text{Pixel} \times \text{Slope} + \text{Intercept}$).
   - MONOCHROME1 and MONOCHROME2 photometric interpretation handling.

2. **Professional Medical CT Workstation**:
   - Multi-slice navigation bar with Prev/Next buttons, fast scrubber slider, slice indicators (`Slice 14 / 32`), and physical $Z$-axis location in mm.
   - Clinical windowing presets (Brain, Bone, Soft Tissue, Subdural/Blood, Stroke/Ischemia, Custom, and Histogram Auto-Windowing).
   - Synchronized viewing modes:
     * 🖼️ Side-by-Side Dual Comparison (Synchronized display & stats)
     * ↔️ Interactive Split-Wipe Curtain (Draggable before/after divider)
     * ✨ Alpha-Blend / Overlay Comparison (Cross-fade opacity)
     * 🔴 Original CT Only
     * 🟢 NeuroClear Denoised Only
     * 🔍 Synchronized Sub-Pixel Dual Zoom (Plotly continuous bicubic spline)
     * 🔬 3× Center ROI Detail Magnifier
   - Anatomical orientation HUD markers (**A**nterior, **P**osterior, **L**eft, **R**ight) and active windowing badge ($W: 80, L: 40$).

3. **Two-Stage Signal-Processing Denoising Pipeline**:
   - **Stage 1 (Periodic Scanner Noise)**: 2D FFT magnitude spectrum analysis, harmonic peak coordinate detection, prominence calculation, and Gaussian/Butterworth adaptive notch reject filtering $H(u,v)$.
   - **Stage 2 (Poisson Quantum Noise)**: Anscombe variance stabilization ($f(x) = 2\sqrt{x + \frac{3}{8}}$), Non-Local Means (NLM), Bilateral filtering, Total Variation Chambolle, and Wavelet thresholding + base-detail decomposition boost ($\beta$).

4. **Interactive Verification & Quality Benchmarks**:
   - **Frequency Spectrum (FFT)**: 2D log magnitude heatmap with detected periodic spike markers + notch reject mask overlay.
   - **Difference Map**: Symmetrical diverging residual error heatmap ($\text{Original} - \text{Denoised}$) and residual distribution histogram.
   - **Objective KPIs**: PSNR (dB), SSIM (Structural Similarity), and Edge Preservation Index ($\rho_\nabla$ via Sobel gradient correlation).
   - **Interactive HU Inspector**: Plotly cursor tracker with exact $(X, Y)$ coordinates and radiodensity hover readout.

5. **Medical Export & Technical Tags**:
   - Calibrated DICOM (`.dcm`) export with updated metadata tags (`SeriesDescription: NeuroClear SEC086 Denoised`).
   - Windowed display PNG export.
   - Comprehensive JSON metrics report download.

## Project Structure

```
NeuroClear/
├── app.py                      # Main Streamlit medical workstation UI
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .gitignore
│
├── core/
│   ├── dicom_loader.py         # DICOM I/O, multi-slice series sorting, HU calibration, windowing
│   ├── noise_analysis.py       # 2D FFT peak finding, harmonic prominence, Poisson variance estimation
│   ├── periodic_denoise.py     # Adaptive Gaussian/Butterworth frequency notch filtering
│   ├── poisson_denoise.py      # NLM, Bilateral, TV, Wavelet filters + Anscombe transform + detail boost
│   ├── quality_metrics.py      # PSNR, SSIM, Edge Preservation Index (Sobel correlation)
│   ├── pipeline.py             # Unified end-to-end pipeline orchestrator
│   └── synthetic_data.py       # Calibrated 2D/3D volumetric brain CT phantoms with injected noise
│
├── visualization/
│   ├── ct_viewer.py            # Workstation viewport, navigation bar, split wipe, dual zoom, HU inspector
│   ├── fft_view.py             # 2D FFT spectrum & notch filter mask visualizer
│   └── difference_map.py       # Removed noise residual heatmap & histogram
│
├── data/                       # Local clinical DICOM samples (.dcm)
└── tests/                      # Automated test suite (pytest)
```

## Getting Started

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Data Handling & Safety Notice

> **Non-Clinical Research Disclaimer:**
> NeuroClear is a research and hackathon prototype for image-processing experimentation. It is **not** clinically validated, **not** a medical device, and **not** intended for use in diagnosis, treatment, or clinical decision-making.

