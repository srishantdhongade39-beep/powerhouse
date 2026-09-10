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

## Current Development Status

**Architecture scaffold only.** This repository currently contains:

- The full project/folder structure
- A minimal, runnable Streamlit shell (`app.py`) with a placeholder
  DICOM upload section
- Stub functions (with docstrings, no logic) for every planned module
  in `core/` and `visualization/`

**Not yet implemented:** DICOM loading, HU conversion, windowing,
noise analysis, periodic-noise notch filtering, Poisson denoising,
quality metrics, and all visualizations. Calling any stub function
currently raises `NotImplementedError` by design.

## Project Structure

```
NeuroClear/
├── app.py                 # Streamlit entry point (shell only)
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/
│   ├── dicom_loader.py        # DICOM I/O, HU conversion, windowing (stubs)
│   ├── noise_analysis.py      # Noise/periodic/Poisson analysis (stubs)
│   ├── periodic_denoise.py    # Notch filtering (stubs)
│   ├── poisson_denoise.py     # Poisson denoising (stubs)
│   ├── quality_metrics.py     # PSNR / SSIM / edge preservation (stubs)
│   └── pipeline.py            # End-to-end orchestration (stub)
│
├── visualization/
│   ├── ct_viewer.py            # CT slice viewer (stub)
│   ├── fft_view.py             # Frequency-domain view (stub)
│   └── difference_map.py       # Before/after difference map (stub)
│
├── data/                   # Local data only — never committed (see .gitignore)
└── tests/                  # Test scaffolding
```

## Getting Started

### 1. Create a Python virtual environment

```bash
python3 -m venv venv
```

Activate it:

- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows (PowerShell):
  ```bash
  venv\Scripts\Activate.ps1
  ```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`)
— open it in your browser.

## Data Handling Notice

`data/` is a local-only working directory. **Do not commit patient
data, DICOM files, or large CT datasets to this repository** — the
`.gitignore` is configured to exclude common medical-imaging and
dataset file types by default.
