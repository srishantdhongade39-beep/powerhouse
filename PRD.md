# Product Requirements Document (Master)

## NeuroClear — SEC086: CT Image Denoising

| Field | Value |
|---|---|
| Project Code | SEC086 |
| Project Name | NeuroClear |
| Document Type | Master PRD |
| Version | 1.0 |
| Status | Draft — Hackathon Prototype |
| Format | 24-hour hackathon build |
| Owner | Project team (TBD) |

---

## 1. Executive Summary

NeuroClear is a prototype, adaptive CT image-quality assistant built for a
24-hour hackathon. It ingests brain CT images from DICOM, detects and
reduces two distinct categories of noise (periodic scanner artifacts and
Poisson/quantum noise), and gives the user an interactive way to inspect
the result and its measured quality against the original. The system runs
entirely on a local laptop, with no cloud dependency, no database, and no
authentication layer.

NeuroClear is an engineering/research demo. It is **not** a medical
device, is **not** clinically validated, and is **not** intended to
inform diagnosis or treatment.

---

## 2. Problem Statement

Brain CT images are commonly degraded by two different noise mechanisms:

1. **Periodic / structured scanner artifacts** — repeating patterns
   introduced by scanner hardware, visible as identifiable peaks in the
   frequency domain.
2. **Poisson (quantum) noise** — signal-dependent noise from low photon
   counts, most pronounced in low-dose acquisitions.

Generic denoising approaches typically address only one of these at a
time, and naive smoothing often blurs clinically relevant anatomical
detail along with the noise. There is no lightweight, local, inspectable
tool that lets an engineer or researcher see *what kind* of noise is
present, apply a targeted fix per noise type, and *quantify* how much
signal fidelity and structure were preserved in the process.

---

## 3. Target Users

| Persona | Description | What they need from NeuroClear |
|---|---|---|
| Hackathon evaluator / judge | Reviews the working prototype in a short demo window | A clear, runnable app that visibly demonstrates the problem → analysis → fix → measurement loop |
| Imaging/ML engineer (primary user) | Wants to experiment with classical denoising techniques on CT slices | Fast local iteration, visual + numeric feedback (PSNR/SSIM/edge preservation), no setup friction |
| Researcher / student | Exploring frequency-domain noise analysis techniques | Transparent, inspectable pipeline (FFT view, difference maps) rather than a black box |

NeuroClear is explicitly **not** built for radiologists, clinicians, or
any diagnostic use case.

---

## 4. Goals & Objectives

- **G1.** Load brain CT images from DICOM and preserve Hounsfield Unit
  (HU) calibration where the source metadata supports it.
- **G2.** Detect periodic/structured noise using frequency-domain (FFT)
  analysis and visualize what was detected.
- **G3.** Reduce periodic noise via targeted (notch) filtering.
- **G4.** Reduce Poisson/quantum noise while preserving anatomical edges
  and structure.
- **G5.** Quantify denoising quality objectively (PSNR, SSIM, edge
  preservation) so improvement is demonstrable, not just visual opinion.
- **G6.** Provide an interactive local viewer (Streamlit) to explore
  original vs. denoised images, the frequency spectrum, and a
  difference map.
- **G7.** Ship a working, demoable prototype within a 24-hour build
  window, running on a standard laptop with no GPU or cloud dependency.

### Non-Goals (Out of Scope)

- ❌ Clinical validation, regulatory compliance, or diagnostic claims of
  any kind.
- ❌ Cloud deployment, hosted service, or multi-user access.
- ❌ Authentication, user accounts, or access control.
- ❌ A database or persistent storage of patient data.
- ❌ Deep-learning / "AI" denoising models — MVP scope is classical
  signal-processing and computer-vision techniques only.
- ❌ Automatic dataset downloading or bundling of real patient data in
  the repository.
- ❌ Support for modalities other than CT, or body regions other than
  brain, in the MVP.
- ❌ Real-time/streaming ingestion from a live scanner.

---

## 5. Success Metrics

Since NeuroClear is a hackathon prototype, success is measured by
**demonstrated functionality**, not production KPIs:

| Metric | Target for demo |
|---|---|
| End-to-end flow works | Upload → HU convert → noise analysis → denoise → metrics → view, with no crashes, on at least one sample DICOM slice |
| Periodic noise detection is visible | FFT view clearly highlights injected/detected periodic peaks |
| Denoising shows measurable improvement | PSNR and SSIM of denoised output are higher than the noisy input, relative to a reference/original image |
| Structure preservation is checkable | Edge-preservation score is reported alongside PSNR/SSIM, not just a single blended score |
| Runs locally without incident | `python -m streamlit run app.py` launches successfully on a standard laptop with only the packages in `requirements.txt` |
| No scope violations | No cloud calls, no auth, no DB, no clinical claims anywhere in the app or docs |

---

## 6. Functional Requirements

Requirements are prioritized P0 (must have for a working demo), P1
(should have if time allows), P2 (nice to have / stretch).

### 6.1 DICOM Ingestion & HU Handling — `core/dicom_loader.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Load a single DICOM file and expose pixel data + metadata | P0 |
| FR-1.2 | Load an ordered series of DICOM slices from a directory | P1 |
| FR-1.3 | Convert raw pixel values to Hounsfield Units using RescaleSlope/RescaleIntercept | P0 |
| FR-1.4 | Apply a display window (level/width), defaulting to a standard brain window | P0 |
| FR-1.5 | Handle missing/incomplete DICOM metadata gracefully (no crash) | P1 |

### 6.2 Noise Analysis — `core/noise_analysis.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | Compute general noise characteristics for a loaded slice | P1 |
| FR-2.2 | Detect periodic noise via frequency-domain (FFT) peak analysis | P0 |
| FR-2.3 | Estimate Poisson noise level (e.g. via homogeneous-region variance) | P0 |

### 6.3 Periodic Noise Removal — `core/periodic_denoise.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Construct a notch filter targeting detected periodic frequency peaks | P0 |
| FR-3.2 | Apply the notch filter to remove periodic noise from a slice | P0 |

### 6.4 Poisson Denoising — `core/poisson_denoise.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Reduce Poisson/quantum noise using an edge-preserving method | P0 |
| FR-4.2 | Expose tunable strength/parameters for the denoising method | P1 |

### 6.5 Quality Metrics — `core/quality_metrics.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | Calculate PSNR between original and denoised image | P0 |
| FR-5.2 | Calculate SSIM between original and denoised image | P0 |
| FR-5.3 | Calculate an edge/structural preservation score | P1 |

### 6.6 Pipeline Orchestration — `core/pipeline.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | Run the full load → analyze → denoise → measure flow from a single entry point | P0 |
| FR-6.2 | Allow configuration of which steps run (e.g. skip periodic removal) | P2 |

### 6.7 Interactive Viewer — `visualization/`, `app.py`

| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | Upload a DICOM file through the Streamlit UI | P0 |
| FR-7.2 | Display the CT slice with windowing controls | P0 |
| FR-7.3 | Display the frequency-domain (FFT) view with detected peaks highlighted | P1 |
| FR-7.4 | Display a before/after difference map | P1 |
| FR-7.5 | Display PSNR / SSIM / edge-preservation results numerically | P0 |
| FR-7.6 | Support multi-slice series navigation | P2 |

---

## 7. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Must run entirely locally — no network calls to cloud services required for core functionality |
| NFR-2 | Must run on a standard laptop CPU (no GPU dependency) |
| NFR-3 | Startup and single-slice processing should be fast enough for live demo use (seconds, not minutes) |
| NFR-4 | Codebase limited to the dependencies in `requirements.txt`; no unapproved new frameworks |
| NFR-5 | No patient data or large datasets committed to version control (`data/` is gitignored) |
| NFR-6 | UI and docs must not make or imply clinical/diagnostic claims |
| NFR-7 | Code should remain readable/modular enough to extend after the hackathon |

---

## 8. Primary User Flow

1. User opens the Streamlit app locally (`python -m streamlit run app.py`).
2. User uploads a brain CT DICOM file.
3. App loads the file, converts pixel data to Hounsfield Units, and
   displays the windowed image.
4. App runs noise analysis: shows the FFT view and flags detected
   periodic noise; estimates Poisson noise level.
5. User triggers denoising (periodic removal + Poisson reduction).
6. App displays the denoised image alongside the original, plus a
   difference map.
7. App reports PSNR, SSIM, and edge-preservation metrics comparing
   original vs. denoised output.
8. User can adjust windowing/parameters and re-run to compare results.

---

## 9. System Architecture Overview

```
NeuroClear/
├── app.py                 # Streamlit UI entry point
├── core/
│   ├── dicom_loader.py        # DICOM I/O, HU conversion, windowing
│   ├── noise_analysis.py      # General/periodic/Poisson noise detection
│   ├── periodic_denoise.py    # Notch filtering
│   ├── poisson_denoise.py     # Edge-preserving denoising
│   ├── quality_metrics.py     # PSNR / SSIM / edge preservation
│   └── pipeline.py            # End-to-end orchestration
├── visualization/
│   ├── ct_viewer.py           # Slice display
│   ├── fft_view.py            # Frequency-domain view
│   └── difference_map.py      # Before/after difference map
├── data/                   # Local-only working data (gitignored)
├── tests/                  # Unit tests
└── docs/
    └── PRD.md              # This document
```

Current repository status: **architecture scaffold complete** (stub
functions with docstrings, no algorithm logic yet). See
`README.md` → "Current Development Status" for the live status.

---

## 10. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| UI | Streamlit | Fast to build, runs locally, no frontend framework needed |
| DICOM I/O | pydicom | Standard Python DICOM library |
| Numerics | NumPy, SciPy | Array ops + FFT/signal processing |
| Classical CV | OpenCV, scikit-image | Filtering, denoising, PSNR/SSIM implementations |
| Plotting | Matplotlib, Plotly | Static + interactive charts (FFT view, metrics) |
| Image I/O | Pillow | Image format handling |

No database, authentication library, or cloud SDK is included by design.

---

## 11. Data Handling & Privacy

- No real patient data is to be committed to the repository at any time.
- `data/` is local-only and excluded via `.gitignore`, along with common
  DICOM/medical-imaging and large-dataset file extensions.
- Any sample data used for the demo should be de-identified/synthetic or
  from a dataset explicitly licensed for open testing.
- The app processes files locally in-memory; it does not upload data to
  any external service.

---

## 12. Assumptions & Constraints

- Build window is fixed at 24 hours — scope is intentionally minimal
  (single modality, single body region, classical methods only).
- Team has access to at least one sample brain CT DICOM file (real or
  synthetic) for development and demo purposes.
- Denoising method choices (e.g. bilateral filter vs. non-local means)
  are not yet finalized and will be selected during implementation based
  on time available and qualitative results.
- Single-user, single-session usage is assumed; no concurrency handling
  is required.

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Time runs out before Poisson denoising is tuned | Demo shows partial pipeline | Prioritize P0 items first (6.1–6.6); keep P1/P2 as stretch |
| Sample DICOM files unavailable or malformed | Blocks testing | Source/validate sample files early, before algorithm work starts |
| Denoising blurs anatomical structure | Undermines the core value proposition | Report edge-preservation score alongside PSNR/SSIM so trade-offs are visible, not hidden |
| Scope creep (e.g. adding a DB or auth "for polish") | Breaks hackathon constraints, wastes time | Explicit Non-Goals section (§4) enforced during build |
| Team over-claims capability in demo | Reputational/ethical risk given medical context | README and UI explicitly state prototype/non-clinical status (already in place) |

---

## 14. Milestones (Indicative 24-Hour Breakdown)

| Time | Milestone |
|---|---|
| Hour 0–2 | Confirm sample data, finalize architecture (done), environment setup |
| Hour 2–6 | Implement `dicom_loader.py` (P0 items) + basic viewer wiring |
| Hour 6–10 | Implement `noise_analysis.py` (periodic + Poisson detection) + FFT view |
| Hour 10–14 | Implement `periodic_denoise.py` and `poisson_denoise.py` |
| Hour 14–17 | Implement `quality_metrics.py`, wire into `pipeline.py` |
| Hour 17–20 | Wire full pipeline into `app.py`, add difference map view |
| Hour 20–22 | Polish UI, add basic tests, verify clean run on a fresh machine |
| Hour 22–24 | Final README/docs pass, rehearse demo |

---

## 15. Open Questions

- Which specific Poisson denoising method will we use (bilateral filter,
  non-local means, or another classical method)? To be decided during
  implementation.
- Do we have a confirmed, license-clear sample DICOM dataset for the
  demo, or do we need to generate synthetic test slices?
- Is multi-slice series support (FR-1.2, FR-7.6) in scope for this
  hackathon, or deferred as a post-hackathon extension?
- What quantitative threshold (if any) counts as "sufficient"
  improvement in PSNR/SSIM for the demo narrative?

---

## 16. Glossary

| Term | Definition |
|---|---|
| **DICOM** | Digital Imaging and Communications in Medicine — the standard file format/protocol for medical images |
| **HU (Hounsfield Unit)** | A standardized scale for CT radiodensity, calibrated per-scan via RescaleSlope/RescaleIntercept |
| **PSNR** | Peak Signal-to-Noise Ratio — a decibel measure of reconstruction fidelity between two images |
| **SSIM** | Structural Similarity Index — a perceptual measure of structural similarity between two images |
| **Poisson / quantum noise** | Signal-dependent noise arising from limited photon counts, common in low-dose CT |
| **Periodic / structured noise** | Repeating artifact patterns from scanner hardware, identifiable as peaks in the frequency spectrum |
| **Notch filter** | A frequency-domain filter that suppresses specific narrow frequency bands (e.g. detected noise peaks) while leaving the rest of the spectrum intact |

---

*This PRD reflects intended scope for the SEC086 hackathon prototype and
will evolve as implementation proceeds. It carries no clinical authority
and should not be used to make claims about diagnostic suitability.*
