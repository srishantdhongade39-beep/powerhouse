"""
NeuroClear - SEC086: CT Image Denoising
-----------------------------------------
Interactive Streamlit Application for Adaptive Brain CT Image Denoising.
Combines frequency-domain periodic artifact removal (adaptive notch filtering)
and spatial Poisson noise reduction (Anscombe transform + edge-preserving filters)
with real-time quality metric benchmarking (PSNR, SSIM, Edge Preservation Index).
"""

from io import BytesIO
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pydicom
from PIL import Image
import streamlit as st

from core.dicom_loader import (
    WINDOW_PRESETS,
    apply_window,
    convert_to_hounsfield_units,
    get_dicom_metadata,
    load_dicom,
)
from core.noise_analysis import analyze_noise, detect_periodic_noise, estimate_poisson_noise
from core.periodic_denoise import create_notch_filter, remove_periodic_noise
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
from visualization.ct_viewer import render_ct_viewer, render_interactive_hu_inspector
from visualization.difference_map import render_difference_map
from visualization.fft_view import render_fft_view

# Streamlit Page Config
st.set_page_config(
    page_title="NeuroClear — SEC086 CT Denoising",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern typography and clean styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize application session state variables."""
    if "hu_slice" not in st.session_state:
        st.session_state.hu_slice = None
    if "clean_slice" not in st.session_state:
        st.session_state.clean_slice = None
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    if "loaded_source_name" not in st.session_state:
        st.session_state.loaded_source_name = None
    if "raw_dataset" not in st.session_state:
        st.session_state.raw_dataset = None


def load_demo_phantom() -> None:
    """Generate and load synthetic brain CT phantom with injected noise."""
    with st.spinner("Generating calibrated Brain CT Phantom with injected artifacts..."):
        noisy, clean, info = generate_brain_ct_phantom(
            size=256,
            add_periodic_artifact=True,
            periodic_amplitude=35.0,
            periodic_freq=(0.12, 0.08),
            add_poisson_noise=True,
            poisson_photon_count=1000.0,
            random_seed=42,
        )
        ds = create_synthetic_dicom_dataset(
            noisy,
            patient_id="SEC086-BRAIN-01",
            series_desc="Synthetic Brain CT with Harmonics & Quantum Noise",
        )
        st.session_state.hu_slice = noisy
        st.session_state.clean_slice = clean
        st.session_state.metadata = get_dicom_metadata(ds)
        st.session_state.raw_dataset = ds
        st.session_state.loaded_source_name = "Synthetic Brain CT Phantom (Injected Noise)"
def load_real_clinical_sample() -> None:
    """Load sample real Brain CT clinical DICOM slice from data directory."""
    from pathlib import Path
    sample_path = Path(__file__).parent / "data" / "sample_real_brain_ct.dcm"
    if sample_path.exists():
        ds = load_dicom(str(sample_path))
        hu = convert_to_hounsfield_units(ds)
        st.session_state.hu_slice = hu
        st.session_state.clean_slice = None
        st.session_state.metadata = get_dicom_metadata(ds)
        st.session_state.raw_dataset = ds
        st.session_state.loaded_source_name = "Real Clinical Brain CT (Patient 1CT1, 128×128)"
        st.session_state.pipeline_results = None


def main() -> None:
    init_session_state()

    # App Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="main-header">🧠 NeuroClear</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">SEC086 — Adaptive Brain CT Image Denoising &amp; Quality Benchmark'
            ' · 24-Hour Hackathon Prototype</div>',
            unsafe_allow_html=True,
        )
    with col_h2:
        st.info(
            "🔬 **Research Prototype**\n\nNon-clinical demo. Classical signal processing & CV only.",
            icon="⚠️",
        )

    # ------------------ SIDEBAR CONTROLS ------------------
    with st.sidebar:
        st.header("1. Image Source")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🧪 Demo Phantom", use_container_width=True, type="primary"):
                load_demo_phantom()
                st.rerun()
        with col_b2:
            if st.button("🏥 Real Clinical CT", use_container_width=True):
                load_real_clinical_sample()
                st.rerun()

        if st.button("🔄 Reset Image", use_container_width=True):
            st.session_state.hu_slice = None
            st.session_state.clean_slice = None
            st.session_state.metadata = None
            st.session_state.pipeline_results = None
            st.session_state.loaded_source_name = None
            st.session_state.raw_dataset = None
            st.rerun()

        st.caption("— or upload a CT slice (.dcm, .png, .jpg) —")
        uploaded_file = st.file_uploader(
            "Upload CT Slice",
            type=["dcm", "dicom", "png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=False,
            help="Upload a Brain CT DICOM file (.dcm) or standard CT image file (.png, .jpg, .tiff).",
        )

        if uploaded_file is not None and st.session_state.loaded_source_name != uploaded_file.name:
            file_name = uploaded_file.name.lower()
            try:
                if file_name.endswith((".dcm", ".dicom")):
                    ds = load_dicom(uploaded_file)
                    hu = convert_to_hounsfield_units(ds)
                    meta = get_dicom_metadata(ds)
                    raw_ds = ds
                else:
                    # Standard image format (PNG, JPG, TIFF)
                    pil_img = Image.open(uploaded_file).convert("L")
                    arr_gray = np.array(pil_img, dtype=np.float32)
                    # Map 8-bit [0, 255] grayscale to standard CT brain window range [0, 80 HU]
                    hu = (arr_gray / 255.0) * 80.0
                    raw_ds = create_synthetic_dicom_dataset(
                        hu,
                        patient_id=f"IMG_{uploaded_file.name[:12]}",
                        series_desc="Imported Image CT Slice",
                    )
                    meta = get_dicom_metadata(raw_ds)
                    meta["series_description"] = f"Imported Image ({uploaded_file.name})"

                st.session_state.hu_slice = hu
                st.session_state.clean_slice = None
                st.session_state.metadata = meta
                st.session_state.raw_dataset = raw_ds
                st.session_state.loaded_source_name = uploaded_file.name
                st.session_state.pipeline_results = None
                st.success(f"Loaded: {uploaded_file.name} ({hu.shape[0]}×{hu.shape[1]})")
            except Exception as e:
                st.error(f"Error loading image: {e}")

        st.divider()

        # Windowing Controls
        st.header("2. Display Windowing")
        preset_names = list(WINDOW_PRESETS.keys()) + ["Custom"]
        selected_preset = st.selectbox(
            "Preset",
            options=preset_names,
            index=0,
            help="Clinical window presets for brain CT radiodensity inspection.",
        )

        if selected_preset in WINDOW_PRESETS:
            default_c = float(WINDOW_PRESETS[selected_preset]["center"])
            default_w = float(WINDOW_PRESETS[selected_preset]["width"])
        else:
            default_c, default_w = 40.0, 80.0

        window_center = st.slider(
            "Window Level / Center (HU)",
            min_value=-500.0,
            max_value=1000.0,
            value=float(default_c),
            step=5.0,
        )
        window_width = st.slider(
            "Window Width (HU)",
            min_value=10.0,
            max_value=2500.0,
            value=float(default_w),
            step=10.0,
        )

        st.divider()

        # Denoising Algorithms Settings
        st.header("3. Denoising Pipeline")

        st.subheader("Periodic Notch Filter")
        enable_periodic = st.checkbox("Enable Periodic Noise Removal", value=True)
        notch_type = st.radio(
            "Notch Filter Profile",
            options=["gaussian", "butterworth"],
            index=0,
            horizontal=True,
            disabled=not enable_periodic,
        )
        notch_radius = st.slider(
            "Notch Bandwidth Radius (D0)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            disabled=not enable_periodic,
        )
        fft_threshold = st.slider(
            "Peak Sensitivity (std factor)",
            min_value=1.5,
            max_value=4.5,
            value=2.5,
            step=0.1,
            disabled=not enable_periodic,
            help="Multiplier of standard deviation above median to classify an FFT peak as periodic noise.",
        )

        st.subheader("Poisson Denoising")
        enable_poisson = st.checkbox("Enable Poisson Denoising", value=True)
        poisson_method = st.selectbox(
            "Method",
            options=["nlm", "bilateral", "tv", "wavelet"],
            format_func=lambda x: {
                "nlm": "Non-Local Means (NLM)",
                "bilateral": "Bilateral Filter",
                "tv": "Total Variation (TV Chambolle)",
                "wavelet": "Wavelet Thresholding (BayesShrink)",
            }.get(x, x),
            index=0,
            disabled=not enable_poisson,
        )
        poisson_strength = st.slider(
            "Denoising Strength",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            disabled=not enable_poisson,
        )
        use_anscombe = st.checkbox(
            "Anscombe Variance Stabilization",
            value=True,
            disabled=not enable_poisson,
            help="Transforms Poisson signal-dependent variance into approximately unit Gaussian noise before filtering.",
        )

        st.divider()
        run_btn = st.button("🚀 Run NeuroClear Denoise", type="primary", use_container_width=True)

    # ------------------ MAIN CONTENT AREA ------------------
    if st.session_state.hu_slice is None:
        st.info(
            "👋 **Welcome to NeuroClear!**\n\n"
            "To begin testing the denoising engine, click **'🧪 Load Demo Phantom'** in the left sidebar "
            "or upload any standard Brain CT DICOM file (`.dcm`).",
            icon="💡",
        )
        st.markdown(
            """
            ### System Capabilities:
            1. **DICOM Ingestion**: Automatically calibrates raw sensor digital numbers into **Hounsfield Units (HU)** using `RescaleSlope` and `RescaleIntercept`.
            2. **Frequency-Domain Peak Detection**: Pinpoints harmonic spikes in the 2D Fast Fourier Transform (FFT) spectrum caused by detector grid interference and scanner rings.
            3. **Targeted Notch Filtering**: Suppresses identified periodic spikes using smooth Gaussian or Butterworth frequency reject filters while preserving core anatomical DC energy.
            4. **Anscombe-Stabilized Poisson Denoising**: Linearizes photon quantum noise variance and applies state-of-the-art edge-preserving spatial filtering (NLM, Bilateral, or TV Chambolle).
            5. **Objective Quality Verification**: Reports PSNR, SSIM, and an **Edge Preservation Index** along with residual difference maps to ensure tissue borders remain crisp.
            """
        )
        return

    hu_slice = st.session_state.hu_slice
    clean_ref = st.session_state.clean_slice

    # Run pipeline if button clicked or if results are not yet present
    if run_btn or st.session_state.pipeline_results is None:
        pipeline_opts = {
            "skip_periodic": not enable_periodic,
            "skip_poisson": not enable_poisson,
            "notch_radius": notch_radius,
            "notch_filter_type": notch_type,
            "threshold_factor": fft_threshold,
            "poisson_method": poisson_method,
            "poisson_strength": poisson_strength,
            "use_anscombe": use_anscombe,
            "window_center": window_center,
            "window_width": window_width,
            "ground_truth": clean_ref,
        }
        with st.spinner("Processing slice through NeuroClear pipeline..."):
            results = run_neuroclear_pipeline(hu_slice, pipeline_opts)
            st.session_state.pipeline_results = results

    results = st.session_state.pipeline_results

    # Active dataset banner
    source_name = st.session_state.loaded_source_name or "Loaded CT Slice"
    st.caption(f"📁 Active Source: **{source_name}** · Matrix: `{hu_slice.shape[0]}×{hu_slice.shape[1]}` pixels")

    # Extract pipeline results
    denoised_hu = results.get("hu_denoised", hu_slice)
    display_orig = results.get("display_original", apply_window(hu_slice, window_center, window_width, as_uint8=True))
    display_denoised = results.get("display_denoised", apply_window(denoised_hu, window_center, window_width, as_uint8=True))
    diff_array = results.get("difference_map", hu_slice - denoised_hu)
    notch_mask = results.get("notch_mask", None)
    initial_noise = results.get("initial_noise", {})
    periodic_analysis = initial_noise.get("periodic", {})
    poisson_est = initial_noise.get("poisson", {})

    # Render Quality KPI Metric Cards
    metrics = results.get("metrics", {})
    gt_metrics = results.get("ground_truth_metrics", None)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        if gt_metrics:
            p_val = gt_metrics.get("output_psnr_db", 0.0)
            p_diff = gt_metrics.get("psnr_improvement_db", 0.0)
            st.metric("PSNR (vs Clean Reference)", f"{p_val:.2f} dB", f"{p_diff:+.2f} dB")
        else:
            psnr_val = metrics.get("psnr_db", 0.0)
            st.metric("PSNR (Fidelity)", f"{psnr_val:.2f} dB", help="Fidelity between input and denoised output.")

    with kpi2:
        if gt_metrics:
            s_val = gt_metrics.get("output_ssim", 0.0)
            s_diff = gt_metrics.get("ssim_improvement", 0.0)
            st.metric("SSIM (Structure)", f"{s_val:.4f}", f"{s_diff:+.4f}")
        else:
            ssim_val = metrics.get("ssim", 0.0)
            st.metric("SSIM (Structure)", f"{ssim_val:.4f}", help="Structural similarity between input and output.")

    with kpi3:
        epi_val = metrics.get("edge_preservation", 0.0)
        st.metric(
            "Edge Preservation Index",
            f"{epi_val:.3f}",
            "Preserved" if epi_val >= 0.75 else "Softened",
            help="Pearson correlation of Sobel gradient magnitudes along anatomical edges (ideal = 1.0).",
        )

    with kpi4:
        peaks_found = periodic_analysis.get("peak_count", 0)
        power_removed = metrics.get("noise_power_removed", 0.0)
        st.metric(
            "Periodic Peaks Notched",
            f"{peaks_found} peaks",
            f"Δσ² = {power_removed:.1f}",
            help="Number of conjugate harmonic peak frequencies suppressed by notch filtering.",
        )

    st.markdown("---")

    # ------------------ WORKSPACE TABS ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ Before vs After CT View",
        "🌐 Frequency Spectrum (FFT)",
        "🔬 Difference & Residual Map",
        "📋 DICOM Metadata & Physics",
    ])

    with tab1:
        render_ct_viewer(
            original_display=display_orig,
            denoised_display=display_denoised,
            hu_original=hu_slice,
            hu_denoised=denoised_hu,
            title_left="Original / Noisy CT Slice",
            title_right=f"NeuroClear Output ({poisson_method.upper()} + Notch)",
        )

        st.markdown("#### 🔍 Interactive HU Pixel Inspector")
        inspect_target = st.radio(
            "Inspect Array:",
            options=["Denoised Output", "Original Input", "Clean Reference" if clean_ref is not None else None],
            horizontal=True,
            index=0,
        )
        if inspect_target == "Denoised Output":
            render_interactive_hu_inspector(denoised_hu, title="Interactive Denoised CT HU Inspector")
        elif inspect_target == "Original Input":
            render_interactive_hu_inspector(hu_slice, title="Interactive Original CT HU Inspector")
        elif inspect_target == "Clean Reference" and clean_ref is not None:
            render_interactive_hu_inspector(clean_ref, title="Interactive Clean Reference HU Inspector")

    with tab2:
        render_fft_view(
            image=hu_slice,
            noise_info=periodic_analysis,
            notch_mask=notch_mask,
        )

        # Peak Table
        detected_peaks = periodic_analysis.get("peaks", [])
        if detected_peaks:
            st.markdown("##### 📍 Detected Harmonic Artifact Coordinates")
            table_data = []
            for idx, p in enumerate(detected_peaks, start=1):
                table_data.append({
                    "Peak #": idx,
                    "Frequency (u, v)": f"({p['u']}, {p['v']})",
                    "Conjugate (-u, -v)": f"({-p['u']}, {-p['v']})",
                    "Radial Freq": f"{p['freq_normalized']:.3f}",
                    "Log Magnitude": f"{p['magnitude']:.2f}",
                    "Prominence": f"{p['prominence']:.2f}",
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.success("No anomalous periodic frequency peaks detected in this slice.")

    with tab3:
        diff_array = hu_slice - denoised_hu
        render_difference_map(
            original_image=hu_slice,
            processed_image=denoised_hu,
            difference_array=diff_array,
        )

        st.markdown("#### 📈 Poisson Noise Level Analysis")
        poisson_est = results.get("poisson_estimation", {})
        if poisson_est:
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated Noise Sigma (σ)", f"{poisson_est.get('estimated_sigma', 0.0):.2f} HU")
            c2.metric("Estimated Tissue SNR", f"{poisson_est.get('snr_db', 0.0):.1f} dB")
            c3.metric("Evaluated Noise Classification", poisson_est.get("noise_level", "Normal").upper())

    with tab4:
        st.markdown("#### 📑 DICOM Header Tags & Calibration")
        meta = st.session_state.metadata or {}
        if meta:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(
                    f"• **Patient Name**: `{meta.get('patient_name', 'De-identified')}`<br>"
                    f"• **Patient ID**: `{meta.get('patient_id', 'N/A')}`<br>"
                    f"• **Modality**: `{meta.get('modality', 'CT')}`<br>"
                    f"• **Series Description**: `{meta.get('series_description', 'N/A')}`<br>"
                    f"• **Matrix Dimensions**: `{meta.get('rows', hu_slice.shape[0])} × {meta.get('columns', hu_slice.shape[1])}`",
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f"• **Rescale Slope**: `{meta.get('rescale_slope', 1.0)}`<br>"
                    f"• **Rescale Intercept**: `{meta.get('rescale_intercept', 0.0)} HU`<br>"
                    f"• **Pixel Spacing**: `{meta.get('pixel_spacing', '1.0 x 1.0 mm')}`<br>"
                    f"• **Slice Thickness**: `{meta.get('slice_thickness', 'N/A')} mm`<br>"
                    f"• **Photometric Interpretation**: `{meta.get('photometric_interpretation', 'MONOCHROME2')}`",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            """
            #### 🔬 Signal Processing Architecture:
            * **Hounsfield Unit Calibration**: $HU = (\\text{StoredPixel} \\times \\text{RescaleSlope}) + \\text{RescaleIntercept}$.
            * **Anscombe Transformation**: For quantum photon counting noise, $f(x) = 2 \\sqrt{x + \\frac{3}{8}}$ stabilizes signal-dependent Poisson variance into additive unit Gaussian variance.
            * **Adaptive Notch Filter**: Suppresses detected harmonic frequency spikes $H(u,v) = \\prod_k \\left(1 - e^{-\\frac{D_k^2}{2 D_0^2}}\\right)$.
            * **Structural Metric (EPI)**: Evaluates edge correlation $\\rho_{\\nabla} = \\frac{\\sum (\\nabla I_{ref} - \\bar{\\nabla}_{ref})(\\nabla I_{proc} - \\bar{\\nabla}_{proc})}{\\sigma_{\\nabla ref} \\sigma_{\\nabla proc}}$ along Sobel gradient contours.
            """
        )

        st.divider()
        st.markdown("#### 💾 Export Denoised CT Data")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            # Export PNG (display windowed)
            img_pil = Image.fromarray(display_denoised)
            buf_png = BytesIO()
            img_pil.save(buf_png, format="PNG")
            st.download_button(
                label="📥 Download Denoised Slice (PNG)",
                data=buf_png.getvalue(),
                file_name="neuroclear_denoised_slice.png",
                mime="image/png",
                use_container_width=True,
            )

        with exp_col2:
            # Export DICOM dataset
            raw_ds = st.session_state.raw_dataset
            if raw_ds is not None:
                try:
                    slope = float(getattr(raw_ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(raw_ds, "RescaleIntercept", 0.0))
                    raw_stored = np.round((denoised_hu - intercept) / slope).astype(np.int16)

                    out_ds = pydicom.Dataset(raw_ds)
                    out_ds.PixelData = raw_stored.tobytes()
                    out_ds.SeriesDescription = "NeuroClear SEC086 Denoised"
                    buf_dcm = BytesIO()
                    pydicom.dcmwrite(buf_dcm, out_ds)
                    st.download_button(
                        label="📥 Download Calibrated DICOM (.dcm)",
                        data=buf_dcm.getvalue(),
                        file_name="neuroclear_denoised.dcm",
                        mime="application/dicom",
                        use_container_width=True,
                    )
                except Exception as ex:
                    st.caption(f"DICOM export formatting notice: {ex}")

    st.markdown("---")
    st.caption("NeuroClear · SEC086 Hackathon Build · 100% Local Processing · No Cloud / No DB / Non-Clinical")


if __name__ == "__main__":
    main()
