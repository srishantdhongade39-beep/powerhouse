"""
ct_viewer.py
--------------
Interactive CT slice viewer supporting:
- Side-by-side comparison (Original vs Denoised)
- Synchronized Sub-Pixel Dual Zoom (Plotly with bicubic spline smoothing, zero pixelation)
- Interactive Before/After Split-Screen Wipe Slider (Juxtapose)
- 3× High-Magnification ROI Detail Magnifier
- Dynamic windowing display and interactive Hounsfield Unit inspection
"""

from typing import Any, Optional
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_ct_viewer(
    original_display: np.ndarray,
    denoised_display: Optional[np.ndarray] = None,
    hu_original: Optional[np.ndarray] = None,
    hu_denoised: Optional[np.ndarray] = None,
    title_left: str = "Original / Noisy CT Slice",
    title_right: str = "NeuroClear Denoised Output",
) -> None:
    """
    Render comprehensive CT slice inspection view within Streamlit with multiple
    interactive modes: Classic Side-by-Side, Interactive Split Wipe, Synchronized Zoom,
    and 3x ROI Magnifier.
    """
    if denoised_display is None:
        st.image(original_display, caption=title_left, use_container_width=True, clamp=True)
        return

    # View Mode Selector
    view_mode = st.radio(
        "Display Mode:",
        options=[
            "🖼️ Side-by-Side",
            "↔️ Interactive Split-Wipe Slider",
            "🔍 Synchronized Sub-Pixel Dual Zoom",
            "🔬 3× ROI Detail Magnifier",
        ],
        horizontal=True,
        index=0,
        help="Choose how you want to inspect and compare the original and denoised CT slices.",
    )

    h, w = original_display.shape[:2]

    # Mode 1: Classic Side-by-Side with Anti-Aliased Sub-Pixel Upsampling Option
    if view_mode == "🖼️ Side-by-Side":
        anti_alias = st.checkbox(
            "Smooth Sub-Pixel Anti-Aliasing (Prevents Pixelation on Zoom)",
            value=True if min(h, w) <= 256 else False,
            help="Applies classical bicubic/Lanczos spline interpolation to display smooth anatomical gradients instead of jagged nearest-neighbor pixel blocks.",
        )

        disp_left = original_display
        disp_right = denoised_display

        if anti_alias and max(h, w) < 768:
            target_dim = 768
            scale = target_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            disp_left = cv2.resize(original_display, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            disp_right = cv2.resize(denoised_display, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### 🔴 {title_left}")
            st.image(disp_left, use_container_width=True, clamp=True)
            if hu_original is not None:
                c_min, c_max = float(np.min(hu_original)), float(np.max(hu_original))
                c_mean, c_std = float(np.mean(hu_original)), float(np.std(hu_original))
                st.caption(f"HU Range: [{c_min:.0f}, {c_max:.0f}] · Mean: {c_mean:.1f} · Std: {c_std:.1f} HU")

        with col2:
            st.markdown(f"##### 🟢 {title_right}")
            st.image(disp_right, use_container_width=True, clamp=True)
            if hu_denoised is not None:
                d_min, d_max = float(np.min(hu_denoised)), float(np.max(hu_denoised))
                d_mean, d_std = float(np.mean(hu_denoised)), float(np.std(hu_denoised))
                st.caption(f"HU Range: [{d_min:.0f}, {d_max:.0f}] · Mean: {d_mean:.1f} · Std: {d_std:.1f} HU")

    # Mode 2: Interactive Before/After Split Wipe Slider (Juxtapose)
    elif view_mode == "↔️ Interactive Split-Wipe Slider":
        st.markdown("Drag the slider below to wipe back and forth across the exact same anatomy:")
        split_pos = st.slider("↔️ Wipe Divider Position (%)", 0, 100, 50, step=1)

        split_col = int(w * split_pos / 100.0)
        split_col = max(0, min(w, split_col))

        # Create 3-channel RGB composite for clean visual divider line
        if len(original_display.shape) == 2:
            comp_left = cv2.cvtColor(original_display, cv2.COLOR_GRAY2RGB)
            comp_right = cv2.cvtColor(denoised_display, cv2.COLOR_GRAY2RGB)
        else:
            comp_left = original_display.copy()
            comp_right = denoised_display.copy()

        composite = np.zeros_like(comp_left)
        composite[:, :split_col] = comp_left[:, :split_col]
        composite[:, split_col:] = comp_right[:, split_col:]

        # Draw vertical separator line (cyan #00E5FF)
        if 0 < split_col < w:
            x_start = max(0, split_col - 2)
            x_end = min(w, split_col + 2)
            composite[:, x_start:x_end] = [0, 229, 255]

        st.image(composite, use_container_width=True, clamp=True)
        col_lbl1, col_lbl2 = st.columns(2)
        with col_lbl1:
            st.caption(f"◀️ Left Side: **{title_left}** (0% to {split_pos}%)")
        with col_lbl2:
            st.caption(f"▶️ Right Side: **{title_right}** ({split_pos}% to 100%)")

    # Mode 3: Synchronized Sub-Pixel Dual Zoom (Plotly with shared axes)
    elif view_mode == "🔍 Synchronized Sub-Pixel Dual Zoom":
        st.markdown(
            "💡 **Click and drag a box** anywhere on either image to zoom into both simultaneously. "
            "Sub-pixel bicubic spline interpolation prevents pixelation."
        )

        hu_left = hu_original if hu_original is not None else original_display.astype(np.float32)
        hu_right = hu_denoised if hu_denoised is not None else denoised_display.astype(np.float32)

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"🔴 {title_left}", f"🟢 {title_right}"],
            horizontal_spacing=0.03,
        )

        # Plotly go.Heatmap with zsmooth='best' provides continuous bicubic spline interpolation
        fig.add_trace(
            go.Heatmap(
                z=hu_left,
                colorscale="gray",
                zsmooth="best",
                showscale=False,
                hoverongaps=False,
                name="Original",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=hu_right,
                colorscale="gray",
                zsmooth="best",
                showscale=True,
                colorbar=dict(title="HU"),
                hoverongaps=False,
                name="Denoised",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(matches="x")
        fig.update_yaxes(matches="y", autorange="reversed")
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=580,
            dragmode="zoom",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Mode 4: 3× High-Magnification ROI Detail Magnifier
    elif view_mode == "🔬 3× ROI Detail Magnifier":
        st.markdown("Inspect fine micro-structure and noise textures under 3× magnification:")
        
        # Center-crop 1/3 of the image
        ch, cw = h // 3, w // 3
        y1, y2 = h // 3, h // 3 + ch
        x1, x2 = w // 3, w // 3 + cw

        roi_left = original_display[y1:y2, x1:x2]
        roi_right = denoised_display[y1:y2, x1:x2]

        # Interpolate with Lanczos4 for sharp continuous texture
        roi_left_mag = cv2.resize(roi_left, (cw * 3, ch * 3), interpolation=cv2.INTER_LANCZOS4)
        roi_right_mag = cv2.resize(roi_right, (cw * 3, ch * 3), interpolation=cv2.INTER_LANCZOS4)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"##### 🔴 {title_left} (3× Center ROI)")
            st.image(roi_left_mag, use_container_width=True, clamp=True)
            st.caption(f"Coordinates: Rows [{y1}..{y2}], Cols [{x1}..{x2}]")
        with col_m2:
            st.markdown(f"##### 🟢 {title_right} (3× Center ROI)")
            st.image(roi_right_mag, use_container_width=True, clamp=True)
            st.caption(f"Continuous Lanczos spline rendering — Notice noise reduction on identical structures.")


def render_interactive_hu_inspector(
    hu_array: np.ndarray,
    title: str = "Interactive CT Pixel & HU Inspector"
) -> None:
    """
    Interactive 2D heatmap viewer with Plotly allowing cursor zoom, pan,
    smooth sub-pixel anti-aliased interpolation, and exact Hounsfield Unit hover reading.
    """
    smooth_toggle = st.checkbox("Smooth Sub-Pixel Interpolation (Prevents Pixelation on Zoom)", value=True)
    zsmooth_val = "best" if smooth_toggle else False

    fig = go.Figure(
        data=go.Heatmap(
            z=hu_array,
            colorscale="gray",
            zsmooth=zsmooth_val,
            colorbar=dict(title="HU"),
            hoverongaps=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        height=540,
        dragmode="zoom",
    )
    st.plotly_chart(fig, use_container_width=True)
