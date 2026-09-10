"""
ct_viewer.py
--------------
Professional medical CT slice viewer for NeuroClear supporting:
- Multi-slice navigation bar with slice counters and spatial Z-location
- Synchronized viewing modes:
  * 🖼️ Side-by-Side Dual Comparison (Synchronized display & stats)
  * ↔️ Interactive Split-Wipe Slider (Draggable comparison curtain)
  * ✨ Alpha Blend / Overlay Comparison (Interactive cross-fade)
  * 🔴 Original CT Only (Full canvas with HUD overlays)
  * 🟢 NeuroClear Denoised Only (Full canvas with HUD overlays)
  * 🔍 Synchronized Sub-Pixel Dual Zoom (Plotly with continuous bicubic spline)
  * 🔬 3× High-Magnification ROI Detail Magnifier
- Anatomical orientation markers (A/P, L/R) and active windowing HUD overlay
- Interactive Hounsfield Unit (HU) pixel inspector with hover coordinate readouts
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_slice_navigation_bar(
    current_index: int,
    total_slices: int,
    slice_location_mm: Optional[float] = None,
    key_prefix: str = "slice_nav",
) -> int:
    """
    Render a responsive medical slice navigation bar with Prev/Next buttons,
    fast slider scrubber, and slice location indicators.
    """
    if total_slices <= 1:
        st.info("ℹ️ Single-slice study loaded (1 of 1). Click **'🧪 3D Volume (16s)'** in the sidebar to load the 16-slice series.")
        return 0

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #00E5FF; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #00E5FF; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                🩻 Axial CT Slice Navigator (Skull Base → Vertex)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_prev, col_slider, col_next, col_info = st.columns([1.2, 4, 1.2, 2.2])

    with col_prev:
        if st.button("◀ Prev Slice", key=f"{key_prefix}_prev", use_container_width=True, disabled=(current_index <= 0)):
            current_index = max(0, current_index - 1)
            st.session_state.active_slice_idx = current_index
            st.rerun()

    with col_next:
        if st.button("Next Slice ▶", key=f"{key_prefix}_next", use_container_width=True, disabled=(current_index >= total_slices - 1)):
            current_index = min(total_slices - 1, current_index + 1)
            st.session_state.active_slice_idx = current_index
            st.rerun()

    with col_slider:
        selected_slice = st.slider(
            "Scrub Slice (Z-Axis)",
            min_value=1,
            max_value=total_slices,
            value=current_index + 1,
            step=1,
            key=f"{key_prefix}_slider",
            label_visibility="collapsed",
            help="Drag to rapidly scrub through CT axial slices.",
        )
        if selected_slice - 1 != current_index:
            current_index = selected_slice - 1
            st.session_state.active_slice_idx = current_index
            st.rerun()

    with col_info:
        loc_str = f"<br><span style='font-size:0.75rem; color:#94A3B8;'>Z-Loc: {slice_location_mm:+.1f} mm</span>" if slice_location_mm is not None else ""
        st.markdown(
            f"<div style='text-align: right; font-weight: 700; color: #00E5FF; font-family: monospace; font-size: 1.05rem; padding-top: 4px;'>"
            f"Slice {current_index + 1} / {total_slices}{loc_str}</div>",
            unsafe_allow_html=True,
        )

    return current_index


def _add_hud_overlay(
    image_uint8: np.ndarray,
    window_center: float,
    window_width: float,
    slice_idx: Optional[int] = None,
    total_slices: Optional[int] = None,
    tag_label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw clean medical orientation markers (A, P, L, R) and HUD window info
    onto a display copy of the image.
    """
    if len(image_uint8.shape) == 2:
        canvas = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image_uint8.copy()

    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.40, min(0.65, w / 600.0))
    thickness = 1
    text_color = (220, 245, 255)  # Soft cyan-white

    # Anatomical Markers
    # Top: Anterior (A)
    cv2.putText(canvas, "A", (w // 2 - 6, int(22 * font_scale + 8)), font, font_scale * 1.1, text_color, thickness + 1, cv2.LINE_AA)
    # Bottom: Posterior (P)
    cv2.putText(canvas, "P", (w // 2 - 6, h - 10), font, font_scale * 1.1, text_color, thickness + 1, cv2.LINE_AA)
    # Right side of patient = Left of screen (R)
    cv2.putText(canvas, "R", (10, h // 2 + 5), font, font_scale * 1.1, text_color, thickness + 1, cv2.LINE_AA)
    # Left side of patient = Right of screen (L)
    cv2.putText(canvas, "L", (w - 22, h // 2 + 5), font, font_scale * 1.1, text_color, thickness + 1, cv2.LINE_AA)

    # Top-Left HUD (Window Info)
    win_txt = f"W:{int(window_width)} L:{int(window_center)}"
    cv2.putText(canvas, win_txt, (12, int(22 * font_scale + 8)), font, font_scale * 0.85, (0, 229, 255), thickness, cv2.LINE_AA)

    # Top-Right HUD (Label / Mode)
    if tag_label:
        (tw, th), _ = cv2.getTextSize(tag_label, font, font_scale * 0.85, thickness)
        cv2.putText(canvas, tag_label, (w - tw - 12, int(22 * font_scale + 8)), font, font_scale * 0.85, (52, 211, 153), thickness, cv2.LINE_AA)

    # Bottom-Left HUD (Slice info)
    if slice_idx is not None and total_slices is not None:
        slice_txt = f"Im: {slice_idx + 1}/{total_slices}"
        cv2.putText(canvas, slice_txt, (12, h - 10), font, font_scale * 0.85, (148, 163, 184), thickness, cv2.LINE_AA)

    return canvas


def render_ct_viewer(
    original_display: np.ndarray,
    denoised_display: Optional[np.ndarray] = None,
    hu_original: Optional[np.ndarray] = None,
    hu_denoised: Optional[np.ndarray] = None,
    window_center: float = 40.0,
    window_width: float = 80.0,
    slice_index: int = 0,
    total_slices: int = 1,
    title_left: str = "ORIGINAL CT (Raw / Unprocessed)",
    title_right: str = "NEUROCLEAR PROCESSED CT (Denoised)",
) -> None:
    """
    Render comprehensive CT slice inspection view within Streamlit with multiple
    interactive modes: Classic Side-by-Side, Interactive Split Wipe, Alpha-Blend Overlay,
    Synchronized Zoom, and 3x ROI Magnifier.
    """
    if denoised_display is None:
        canvas_orig = _add_hud_overlay(
            original_display,
            window_center=window_center,
            window_width=window_width,
            slice_idx=slice_index,
            total_slices=total_slices,
            tag_label="ORIGINAL CT",
        )
        st.image(canvas_orig, caption=f"{title_left} (Slice {slice_index + 1}/{total_slices})", use_container_width=True, clamp=True)
        return

    # View Mode Selector
    view_mode = st.radio(
        "Viewer Inspection Mode:",
        options=[
            "🖼️ Side-by-Side Synchronized",
            "↔️ Interactive Split-Wipe Slider",
            "✨ Alpha Blend / Overlay",
            "🔴 Original CT Only",
            "🟢 NeuroClear Denoised Only",
            "🔍 Synchronized Sub-Pixel Dual Zoom",
            "🔬 3× Center ROI Magnifier",
        ],
        horizontal=True,
        index=0,
        help="Choose how you want to inspect and compare the original CT slice against the NeuroClear denoised output.",
    )

    h, w = original_display.shape[:2]

    # Prepare HUD-augmented images
    hud_orig = _add_hud_overlay(
        original_display,
        window_center=window_center,
        window_width=window_width,
        slice_idx=slice_index,
        total_slices=total_slices,
        tag_label="ORIGINAL CT",
    )
    hud_denoised = _add_hud_overlay(
        denoised_display,
        window_center=window_center,
        window_width=window_width,
        slice_idx=slice_index,
        total_slices=total_slices,
        tag_label="NEUROCLEAR PROCESSED CT",
    )

    # 1. Mode: Side-by-Side Synchronized
    if view_mode == "🖼️ Side-by-Side Synchronized":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='font-weight:700; color:#EF4444;'>🔴 {title_left}</div>", unsafe_allow_html=True)
            st.image(hud_orig, use_container_width=True, clamp=True)
            if hu_original is not None:
                c_min, c_max = float(np.min(hu_original)), float(np.max(hu_original))
                c_mean, c_std = float(np.mean(hu_original)), float(np.std(hu_original))
                st.caption(f"HU Range: `[{c_min:.0f}, {c_max:.0f}]` · Mean: `{c_mean:.1f}` · Std: `{c_std:.1f} HU`")

        with col2:
            st.markdown(f"<div style='font-weight:700; color:#10B981;'>🟢 {title_right}</div>", unsafe_allow_html=True)
            st.image(hud_denoised, use_container_width=True, clamp=True)
            if hu_denoised is not None:
                d_min, d_max = float(np.min(hu_denoised)), float(np.max(hu_denoised))
                d_mean, d_std = float(np.mean(hu_denoised)), float(np.std(hu_denoised))
                st.caption(f"HU Range: `[{d_min:.0f}, {d_max:.0f}]` · Mean: `{d_mean:.1f}` · Std: `{d_std:.1f} HU`")

    # 2. Mode: Interactive Split Wipe Slider
    elif view_mode == "↔️ Interactive Split-Wipe Slider":
        st.markdown("Drag the slider below to smoothly wipe between the Original and NeuroClear denoised slice:")
        split_pos = st.slider("↔️ Wipe Curtain Divider (%)", 0, 100, 50, step=1)

        split_col = int(w * split_pos / 100.0)
        split_col = max(0, min(w, split_col))

        if len(original_display.shape) == 2:
            comp_left = cv2.cvtColor(original_display, cv2.COLOR_GRAY2RGB)
            comp_right = cv2.cvtColor(denoised_display, cv2.COLOR_GRAY2RGB)
        else:
            comp_left = original_display.copy()
            comp_right = denoised_display.copy()

        composite = np.zeros_like(comp_left)
        composite[:, :split_col] = comp_left[:, :split_col]
        composite[:, split_col:] = comp_right[:, split_col:]

        # Draw glowing divider line (cyan #00E5FF)
        if 0 < split_col < w:
            x_start = max(0, split_col - 2)
            x_end = min(w, split_col + 2)
            composite[:, x_start:x_end] = [0, 229, 255]

        hud_comp = _add_hud_overlay(
            composite,
            window_center=window_center,
            window_width=window_width,
            slice_idx=slice_index,
            total_slices=total_slices,
            tag_label=f"WIPE: {split_pos}%",
        )
        st.image(hud_comp, use_container_width=True, clamp=True)
        col_lbl1, col_lbl2 = st.columns(2)
        with col_lbl1:
            st.caption(f"◀️ Left: **Original / Noisy** (0% to {split_pos}%)")
        with col_lbl2:
            st.caption(f"▶️ Right: **NeuroClear Denoised** ({split_pos}% to 100%)")

    # 3. Mode: Alpha Blend / Overlay Comparison
    elif view_mode == "✨ Alpha Blend / Overlay":
        st.markdown("Cross-fade between Original and NeuroClear denoised slice to observe edge retention:")
        alpha = st.slider("🎨 Denoised Blend Opacity (%)", 0, 100, 75, step=5) / 100.0

        blended = cv2.addWeighted(original_display, 1.0 - alpha, denoised_display, alpha, 0.0)
        hud_blended = _add_hud_overlay(
            blended,
            window_center=window_center,
            window_width=window_width,
            slice_idx=slice_index,
            total_slices=total_slices,
            tag_label=f"BLEND: {int(alpha*100)}% CLEAN",
        )
        st.image(hud_blended, use_container_width=True, clamp=True)
        st.caption(f"Overlay Ratio: **{int((1.0 - alpha)*100)}% Original** + **{int(alpha*100)}% NeuroClear Denoised**")

    # 4. Mode: Original Only
    elif view_mode == "🔴 Original CT Only":
        st.image(hud_orig, use_container_width=True, clamp=True)
        if hu_original is not None:
            c_min, c_max = float(np.min(hu_original)), float(np.max(hu_original))
            st.caption(f"Original CT Slice · HU Range: `[{c_min:.0f}, {c_max:.0f}]` · Matrix: `{w}×{h}`")

    # 5. Mode: NeuroClear Denoised Only
    elif view_mode == "🟢 NeuroClear Denoised Only":
        st.image(hud_denoised, use_container_width=True, clamp=True)
        if hu_denoised is not None:
            d_min, d_max = float(np.min(hu_denoised)), float(np.max(hu_denoised))
            st.caption(f"NeuroClear Cleaned CT Slice · HU Range: `[{d_min:.0f}, {d_max:.0f}]` · Matrix: `{w}×{h}`")

    # 6. Mode: Synchronized Sub-Pixel Dual Zoom
    elif view_mode == "🔍 Synchronized Sub-Pixel Dual Zoom":
        st.markdown(
            "💡 **Click and drag a box** on either image to zoom both simultaneously. "
            "Continuous bicubic spline interpolation prevents pixelation."
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

        fig.update_xaxes(matches="x", showgrid=False, zeroline=False)
        fig.update_yaxes(matches="y", autorange="reversed", showgrid=False, zeroline=False)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#0A0E17",
            margin=dict(l=10, r=10, t=40, b=10),
            height=560,
            dragmode="zoom",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 7. Mode: 3× ROI Detail Magnifier
    elif view_mode == "🔬 3× Center ROI Magnifier":
        st.markdown("Inspect fine anatomical micro-structure and noise textures under 3× magnification:")
        ch, cw = h // 3, w // 3
        y1, y2 = h // 3, h // 3 + ch
        x1, x2 = w // 3, w // 3 + cw

        roi_left = original_display[y1:y2, x1:x2]
        roi_right = denoised_display[y1:y2, x1:x2]

        roi_left_mag = cv2.resize(roi_left, (cw * 3, ch * 3), interpolation=cv2.INTER_LANCZOS4)
        roi_right_mag = cv2.resize(roi_right, (cw * 3, ch * 3), interpolation=cv2.INTER_LANCZOS4)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"##### 🔴 {title_left} (3× Center ROI)")
            st.image(roi_left_mag, use_container_width=True, clamp=True)
            st.caption(f"Coordinates: Rows `[{y1}..{y2}]`, Cols `[{x1}..{x2}]`")
        with col_m2:
            st.markdown(f"##### 🟢 {title_right} (3× Center ROI)")
            st.image(roi_right_mag, use_container_width=True, clamp=True)
            st.caption("Continuous Lanczos spline rendering — Notice noise reduction on identical structures.")


def render_interactive_hu_inspector(
    hu_array: np.ndarray,
    title: str = "Interactive CT Pixel & HU Inspector",
    pixel_spacing_mm: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Interactive 2D heatmap viewer with Plotly allowing cursor zoom, pan,
    smooth sub-pixel anti-aliased interpolation, and exact Hounsfield Unit hover reading.
    """
    smooth_toggle = st.checkbox("Smooth Sub-Pixel Interpolation (Prevents Pixelation on Zoom)", value=True, key="insp_smooth")
    zsmooth_val = "best" if smooth_toggle else False

    fig = go.Figure(
        data=go.Heatmap(
            z=hu_array,
            colorscale="gray",
            zsmooth=zsmooth_val,
            colorbar=dict(title="HU"),
            hoverongaps=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br><b>Radiodensity: %{z:.1f} HU</b><extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#0A0E17",
        margin=dict(l=10, r=10, t=40, b=10),
        height=540,
        dragmode="zoom",
    )
    st.plotly_chart(fig, use_container_width=True)

    if pixel_spacing_mm:
        st.caption(f"📏 Calibrated Pixel Spacing: `{pixel_spacing_mm[0]:.3f} mm × {pixel_spacing_mm[1]:.3f} mm` per pixel.")

