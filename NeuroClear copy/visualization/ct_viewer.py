"""
ct_viewer.py
--------------
Interactive CT slice viewer supporting side-by-side comparison (Original vs Denoised),
dynamic windowing display, and interactive hover inspection of Hounsfield Units.
"""

from typing import Any, Optional
import numpy as np
import plotly.express as px
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
    Render side-by-side CT slice inspection view within Streamlit.

    Args:
        original_display: Windowed uint8 or [0,1] float array for display.
        denoised_display: Windowed array for denoised slice.
        hu_original: Raw calibrated HU array for hover inspection.
        hu_denoised: Raw calibrated HU array for hover inspection.
        title_left: Header for left slice.
        title_right: Header for right slice.
    """
    if denoised_display is None:
        st.image(original_display, caption=title_left, use_container_width=True, clamp=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"##### 🔴 {title_left}")
        st.image(original_display, use_container_width=True, clamp=True)
        if hu_original is not None:
            c_min, c_max = float(np.min(hu_original)), float(np.max(hu_original))
            c_mean, c_std = float(np.mean(hu_original)), float(np.std(hu_original))
            st.caption(f"HU Range: [{c_min:.0f}, {c_max:.0f}] · Mean: {c_mean:.1f} · Std: {c_std:.1f} HU")

    with col2:
        st.markdown(f"##### 🟢 {title_right}")
        st.image(denoised_display, use_container_width=True, clamp=True)
        if hu_denoised is not None:
            d_min, d_max = float(np.min(hu_denoised)), float(np.max(hu_denoised))
            d_mean, d_std = float(np.mean(hu_denoised)), float(np.std(hu_denoised))
            st.caption(f"HU Range: [{d_min:.0f}, {d_max:.0f}] · Mean: {d_mean:.1f} · Std: {d_std:.1f} HU")


def render_interactive_hu_inspector(
    hu_array: np.ndarray,
    title: str = "Interactive CT Pixel & HU Inspector"
) -> None:
    """
    Interactive 2D heatmap viewer with Plotly allowing cursor zoom, pan,
    and exact Hounsfield Unit hover reading at any pixel coordinate.
    """
    fig = px.imshow(
        hu_array,
        color_continuous_scale="gray",
        labels={"x": "X (column)", "y": "Y (row)", "color": "HU"},
        title=title,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
        coloraxis_colorbar=dict(title="Hounsfield Units (HU)"),
    )
    st.plotly_chart(fig, use_container_width=True)
