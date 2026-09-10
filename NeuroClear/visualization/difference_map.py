"""
difference_map.py
--------------------
Before/after difference map visualization.
Reveals exactly what was removed by the denoising pipeline (scanner rings,
stripes, quantum photon noise). If the difference map shows no anatomical
outlines, structural fidelity is verified.
"""

from typing import Any, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_difference_map(
    original_image: Optional[np.ndarray] = None,
    processed_image: Optional[np.ndarray] = None,
    difference_array: Optional[np.ndarray] = None,
) -> None:
    """
    Render visual difference map and residual error histogram in Streamlit.

    Args:
        original_image: 2D numpy array (original slice).
        processed_image: 2D numpy array (denoised slice).
        difference_array: Optional pre-calculated difference (original - processed).
    """
    if difference_array is None:
        if original_image is None or processed_image is None:
            st.warning("Both original and denoised slices are required to compute difference map.")
            return
        diff = np.asarray(original_image, dtype=np.float32) - np.asarray(processed_image, dtype=np.float32)
    else:
        diff = np.asarray(difference_array, dtype=np.float32)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("##### 🔬 Removed Noise & Artifact Residual Map (Original − Denoised)")

        # Symmetrical diverging scale centered at 0
        v_abs = float(np.percentile(np.abs(diff), 99.5))
        v_abs = max(1.0, v_abs)

        fig_diff = px.imshow(
            diff,
            color_continuous_scale="RdBu_r",
            zmin=-v_abs,
            zmax=v_abs,
            labels={"x": "X", "y": "Y", "color": "Δ HU"},
        )
        fig_diff.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_colorbar=dict(title="HU Residual"),
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        st.caption(
            "💡 **Clinical Verification Tip**: A high-quality denoising result will show "
            "uniform noise grain and periodic stripe artifacts in the difference map. "
            "If brain tissue or bone boundaries appear sharply here, anatomical edges are being eroded."
        )

    with col2:
        st.markdown("##### 📊 Residual Distribution Histogram")

        diff_flat = diff.ravel()
        hist, bin_edges = np.histogram(diff_flat, bins=60)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                marker_color="#2E86AB",
                name="Residual Counts",
            )
        )
        fig_hist.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Difference (HU)",
            yaxis_title="Pixel Count",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        mean_diff = float(np.mean(diff_flat))
        std_diff = float(np.std(diff_flat))
        st.markdown(
            f"**Residual Statistics:**<br>"
            f"• Residual Mean: `{mean_diff:+.3f} HU` *(ideal ≈ 0)*<br>"
            f"• Residual Std Dev: `{std_diff:.2f} HU` *(noise power removed)*",
            unsafe_allow_html=True,
        )
