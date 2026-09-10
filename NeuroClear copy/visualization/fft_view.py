"""
fft_view.py
-------------
Frequency-domain (2D FFT magnitude spectrum) visualization.
Displays the power spectrum with detected periodic noise peaks highlighted
and overlays the corresponding notch filter reject mask.
"""

from typing import Any, Dict, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_fft_view(
    image: Optional[np.ndarray] = None,
    noise_info: Optional[Dict[str, Any]] = None,
    notch_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Render frequency-domain analysis dashboard in Streamlit.

    Args:
        image: 2D numpy array to transform if noise_info not pre-calculated.
        noise_info: Output dict from noise_analysis.detect_periodic_noise().
        notch_mask: 2D numpy array of applied notch filter mask.
    """
    if noise_info is None:
        if image is None:
            st.warning("No image or FFT spectrum available to render.")
            return
        from core.noise_analysis import detect_periodic_noise
        noise_info = detect_periodic_noise(image)

    log_mag = noise_info.get("log_magnitude", None)
    if log_mag is None and image is not None:
        fft_val = np.fft.fftshift(np.fft.fft2(image))
        log_mag = np.log1p(np.abs(fft_val))

    if log_mag is None:
        st.error("Cannot compute FFT spectrum.")
        return

    peaks = noise_info.get("peaks", [])
    h, w = log_mag.shape

    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        st.markdown("##### 🌐 2D FFT Magnitude Spectrum (Log Scale)")

        # Create Plotly Heatmap
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=log_mag,
                colorscale="Viridis",
                colorbar=dict(title="Log |F|"),
                hoverinfo="x+y+z",
            )
        )

        # Overlay detected artifact peak coordinates
        if peaks:
            peak_x = []
            peak_y = []
            hover_texts = []

            for p in peaks:
                # Primary peak
                r, c = p["row"], p["col"]
                u, v = p["u"], p["v"]
                peak_x.append(c)
                peak_y.append(r)
                hover_texts.append(f"Peak (u={u}, v={v})<br>Freq: {p['freq_normalized']:.3f}<br>Prominence: {p['prominence']:.2f}")

                # Conjugate peak
                conj = p.get("conjugate", {})
                if conj:
                    peak_x.append(conj["col"])
                    peak_y.append(conj["row"])
                    hover_texts.append(f"Conjugate (-u={-u}, -v={-v})")

            fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=14,
                        color="red",
                        line=dict(width=2.5, color="#FF2A2A"),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                    name="Detected Artifact Peaks",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#0A0E17",
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        if peaks:
            st.warning(f"⚠️ **{len(peaks)} Periodic Hardware Spike Pair(s) Detected** (Gantry vibration / power harmonics active).")
        else:
            st.success("✅ **Nominal Scanner Frequency Profile (0 Periodic Harmonics Detected)**")
            st.caption("Standard clinical diagnostic scans do not suffer from motor vibration harmonics. Spatial quantum photon noise reduction is actively processed by the edge-preserving engine.")

    with col2:
        st.markdown("##### 🛡️ Frequency Notch Reject Mask H(u, v)")
        if notch_mask is not None:
            mask_fig = go.Figure(
                data=go.Heatmap(
                    z=notch_mask,
                    colorscale="Blues_r",
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="Pass (1) / Stop (0)"),
                )
            )
            mask_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0A0E17",
                plot_bgcolor="#0A0E17",
                height=460,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
            )
            st.plotly_chart(mask_fig, use_container_width=True)
            if peaks:
                st.caption("Notch filter zeroes out detected hardware vibration spike coordinates while preserving 100% of anatomical frequency spectrum.")
            else:
                st.caption("Pass-Through Mask (1.0 across all frequencies): Full anatomical spectrum is preserved without frequency attenuation.")
        else:
            st.info("Run the denoising pipeline to generate the notch reject filter mask.")

