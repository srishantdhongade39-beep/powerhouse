"""
model_3d.py
-----------
Interactive 3D anatomical modeling and volumetric visualization for NeuroClear:
- For multi-slice CT series: 3D interactive volumetric isosurface mesh (Bone, Soft Tissue, Lesions)
- For single-slice uploaded CT/DICOM images & active slices: Interactive 3D Radiodensity Topographic Surface Mesh
  where tissue radiodensity (HU) is converted into physical 3D elevation (Z-height), allowing full 360° rotation,
  tilt, and depth inspection of tumors, lesions, and bone architecture.
"""

from typing import Any, List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st


COLORSCALE_MAP = {
    "Bone": [
        [0.0, "#05080f"],
        [0.2, "#1a2536"],
        [0.5, "#4a5d78"],
        [0.8, "#c4baa8"],
        [1.0, "#ffffff"],
    ],
    "Gray": "gray",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Thermal": "thermal",
    "Hot": "hot",
    "Magma": "magma",
    "Cividis": "cividis",
}


def render_3d_model(
    volume_hu: List[np.ndarray],
    active_slice_idx: int = 0,
    window_center: float = 40.0,
    window_width: float = 80.0,
    source_name: Optional[str] = None,
    denoised_hu: Optional[np.ndarray] = None,
) -> None:
    """
    Render an interactive 3D model in Streamlit based on the currently loaded image or series.
    Adapts dynamically to both single-slice uploads and multi-slice 3D series.
    """
    if not volume_hu:
        st.info("No CT data loaded for 3D reconstruction. Please load a study or upload a CT image in the sidebar.")
        return

    num_slices = len(volume_hu)
    src_title = source_name or "Active CT Study"
    active_idx = min(max(0, active_slice_idx), num_slices - 1)

    st.markdown(f"#### 🧊 Interactive 3D Anatomical Model — `{src_title}`")

    # Determine display mode
    if num_slices > 1:
        view_mode = st.radio(
            "3D Visualization Architecture:",
            ["🧊 3D Volumetric Isosurface (Full Volume)", f"🗺️ 3D Topographic Surface (Active Slice #{active_idx + 1})"],
            horizontal=True,
            key="radio_3d_view_mode",
        )
    else:
        view_mode = "🗺️ 3D Topographic Surface"

    # Case 1: Multi-slice volumetric isosurface study
    if view_mode.startswith("🧊"):
        st.caption(f"3D Volumetric Isosurface Model reconstructed from **{num_slices} axial CT slices**.")

        # Subsample for smooth 60fps browser rendering
        vol_stack = np.array(volume_hu, dtype=np.float32)
        nz, ny, nx = vol_stack.shape

        step = max(1, nx // 128)
        sub_vol = vol_stack[:, ::step, ::step]
        sz, sy, sx = sub_vol.shape

        z_grid, y_grid, x_grid = np.mgrid[:sz, :sy, :sx]

        col_3d_ctl1, col_3d_ctl2 = st.columns([2, 2])
        with col_3d_ctl1:
            tissue_target = st.selectbox(
                "3D Anatomical Target:",
                ["All Tissues (Transparent)", "Dense Bone & Skull (>300 HU)", "Brain Soft Tissue (20-100 HU)", "Custom HU Range"],
                index=0,
                key="select_vol_tissue",
            )
        with col_3d_ctl2:
            colorscale_name = st.selectbox(
                "3D Colormap:",
                ["Bone", "Viridis", "Plasma", "Thermal", "Gray"],
                index=0,
                key="select_vol_cmap",
            )

        if tissue_target == "Dense Bone & Skull (>300 HU)":
            isomin, isomax = 300.0, float(np.max(sub_vol))
            opacity_val = 0.35
        elif tissue_target == "Brain Soft Tissue (20-100 HU)":
            isomin, isomax = 15.0, 120.0
            opacity_val = 0.20
        elif tissue_target == "All Tissues (Transparent)":
            isomin, isomax = -200.0, float(np.max(sub_vol))
            opacity_val = 0.15
        else:
            isomin = float(window_center - window_width / 2.0)
            isomax = float(window_center + window_width / 2.0)
            opacity_val = 0.25

        vol_cmap = COLORSCALE_MAP.get(colorscale_name, "viridis")

        fig_vol = go.Figure(
            data=go.Volume(
                x=x_grid.flatten(),
                y=y_grid.flatten(),
                z=z_grid.flatten(),
                value=sub_vol.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=opacity_val,
                surface_count=8,
                colorscale=vol_cmap,
                colorbar=dict(title="HU"),
            )
        )

        fig_vol.update_layout(
            scene=dict(
                xaxis_title="X (Right-Left)",
                yaxis_title="Y (Ant-Post)",
                zaxis_title="Z (Axial Slice)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.4),
                ),
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=620,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Case 2: 3D Topographic Surface Mesh (single slice or active slice)
    else:
        st.markdown(
            "💡 **3D Radiodensity Topographic Surface Model**: "
            "The active CT slice is projected into 3D Euclidean space where tissue radiodensity (HU) is converted into **physical 3D elevation (Z-height)**. "
            "Dense bone forms towering 3D walls, soft brain parenchyma forms undulating valleys, and tumors/lesions appear as distinct 3D topological relief structures. "
            "**Click, hold, and drag with your mouse to rotate 360° in 3D.**"
        )

        col_src, col_s1, col_s2, col_s3 = st.columns([2, 1.5, 1.5, 1.5])
        with col_src:
            if denoised_hu is not None:
                surface_source = st.radio(
                    "Surface Source:",
                    ["Denoised Output (Clean)", "Original Input (Noisy)"],
                    horizontal=True,
                    key="radio_surf_source",
                )
                slice_data = denoised_hu if "Clean" in surface_source else volume_hu[active_idx]
            else:
                slice_data = volume_hu[active_idx]

        with col_s1:
            color_scheme = st.selectbox(
                "Surface Palette:",
                ["Bone", "Viridis", "Plasma", "Thermal", "Magma", "Gray"],
                index=0,
                key="select_surf_cmap",
            )
        with col_s2:
            elev_scale = st.slider("3D Elevation Relief:", 0.2, 3.0, 1.0, 0.1, key="slider_surf_elev")
        with col_s3:
            lighting = st.selectbox(
                "Lighting & Shading:",
                ["Realistic Specular", "Smooth Diffuse", "Flat"],
                index=0,
                key="select_surf_light",
            )

        h, w = slice_data.shape

        # Downsample grid for ultra-responsive 60fps interaction
        step = max(1, max(h, w) // 180)
        sub_slice = slice_data[::step, ::step].astype(np.float64) * elev_scale

        x_coords = np.arange(0, w, step)
        y_coords = np.arange(0, h, step)

        lighting_opts = dict(
            ambient=0.45,
            diffuse=0.65,
            roughness=0.4,
            specular=1.2 if lighting == "Realistic Specular" else 0.15,
            fresnel=0.35,
        )

        surf_cmap = COLORSCALE_MAP.get(color_scheme, "viridis")

        fig_surf = go.Figure(
            data=[
                go.Surface(
                    x=x_coords,
                    y=y_coords,
                    z=sub_slice,
                    colorscale=surf_cmap,
                    lighting=lighting_opts,
                    colorbar=dict(title="HU Elevation"),
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="#00E5FF", project_z=True)
                    ),
                )
            ]
        )

        fig_surf.update_layout(
            title=f"3D Topographic Surface: {src_title} (Slice #{active_idx + 1}/{num_slices})",
            scene=dict(
                xaxis_title="X (Columns)",
                yaxis_title="Y (Rows)",
                zaxis_title="HU Elevation (Z)",
                camera=dict(
                    eye=dict(x=-1.5, y=-1.5, z=1.2),
                ),
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            height=640,
        )
        st.plotly_chart(fig_surf, use_container_width=True)
