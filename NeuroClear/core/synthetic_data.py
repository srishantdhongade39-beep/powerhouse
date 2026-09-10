"""
synthetic_data.py
-------------------
Generates realistic brain CT phantoms calibrated in true Hounsfield Units (HU)
with injected periodic scanner artifacts (frequency spikes) and quantum
(Poisson) noise.

Provides an immediate, self-contained test slice for evaluation and live demos.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def generate_brain_ct_phantom(
    size: int = 512,
    add_periodic_artifact: bool = True,
    periodic_freq: Tuple[float, float] = (0.12, 0.08),
    periodic_amplitude: float = 25.0,
    add_poisson_noise: bool = True,
    poisson_photon_count: float = 1200.0,
    random_seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a clinically calibrated synthetic Brain CT slice in Hounsfield Units (HU).

    Tissues modeled (Standard HU scale):
    - Air background: -1000 HU
    - Cranial Skull Bone: +900 to +1100 HU
    - Gray Matter: +40 HU
    - White Matter: +30 HU
    - Ventricles (CSF): +15 HU
    - Basal Ganglia / Deep structures: +45 HU

    Args:
        size: Dimension of square image (e.g. 512x512).
        add_periodic_artifact: Injects sinusoidal scanner hardware interference.
        periodic_freq: Normalized spatial frequencies (fy, fx).
        periodic_amplitude: Amplitude of periodic wave in HU.
        add_poisson_noise: Injects signal-dependent photon quantum noise.
        poisson_photon_count: Photon flux simulation factor (lower = noisier).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - noisy_hu: 2D numpy array with artifacts and noise in HU.
            - clean_ground_truth: Clean 2D numpy array in HU.
            - info: Metadata dict describing injected artifact parameters.
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    h, w = size, size
    y, x = np.mgrid[-1.0:1.0:complex(0, h), -1.0:1.0:complex(0, w)]

    # Start with air
    clean = np.full((h, w), -1000.0, dtype=np.float32)

    # 1. Outer Skull Bone (+1000 HU)
    # Scaled ellipse
    skull_outer = (x / 0.72) ** 2 + (y / 0.88) ** 2 <= 1.0
    clean[skull_outer] = 1000.0

    # 2. Inner Skull / Brain cavity
    skull_inner = (x / 0.66) ** 2 + (y / 0.82) ** 2 <= 1.0
    clean[skull_inner] = 40.0  # Gray matter baseline

    # 3. White matter interior
    white_matter = (x / 0.52) ** 2 + (y / 0.65) ** 2 <= 1.0
    clean[white_matter] = 30.0

    # 4. Basal Ganglia / Thalamus (oval structures)
    bg_left = ((x + 0.16) / 0.10) ** 2 + ((y - 0.05) / 0.18) ** 2 <= 1.0
    bg_right = ((x - 0.16) / 0.10) ** 2 + ((y - 0.05) / 0.18) ** 2 <= 1.0
    clean[bg_left | bg_right] = 45.0

    # 5. Lateral Ventricles (CSF, +15 HU, crescent / curved slits)
    ventricle_left = (
        ((x + 0.06) / 0.04) ** 2 + ((y + 0.08) / 0.22) ** 2 <= 1.0
    ) & (x < -0.02)
    ventricle_right = (
        ((x - 0.06) / 0.04) ** 2 + ((y + 0.08) / 0.22) ** 2 <= 1.0
    ) & (x > 0.02)
    clean[ventricle_left | ventricle_right] = 15.0

    # 6. Third Ventricle (center slit)
    third_vent = (np.abs(x) < 0.015) & (y > -0.05) & (y < 0.15)
    clean[third_vent] = 15.0

    # 7. Subtle tissue heterogeneity
    clean[skull_inner] += rng.normal(0.0, 1.2, size=clean[skull_inner].shape).astype(np.float32)

    noisy = clean.copy()
    injected_info: Dict[str, Any] = {
        "periodic_injected": False,
        "poisson_injected": False,
        "periodic_amplitude": periodic_amplitude,
        "periodic_freq": periodic_freq,
    }

    # Inject Periodic Scanner Interference (Harmonics creating distinct frequency peaks)
    if add_periodic_artifact:
        fy, fx = periodic_freq
        # Harmonic components representing detector grid / scanner ring interference
        wave1 = periodic_amplitude * np.cos(2.0 * np.pi * (fx * np.arange(w)[None, :] + fy * np.arange(h)[:, None]))
        wave2 = (periodic_amplitude * 0.45) * np.sin(2.0 * np.pi * ((-fy * 0.7) * np.arange(h)[:, None] + (fx * 1.3) * np.arange(w)[None, :]))
        # Only inject inside the scan field of view
        fov_mask = (x**2 + y**2) <= 0.95
        noisy[fov_mask] += (wave1 + wave2)[fov_mask].astype(np.float32)
        injected_info["periodic_injected"] = True

    # Inject Signal-Dependent Quantum (Poisson) Noise
    if add_poisson_noise:
        # In CT physics, measured raw intensities follow Poisson photon counting.
        # Higher attenuation regions (bone) have fewer photons (higher noise).
        # In HU space, we model quantum noise variance proportional to tissue attenuation:
        head_mask = skull_outer
        # Shift HU to positive attenuation proxy: mu ~ (HU + 1000) / 1000
        mu = np.maximum(0.05, (clean + 1000.0) / 1000.0)
        # Quantum noise standard deviation is higher in low photon counts
        base_sigma = 18.0 * (1200.0 / max(100.0, poisson_photon_count))
        noise_sigma_map = base_sigma * np.sqrt(mu)
        quantum_noise = rng.normal(0.0, 1.0, size=(h, w)) * noise_sigma_map
        noisy[head_mask] += quantum_noise[head_mask].astype(np.float32)
        injected_info["poisson_injected"] = True
        injected_info["photon_count"] = poisson_photon_count

    return noisy, clean, injected_info


def create_synthetic_dicom_dataset(
    hu_array: np.ndarray,
    patient_id: str = "PHANTOM_SEC086",
    series_desc: str = "Demo Brain CT (SEC086 Synthetic)",
) -> pydicom.Dataset:
    """
    Wrap a 2D HU array into a fully valid in-memory pydicom.Dataset.
    RescaleSlope = 1.0, RescaleIntercept = -1024.0.
    """
    h, w = hu_array.shape
    slope = 1.0
    intercept = -1024.0

    # Raw stored pixel values: pixel = (HU - intercept) / slope
    raw_pixels = np.round((hu_array - intercept) / slope).astype(np.int16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.PatientName = "NeuroClear^DemoPhantom"
    ds.PatientID = patient_id
    ds.Modality = "CT"
    ds.SeriesDescription = series_desc
    ds.StudyDate = "20260910"
    ds.InstanceNumber = 1

    ds.Rows = h
    ds.Columns = w
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.WindowCenter = 40.0
    ds.WindowWidth = 80.0
    ds.PixelSpacing = [0.45, 0.45]
    ds.SliceThickness = 2.5

    ds.PixelData = raw_pixels.tobytes()
    return ds


def generate_brain_ct_volume(
    num_slices: int = 16,
    size: int = 256,
    add_periodic_artifact: bool = False,
    add_poisson_noise: bool = True,
    poisson_photon_count: float = 1600.0,
    random_seed: Optional[int] = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[pydicom.Dataset]]:
    """
    Generate a full 3D volumetric multi-slice Brain CT study with anatomical progression
    along the axial Z-axis (skull base -> mid ventricles -> centrum semiovale -> vertex).

    Args:
        num_slices: Total number of axial CT slices (e.g. 16 or 24).
        size: Matrix dimensions for each slice (e.g. 256x256).
        add_periodic_artifact: Inject periodic scanner ring/stripe noise.
        add_poisson_noise: Inject signal-dependent quantum noise.
        poisson_photon_count: Photon count scaling factor.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (noisy_slices, clean_slices, dicom_datasets).
    """
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random.RandomState()

    noisy_slices = []
    clean_slices = []
    dicom_datasets = []

    study_uid = generate_uid()
    series_uid = generate_uid()
    slice_thickness = 3.0  # mm

    h, w = size, size
    y, x = np.mgrid[-1.0:1.0:complex(0, h), -1.0:1.0:complex(0, w)]

    for idx in range(num_slices):
        z_norm = idx / max(1, num_slices - 1)  # 0.0 (skull base) to 1.0 (vertex)
        z_pos = (idx - num_slices / 2.0) * slice_thickness

        # Anatomical radius varies along Z (narrow at base, wider in middle, narrowing at vertex)
        head_scale_x = 0.65 + 0.12 * np.sin(np.pi * z_norm)
        head_scale_y = 0.80 + 0.12 * np.sin(np.pi * z_norm)

        clean = np.full((h, w), -1000.0, dtype=np.float32)

        # 1. Outer Skull Bone (+1000 HU)
        skull_outer = (x / head_scale_x) ** 2 + (y / head_scale_y) ** 2 <= 1.0
        clean[skull_outer] = 1000.0

        # 2. Inner Skull / Brain cavity
        bone_thickness = 0.06 - 0.02 * z_norm  # Thicker at base
        inner_scale_x = max(0.2, head_scale_x - bone_thickness)
        inner_scale_y = max(0.2, head_scale_y - bone_thickness)
        skull_inner = (x / inner_scale_x) ** 2 + (y / inner_scale_y) ** 2 <= 1.0
        clean[skull_inner] = 40.0  # Gray matter baseline

        # 3. White matter
        wm_scale_x = max(0.15, inner_scale_x * 0.80)
        wm_scale_y = max(0.15, inner_scale_y * 0.80)
        white_matter = (x / wm_scale_x) ** 2 + (y / wm_scale_y) ** 2 <= 1.0
        clean[white_matter] = 30.0

        # 4. Basal Ganglia (Mid slices ~ z_norm 0.35 to 0.65)
        if 0.30 <= z_norm <= 0.70:
            bg_left = ((x + 0.15) / 0.09) ** 2 + ((y - 0.03) / 0.15) ** 2 <= 1.0
            bg_right = ((x - 0.15) / 0.09) ** 2 + ((y - 0.03) / 0.15) ** 2 <= 1.0
            clean[bg_left | bg_right] = 45.0

        # 5. Ventricles (Vary significantly with slice height)
        if 0.25 <= z_norm <= 0.75:
            vent_size = np.sin(np.pi * (z_norm - 0.25) / 0.50)
            vw = 0.05 * (0.4 + 0.6 * vent_size)
            vh = 0.22 * (0.4 + 0.6 * vent_size)
            vent_left = (((x + 0.055) / max(0.01, vw)) ** 2 + ((y + 0.05) / max(0.01, vh)) ** 2 <= 1.0) & (x < -0.015)
            vent_right = (((x - 0.055) / max(0.01, vw)) ** 2 + ((y + 0.05) / max(0.01, vh)) ** 2 <= 1.0) & (x > 0.015)
            clean[vent_left | vent_right] = 15.0  # CSF

            # 3rd ventricle
            if 0.35 <= z_norm <= 0.60:
                third_v = (np.abs(x) < 0.012) & (y > -0.04) & (y < 0.12)
                clean[third_v] = 15.0

        # 6. Subtle parenchyma texture
        clean[skull_inner] += rng.normal(0.0, 1.0, size=clean[skull_inner].shape).astype(np.float32)

        noisy = clean.copy()

        # Periodic scanner ring/stripe artifact
        if add_periodic_artifact:
            fx, fy = 0.12, 0.08
            wave1 = 28.0 * np.cos(2.0 * np.pi * (fx * np.arange(w)[None, :] + fy * np.arange(h)[:, None]))
            wave2 = 12.0 * np.sin(2.0 * np.pi * ((-fy * 0.7) * np.arange(h)[:, None] + (fx * 1.3) * np.arange(w)[None, :]))
            fov_mask = (x**2 + y**2) <= 0.95
            noisy[fov_mask] += (wave1 + wave2)[fov_mask].astype(np.float32)

        # Poisson quantum noise
        if add_poisson_noise:
            mu = np.maximum(0.05, (clean + 1000.0) / 1000.0)
            base_sigma = 18.0 * (1200.0 / max(100.0, poisson_photon_count))
            noise_sigma_map = base_sigma * np.sqrt(mu)
            quantum = rng.normal(0.0, 1.0, size=(h, w)) * noise_sigma_map
            noisy[skull_outer] += quantum[skull_outer].astype(np.float32)

        # Create DICOM Dataset for this slice
        ds = create_synthetic_dicom_dataset(
            noisy,
            patient_id="NC-VOL-BRAIN3D",
            series_desc=f"Axial Brain CT 3D Series ({num_slices} Slices)",
        )
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = idx + 1
        ds.SliceLocation = float(z_pos)
        ds.ImagePositionPatient = [-w * 0.45 / 2.0, -h * 0.45 / 2.0, float(z_pos)]
        ds.SliceThickness = float(slice_thickness)

        noisy_slices.append(noisy)
        clean_slices.append(clean)
        dicom_datasets.append(ds)

    return noisy_slices, clean_slices, dicom_datasets

