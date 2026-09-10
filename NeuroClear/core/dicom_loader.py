"""
dicom_loader.py
----------------
Responsible for reading brain CT images from DICOM files/series,
converting pixel data into calibrated Hounsfield Unit (HU) values,
and applying display windowing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pydicom

WINDOW_PRESETS: Dict[str, Dict[str, Any]] = {
    "Brain": {
        "center": 40.0,
        "width": 80.0,
        "description": "Standard Brain Parenchyma (0 to 80 HU)",
    },
    "Subdural / Blood": {
        "center": 75.0,
        "width": 100.0,
        "description": "Subdural Hematoma & Acute Blood (25 to 125 HU)",
    },
    "Stroke / Ischemia": {
        "center": 32.0,
        "width": 8.0,
        "description": "Early Ischemic Stroke Sensitivity (28 to 36 HU)",
    },
    "Soft Tissue": {
        "center": 50.0,
        "width": 350.0,
        "description": "General Soft Tissue & Chest (-125 to 225 HU)",
    },
    "Chest / Mediastinum": {
        "center": 40.0,
        "width": 400.0,
        "description": "Chest & Mediastinal Structures (-160 to 240 HU)",
    },
    "Lung": {
        "center": -600.0,
        "width": 1500.0,
        "description": "Lung Parenchyma & Air Spaces (-1350 to 150 HU)",
    },
    "Abdomen": {
        "center": 60.0,
        "width": 400.0,
        "description": "Abdominal Organs & Viscera (-140 to 260 HU)",
    },
    "Bone": {
        "center": 400.0,
        "width": 1800.0,
        "description": "Cranial Bone & Fractures (-500 to 1300 HU)",
    },
}


def _extract_tag_scalar(val: Any, default: float) -> float:
    """Helper to safely extract scalar float from DICOM tag which might be list or DS."""
    if val is None:
        return default
    try:
        if isinstance(val, (list, tuple, pydicom.multival.MultiValue)):
            return float(val[0])
        return float(val)
    except (ValueError, TypeError):
        return default


def get_dicom_metadata(dicom_dataset: pydicom.Dataset) -> Dict[str, Any]:
    """
    Extract key clinical, imaging, and calibration metadata from a DICOM dataset.
    """
    slope = _extract_tag_scalar(getattr(dicom_dataset, "RescaleSlope", None), 1.0)
    intercept = _extract_tag_scalar(getattr(dicom_dataset, "RescaleIntercept", None), 0.0)
    center = _extract_tag_scalar(getattr(dicom_dataset, "WindowCenter", None), 40.0)
    width = _extract_tag_scalar(getattr(dicom_dataset, "WindowWidth", None), 80.0)

    pixel_spacing = getattr(dicom_dataset, "PixelSpacing", [1.0, 1.0])
    if isinstance(pixel_spacing, (list, tuple, pydicom.multival.MultiValue)):
        spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    else:
        spacing = [1.0, 1.0]

    return {
        "patient_id": str(getattr(dicom_dataset, "PatientID", "De-identified")),
        "study_date": str(getattr(dicom_dataset, "StudyDate", "Unknown")),
        "modality": str(getattr(dicom_dataset, "Modality", "CT")),
        "series_description": str(getattr(dicom_dataset, "SeriesDescription", "Brain CT")),
        "rows": int(getattr(dicom_dataset, "Rows", 512)),
        "columns": int(getattr(dicom_dataset, "Columns", 512)),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "default_window_center": center,
        "default_window_width": width,
        "pixel_spacing": spacing,
        "slice_thickness": _extract_tag_scalar(getattr(dicom_dataset, "SliceThickness", None), 1.0),
        "photometric_interpretation": str(getattr(dicom_dataset, "PhotometricInterpretation", "MONOCHROME2")),
    }


def load_dicom(file_or_path: Union[str, Path, Any]) -> pydicom.Dataset:
    """
    Load a single DICOM file from disk path or file-like buffer (e.g. Streamlit BytesIO).

    Args:
        file_or_path: Path to a .dcm file or a readable file-like binary stream.

    Returns:
        pydicom.Dataset representing the loaded file.

    Raises:
        ValueError: If the file cannot be parsed or lacks pixel data.
        FileNotFoundError: If the file path does not exist.
    """
    try:
        ds = pydicom.dcmread(file_or_path, force=True)
    except Exception as exc:
        raise ValueError(f"Failed to read DICOM file: {exc}") from exc

    if not hasattr(ds, "pixel_array") and "PixelData" not in ds:
        raise ValueError("Provided DICOM file contains no pixel data.")

    return ds


def load_dicom_series(directory_path: Union[str, Path]) -> List[pydicom.Dataset]:
    """
    Load an ordered series of DICOM slices from a directory.
    Sorts slices by InstanceNumber or SliceLocation.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    slices = []
    for file_name in sorted(os.listdir(dir_path)):
        full_path = dir_path / file_name
        if full_path.is_file():
            try:
                ds = pydicom.dcmread(str(full_path), force=True)
                if hasattr(ds, "pixel_array") or "PixelData" in ds:
                    slices.append(ds)
            except Exception:
                continue

    if not slices:
        raise ValueError(f"No valid DICOM slice files found in: {directory_path}")

    return sort_dicom_slices(slices)


def sort_dicom_slices(slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    Sort a list of DICOM slice datasets anatomically along the axial Z-axis
    using ImagePositionPatient[2], SliceLocation, or InstanceNumber.
    """
    def sort_key(s: pydicom.Dataset):
        if hasattr(s, "ImagePositionPatient") and len(s.ImagePositionPatient) >= 3:
            try:
                return (0, float(s.ImagePositionPatient[2]))
            except Exception:
                pass
        if hasattr(s, "SliceLocation") and s.SliceLocation is not None:
            try:
                return (1, float(s.SliceLocation))
            except Exception:
                pass
        return (2, int(getattr(s, "InstanceNumber", 0)))

    return sorted(slices, key=sort_key)


def load_dicom_files_list(files_list: List[Any]) -> List[pydicom.Dataset]:
    """
    Load a list of DICOM file objects (such as Streamlit UploadedFile objects or file paths)
    and return an anatomically ordered list of pydicom.Dataset objects.
    """
    valid_slices = []
    for f in files_list:
        try:
            ds = load_dicom(f)
            valid_slices.append(ds)
        except Exception:
            continue

    if not valid_slices:
        raise ValueError("No valid DICOM datasets with pixel data could be read from the uploaded files.")

    return sort_dicom_slices(valid_slices)



def convert_to_hounsfield_units(
    dicom_dataset: pydicom.Dataset,
    pixel_array: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert raw stored CT pixel values to calibrated Hounsfield Units (HU).
    Formula: HU = pixel_value * RescaleSlope + RescaleIntercept.
    Handles MONOCHROME1 inversion if present.
    """
    if pixel_array is None:
        raw_pixels = dicom_dataset.pixel_array.astype(np.float32)
    else:
        raw_pixels = np.asarray(pixel_array, dtype=np.float32)

    # Invert MONOCHROME1 if applicable (where 0 is white, max is black)
    photometric = getattr(dicom_dataset, "PhotometricInterpretation", "MONOCHROME2")
    if str(photometric).strip() == "MONOCHROME1":
        raw_pixels = np.max(raw_pixels) - raw_pixels

    slope = _extract_tag_scalar(getattr(dicom_dataset, "RescaleSlope", None), 1.0)
    intercept = _extract_tag_scalar(getattr(dicom_dataset, "RescaleIntercept", None), 0.0)

    hu_array = (raw_pixels * slope) + intercept
    return hu_array.astype(np.float32)


def apply_window(
    hu_array: np.ndarray,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    as_uint8: bool = False
) -> np.ndarray:
    """
    Apply clinical windowing (level/width) to an HU array for display.
    Values outside [center - width/2, center + width/2] are clipped.

    Args:
        hu_array: 2D numpy array of Hounsfield Units.
        window_center: Window level (center). Default 40.0 (Brain).
        window_width: Window width. Default 80.0 (Brain).
        as_uint8: If True, returns uint8 array in [0, 255]; otherwise float32 in [0.0, 1.0].

    Returns:
        Windowed 2D numpy array.
    """
    center = 40.0 if window_center is None else float(window_center)
    width = 80.0 if window_width is None else max(1.0, float(window_width))

    lower = center - (width / 2.0)
    upper = center + (width / 2.0)

    clipped = np.clip(hu_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower)

    if as_uint8:
        return np.round(normalized * 255.0).astype(np.uint8)
    return normalized.astype(np.float32)
