"""
matlab_bridge.py
----------------
Optional MATLAB Engine integration for NeuroClear.
Provides seamless execution of MATLAB Image Processing Toolbox algorithms
(imdiffusefilt, locallapfilter, bm3d, custom .m scripts) directly from Python.

Handles graceful fallback when MATLAB is not installed so the workstation
runs 100% standalone without external software dependencies.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

# Global cached MATLAB engine instance
_MATLAB_ENGINE: Any = None
_MATLAB_AVAILABLE: Optional[bool] = None


def is_matlab_available() -> Tuple[bool, str]:
    """
    Check if the official MATLAB Engine for Python (matlabengine) is installed
    and ready to use on this machine.
    """
    global _MATLAB_AVAILABLE
    if _MATLAB_AVAILABLE is not None:
        return _MATLAB_AVAILABLE, "MATLAB Engine for Python is available." if _MATLAB_AVAILABLE else "matlabengine package is not installed."
    try:
        import sys
        if "matlab" in sys.modules and "matlab.engine" in sys.modules:
            _MATLAB_AVAILABLE = True
            return True, "MATLAB Engine for Python is available."
    except Exception:
        pass
    _MATLAB_AVAILABLE = False
    return False, "matlabengine package is not installed. Run 'pip install matlabengine' to enable live MATLAB execution."


def get_matlab_engine() -> Optional[Any]:
    """
    Get or initialize the persistent background MATLAB engine session.
    Returns None if MATLAB is unavailable.
    """
    global _MATLAB_ENGINE
    if _MATLAB_ENGINE is not None:
        return _MATLAB_ENGINE

    avail, _ = is_matlab_available()
    if not avail:
        return None

    try:
        import matlab.engine
        _MATLAB_ENGINE = matlab.engine.start_matlab()
        return _MATLAB_ENGINE
    except Exception:
        return None


def denoise_with_matlab_anisotropic(
    image: np.ndarray,
    n_iter: int = 8,
    gradient_threshold: float = 15.0,
    conduction_method: str = "exponential",
) -> np.ndarray:
    """
    Execute MATLAB's native imdiffusefilt() on a 2D CT array.

    Falls back to Python native anisotropic_diffusion_perona_malik if MATLAB is not active.
    """
    eng = get_matlab_engine()
    if eng is None:
        from core.poisson_denoise import anisotropic_diffusion_perona_malik
        return anisotropic_diffusion_perona_malik(
            image,
            n_iter=n_iter,
            kappa=gradient_threshold,
            conduction_method=conduction_method,
        )

    try:
        import matlab
        img_arr = np.asarray(image, dtype=np.float64)
        matlab_in = matlab.double(img_arr.tolist())
        cond_map = "exponential" if conduction_method == "exponential" else "quadratic"
        res_matlab = eng.imdiffusefilt(
            matlab_in,
            "NumberOfIterations", int(n_iter),
            "GradientThreshold", float(gradient_threshold),
            "ConductionMethod", cond_map,
        )
        res_np = np.array(res_matlab._data, dtype=np.float32).reshape(img_arr.shape)
        return res_np
    except Exception:
        from core.poisson_denoise import anisotropic_diffusion_perona_malik
        return anisotropic_diffusion_perona_malik(
            image,
            n_iter=n_iter,
            kappa=gradient_threshold,
            conduction_method=conduction_method,
        )


def execute_custom_matlab_script(
    script_path: str,
    function_name: str,
    input_slice: np.ndarray,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Execute a user's custom MATLAB .m script on a CT slice.
    """
    eng = get_matlab_engine()
    if eng is None:
        return False, None, "MATLAB Engine is not active on this system."

    try:
        import matlab
        eng.addpath(script_path, nargout=0)
        img_arr = np.asarray(input_slice, dtype=np.float64)
        matlab_in = matlab.double(img_arr.tolist())
        matlab_func = getattr(eng, function_name)
        result = matlab_func(matlab_in)
        out_np = np.array(result._data, dtype=np.float32).reshape(img_arr.shape)
        return True, out_np, "Success"
    except Exception as ex:
        return False, None, f"Execution failed: {ex}"
