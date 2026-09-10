"""
output_validation.py
--------------------
Post-processing safety validation module informed by IEC 62304 and ISO 14971 principles.
Performs quantitative data integrity, dimensional consistency, and structural preservation checks
before displaying or exporting processed CT slices.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.quality_metrics import calculate_edge_preservation


def validate_pipeline_output(
    input_hu: np.ndarray,
    output_hu: np.ndarray,
    min_edge_preservation: float = 0.50,
    max_mean_shift_hu: float = 50.0,
    expected_hu_range: Tuple[float, float] = (-1500.0, 4000.0),
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Validate the numerical and structural integrity of a processed CT slice.
    """
    if "original_hu" in kwargs and input_hu is None:
        input_hu = kwargs["original_hu"]
    if "candidate_hu" in kwargs and output_hu is None:
        output_hu = kwargs["candidate_hu"]

    checks: List[Dict[str, Any]] = []
    messages: List[str] = []
    is_valid = True
    fallback_applied = False

    # 1. Existence Check
    if output_hu is None or not isinstance(output_hu, np.ndarray):
        checks.append({"name": "Existence Check", "passed": False, "details": "Output array is null or invalid type"})
        messages.append("ERROR: Output array is null or invalid type.")
        return {
            "is_valid": False,
            "status": "FAILED",
            "checks": checks,
            "messages": messages,
            "edge_preservation": 0.0,
            "safe_output_hu": input_hu.copy() if input_hu is not None else np.zeros((128, 128), dtype=np.float32),
            "fallback_applied": True,
        }
    checks.append({"name": "Existence & Non-Empty Check", "passed": True, "details": f"Array populated with shape {output_hu.shape}"})

    # 2. Dimensional Consistency
    if input_hu is not None and output_hu.shape != input_hu.shape:
        checks.append({"name": "Dimensional Consistency", "passed": False, "details": f"Shape mismatch: {output_hu.shape} vs input {input_hu.shape}"})
        messages.append(f"ERROR: Shape mismatch. Output {output_hu.shape} != Input {input_hu.shape}.")
        is_valid = False
    else:
        checks.append({"name": "Dimensional Consistency", "passed": True, "details": f"Exact matrix dimension match ({output_hu.shape})"})

    # 3. Numerical Stability (NaN / Inf)
    has_nan = bool(np.isnan(output_hu).any())
    has_inf = bool(np.isinf(output_hu).any())
    if has_nan or has_inf:
        checks.append({"name": "Numerical Stability (NaN / Inf)", "passed": False, "details": f"NaN detected: {has_nan}, Inf detected: {has_inf}"})
        messages.append(f"ERROR: Numerical instability detected (NaN: {has_nan}, Inf: {has_inf}).")
        is_valid = False
    else:
        checks.append({"name": "Numerical Stability (NaN / Inf)", "passed": True, "details": "Zero NaN or Inf floating point elements detected"})

    # 4. Physiological HU Bounds
    min_hu, max_hu = float(np.min(output_hu)), float(np.max(output_hu))
    exp_min, exp_max = expected_hu_range
    if min_hu < exp_min or max_hu > exp_max:
        checks.append({"name": "Physiological HU Range", "passed": False, "details": f"Range [{min_hu:.1f}, {max_hu:.1f}] outside physiological bounds [{exp_min}, {exp_max}]"})
        messages.append(f"WARNING: HU values [{min_hu:.1f}, {max_hu:.1f}] exceed normal range [{exp_min}, {exp_max}].")
    else:
        checks.append({"name": "Physiological HU Range", "passed": True, "details": f"Radiodensity [{min_hu:.1f}, {max_hu:.1f}] HU within physiological bounds"})

    # 5. Mean Attenuation Shift (Baseline Drift)
    mean_shift = 0.0
    if input_hu is not None and input_hu.shape == output_hu.shape and not has_nan and not has_inf:
        mean_shift = float(np.abs(np.mean(output_hu) - np.mean(input_hu)))
        if mean_shift > max_mean_shift_hu:
            checks.append({"name": "Mean Radiodensity Stability", "passed": False, "details": f"Global mean shift {mean_shift:.2f} HU exceeds max allowable {max_mean_shift_hu:.1f} HU"})
            messages.append(f"WARNING: Global mean shift {mean_shift:.2f} HU is excessive.")
        else:
            checks.append({"name": "Mean Radiodensity Stability", "passed": True, "details": f"Global baseline drift {mean_shift:.3f} HU within safe limits (< {max_mean_shift_hu:.1f} HU)"})

    # 6. Edge Preservation Verification (EPI)
    epi = 1.0
    if input_hu is not None and input_hu.shape == output_hu.shape and not has_nan and not has_inf:
        try:
            epi = float(calculate_edge_preservation(input_hu, output_hu))
        except Exception:
            epi = 0.0

        if epi < min_edge_preservation:
            checks.append({"name": "Edge Preservation Index (EPI)", "passed": False, "details": f"EPI {epi:.3f} below safety threshold {min_edge_preservation:.2f}"})
            messages.append(f"ERROR: Edge Preservation Index ({epi:.3f}) below safety threshold ({min_edge_preservation:.2f}).")
            is_valid = False
        else:
            checks.append({"name": "Edge Preservation Index (EPI)", "passed": True, "details": f"EPI {epi:.3f} >= {min_edge_preservation:.2f} (anatomical borders preserved)"})

    # Safe Fallback Resolution
    if not is_valid:
        status = "FAILED"
        safe_output = input_hu.copy() if input_hu is not None else output_hu
        fallback_applied = True
        messages.append("CRITICAL: Output validation gate failed. Original input slice safely retained.")
    else:
        status = "PASSED"
        safe_output = output_hu
        fallback_applied = False
        messages.append("All output integrity and structural verification checks passed successfully.")

    return {
        "is_valid": is_valid,
        "status": status,
        "checks": checks,
        "messages": messages,
        "edge_preservation": float(epi),
        "mean_shift_hu": float(mean_shift),
        "safe_output_hu": safe_output,
        "fallback_applied": fallback_applied,
    }


def generate_algorithm_decision_trace(
    noise_analysis: Dict[str, Any],
    options_applied: Dict[str, Any],
    validation_result: Dict[str, Any],
    execution_time_seconds: float,
) -> Dict[str, Any]:
    """
    Generate an auditable, transparent record of all algorithmic decisions made during processing.
    """
    periodic_info = noise_analysis.get("periodic", {})
    poisson_info = noise_analysis.get("poisson", {})

    # Periodic Decision Trail
    peaks = periodic_info.get("peaks", [])
    peak_count = len(peaks)
    periodic_detected = periodic_info.get("detected", False)
    
    # Calculate confidence based on maximum peak prominence
    if peaks:
        max_prom = max([p.get("prominence", 0.0) for p in peaks])
        periodic_confidence = min(99.0, max(50.0, 50.0 + max_prom * 10.0))
    else:
        periodic_confidence = 10.0

    if not options_applied.get("skip_periodic", False) and periodic_detected:
        periodic_action = f"Adaptive Notch Filter Applied ({options_applied.get('notch_filter_type', 'gaussian').capitalize()}, Radius={options_applied.get('notch_radius', 5.0)})"
    elif options_applied.get("skip_periodic", False):
        periodic_action = "Periodic Filter Bypassed (Disabled by User Configuration)"
    else:
        periodic_action = "Periodic Filter Skipped (No Significant Harmonic Spikes Detected)"

    # Poisson Decision Trail
    poisson_level = poisson_info.get("noise_level", "Moderate")
    sigma_est = poisson_info.get("estimated_sigma", 0.0)
    
    if not options_applied.get("skip_poisson", False):
        method_name = str(options_applied.get("poisson_method", "bilateral")).upper()
        strength = options_applied.get("poisson_strength", 1.0)
        anscombe = " + Anscombe Stabilization" if options_applied.get("use_anscombe", False) else ""
        poisson_action = f"{method_name} Filter Applied (Strength={strength:.2f}{anscombe})"
    else:
        poisson_action = "Poisson Denoising Skipped (Disabled by User Configuration)"

    return {
        "version": "v0.1.0",
        "timestamp": np.datetime64("now").astype(str),
        "execution_time_ms": round(execution_time_seconds * 1000.0, 2),
        "status": validation_result.get("status", "PASSED"),
        "validation_passed": validation_result.get("is_valid", True),
        "fallback_applied": validation_result.get("fallback_applied", False),
        "method_selected": options_applied.get("poisson_method", "bilateral"),
        "periodic_filter_applied": not options_applied.get("skip_periodic", False) and periodic_detected,
        "poisson_filter_applied": not options_applied.get("skip_poisson", False),
        "edge_preservation_index": round(float(validation_result.get("edge_preservation", 1.0)), 4),
        "mean_shift_hu": round(float(validation_result.get("mean_shift_hu", 0.0)), 4),
        "validation_notes": validation_result.get("messages", []),
        "periodic_decision": {
            "status": "DETECTED" if periodic_detected else "NOT SIGNIFICANT",
            "peak_count": peak_count,
            "confidence_percent": round(periodic_confidence, 1),
            "action": periodic_action,
        },
        "poisson_decision": {
            "classified_noise_level": str(poisson_level).upper(),
            "estimated_sigma_hu": round(float(sigma_est), 2),
            "action": poisson_action,
        },
        "validation_decision": {
            "status": validation_result.get("status", "PASSED"),
            "edge_preservation_score": round(validation_result.get("edge_preservation", 1.0) * 100.0, 1),
            "fallback_applied": validation_result.get("fallback_applied", False),
            "messages": validation_result.get("messages", []),
        },
    }
