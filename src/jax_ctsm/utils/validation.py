"""Validation utilities for physical processes."""

import jax.numpy as jnp
from typing import Dict, Tuple


def check_mass_balance(
    state_before: jnp.ndarray,
    state_after: jnp.ndarray,
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    dt: float,
    tolerance: float = 1e-10,
) -> Tuple[bool, float]:
    """Check mass balance for a state update.
    
    Verifies: state_after = state_before + (inputs - outputs) * dt
    
    Args:
        state_before: State before update [...]
        state_after: State after update [...]
        inputs: Input fluxes [.../s]
        outputs: Output fluxes [.../s]
        dt: Time step [s]
        tolerance: Acceptable error
        
    Returns:
        (is_balanced, max_error): Whether balanced and maximum error
    """
    expected = state_before + (inputs - outputs) * dt
    error = jnp.abs(state_after - expected)
    max_error = jnp.max(error)
    
    is_balanced = max_error < tolerance
    
    return is_balanced, float(max_error)


def validate_fluxes(fluxes: Dict[str, jnp.ndarray]) -> Dict[str, bool]:
    """Validate flux values for physical correctness.
    
    Checks:
    - No NaN values
    - No Inf values
    - Respiration fluxes are non-negative
    
    Args:
        fluxes: Dictionary of flux names to arrays
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    for name, flux in fluxes.items():
        # Check for NaN
        has_nan = jnp.any(jnp.isnan(flux))
        
        # Check for Inf
        has_inf = jnp.any(jnp.isinf(flux))
        
        # Check if respiration (should be non-negative)
        if "mr" in name.lower() or "respiration" in name.lower():
            is_negative = jnp.any(flux < 0)
            results[name] = not (has_nan or has_inf or is_negative)
        else:
            results[name] = not (has_nan or has_inf)
    
    return results


def summarize_fluxes(
    fluxes: Dict[str, jnp.ndarray],
    dt: float = 86400.0,
) -> str:
    """Create summary string of flux values.
    
    Args:
        fluxes: Dictionary of flux arrays
        dt: Time step for conversion [s] (default: 1 day)
        
    Returns:
        Formatted summary string
    """
    lines = ["Flux Summary:"]
    lines.append("=" * 50)
    
    for name, flux in fluxes.items():
        mean_val = float(jnp.mean(flux))
        min_val = float(jnp.min(flux))
        max_val = float(jnp.max(flux))
        
        # Convert to daily if per-second
        if "/s" in name:
            mean_val *= dt
            min_val *= dt
            max_val *= dt
            unit = name.split("/s")[0].split()[-1] + "/day"
        else:
            unit = name.split()[-1] if " " in name else ""
        
        lines.append(
            f"{name:30s}: mean={mean_val:10.6f}, "
            f"min={min_val:10.6f}, max={max_val:10.6f} {unit}"
        )
    
    return "\n".join(lines)
