"""JAX-CTSM: Community Terrestrial Systems Model in JAX.

A differentiable, GPU-accelerated reimplementation of CTSM ecosystem dynamics.
"""

__version__ = "0.1.0"

from jax_ctsm.core import (
    PatchState,
    ColumnState,
    NitrogenState,
    CarbonFlux,
    SpatialInfo,
)
from jax_ctsm.physics import calculate_maintenance_respiration
from jax_ctsm.params import RespirationParams

__all__ = [
    "PatchState",
    "ColumnState", 
    "NitrogenState",
    "CarbonFlux",
    "SpatialInfo",
    "calculate_maintenance_respiration",
    "RespirationParams",
]
