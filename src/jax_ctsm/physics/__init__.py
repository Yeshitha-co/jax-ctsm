"""Physical processes for JAX-CTSM."""

from jax_ctsm.physics.maintenance_respiration import (
    calculate_maintenance_respiration,
    leaf_maintenance_respiration,
    root_maintenance_respiration,
    stem_maintenance_respiration,
)

__all__ = [
    "calculate_maintenance_respiration",
    "leaf_maintenance_respiration", 
    "root_maintenance_respiration",
    "stem_maintenance_respiration",
]
