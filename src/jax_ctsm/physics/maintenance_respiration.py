"""
Maintenance Respiration Calculations.

Translated from CTSM's CNMRespMod.F90

Maintenance respiration (MR) represents the carbon cost of maintaining existing
plant tissues. It scales with tissue nitrogen content and temperature.

Key equations:
    MR = N_tissue * base_rate * Q10^((T-Tref)/10)
    
Where:
    - N_tissue: Nitrogen content of tissue [gN/m2]
    - base_rate: Base respiration rate [gC/gN/s]
    - Q10: Temperature sensitivity (typically 1.5)
    - T: Tissue temperature [°C]
    - Tref: Reference temperature (20°C)
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax_ctsm.core.hierarchy import (
    PatchState,
    CarbonFlux,
    TemperatureState,
    column_to_patch,
)
from jax_ctsm.params.respiration import RespirationParams


def leaf_maintenance_respiration(
    lai_sun: jnp.ndarray,
    lai_shade: jnp.ndarray,
    lmr_sun: jnp.ndarray,
    lmr_shade: jnp.ndarray,
    frac_veg_nosno: jnp.ndarray,
    params: RespirationParams,
) -> jnp.ndarray:
    """Calculate leaf maintenance respiration.
    
    Leaf MR comes from the photosynthesis model and is already calculated
    as a function of leaf nitrogen and temperature. We just convert units
    and account for snow cover.
    
    Args:
        lai_sun: Sunlit leaf area index [m2/m2] [n_patches]
        lai_shade: Shaded leaf area index [m2/m2] [n_patches]
        lmr_sun: Sunlit leaf MR [umol CO2/m2 leaf/s] [n_patches]
        lmr_shade: Shaded leaf MR [umol CO2/m2 leaf/s] [n_patches]
        frac_veg_nosno: Fraction without snow cover [0 or 1] [n_patches]
        params: Respiration parameters
        
    Returns:
        Leaf MR flux [gC/m2 ground/s] [n_patches]
        
    Note:
        When snow covered (frac_veg_nosno=0), leaf MR is set to zero.
    """
    # Convert from umol CO2/m2 leaf/s to gC/m2 ground/s
    # Multiply by LAI to get ground area basis
    leaf_mr_sun = lmr_sun * lai_sun * params.umol_to_gC
    leaf_mr_shade = lmr_shade * lai_shade * params.umol_to_gC
    
    # Total leaf MR
    leaf_mr = leaf_mr_sun + leaf_mr_shade
    
    # Zero out if snow covered
    leaf_mr = jnp.where(frac_veg_nosno > 0, leaf_mr, 0.0)
    
    return leaf_mr


def stem_maintenance_respiration(
    stem_n: jnp.ndarray,
    temperature: jnp.ndarray,
    t_10day: jnp.ndarray,
    is_woody: jnp.ndarray,
    params: RespirationParams,
) -> jnp.ndarray:
    """Calculate live stem maintenance respiration.
    
    Args:
        stem_n: Live stem nitrogen [gN/m2] [n_patches]
        temperature: 2m air temperature [K] [n_patches]
        t_10day: 10-day running mean temperature [K] [n_patches]
        is_woody: Whether PFT is woody [boolean] [n_patches]
        params: Respiration parameters
        
    Returns:
        Stem MR flux [gC/m2/s] [n_patches]
    """
    # Get temperature correction
    temp_corr = params.get_temp_correction(temperature)
    
    # Get acclimation factor if enabled
    acclim_factor = params.get_acclimation_factor(t_10day)
    
    # Base rate with acclimation
    br_adjusted = params.br * acclim_factor
    
    # Calculate MR
    stem_mr = stem_n * br_adjusted * temp_corr
    
    # Only apply to woody plants
    stem_mr = jnp.where(is_woody, stem_mr, 0.0)
    
    return stem_mr


def coarse_root_maintenance_respiration(
    croot_n: jnp.ndarray,
    temperature: jnp.ndarray,
    t_10day: jnp.ndarray,
    is_woody: jnp.ndarray,
    params: RespirationParams,
) -> jnp.ndarray:
    """Calculate live coarse root maintenance respiration.
    
    Args:
        croot_n: Live coarse root nitrogen [gN/m2] [n_patches]
        temperature: 2m air temperature [K] [n_patches]
        t_10day: 10-day running mean temperature [K] [n_patches]
        is_woody: Whether PFT is woody [boolean] [n_patches]
        params: Respiration parameters
        
    Returns:
        Coarse root MR flux [gC/m2/s] [n_patches]
    """
    # Get temperature correction
    temp_corr = params.get_temp_correction(temperature)
    
    # Get acclimation factor if enabled
    acclim_factor = params.get_acclimation_factor(t_10day)
    
    # Base rate with acclimation (use br_root)
    br_adjusted = params.br_root * acclim_factor
    
    # Calculate MR
    croot_mr = croot_n * br_adjusted * temp_corr
    
    # Only apply to woody plants
    croot_mr = jnp.where(is_woody, croot_mr, 0.0)
    
    return croot_mr


def reproductive_maintenance_respiration(
    repro_n: jnp.ndarray,
    temperature: jnp.ndarray,
    t_10day: jnp.ndarray,
    is_crop: jnp.ndarray,
    params: RespirationParams,
) -> jnp.ndarray:
    """Calculate reproductive tissue (grain) maintenance respiration.
    
    Args:
        repro_n: Reproductive nitrogen [gN/m2] [n_patches, n_repr]
        temperature: 2m air temperature [K] [n_patches]
        t_10day: 10-day running mean temperature [K] [n_patches]
        is_crop: Whether PFT is crop [boolean] [n_patches]
        params: Respiration parameters
        
    Returns:
        Reproductive MR flux [gC/m2/s] [n_patches, n_repr]
    """
    # Get temperature correction [n_patches]
    temp_corr = params.get_temp_correction(temperature)
    
    # Get acclimation factor if enabled [n_patches]
    acclim_factor = params.get_acclimation_factor(t_10day)
    
    # Base rate with acclimation [n_patches]
    br_adjusted = params.br * acclim_factor
    
    # Broadcast to match repro_n shape [n_patches, 1]
    temp_corr = jnp.expand_dims(temp_corr, axis=-1)
    br_adjusted = jnp.expand_dims(br_adjusted, axis=-1)
    is_crop_expanded = jnp.expand_dims(is_crop, axis=-1)
    
    # Calculate MR [n_patches, n_repr]
    repro_mr = repro_n * br_adjusted * temp_corr
    
    # Only apply to crops
    repro_mr = jnp.where(is_crop_expanded, repro_mr, 0.0)
    
    return repro_mr


def root_maintenance_respiration(
    root_n: jnp.ndarray,
    root_frac: jnp.ndarray,
    soil_temp: jnp.ndarray,
    column_indices: jnp.ndarray,
    t_10day: jnp.ndarray,
    params: RespirationParams,
) -> jnp.ndarray:
    """Calculate fine root maintenance respiration with depth distribution.
    
    Fine roots are distributed vertically in soil, so we weight by root
    fraction in each layer and use layer-specific soil temperature.
    
    Args:
        root_n: Fine root nitrogen [gN/m2] [n_patches]
        root_frac: Fraction of roots in each layer [n_patches, n_levgrnd]
        soil_temp: Soil temperature by layer [K] [n_columns, n_levgrnd]
        column_indices: Parent column for each patch [n_patches]
        t_10day: 10-day running mean temperature [K] [n_patches]
        params: Respiration parameters
        
    Returns:
        Fine root MR flux [gC/m2/s] [n_patches]
        
    Note:
        root_frac should sum to 1.0 over all layers for each patch.
    """
    # Get acclimation factor if enabled [n_patches]
    acclim_factor = params.get_acclimation_factor(t_10day)
    
    # Base rate with acclimation [n_patches]
    br_adjusted = params.br_root * acclim_factor
    
    # Get soil temperature for each patch [n_patches, n_levgrnd]
    patch_soil_temp = soil_temp[column_indices, :]
    
    # Temperature correction for each layer [n_patches, n_levgrnd]
    # Using params.q10 and reference temp 20°C
    temp_c = patch_soil_temp - 273.15
    temp_corr_layers = params.q10 ** ((temp_c - params.reference_temp) / 10.0)
    
    # Calculate MR for each layer weighted by root fraction
    # [n_patches, n_levgrnd]
    mr_by_layer = (
        jnp.expand_dims(root_n * br_adjusted, axis=-1)
        * temp_corr_layers
        * root_frac
    )
    
    # Sum over all layers [n_patches]
    root_mr = jnp.sum(mr_by_layer, axis=-1)
    
    return root_mr


def calculate_maintenance_respiration(
    patch_state: PatchState,
    params: RespirationParams,
) -> CarbonFlux:
    """Calculate all maintenance respiration fluxes for patches.
    
    This is the main entry point for maintenance respiration calculations.
    It orchestrates all tissue-specific calculations.
    
    Args:
        patch_state: Complete patch state
        params: Respiration parameters
        
    Returns:
        Carbon fluxes with MR components filled
        
    Note:
        This is a pure function - it does not modify patch_state.
        All calculations are vectorized over patches using JAX arrays.
    """
    # Extract state components
    canopy = patch_state.canopy
    nitrogen = patch_state.nitrogen
    temp = patch_state.temperature
    soil = patch_state.soil
    pft_params = patch_state.pft_params
    spatial = patch_state.spatial
    
    # Get PFT type flags
    is_woody = pft_params.get("is_woody", jnp.zeros_like(spatial.pft_type, dtype=bool))
    is_crop = pft_params.get("is_crop", jnp.zeros_like(spatial.pft_type, dtype=bool))
    
    # Calculate leaf MR
    leaf_mr = leaf_maintenance_respiration(
        canopy.lai_sun,
        canopy.lai_shade,
        canopy.lmr_sun,
        canopy.lmr_shade,
        canopy.frac_veg_nosno,
        params,
    )
    
    # Calculate fine root MR (with vertical distribution)
    froot_mr = root_maintenance_respiration(
        nitrogen.frootn,
        soil.crootfr,
        temp.t_soisno,
        spatial.column_index,
        temp.t_10day,
        params,
    )
    
    # Calculate live stem MR (woody plants only)
    livestem_mr = stem_maintenance_respiration(
        nitrogen.livestemn,
        temp.t_ref2m,
        temp.t_10day,
        is_woody,
        params,
    )
    
    # Calculate live coarse root MR (woody plants only)
    livecroot_mr = coarse_root_maintenance_respiration(
        nitrogen.livecrootn,
        temp.t_ref2m,
        temp.t_10day,
        is_woody,
        params,
    )
    
    # Calculate reproductive tissue MR (crops only)
    reproductive_mr = reproductive_maintenance_respiration(
        nitrogen.reproductiven,
        temp.t_ref2m,
        temp.t_10day,
        is_crop,
        params,
    )
    
    # Create and return flux structure
    # Note: GPP is not calculated here (comes from photosynthesis)
    n_patches = leaf_mr.shape[0]
    
    return CarbonFlux(
        gpp=jnp.zeros(n_patches),  # Filled by photosynthesis module
        leaf_mr=leaf_mr,
        froot_mr=froot_mr,
        livestem_mr=livestem_mr,
        livecroot_mr=livecroot_mr,
        reproductive_mr=reproductive_mr,
    )


# Vectorized version for batched processing
@jax.jit
def calculate_maintenance_respiration_batch(
    patch_states: PatchState,
    params: RespirationParams,
) -> CarbonFlux:
    """JIT-compiled batch version of maintenance respiration.
    
    Args:
        patch_states: Batch of patch states (all arrays have leading batch dim)
        params: Respiration parameters (shared across batch)
        
    Returns:
        Batch of carbon fluxes
    """
    return calculate_maintenance_respiration(patch_states, params)
