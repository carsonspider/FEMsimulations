"""
Blended TPMS Generator: Skeletal-to-Sheet Gradient Structures

This module implements functionally graded TPMS structures that smoothly transition
from dense skeletal (solid) structures at the bottom to porous sheet (thin-walled)
structures at the top, following the methodology described in:

"Revealing the apparent and local mechanical properties of heterogeneous lattice:
a multi-scale study of functionally graded scaffold"

Key Concepts:
- Skeletal structure: Fills volume where G > -t_skeletal (dense, thick struts)
- Sheet structure: Keeps material where |G| < t_sheet (thin walls, porous)
- Sigmoid gradient: Smoothly transitions between skeletal (bottom) and sheet (top)
- Graded threshold: t_graded = t_skeletal * (1 - gradient) + t_sheet * gradient

This creates scaffolds with continuous density variation from bottom to top,
avoiding abrupt transitions that create mechanical weak points.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage import measure
from stl import mesh

# Import base functions from active_gyroid_gen
from active_gyroid_gen import (
    GyroidParameters,
    validate_params,
    compute_domain_lengths,
    generate_coordinate_grid,
    tpms_field,
    add_bounding_box,
    marching_cubes_mesh,
    export_stl,
    visualise
)


@dataclass
class BlendedTPMSParameters:
    """
    Parameters for blended skeletal-to-sheet TPMS structures.
    
    Extends GyroidParameters with skeletal/sheet blending parameters.
    
    Attributes
    ----------
    base_params : GyroidParameters
        Base TPMS generation parameters (geometry, porosity, etc.)
    t_skeletal : float
        Threshold for skeletal region (NEGATIVE value, larger magnitude = denser).
        Typical range: -0.7 to -0.3. Default: -0.5
        Rule: Material if G > t_skeletal (fills enclosed volume)
        Example: t_skeletal = -0.5 means G > -0.5 (keeps positive G values)
    t_sheet : float
        Threshold for sheet region (POSITIVE value, smaller = thinner walls).
        Typical range: 0.05-0.2. Default: 0.1
        Rule: Material if |G| < t_sheet (thin surface wrapping)
    steepness : float
        Sigmoid gradient transition sharpness (higher = more abrupt change).
        Typical range: 3-10. Default: 5
        Controls how quickly the transition occurs from skeletal to sheet
    use_blending : bool
        If True, use skeletal/sheet blending. If False, use standard porosity-based method.
        Default: True
    """
    base_params: GyroidParameters
    t_skeletal: float = -0.5  # Threshold for skeletal (dense) region (NEGATIVE)
    t_sheet: float = 0.1      # Threshold for sheet (porous) region (POSITIVE)
    steepness: float = 5.0    # Sigmoid gradient steepness
    use_blending: bool = True # Enable skeletal/sheet blending


def sigmoid_gradient(z: np.ndarray, z_min: float, z_max: float, steepness: float = 5.0) -> np.ndarray:
    """
    Compute sigmoid gradient function for smooth transition from 0 to 1.
    
    The sigmoid function creates a smooth S-shaped transition:
    - At z_min: gradient ≈ 0 (skeletal structure)
    - At z_max: gradient ≈ 1 (sheet structure)
    - Transition occurs around the midpoint
    
    Parameters
    ----------
    z : np.ndarray
        3D array of z-coordinates (or 1D array for single axis)
    z_min : float
        Minimum z value (bottom of structure)
    z_max : float
        Maximum z value (top of structure)
    steepness : float
        Controls transition sharpness (higher = more abrupt)
        Default: 5.0
    
    Returns
    -------
    np.ndarray
        Gradient values in range [0, 1], same shape as z
    """
    # Normalize z to [0, 1] range
    z_normalized = (z - z_min) / (z_max - z_min + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Sigmoid function: 1 / (1 + exp(-steepness * (z_norm - 0.5)))
    # This creates S-curve from 0 to 1, centered at z_norm = 0.5
    gradient = 1.0 / (1.0 + np.exp(-steepness * (z_normalized - 0.5)))
    
    return gradient


def generate_blended_volume(params: BlendedTPMSParameters) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, np.ndarray]]:
    """
    Generate a binary TPMS volume using skeletal-to-sheet blending.
    
    This function creates a functionally graded scaffold that transitions
    smoothly from dense skeletal structure (bottom) to porous sheet structure (top).
    
    Process:
    1. Generate TPMS field G(x,y,z) using base parameters
    2. Compute sigmoid gradient based on z-position
    3. Interpolate threshold: t_graded = t_skeletal * (1 - gradient) + t_sheet * gradient
    4. Apply blending rule:
       - Skeletal rule (bottom): Material if G > -t_skeletal
       - Sheet rule (top): Material if |G| < t_sheet
       - Blended: Material if G > -t_graded OR |G| < t_graded (depending on region)
    
    Parameters
    ----------
    params : BlendedTPMSParameters
        Parameters including base TPMS params and blending parameters
    
    Returns
    -------
    volume : np.ndarray
        3D boolean array (True = material, False = void)
    spacing : Tuple[float, float, float]
        Voxel spacing in each direction (mm)
    metadata : Dict[str, np.ndarray]
        Metadata including porosity profile, gradient values, etc.
    """
    # Validate base parameters
    base_params = validate_params(params.base_params)
    
    # Generate coordinate grid
    grid = generate_coordinate_grid(base_params)
    X, Y, Z = grid
    
    # Compute TPMS field G(x,y,z) - normalized to [-1, 1]
    G = tpms_field(base_params, grid)
    
    # Get domain dimensions
    lx, ly, lz = compute_domain_lengths(base_params)
    nx, ny, nz = G.shape
    
    # Compute sigmoid gradient based on z-position
    z_min = np.min(Z)
    z_max = np.max(Z)
    gradient = sigmoid_gradient(Z, z_min, z_max, params.steepness)
    
    # Interpolate threshold: transitions from t_skeletal (bottom) to t_sheet (top)
    # gradient = 0 (bottom) -> t_graded = t_skeletal
    # gradient = 1 (top) -> t_graded = t_sheet
    t_graded = params.t_skeletal * (1.0 - gradient) + params.t_sheet * gradient
    
    # Apply blending rule as specified:
    # - Skeletal rule: Material if G > -t_skeletal (fills enclosed volume)
    #   Example: t_skeletal = -0.5, so G > 0.5 (keeps positive side)
    # - Sheet rule: Material if |G| < t_sheet (thin surface)
    #   Example: t_sheet = 0.1, so |G| < 0.1 (keeps near surface)
    
    # Compute both rules
    skeletal_rule = G > -params.t_skeletal  # G > 0.5 if t_skeletal = -0.5
    sheet_rule = np.abs(G) < params.t_sheet  # |G| < 0.1 if t_sheet = 0.1
    
    # Use gradient to decide which rule to apply at each point:
    # - If gradient < 0.5 (lower region): Use skeletal rule (dense bottom)
    # - If gradient >= 0.5 (upper region): Use sheet rule (porous top)
    volume = np.zeros((nx, ny, nz), dtype=bool)
    
    lower_region = gradient < 0.5
    upper_region = gradient >= 0.5
    
    volume[lower_region] = skeletal_rule[lower_region]  # Skeletal at bottom
    volume[upper_region] = sheet_rule[upper_region]      # Sheet at top
    
    # Apply Gaussian smoothing if requested
    if base_params.smoothness > 0:
        volume = ndimage.binary_opening(volume)  # Remove small artifacts
        volume = ndimage.binary_closing(volume)  # Fill small holes
    
    # Add bounding box walls
    spacing = (lx / nx, ly / ny, lz / nz)
    volume = add_bounding_box(volume, spacing, base_params.wall_thickness)
    
    # Compute porosity profile along z-direction
    porosity_profile = np.zeros(nz)
    for k in range(nz):
        layer = volume[:, :, k]
        porosity_profile[k] = 1.0 - np.mean(layer)
    
    # Metadata
    metadata: Dict[str, np.ndarray] = {
        "porosity_targets": porosity_profile,  # Achieved porosity at each layer
        "porosity_achieved": porosity_profile,
        "gradient_values": gradient[:, 0, :].mean(axis=0),  # Average gradient per z-layer
        "threshold_graded": t_graded[:, 0, :].mean(axis=0),  # Average threshold per z-layer
        "average_porosity": np.mean(porosity_profile),
        "volume_fraction": np.mean(volume),
        "lengths": np.array([lx, ly, lz]),
        "spacing": np.array(spacing),
        "t_skeletal": params.t_skeletal,
        "t_sheet": params.t_sheet,
        "steepness": params.steepness,
    }
    
    return volume, spacing, metadata


def create_blended_tpms(params: BlendedTPMSParameters, output_dir: Path, show_plot: bool = True) -> Path:
    """
    Generate, export, and optionally visualize a blended skeletal-to-sheet TPMS structure.
    
    Parameters
    ----------
    params : BlendedTPMSParameters
        Parameters for blended TPMS generation
    output_dir : Path
        Directory to save STL file
    show_plot : bool
        Whether to display visualization (default: True)
    
    Returns
    -------
    Path
        Path to generated STL file
    """
    # Generate volume
    volume, spacing, metadata = generate_blended_volume(params)
    
    # Validate volume has material
    solid_voxels = np.sum(volume)
    total_voxels = volume.size
    solid_fraction = solid_voxels / total_voxels if total_voxels > 0 else 0.0
    
    if solid_voxels == 0:
        raise ValueError(
            f"Generated volume is empty! Check parameters:\n"
            f"  - t_skeletal={params.t_skeletal}, t_sheet={params.t_sheet}\n"
            f"  - steepness={params.steepness}\n"
            f"  - nsteps={params.base_params.nsteps}"
        )
    
    if solid_fraction < 0.01:
        raise ValueError(
            f"Volume is too sparse ({solid_fraction*100:.1f}% solid)! "
            f"Try adjusting t_skeletal (current: {params.t_skeletal}) "
            f"or t_sheet (current: {params.t_sheet})"
        )
    
    # Extract mesh using marching cubes
    verts, faces, *_ = marching_cubes_mesh(volume, spacing, params.base_params)
    
    # Validate mesh
    if len(faces) == 0:
        raise ValueError(
            f"Marching cubes produced no faces! Volume might be too sparse.\n"
            f"  - Solid fraction: {solid_fraction*100:.1f}%\n"
            f"  - Try adjusting t_skeletal or t_sheet"
        )
    
    # Export STL
    tpms_type = params.base_params.tpms_type.lower() if hasattr(params.base_params, 'tpms_type') else 'gyroid'
    stl_path = export_stl(verts, faces, output_dir, tpms_type=f"{tpms_type}_blended")
    
    # Validate STL file
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file was not created at {stl_path}")
    
    file_size = stl_path.stat().st_size
    if file_size < 5000:
        raise ValueError(
            f"STL file is too small ({file_size} bytes)! "
            f"Mesh generation likely failed."
        )
    
    # Visualize if requested
    if show_plot:
        # Use base visualization function
        visualise(params.base_params, volume, verts, faces, spacing, metadata)
    
    return stl_path


def main():
    """Example usage of blended TPMS generator."""
    from active_gyroid_gen import DEFAULT_PARAMS
    
    # Create base parameters
    base_params = GyroidParameters(
        numx=3,
        numy=3,
        numz=3,
        unit_cell_size=3.0,
        nsteps=30,
        porosity_min=0.3,
        porosity_max=0.7,
        grad=0,  # Not used in blending mode
        func_degree=1,
        delta=0.2,
        smoothness=0.8,
        marching_step=1,
        wall_thickness=0.5,
        tpms_type='gyroid'
    )
    
    # Create blended parameters
    blended_params = BlendedTPMSParameters(
        base_params=base_params,
        t_skeletal=-0.5,  # Dense bottom (NEGATIVE value)
        t_sheet=0.1,      # Porous top (POSITIVE value)
        steepness=5.0,    # Moderate transition
        use_blending=True
    )
    
    # Generate structure
    output_dir = Path.cwd() / "gyroid_outputs"
    stl_path = create_blended_tpms(blended_params, output_dir, show_plot=True)
    
    print(f"\nGenerated blended TPMS structure:")
    print(f"  t_skeletal: {blended_params.t_skeletal}")
    print(f"  t_sheet: {blended_params.t_sheet}")
    print(f"  steepness: {blended_params.steepness}")
    print(f"STL exported to: {stl_path}")


if __name__ == "__main__":
    main()

