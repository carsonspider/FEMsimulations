"""
Helper functions for TPMS generation.

This module provides utility functions for generating TPMS fields, meshes,
and exporting STL files. It integrates with surfaces.py for TPMS equations
and types.py for type definitions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure
from stl import mesh

from .types import NDARRAY
from .parameters import GyroidParameters
from . import surfaces


def validate_params(params: GyroidParameters) -> GyroidParameters:
    """
    Validate and sanitize input parameters with appropriate clamping.
    
    This function performs comprehensive validation of all parameters,
    ensuring they fall within acceptable ranges for stable computation.
    Invalid inputs are clamped to safe defaults or raise descriptive errors.
    
    Parameters
    ----------
    params : GyroidParameters
        Input parameter bundle (may contain invalid values).
    
    Returns
    -------
    GyroidParameters
        Validated and sanitized parameter bundle with all values
        within acceptable ranges.
    
    Raises
    ------
    ValueError
        If nsteps < 4 (insufficient resolution for gradients),
        func_degree not in {0, 1, 2}, or grad not in {0, 1}.
    """
    # Check minimum resolution requirement for gradient computation
    if params.nsteps < 4:
        raise ValueError("nsteps must be at least 4 to support porosity gradients.")
    
    # Validate gradient function degree (must be 0, 1, or 2)
    if params.func_degree not in (0, 1, 2):
        raise ValueError("func_degree must be 0 (constant), 1 (linear), or 2 (quadratic).")
    
    # Validate gradient flag (must be 0 or 1)
    if params.grad not in (0, 1):
        raise ValueError("grad must be 0 (constant porosity) or 1 (graded porosity).")
    
    # Validate TPMS type if present
    if hasattr(params, 'tpms_type'):
        valid_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
        if params.tpms_type.lower() not in valid_types:
            print(f"Warning: Unknown TPMS type '{params.tpms_type}', defaulting to 'gyroid'")
            params.tpms_type = 'gyroid'
    
    # Clamp porosity values to valid range [0, 1]
    poro_min = float(np.clip(params.porosity_min, 0.0, 1.0))
    poro_max = float(np.clip(params.porosity_max, 0.0, 1.0))
    
    # Ensure porosity_min <= porosity_max (swap if necessary)
    if poro_min > poro_max:
        poro_min, poro_max = poro_max, poro_min
    
    # Get TPMS type if present, default to 'gyroid'
    tpms_type = 'gyroid'
    if hasattr(params, 'tpms_type'):
        tpms_type = params.tpms_type.lower()
        valid_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
        if tpms_type not in valid_types:
            tpms_type = 'gyroid'
    
    # Return validated parameter bundle with type coercion and clamping
    return GyroidParameters(
        numx=int(params.numx),
        numy=int(params.numy),
        numz=int(params.numz),
        unit_cell_size=float(params.unit_cell_size),
        nsteps=int(params.nsteps),
        porosity_min=poro_min,
        porosity_max=poro_max,
        grad=int(params.grad),
        func_degree=int(params.func_degree),
        delta=float(max(params.delta, 1e-4)),
        smoothness=float(max(params.smoothness, 0.0)),
        marching_step=int(max(params.marching_step, 1)),
        wall_thickness=float(max(params.wall_thickness, 0.0)),
        tpms_type=tpms_type,
    )


def compute_domain_lengths(params: GyroidParameters) -> Tuple[float, float, float]:
    """
    Compute the physical domain dimensions from cell counts and cell size.
    
    Lx, Ly, Lz are obtained by multiplying the number of unit cells by the
    physical size of each unit cell. These lengths correspond to the actual
    physical dimensions of the generated structure in millimetres.
    
    Parameters
    ----------
    params : GyroidParameters
        Parameter bundle containing numx, numy, numz, and unit_cell_size.
    
    Returns
    -------
    Tuple[float, float, float]
        Physical domain lengths (Lx, Ly, Lz) in millimetres.
    """
    lx = params.numx * params.unit_cell_size
    ly = params.numy * params.unit_cell_size
    lz = params.numz * params.unit_cell_size
    return lx, ly, lz


def generate_coordinate_grid(params: GyroidParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a periodic voxel grid covering the design domain."""
    lx, ly, lz = compute_domain_lengths(params)
    nx = params.numx * params.nsteps
    ny = params.numy * params.nsteps
    nz = params.numz * params.nsteps
    
    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    z = np.linspace(0, lz, nz, endpoint=False)
    
    return np.meshgrid(x, y, z, indexing="ij")


def tpms_field(params: GyroidParameters, grid: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Evaluate the implicit TPMS field (normalised to [-1, 1]).
    
    Uses surface functions from surfaces.py and supports multiple TPMS lattice types.
    
    Parameters
    ----------
    params : GyroidParameters
        Parameters including tpms_type
    grid : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Coordinate grids (x, y, z)
    
    Returns
    -------
    np.ndarray
        Normalized TPMS field values in [-1, 1]
    """
    x, y, z = grid
    lx, ly, lz = compute_domain_lengths(params)

    kx = 2.0 * np.pi / lx
    ky = 2.0 * np.pi / ly
    kz = 2.0 * np.pi / lz

    # Scale coordinates for TPMS equations
    x_scaled = kx * x
    y_scaled = ky * y
    z_scaled = kz * z

    # Get TPMS type (case-insensitive, default to gyroid)
    tpms_type = params.tpms_type.lower() if hasattr(params, 'tpms_type') else 'gyroid'
    
    # Map TPMS type to surface function from surfaces.py
    if tpms_type == 'gyroid':
        field = surfaces.gyroid(x_scaled, y_scaled, z_scaled)
    elif tpms_type == 'schwarz':
        field = surfaces.schwarz_p(x_scaled, y_scaled, z_scaled)
    elif tpms_type == 'diamond':
        field = surfaces.diamond(x_scaled, y_scaled, z_scaled)
    elif tpms_type == 'lidinoid':
        field = surfaces.l_surface(x_scaled, y_scaled, z_scaled)
    elif tpms_type == 'split-p':
        # Split-P is not in surfaces.py, use approximation
        field = (
            1.1 * (
                np.sin(2 * x_scaled) * np.cos(y_scaled) * np.sin(z_scaled)
                + np.sin(x_scaled) * np.sin(2 * y_scaled) * np.cos(z_scaled)
                + np.cos(x_scaled) * np.sin(y_scaled) * np.sin(2 * z_scaled)
            )
            - 0.2 * (
                np.cos(2 * x_scaled) * np.cos(2 * y_scaled)
                + np.cos(2 * y_scaled) * np.cos(2 * z_scaled)
                + np.cos(2 * z_scaled) * np.cos(2 * x_scaled)
            )
            - 0.4 * (
                np.cos(2 * x_scaled) + np.cos(2 * y_scaled) + np.cos(2 * z_scaled)
            )
        ) / 5.0
    else:
        # Default to gyroid if unknown type
        print(f"Warning: Unknown TPMS type '{tpms_type}', defaulting to 'gyroid'")
        field = surfaces.gyroid(x_scaled, y_scaled, z_scaled)

    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(field))
    if max_abs > 0:
        field = field / max_abs

    # Apply Gaussian smoothing if requested
    if params.smoothness > 0:
        field = ndimage.gaussian_filter(field, sigma=params.smoothness)
        max_abs = np.max(np.abs(field))
        if max_abs > 0:
            field = field / max_abs

    return np.clip(field, -1.0, 1.0)


def add_bounding_box(volume: np.ndarray, spacing: Tuple[float, float, float], wall_thickness: float) -> np.ndarray:
    """Add solid bounding box walls to enclose the TPMS structure."""
    nx, ny, nz = volume.shape
    dx, dy, dz = spacing
    
    # Calculate wall thickness in voxels for each direction
    wall_x = max(1, int(np.ceil(wall_thickness / dx)))
    wall_y = max(1, int(np.ceil(wall_thickness / dy)))
    wall_z = max(1, int(np.ceil(wall_thickness / dz)))
    
    # Create a copy to avoid modifying the original
    enclosed_volume = volume.copy()
    
    # Add walls on all 6 faces
    # Bottom and top faces (z-direction)
    enclosed_volume[:, :, :wall_z] = True
    enclosed_volume[:, :, -wall_z:] = True
    
    # Front and back faces (y-direction)
    enclosed_volume[:, :wall_y, :] = True
    enclosed_volume[:, -wall_y:, :] = True
    
    # Left and right faces (x-direction)
    enclosed_volume[:wall_x, :, :] = True
    enclosed_volume[-wall_x:, :, :] = True
    
    return enclosed_volume


def marching_cubes_mesh(volume: np.ndarray, spacing: Tuple[float, float, float], params: GyroidParameters):
    """Extract a triangle mesh from the voxelised TPMS."""
    verts, faces, normals, values = measure.marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=spacing,
        step_size=params.marching_step,
    )
    return verts, faces, normals, values


def export_stl(verts: np.ndarray, faces: np.ndarray, output_dir: Path, tpms_type: str = 'gyroid') -> Path:
    """Write the mesh to an STL file and return the resulting path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stl_path = output_dir / f"{tpms_type}_{timestamp}.stl"

    tpms_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            tpms_mesh.vectors[i][j] = verts[face[j], :]
    tpms_mesh.save(stl_path.as_posix())
    return stl_path


def solve_layer_threshold(layer_field: np.ndarray, solid_target: float, delta: float) -> Tuple[float, float]:
    """Binary search the threshold for a single voxel layer."""
    low, high = -1.0, 1.0
    tolerance = max(delta, 1e-6)
    best_mid = 0.0
    best_fraction = 0.0

    for _ in range(60):
        mid = 0.5 * (low + high)
        solid_fraction = float(np.mean(layer_field > mid))
        error = solid_fraction - solid_target
        best_mid = mid
        best_fraction = solid_fraction
        if abs(error) <= tolerance:
            break
        if error > 0:
            low = mid
        else:
            high = mid

    return best_mid, best_fraction

