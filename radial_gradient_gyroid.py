"""
Radial Gradient Gyroid Generator

This module implements gyroid structures with radial porosity gradients,
following the methodology described in:

"Influence of Porosity Gradient Distribution on Mechanical and Biological 
Properties of Gyroid-Based Zn-2Mg Scaffolds for Bone Tissue Engineering"

Key Concepts:
- Forward Radial Gradient: Porosity increases from center (dense core) to edge (porous shell)
  Example: 50% center → 70% edge
- Reverse Radial Gradient: Porosity decreases from center (porous core) to edge (dense shell)
  Example: 70% center → 50% edge

The radial gradient is computed as:
  r = sqrt(x² + y²)  # Distance from center
  r_norm = r / r_max  # Normalized distance [0, 1]
  d(r) = d_center + (d_edge - d_center) * r_norm  # Threshold varies with radius

This creates scaffolds with continuous density variation from center to edge,
providing different mechanical properties at different radial positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage import measure
from stl import mesh

# Import from lib modules
from .parameters import GyroidParameters
from .tpms_helpers import (
    validate_params,
    compute_domain_lengths,
    generate_coordinate_grid,
    tpms_field,
    add_bounding_box,
    marching_cubes_mesh,
    export_stl,
    solve_layer_threshold,
)


@dataclass
class RadialGradientParameters:
    """
    Parameters for radial gradient gyroid structures.
    
    Attributes
    ----------
    base_params : GyroidParameters
        Base TPMS generation parameters (geometry, etc.)
    porosity_center : float
        Porosity at the center (r=0) in range [0, 1]
        For forward gradient: lower porosity (denser, e.g., 0.50)
        For reverse gradient: higher porosity (more porous, e.g., 0.70)
    porosity_edge : float
        Porosity at the edge (r=r_max) in range [0, 1]
        For forward gradient: higher porosity (more porous, e.g., 0.70)
        For reverse gradient: lower porosity (denser, e.g., 0.50)
    gradient_type : Literal['forward', 'reverse']
        'forward': Porosity increases from center to edge (dense core → porous shell)
        'reverse': Porosity decreases from center to edge (porous core → dense shell)
    use_porosity_to_threshold : bool
        If True, use binary search to find thresholds from porosity targets.
        If False, use direct threshold mapping (requires threshold_center and threshold_edge).
        Default: True
    threshold_center : float, optional
        Direct threshold value at center (only used if use_porosity_to_threshold=False)
        Typical values: -0.2065 for 50% porosity, -0.4108 for 70% porosity
    threshold_edge : float, optional
        Direct threshold value at edge (only used if use_porosity_to_threshold=False)
    """
    base_params: GyroidParameters
    porosity_center: float = 0.50  # Porosity at center
    porosity_edge: float = 0.70    # Porosity at edge
    gradient_type: Literal['forward', 'reverse'] = 'forward'
    use_porosity_to_threshold: bool = True
    threshold_center: float = None
    threshold_edge: float = None


def porosity_to_threshold_approx(porosity: float) -> float:
    """
    Approximate mapping from porosity percentage to threshold value.
    
    Based on empirical values from the paper:
    - 50% porosity → d ≈ -0.2065
    - 70% porosity → d ≈ -0.4108
    
    Uses linear interpolation for intermediate values.
    
    Parameters
    ----------
    porosity : float
        Porosity value in [0, 1]
    
    Returns
    -------
    float
        Approximate threshold value
    """
    # Known porosity-to-threshold mappings
    porosities = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    thresholds = np.array([-0.10, -0.15, -0.2065, -0.30, -0.4108, -0.55])
    
    # Clamp porosity to valid range
    porosity = np.clip(porosity, 0.0, 1.0)
    
    # Linear interpolation
    threshold = np.interp(porosity, porosities, thresholds)
    return float(threshold)


def compute_radial_distance(grid: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Compute radial distance from center for each point in the grid.
    
    Parameters
    ----------
    grid : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Coordinate grids (X, Y, Z)
    
    Returns
    -------
    r : np.ndarray
        Radial distance from center for each point
    r_max : float
        Maximum radial distance (distance to corner)
    """
    X, Y, Z = grid
    
    # Find center coordinates
    x_center = (np.max(X) + np.min(X)) / 2.0
    y_center = (np.max(Y) + np.min(Y)) / 2.0
    
    # Compute radial distance: r = sqrt((x - x_center)² + (y - y_center)²)
    r = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    
    # Maximum radial distance (distance to corner)
    r_max = np.sqrt(
        (np.max(X) - x_center)**2 + (np.max(Y) - y_center)**2
    )
    
    return r, r_max


def compute_radial_threshold(
    r: np.ndarray,
    r_max: float,
    porosity_center: float,
    porosity_edge: float,
    use_porosity_to_threshold: bool = True,
    threshold_center: float = None,
    threshold_edge: float = None,
) -> np.ndarray:
    """
    Compute threshold values that vary with radial distance.
    
    The threshold d(r) is computed as:
        d(r) = d_center + (d_edge - d_center) * (r / r_max)
    
    Parameters
    ----------
    r : np.ndarray
        Radial distance from center for each point
    r_max : float
        Maximum radial distance
    porosity_center : float
        Target porosity at center
    porosity_edge : float
        Target porosity at edge
    use_porosity_to_threshold : bool
        If True, convert porosity to threshold using approximation
        If False, use provided threshold values directly
    threshold_center : float, optional
        Direct threshold at center (if use_porosity_to_threshold=False)
    threshold_edge : float, optional
        Direct threshold at edge (if use_porosity_to_threshold=False)
    
    Returns
    -------
    np.ndarray
        Threshold values d(r) for each point, same shape as r
    """
    # Normalize radial distance to [0, 1]
    r_norm = np.clip(r / (r_max + 1e-10), 0.0, 1.0)
    
    # Get threshold values
    if use_porosity_to_threshold:
        d_center = porosity_to_threshold_approx(porosity_center)
        d_edge = porosity_to_threshold_approx(porosity_edge)
    else:
        if threshold_center is None or threshold_edge is None:
            raise ValueError(
                "threshold_center and threshold_edge must be provided "
                "when use_porosity_to_threshold=False"
            )
        d_center = threshold_center
        d_edge = threshold_edge
    
    # Linear interpolation: d(r) = d_center + (d_edge - d_center) * r_norm
    d_r = d_center + (d_edge - d_center) * r_norm
    
    return d_r


def generate_radial_gradient_volume(
    params: RadialGradientParameters
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, np.ndarray]]:
    """
    Generate a binary TPMS volume with radial porosity gradient.
    
    Process:
    1. Generate TPMS field G(x,y,z) using base parameters
    2. Compute radial distance r = sqrt((x-x_center)² + (y-y_center)²)
    3. Normalize: r_norm = r / r_max
    4. Compute threshold: d(r) = d_center + (d_edge - d_center) * r_norm
    5. Apply threshold: Material if G(x,y,z) > d(r)
    
    Parameters
    ----------
    params : RadialGradientParameters
        Parameters including base TPMS params and radial gradient parameters
    
    Returns
    -------
    volume : np.ndarray
        3D boolean array (True = material, False = void)
    spacing : Tuple[float, float, float]
        Voxel spacing in each direction (mm)
    metadata : Dict[str, np.ndarray]
        Metadata including porosity profile, radial distances, etc.
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
    
    # Compute radial distance from center
    r, r_max = compute_radial_distance(grid)
    
    # Determine porosity values based on gradient type
    if params.gradient_type == 'forward':
        # Forward: dense center → porous edge
        porosity_center = params.porosity_center  # Lower porosity (denser)
        porosity_edge = params.porosity_edge      # Higher porosity (more porous)
    else:  # reverse
        # Reverse: porous center → dense edge
        porosity_center = params.porosity_edge    # Higher porosity (more porous)
        porosity_edge = params.porosity_center    # Lower porosity (denser)
    
    # Compute radial threshold d(r)
    d_r = compute_radial_threshold(
        r, r_max,
        porosity_center, porosity_edge,
        params.use_porosity_to_threshold,
        params.threshold_center,
        params.threshold_edge,
    )
    
    # Apply threshold: Material if G(x,y,z) > d(r)
    volume = G > d_r
    
    # Apply Gaussian smoothing if requested
    if base_params.smoothness > 0:
        volume = ndimage.binary_opening(volume)  # Remove small artifacts
        volume = ndimage.binary_closing(volume)  # Fill small holes
    
    # Add bounding box walls
    spacing = (lx / nx, ly / ny, lz / nz)
    volume = add_bounding_box(volume, spacing, base_params.wall_thickness)
    
    # Compute porosity profile along radial direction
    # Create radial bins
    n_bins = 20
    r_bins = np.linspace(0, r_max, n_bins + 1)
    porosity_profile = np.zeros(n_bins)
    radial_positions = np.zeros(n_bins)
    
    for i in range(n_bins):
        r_min = r_bins[i]
        r_max_bin = r_bins[i + 1]
        mask = (r >= r_min) & (r < r_max_bin)
        if np.any(mask):
            porosity_profile[i] = 1.0 - np.mean(volume[mask])
            radial_positions[i] = (r_min + r_max_bin) / 2.0
    
    # Compute average porosity
    average_porosity = 1.0 - np.mean(volume)
    
    # Metadata
    metadata: Dict[str, np.ndarray] = {
        "porosity_profile": porosity_profile,  # Porosity at each radial bin
        "radial_positions": radial_positions,  # Radial positions of bins
        "porosity_center": porosity_center,
        "porosity_edge": porosity_edge,
        "average_porosity": average_porosity,
        "volume_fraction": np.mean(volume),
        "lengths": np.array([lx, ly, lz]),
        "spacing": np.array(spacing),
        "r_max": r_max,
        "gradient_type": params.gradient_type,
    }
    
    return volume, spacing, metadata


def create_radial_gradient_gyroid(
    params: RadialGradientParameters,
    output_dir: Path,
    show_plot: bool = True
) -> Path:
    """
    Generate, export, and optionally visualize a radial gradient gyroid structure.
    
    Parameters
    ----------
    params : RadialGradientParameters
        Parameters for radial gradient gyroid generation
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
    volume, spacing, metadata = generate_radial_gradient_volume(params)
    
    # Validate volume has material
    solid_voxels = np.sum(volume)
    total_voxels = volume.size
    solid_fraction = solid_voxels / total_voxels if total_voxels > 0 else 0.0
    
    if solid_voxels == 0:
        raise ValueError(
            f"Generated volume is empty! Check parameters:\n"
            f"  - porosity_center={params.porosity_center}, "
            f"porosity_edge={params.porosity_edge}\n"
            f"  - gradient_type={params.gradient_type}\n"
            f"  - nsteps={params.base_params.nsteps}"
        )
    
    if solid_fraction < 0.01:
        raise ValueError(
            f"Volume is too sparse ({solid_fraction*100:.1f}% solid)! "
            f"Try adjusting porosity values or increasing nsteps"
        )
    
    # Extract mesh using marching cubes
    verts, faces, *_ = marching_cubes_mesh(volume, spacing, params.base_params)
    
    # Validate mesh
    if len(faces) == 0:
        raise ValueError(
            f"Marching cubes produced no faces! Volume might be too sparse.\n"
            f"  - Solid fraction: {solid_fraction*100:.1f}%\n"
            f"  - Try adjusting porosity values"
        )
    
    # Export STL
    tpms_type = params.base_params.tpms_type.lower() if hasattr(params.base_params, 'tpms_type') else 'gyroid'
    gradient_suffix = f"radial_{params.gradient_type}"
    stl_path = export_stl(verts, faces, output_dir, tpms_type=f"{tpms_type}_{gradient_suffix}")
    
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
        visualise_radial_gradient(params, volume, verts, faces, spacing, metadata)
    
    return stl_path


def visualise_radial_gradient(
    params: RadialGradientParameters,
    volume: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    spacing: Tuple[float, float, float],
    metadata: Dict[str, np.ndarray],
):
    """Visualize radial gradient gyroid with 3D preview and radial porosity profile."""
    lx, ly, lz = metadata["lengths"]
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D preview
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    poly = Poly3DCollection(verts[faces], alpha=0.65, facecolor="#1f77b4", edgecolor="none")
    ax3d.add_collection3d(poly)
    ax3d.set_xlim(0, lx)
    ax3d.set_ylim(0, ly)
    ax3d.set_zlim(0, lz)
    ax3d.set_box_aspect((lx, ly, lz))
    tpms_type = params.base_params.tpms_type.lower() if hasattr(params.base_params, 'tpms_type') else 'gyroid'
    ax3d.set_title(f"{tpms_type.capitalize()} Radial Gradient ({params.gradient_type})")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    
    # Side view projection
    ax_side = fig.add_subplot(2, 3, 2)
    side_projection = np.max(volume, axis=0).astype(float)
    im_side = ax_side.imshow(side_projection.T, origin="lower", cmap="viridis")
    ax_side.set_title("Side View (X Projection)")
    ax_side.set_xlabel("Y voxels")
    ax_side.set_ylabel("Z voxels")
    fig.colorbar(im_side, ax=ax_side, fraction=0.046, pad=0.04)
    
    # Top view (radial gradient visible)
    ax_top = fig.add_subplot(2, 3, 3)
    mid_z = volume.shape[2] // 2
    top_view = volume[:, :, mid_z].astype(float)
    im_top = ax_top.imshow(top_view.T, origin="lower", cmap="plasma")
    ax_top.set_title(f"Top View (Z={mid_z}, Radial Gradient)")
    ax_top.set_xlabel("X voxels")
    ax_top.set_ylabel("Y voxels")
    fig.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.04)
    
    # Radial porosity profile
    ax_profile = fig.add_subplot(2, 3, 5)
    radial_positions = metadata["radial_positions"]
    porosity_profile = metadata["porosity_profile"]
    ax_profile.plot(radial_positions, porosity_profile, 'o-', linewidth=2, markersize=6)
    ax_profile.set_xlabel("Radial Distance (mm)")
    ax_profile.set_ylabel("Porosity")
    ax_profile.set_title("Radial Porosity Profile")
    ax_profile.grid(True, alpha=0.3)
    ax_profile.axhline(metadata["porosity_center"], color='r', linestyle='--', 
                       label=f'Center: {metadata["porosity_center"]:.2%}')
    ax_profile.axhline(metadata["porosity_edge"], color='b', linestyle='--', 
                       label=f'Edge: {metadata["porosity_edge"]:.2%}')
    ax_profile.legend()
    
    # Cross-section at different radial positions
    ax_cross = fig.add_subplot(2, 3, 6)
    # Show a slice through the center
    center_x = volume.shape[0] // 2
    cross_section = volume[center_x, :, :].astype(float)
    im_cross = ax_cross.imshow(cross_section.T, origin="lower", cmap="plasma")
    ax_cross.set_title("Cross-Section (X=center)")
    ax_cross.set_xlabel("Y voxels")
    ax_cross.set_ylabel("Z voxels")
    fig.colorbar(im_cross, ax=ax_cross, fraction=0.046, pad=0.04)
    
    # Add text with parameters
    inputs_caption = (
        f"numx/numy/numz: {params.base_params.numx}/{params.base_params.numy}/{params.base_params.numz}\n"
        f"unit_cell_size: {params.base_params.unit_cell_size:.2f} mm\n"
        f"nsteps: {params.base_params.nsteps}\n"
        f"gradient_type: {params.gradient_type}\n"
        f"porosity_center: {params.porosity_center:.2%}\n"
        f"porosity_edge: {params.porosity_edge:.2%}"
    )
    outputs_caption = (
        f"avg_porosity: {metadata['average_porosity']:.3f}\n"
        f"volume_fraction: {metadata['volume_fraction']:.3f}\n"
        f"domain (mm): {lx:.1f} × {ly:.1f} × {lz:.1f}\n"
        f"r_max: {metadata['r_max']:.2f} mm"
    )
    fig.text(0.05, 0.02, f"Inputs:\n{inputs_caption}", ha="left", va="bottom", 
             fontsize=9, family="monospace")
    fig.text(0.55, 0.02, f"Outputs:\n{outputs_caption}", ha="left", va="bottom", 
             fontsize=9, family="monospace")
    
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def main():
    """Example usage of radial gradient gyroid generator."""
    # Create base parameters
    base_params = GyroidParameters(
        numx=5,
        numy=5,
        numz=5,
        unit_cell_size=3.0,
        nsteps=40,
        porosity_min=0.3,  # Not used in radial gradient, but required
        porosity_max=0.7,  # Not used in radial gradient, but required
        grad=0,  # No z-direction gradient
        func_degree=1,
        delta=0.2,
        smoothness=0.8,
        marching_step=1,
        wall_thickness=0.5,
        tpms_type='gyroid'
    )
    
    # Example 1: Forward radial gradient (dense center → porous edge)
    print("Generating forward radial gradient gyroid (50% center → 70% edge)...")
    forward_params = RadialGradientParameters(
        base_params=base_params,
        porosity_center=0.50,  # Dense center
        porosity_edge=0.70,    # Porous edge
        gradient_type='forward',
        use_porosity_to_threshold=True,
    )
    
    output_dir = Path.cwd() / "gyroid_outputs"
    stl_path_forward = create_radial_gradient_gyroid(forward_params, output_dir, show_plot=True)
    print(f"Forward gradient STL exported to: {stl_path_forward}\n")
    
    # Example 2: Reverse radial gradient (porous center → dense edge)
    print("Generating reverse radial gradient gyroid (70% center → 50% edge)...")
    reverse_params = RadialGradientParameters(
        base_params=base_params,
        porosity_center=0.70,  # Porous center
        porosity_edge=0.50,     # Dense edge
        gradient_type='reverse',
        use_porosity_to_threshold=True,
    )
    
    stl_path_reverse = create_radial_gradient_gyroid(reverse_params, output_dir, show_plot=True)
    print(f"Reverse gradient STL exported to: {stl_path_reverse}\n")


if __name__ == "__main__":
    main()

