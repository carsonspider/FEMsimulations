"""
Parameter definitions for TPMS generation.

This module contains dataclasses for configuring TPMS (Triply Periodic Minimal Surface)
lattice structure generation.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class GyroidParameters:
    """
    Parameter bundle for the graded TPMS generator.
    
    This dataclass encapsulates all parameters needed to generate TPMS
    (Triply Periodic Minimal Surface) lattice structures including:
    - Gyroid (Schoen gyroid)
    - Schwarz (Schwarz P-surface / Primitive)
    - Diamond (Schwarz D-surface)
    - Lidinoid
    - Split-P
    
    All geometric parameters are defined in physical units (millimetres)
    to ensure direct manufacturing compatibility.
    
    Attributes
    ----------
    numx : int
        Number of unit cells repeated along the x-axis.
        Together with unit_cell_size, determines Lx (domain length in x).
    numy : int
        Number of unit cells repeated along the y-axis.
        Together with unit_cell_size, determines Ly (domain length in y).
    numz : int
        Number of unit cells repeated along the z-axis (build direction).
        Together with unit_cell_size, determines Lz (domain length in z).
        This is the direction of porosity grading.
    unit_cell_size : float
        Physical dimension of a single unit cell in millimetres.
        The total domain size is: Lx = numx × unit_cell_size (and similarly for y, z).
    nsteps : int
        Number of voxels used to discretize the TPMS in each direction
        per unit cell. Higher values yield finer resolution but increase
        computation time and mesh complexity. Minimum: 4 (required for gradients).
    porosity_min : float
        Minimum porosity value in the range [0, 1]. For graded structures,
        this is typically achieved at the bottom (z=0) layer.
        0.0 = fully solid, 1.0 = fully void.
    porosity_max : float
        Maximum porosity value in the range [0, 1]. For graded structures,
        this is typically achieved at the top (z=Lz) layer.
        0.0 = fully solid, 1.0 = fully void.
        Must be >= porosity_min.
    grad : int
        Gradient flag: 0 = constant porosity (no grading),
                       1 = graded porosity along z-direction.
        When grad=1, porosity varies from porosity_max (at z=0) to porosity_min
        (at z=Lz) according to func_degree.
    func_degree : int
        Polynomial degree of the porosity gradient function:
        0 = constant (uniform porosity = porosity_max),
        1 = linear gradient (straight line from max to min),
        2 = quadratic gradient (parabolic transition).
        Only used when grad=1.
    delta : float
        Tolerance parameter controlling the maximum allowable deviation
        between actual and target porosity for each layer. Smaller values
        (e.g., 0.02 = 2%) yield more accurate porosity control but increase
        computation time. Typical range: 0.01-0.05.
    smoothness : float
        Standard deviation (in voxels) for optional Gaussian smoothing
        applied to the implicit TPMS field before thresholding. This
        reduces voxel-scale artifacts and stair-stepping. 0.0 = no smoothing.
        Typical range: 0.5-1.0 voxels.
    marching_step : int
        Step size parameter for the marching cubes algorithm. Higher values
        reduce triangle count (lighter mesh) but lower surface quality.
        step_size=1 uses all voxels (maximum quality).
        step_size=2 uses every other voxel (faster, lower resolution).
    wall_thickness : float, optional
        Thickness of the bounding box walls in millimetres. These solid walls
        enclose the TPMS lattice on all 6 faces, making the structure
        watertight and suitable for simulation/printing. Default: 0.5 mm.
        Set to 0.0 to disable bounding box (not recommended for printing).
    tpms_type : str, optional
        Type of TPMS structure to generate. Options:
        - 'gyroid': Schoen gyroid (default)
        - 'schwarz': Schwarz P-surface / Primitive
        - 'diamond': Schwarz D-surface / Diamond
        - 'lidinoid': Lidinoid surface
        - 'split-p': Split-P surface
    """

    numx: int
    numy: int
    numz: int
    unit_cell_size: float
    nsteps: int
    porosity_min: float  # 0–1
    porosity_max: float  # 0–1
    grad: int  # 0 = constant, 1 = graded in z
    func_degree: int  # 0 constant, 1 linear, 2 quadratic
    delta: float  # allowable porosity deviation
    smoothness: float  # Gaussian sigma (voxels)
    marching_step: int
    wall_thickness: float = 0.5  # wall thickness in mm for bounding box
    tpms_type: str = 'gyroid'  # TPMS structure type

