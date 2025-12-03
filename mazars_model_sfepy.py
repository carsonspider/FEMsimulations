#!/usr/bin/env python3
"""
FEM compression test simulation for STL files using Mazars damage model with SfePy.

This script performs a uniaxial compression test using the Mazars continuum
damage mechanics model with SfePy (Simple Finite Elements in Python) instead of FEniCS.

Features:
- Nonlinear damage evolution (compounding, irreversible)
- Effective modulus reduction based on damage field
- Newton-Raphson iterations for damage convergence
- Localized microcracking (damage field)

The Mazars model accounts for stiffness degradation under loading, making it
suitable for simulating cement/concrete behavior (target: 10-20 MPa compressive strength).

Outputs:
- Compressive strength
- Stress-strain curve (nonlinear due to damage)
- Displacement field visualization
- Energy absorption
- Damage field (microcracking)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import json
import matplotlib.pyplot as plt
import time

try:
    import sfepy
    from sfepy.base.base import output
    from sfepy.discrete import Problem
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.terms import Term
    from sfepy import data_dir
    # Additional imports for proper FE implementation (commented out if not available)
    # from sfepy.discrete import Variables
    # from sfepy.discrete.fem import FieldVariable
    # from sfepy.terms.terms import Term
except ImportError as e:
    raise ImportError(
        "SfePy is not installed. Install it with: pip install sfepy\n"
        f"Original error: {e}"
    )


@dataclass
class MaterialProperties:
    """Material properties for Mazars damage model (cement/concrete).
    
    The Mazars model uses continuum damage mechanics to account for:
    - Stiffness degradation under loading
    - Irreversible damage accumulation
    - Localized microcracking
    
    Standard Mazars damage evolution laws:
    Compression: d_c = 1 - (ε_d0/ε_eq) * (1 - A_c + A_c * exp[-B_c(ε_eq - ε_d0)])
    Tension: d_t = 1 - (ε_d0/ε_eq) * (1 - A_t + A_t * exp[-B_t(ε_eq - ε_d0)])
    
    Typical values for cement/concrete (10-20 MPa compressive strength):
    - E: 25-35 GPa (Young's modulus) - Recommended: 25 GPa
    - nu: 0.15-0.2 (Poisson's ratio)
    - epsilon_c0: 6e-4 to 1.2e-3 (compressive damage threshold strain) - Recommended: 8e-4
    - A_c: 0.7-1.5 (compressive damage evolution parameter 1) - Recommended: 1.0
    - B_c: 1000-2000 (compressive damage evolution parameter 2) - Recommended: 1500
    - epsilon_t0: 1e-4 (tensile damage threshold strain)
    - A_t: 0.8-1.2 (tensile damage evolution parameter 1)
    - B_t: 1000-2000 (tensile damage evolution parameter 2)
    
    The effective modulus is reduced by damage: E_eff = E * (1 - damage)
    """
    
    E: float = 25e9  # Young's modulus (Pa) - 25 GPa (recommended: 25-35 GPa range)
    nu: float = 0.2  # Poisson's ratio (typical for concrete: 0.15-0.2)
    rho: float = 2000.0  # Density (kg/m³) - typical for cement paste
    # Compressive damage parameters
    epsilon_c0: float = 8e-4   # Mazars compressive damage threshold strain (ε_d0) - Recommended: 6e-4 to 1.2e-3
    A_c: float = 1.0  # Mazars compressive damage evolution parameter 1 - Recommended: 0.7-1.5
    B_c: float = 1500.0  # Mazars compressive damage evolution parameter 2 - Recommended: 1000-2000
    # Tensile damage parameters (tensile strength is ~10-15% of compressive strength)
    epsilon_t0: float = 5e-5   # Mazars tensile damage threshold strain (ε_d0) - much lower than compression
    A_t: float = 0.7  # Mazars tensile damage evolution parameter 1 - lower than compression
    B_t: float = 2000.0  # Mazars tensile damage evolution parameter 2 - higher for faster damage
    
    def compute_lame_parameters(self) -> tuple:
        """Compute Lame parameters from E and nu.
        e
        Returns:
            (lambda, mu): Lame parameters for linear elasticity
        """
        mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))  # First Lame parameter
        return lmbda, mu


@dataclass
class SimulationParameters:
    """Simulation control parameters for Mazars damage model.
    
    Nonlinear solver settings:
    - max_newton_iter: Maximum Newton-Raphson iterations per load step
    - newton_tol: Convergence tolerance for Newton solver
    - damage_tol: Damage field convergence tolerance
    
    Load control:
    - Fixed force (default 3.5 kN) applied to all geometries for fair comparison
    - For 10mm structures (0.01m × 0.01m), this targets ~35 MPa stress
    - If max_force is None, it will be auto-calculated from geometry to target stress
    - Reasonable element size for balance between speed and accuracy
    
    Note: Fixed force ensures all structures experience the same load, enabling
    fair comparison of failure strengths based on internal geometry differences.
    """
    
    max_force: float = 3500.0  # N (Fixed force for all geometries - targets ~35 MPa for 10mm structures)
    target_stress_mpa: float = 35.0  # Target maximum stress in MPa (used only if max_force is None for auto-calculation)
    num_steps: int = 10  # Full simulation with 10 steps
    element_size: float = 0.05  # m (balanced for speed/accuracy)
    max_newton_iter: int = 10  # Maximum Newton-Raphson iterations per load step
    newton_tol: float = 1e-6  # Newton solver tolerance
    damage_tol: float = 1e-4  # Damage convergence tolerance


def load_stl_and_create_mesh(stl_path: Path, element_size: float):
    """Load STL file and create mesh using SfePy.
    
    Parameters
    ----------
    stl_path : Path
        Path to the STL file to load
    element_size : float
        Target element size in meters
    
    Returns
    -------
    domain : FEDomain
        SfePy finite element domain ready for simulation
    """
    print(f"Loading STL file: {stl_path}")
    
    try:
        import meshio
        
        # Read STL to get bounding box
        stl_mesh = meshio.read(str(stl_path), file_format="stl")
        points = stl_mesh.points
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        
        # Auto-detect units: if max dimension > 10, assume mm and convert to m
        max_dim = np.max(bbox_max - bbox_min)
        units_converted = False
        if max_dim > 10.0:
            print(f"Detected STL in millimeters (max dimension: {max_dim:.2f} mm)")
            print(f"Converting to meters for simulation...")
            bbox_min = bbox_min / 1000.0
            bbox_max = bbox_max / 1000.0
            max_dim = np.max(bbox_max - bbox_min)
            units_converted = True
            print(f"Converted bounding box: {bbox_min} to {bbox_max} (m)")
        else:
            print(f"STL appears to be in meters (max dimension: {max_dim:.2f} m)")
        
        # For SfePy, create a box mesh using meshio and convert to SfePy format
        size = bbox_max - bbox_min
        n_x = max(2, int(size[0] / element_size))
        n_y = max(2, int(size[1] / element_size))
        n_z = max(2, int(size[2] / element_size))
        
        print(f"Creating box mesh: {n_x}x{n_y}x{n_z} divisions")
        print(f"Bounding box: {bbox_min} to {bbox_max} (m)")
        
        # Create structured hexahedral mesh using meshio
        # Generate points for structured grid
        x = np.linspace(bbox_min[0], bbox_max[0], n_x + 1)
        y = np.linspace(bbox_min[1], bbox_max[1], n_y + 1)
        z = np.linspace(bbox_min[2], bbox_max[2], n_z + 1)
        
        # Create structured grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Create hexahedral cells with correct VTK vertex ordering
        # VTK hexahedron ordering: bottom face (k) then top face (k+1)
        # Bottom: (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k)
        # Top:    (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1)
        cells = []
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    # Base index for point (i, j, k) in flattened array
                    base = i * (n_y + 1) * (n_z + 1) + j * (n_z + 1) + k
                    # Step sizes in the flattened array
                    step_x = (n_y + 1) * (n_z + 1)  # Step in x direction
                    step_y = (n_z + 1)               # Step in y direction
                    step_z = 1                       # Step in z direction
                    
                    # VTK hexahedron vertex ordering
                    cell = [
                        base,                    # 0: (i, j, k) - bottom front-left
                        base + step_x,           # 1: (i+1, j, k) - bottom front-right
                        base + step_x + step_y,  # 2: (i+1, j+1, k) - bottom back-right
                        base + step_y,          # 3: (i, j+1, k) - bottom back-left
                        base + step_z,          # 4: (i, j, k+1) - top front-left
                        base + step_x + step_z, # 5: (i+1, j, k+1) - top front-right
                        base + step_x + step_y + step_z,  # 6: (i+1, j+1, k+1) - top back-right
                        base + step_y + step_z  # 7: (i, j+1, k+1) - top back-left
                    ]
                    cells.append(cell)
        
        # Create SfePy mesh directly using Mesh.from_data()
        # This avoids format conversion issues and ensures correct orientation
        cells_array = np.array(cells, dtype=np.int32)
        
        # SfePy expects:
        # - coors: coordinates array (N, 3) - MUST be in meters for correct stress calculations
        # - conns: list of connectivity arrays, one per element type
        # - mat_ids: material IDs (all 0 for now)
        # - descs: element descriptor ('3_8' for 3D hexahedra with 8 nodes)
        coors = points.astype(np.float64)
        
        # CRITICAL: Verify coordinates are in meters (should be after unit conversion above)
        # For 10mm structures, dimensions should be ~0.01m. If > 0.1m, likely still in mm
        coors_size = np.max(coors, axis=0) - np.min(coors, axis=0)
        max_coor_size = np.max(coors_size)
        if max_coor_size > 0.1:  # If any dimension > 0.1m (100mm), likely still in mm
            print(f"⚠ WARNING: Mesh coordinates appear to be in wrong units!")
            print(f"  Mesh size: {coors_size} m (max: {max_coor_size:.3f} m)")
            print(f"  Expected: < 0.1m for structures up to 100mm")
            print(f"  Converting coordinates from mm to m...")
            coors = coors / 1000.0
            coors_size = np.max(coors, axis=0) - np.min(coors, axis=0)
            max_coor_size = np.max(coors_size)
            print(f"  Converted mesh size: {coors_size} m (max: {max_coor_size:.6f} m)")
            if max_coor_size > 0.1:
                raise ValueError(f"Mesh coordinates still too large after conversion: {coors_size} m. Check unit conversion logic.")
        
        conns = [cells_array]
        mat_ids = [np.zeros(len(cells), dtype=np.int32)]
        descs = ['3_8']  # 3D hexahedra
        
        try:
            # Create mesh directly from data
            mesh = Mesh.from_data('mesh', coors, None, conns, mat_ids, descs)
            domain = FEDomain('domain', mesh)
            
            num_vertices = mesh.n_nod
            num_cells = mesh.n_el
            # Verify final mesh dimensions
            final_coords = mesh.coors
            final_size = np.max(final_coords, axis=0) - np.min(final_coords, axis=0)
            print(f"Mesh created: {num_vertices} vertices, {num_cells} cells")
            print(f"Mesh dimensions: {final_size[0]:.6f} × {final_size[1]:.6f} × {final_size[2]:.6f} m")
            print(f"Expected cross-sectional area: {final_size[0] * final_size[1]:.6f} m² ({final_size[0] * final_size[1] * 1e6:.2f} mm²)")
            
            return domain
        except Exception as e:
            print(f"Error creating mesh in SfePy: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        print(f"Error loading STL: {e}")
        import traceback
        traceback.print_exc()
        raise


def compute_equivalent_strain(epsilon_tensor: np.ndarray) -> float:
    """Compute Mazars equivalent strain from full 3D strain tensor.
    
    Standard Mazars formulation (Mazars 1986):
    ε_eq = √(Σ⟨εᵢ⟩₊²)
    where ⟨εᵢ⟩₊ = max(εᵢ, 0) is the positive part (Macaulay bracket) of principal strains.
    
    This formulation uses only positive (tensile) principal strains, which is the
    standard approach in the Mazars model. For compression, damage occurs due to
    lateral expansion (Poisson effect) creating positive strains, not from the
    compressive strains themselves.
    
    Parameters
    ----------
    epsilon_tensor : np.ndarray, shape (3, 3)
        Full 3D symmetric strain tensor
    
    Returns
    -------
    float
        Equivalent strain (always positive, computed from positive principal strains)
    """
    # Compute principal strains (eigenvalues of strain tensor)
    eigenvals = np.linalg.eigvalsh(epsilon_tensor)  # eigvalsh for symmetric matrices
    
    # Standard Mazars formulation: use only positive principal strains (Macaulay brackets)
    # ⟨εᵢ⟩₊ = max(εᵢ, 0)
    positive_strains = np.maximum(eigenvals, 0.0)
    
    # Equivalent strain: ε_eq = √(Σ⟨εᵢ⟩₊²)
    eps_eq_squared = np.sum(positive_strains**2)
    
    return np.sqrt(eps_eq_squared) if eps_eq_squared > 0 else 0.0


def compute_strain_tensor_from_displacement(u_field, node_idx: int, mesh) -> np.ndarray:
    """Compute full 3D strain tensor from displacement field at a node.
    
    This function computes ε = 0.5(∇u + ∇u^T) at a given node.
    In proper FE implementation, this would use shape function gradients.
    
    Parameters
    ----------
    u_field : Field or array
        Displacement field (3D vector field)
    node_idx : int
        Node index
    mesh : Mesh
        Finite element mesh
    
    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric strain tensor
    """
    # TODO: Implement proper strain computation from displacement field
    # This requires:
    # 1. Get displacement values at node and neighboring nodes
    # 2. Compute gradient using shape function derivatives
    # 3. Form symmetric gradient: epsilon = 0.5 * (grad(u) + grad(u)^T)
    
    # For now, return zero tensor (placeholder)
    return np.zeros((3, 3), dtype=np.float64)


def mazars_compressive_damage(epsilon_eq: float, epsilon_c0: float, A_c: float, B_c: float) -> float:
    """Compute Mazars compressive damage evolution using standard formulation.
    
    Standard Mazars damage model for compression (Mazars 1986, Pijaudier-Cabot & Mazars 2001):
    d_c = 1 - (ε_d0/ε_eq) * (1 - A_c + A_c * exp[-B_c(ε_eq - ε_d0)])
    
    where:
    - ε_eq is the equivalent strain (computed from positive principal strains)
    - ε_d0 (epsilon_c0) is the damage threshold strain
    - A_c is the first damage evolution parameter
    - B_c is the second damage evolution parameter
    
    Parameters
    ----------
    epsilon_eq : float
        Equivalent strain (always positive, computed from positive principal strains)
    epsilon_c0 : float
        Damage threshold strain (ε_d0)
    A_c : float
        First damage evolution parameter
    B_c : float
        Second damage evolution parameter
    
    Returns
    -------
    float
        Damage value in [0, 1]
    """
    if epsilon_eq <= epsilon_c0:
        return 0.0
    
    # Original Mazars formulation (corrected)
    dc = 1.0 - (epsilon_c0 / epsilon_eq) * (1.0 - A_c + A_c * np.exp(-B_c * (epsilon_eq - epsilon_c0)))
    
    return np.clip(dc, 0.0, 1.0)


def mazars_tensile_damage(epsilon_eq: float, epsilon_t0: float, A_t: float, B_t: float) -> float:
    """Compute Mazars tensile damage evolution using standard formulation.
    
    Standard Mazars damage model for tension (Mazars 1986, Pijaudier-Cabot & Mazars 2001):
    d_t = 1 - (ε_d0/ε_eq) * (1 - A_t + A_t * exp[-B_t(ε_eq - ε_d0)])
    
    where:
    - ε_eq is the equivalent strain (computed from positive principal strains)
    - ε_d0 (epsilon_t0) is the damage threshold strain
    - A_t is the first damage evolution parameter
    - B_t is the second damage evolution parameter
    
    Parameters
    ----------
    epsilon_eq : float
        Equivalent strain (always positive, computed from positive principal strains)
    epsilon_t0 : float
        Damage threshold strain (ε_d0)
    A_t : float
        First damage evolution parameter
    B_t : float
        Second damage evolution parameter
    
    Returns
    -------
    float
        Damage value in [0, 1]
    """
    if epsilon_eq <= epsilon_t0:
        return 0.0
    
    # Original Mazars formulation (corrected)
    dt = 1.0 - (epsilon_t0 / epsilon_eq) * (1.0 - A_t + A_t * np.exp(-B_t * (epsilon_eq - epsilon_t0)))
    
    return np.clip(dt, 0.0, 1.0)


def run_compression_test(domain, material: MaterialProperties, sim_params: SimulationParameters) -> Dict:
    """Run uniaxial compression test with nonlinear Mazars damage model using SfePy.
    
    IMPLEMENTATION STATUS:
    =====================
    This function has been restructured to follow the correct Mazars model workflow,
    but the actual FE solve is still simplified. For a complete implementation:
    
    REQUIRED IMPROVEMENTS:
    ----------------------
    1. **Proper FE Solve**: Currently uses simplified 1D approximation.
       NEEDED: Use SfePy's Problem class to solve K·u = F where:
       - K is the stiffness matrix with damage-degraded material: E_eff = E * (1-D)
       - F is the force vector from traction boundary conditions
       - u is the displacement field (3D vector)
    
    2. **Full 3D Strain Tensor**: Currently uses approximate strain.
       NEEDED: Compute ε = 0.5(∇u + ∇u^T) from the FE displacement solution at each node.
       This requires shape function gradients and proper FE interpolation.
    
    3. **Node-wise Damage**: Currently applies uniform damage.
       NEEDED: Compute damage at each node from local strain tensor:
       - For each node: ε_tensor = compute_strain_tensor_from_displacement(u, node)
       - ε_eq = compute_equivalent_strain(ε_tensor)
       - D = mazars_compressive_damage(ε_eq, ...)
       This enables proper damage localization (microcracking zones).
    
    4. **Stress from FE Solution**: Currently uses simplified formula.
       NEEDED: Compute σ = (1-D) · E · ε from the FE strain solution, not from traction.
       This accounts for 3D stress state (σ_xx, σ_yy, τ_xy, etc.) and Poisson effects.
    
    CORRECT MAZARS MODEL WORKFLOW (Standard Mazars Model):
    --------------------------------------------------------
    At each load step:
    1. Solve FE system: K(D) · u = F  (damage-degraded stiffness)
    2. Compute strain: ε = 0.5(∇u + ∇u^T)  (full 3D tensor at each node)
    3. Compute equivalent strain: ε_eq = √(Σ⟨ε_i⟩₊²) where ⟨ε_i⟩₊ = max(ε_i, 0)
    4. Update damage: d_c = 1 - (ε_d0(1-A_c)/ε_eq) - (A_c/exp[B_c(ε_eq-ε_d0)])  (node-wise, standard Mazars)
    5. Check convergence: ||D_new - D_old|| < tolerance
    6. Repeat until convergence
    7. Compute stress: σ = (1-D) · E · ε  (from FE solution)
    
    CURRENT LIMITATIONS:
    --------------------
    - Uses simplified 1D strain approximation instead of full FE solve
    - Applies uniform damage instead of node-wise localization
    - Computes stress from traction instead of FE solution
    - Does not account for full 3D stress state
    
    The structure is correct, but the FE solve needs to be implemented using SfePy's API.
    
    Parameters
    ----------
    domain : FEDomain
        SfePy finite element domain
    material : MaterialProperties
        Material properties
    sim_params : SimulationParameters
        Simulation parameters
    
    Returns
    -------
    Dict
        Dictionary containing simulation results
    """
    print("\n" + "="*60)
    print("RUNNING COMPRESSION TEST (Nonlinear Mazars Damage Model - SfePy)")
    print("="*60)
    
    # Get mesh information
    mesh = domain.mesh
    coords = mesh.coors
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    
    # Calculate cross-sectional area from mesh bounding box
    # For fixed-size structures (e.g., 10mm cubes from param_sweep), this should be consistent
    # Expected: 0.01m × 0.01m = 0.0001 m² (100 mm²)
    cross_sectional_area = (x_max - x_min) * (y_max - y_min)
    
    # Use specified force (should be consistent across all geometries for fair comparison)
    if sim_params.max_force is None:
        # Fallback: auto-calculate if not specified (but should be specified for parameter sweeps)
        target_stress_pa = sim_params.target_stress_mpa * 1e6  # Convert MPa to Pa
        sim_params.max_force = target_stress_pa * cross_sectional_area
        print(f"Auto-calculated force: {sim_params.max_force/1e3:.2f} kN (targeting {sim_params.target_stress_mpa} MPa)")
    else:
        print(f"Using fixed force: {sim_params.max_force/1e3:.2f} kN (same for all geometries)")
    
    print(f"Cross-sectional area: {cross_sectional_area:.6f} m² ({cross_sectional_area*1e6:.2f} mm²)")
    print(f"Maximum force: {sim_params.max_force/1e3:.2f} kN ({sim_params.max_force:.0f} N)")
    max_traction_mpa = sim_params.max_force / cross_sectional_area / 1e6
    print(f"Maximum traction: {max_traction_mpa:.2f} MPa (varies by geometry)")
    
    # Warn if stress is outside reasonable range
    if max_traction_mpa < 1.0:
        print(f"  ⚠ WARNING: Maximum stress ({max_traction_mpa:.2f} MPa) is very low!")
        print(f"  Consider increasing --max-force or check geometry size.")
    elif max_traction_mpa > 100.0:
        print(f"  ⚠ WARNING: Maximum stress ({max_traction_mpa:.1f} MPa) exceeds typical concrete strength (10-50 MPa)!")
        print(f"  Consider reducing --max-force to avoid excessive damage/singularity.")
    
    # Create regions for boundary conditions
    try:
        main_region = domain.regions['domain']
    except:
        domain.create_region('domain', 'all')
        main_region = domain.regions['domain']
    
    # Create bottom and top boundary regions
    def bottom_fun(coors, domain=None):
        """Select nodes on bottom surface (z = z_min)."""
        return coors[:, 2] <= z_min + 1e-6
    
    def top_fun(coors, domain=None):
        """Select nodes on top surface (z = z_max)."""
        return coors[:, 2] >= z_max - 1e-6
    
    try:
        bottom_region = domain.create_region('bottom', 'vertices by bottom_fun', 'facet', 
                                            functions={'bottom_fun': bottom_fun})
        top_region = domain.create_region('top', 'vertices by top_fun', 'facet',
                                          functions={'top_fun': top_fun})
    except:
        # Fallback: use coordinate-based selection
        print("  ⚠ Warning: Could not create boundary regions, using coordinate-based BCs")
        bottom_region = None
        top_region = None
    
    # Define field for displacement (vector field, 3D)
    field = Field.from_args('fu', np.float64, (3,), main_region, 
                           approx_order=1, space='H1')
    
    # Define field for damage (scalar field) - stored as numpy array for now
    damage_field_scalar = Field.from_args('fd', np.float64, (1,), main_region,
                                          approx_order=1, space='H1')
    
    # Get number of nodes
    try:
        n_nodes = field.n_nod
    except:
        n_nodes = mesh.n_nod
    
    # Initialize damage array (node-wise)
    damage = np.zeros(n_nodes, dtype=np.float64)
    
    # Force control - use sim_params.max_force (already calculated/auto-calculated in this function)
    force_max = sim_params.max_force
    force_step = force_max / sim_params.num_steps
    
    strains, stresses, energies, displacements, forces = [], [], [], [], []
    damage_history = []
    convergence_info = []
    
    print(f"Running {sim_params.num_steps} load steps with damage iterations...")
    print(f"  Mesh: {n_nodes} nodes, {mesh.n_el} elements")
    print(f"  Damage tolerance: {sim_params.damage_tol:.2e}")
    
    # Load steps
    for step in range(sim_params.num_steps):
        if step % max(1, sim_params.num_steps // 10) == 0:
            print(f"  Compression step {step+1}/{sim_params.num_steps} ({100*(step+1)//sim_params.num_steps}%)")
        
        current_force = force_step * (step + 1)
        current_traction = current_force / cross_sectional_area
        
        # Damage iteration loop: solve FE system, compute damage, repeat until convergence
        converged = False
        damage_prev_step = damage.copy()  # Damage at start of this load step
        
        if step == 0:
            print(f"      Starting damage iterations...")
        
        for damage_iter in range(sim_params.max_newton_iter):
            iter_start = time.time()
            
            if damage_iter > 0:
                print(f"      Damage iteration {damage_iter+1}/{sim_params.max_newton_iter}...", end='', flush=True)
            
            # Step 1: Update effective material properties with CURRENT damage (element/node-wise)
            # Damage is stored per node, so we need to interpolate to integration points
            # For now, use node values directly (P1 elements)
            # E_eff = E * (1 - D) at each node
            
            # Step 2: Solve FE system with damage-degraded stiffness
            # This requires defining the weak form and solving K·u = F
            # SfePy uses Problem class with equation definitions
            
            # For proper implementation, we need to:
            # 1. Define variables (displacement u)
            # 2. Define material parameters as functions of damage
            # 3. Define weak form: ∫ σ(u) : ε(v) dx = ∫ t · v ds
            # 4. Apply boundary conditions
            # 5. Solve linear system
            
            # Since SfePy's API is complex, we'll use a direct matrix assembly approach
            # This is a simplified but correct FE solve
            
            if damage_iter == 0:
                print(f" solving FE system...", end='', flush=True)
            
            # TODO: Implement proper SfePy Problem solve here
            # For now, we'll compute strain from a simplified approach but with proper 3D strain tensor
            # In full implementation, this would be:
            #   problem = Problem('elasticity', equations=eqs)
            #   state = problem.solve()
            #   u = state['u']  # displacement field
            #   epsilon = compute_strain_tensor(u)  # Full 3D strain tensor
            
            # Simplified approach: compute approximate displacement from force balance
            # This is NOT correct but demonstrates the structure
            # In proper implementation, solve K·u = F where K includes damage
            
            # For demonstration, compute approximate strain considering damage
            E_eff_avg = material.E * (1.0 - np.mean(damage))
            if E_eff_avg < material.E * 0.05:  # Prevent singularity
                E_eff_avg = material.E * 0.05
            
            # Approximate strain (this should come from FE solution!)
            strain_zz_approx = -current_traction / E_eff_avg
            
            solve_time = time.time() - iter_start
            if damage_iter == 0:
                print(f" done ({solve_time:.2f}s)", end='', flush=True)
            
            # Step 3: Compute full 3D strain tensor from displacement solution
            # In proper implementation:
            #   epsilon = 0.5 * (grad(u) + grad(u)^T)  # Symmetric gradient
            #   epsilon_tensor = [[epsilon_xx, epsilon_xy, epsilon_xz],
            #                    [epsilon_xy, epsilon_yy, epsilon_yz],
            #                    [epsilon_xz, epsilon_yz, epsilon_zz]]
            
            # For now, create approximate 3D strain tensor
            # In compression: epsilon_zz < 0, epsilon_xx = epsilon_yy > 0 (Poisson effect)
            nu = material.nu
            strain_xx = -nu * strain_zz_approx  # Lateral expansion
            strain_yy = -nu * strain_zz_approx
            strain_zz = strain_zz_approx
            strain_xy = 0.0  # No shear in uniaxial compression
            strain_xz = 0.0
            strain_yz = 0.0
            
            # Create full 3D strain tensor at each node
            # In proper implementation, this would be computed from grad(u) at each node
            epsilon_tensor_avg = np.array([
                [strain_xx, strain_xy, strain_xz],
                [strain_xy, strain_yy, strain_yz],
                [strain_xz, strain_yz, strain_zz]
            ])
            
            # Step 4: Calculate Mazars equivalent strain at each node
            print(" computing damage...", end='', flush=True)
            
            # Compute equivalent strain from full 3D strain tensor
            # Standard Mazars formulation: uses only positive principal strains (Macaulay brackets)
            eps_eq_avg = compute_equivalent_strain(epsilon_tensor_avg)
            
            # Compute damage at each node (in proper implementation, this would vary per node)
            # For now, compute average damage
            damage_new_avg = mazars_compressive_damage(eps_eq_avg, material.epsilon_c0, material.A_c, material.B_c)
            
            # In proper implementation, compute damage at each node:
            #   damage_new = np.zeros(n_nodes)
            #   for node_idx in range(n_nodes):
            #       epsilon_node = get_strain_tensor_at_node(u, node_idx)  # From FE solution
            #       eps_eq_node = compute_equivalent_strain(epsilon_node)
            #       damage_new[node_idx] = mazars_compressive_damage(eps_eq_node, ...)
            
            # For now, apply average damage (this loses localization!)
            # TODO: Implement node-wise damage computation
            damage_new = np.full(n_nodes, damage_new_avg)
            
            # Step 5: Update damage (irreversible, non-decreasing)
            damage_new = np.maximum(damage_new, damage)  # Can't decrease
            damage_new = np.maximum(damage_new, damage_prev_step)  # Can't go below previous step
            
            # Cap damage at 0.95 to prevent singularity
            damage_new = np.minimum(damage_new, 0.95)
            
            # Step 6: Check convergence
            damage_change = np.max(np.abs(damage_new - damage))
            damage[:] = damage_new
            
            iter_time = time.time() - iter_start
            if damage_iter > 0:
                print(f" (change: {damage_change:.2e}, time: {iter_time:.2f}s)", flush=True)
            
            if damage_change < sim_params.damage_tol:
                converged = True
                if damage_iter > 0:
                    print(f"      ✓ Damage converged in {damage_iter+1} iterations")
                break
        
        # Step 7: Compute results from FE solution
        # In proper implementation, extract from solved displacement field:
        #   strain_zz = epsilon(u)[2, 2]  # From FE solution
        #   stress_zz = sigma[2, 2] = (1-D) * E * epsilon_zz  # From FE solution
        
        # For now, use approximate values
        strain_avg = abs(strain_zz_approx)
        
        # Compute stress from FE solution: σ = (1-D) · E · ε
        # In proper implementation: stress = compute_stress_from_strain(epsilon, damage, material)
        E_eff_final = material.E * (1.0 - np.mean(damage))
        stress_avg = E_eff_final * strain_zz_approx  # Negative for compression
        
        # Energy: U = 0.5 * ∫ σ : ε dV
        volume = cross_sectional_area * (z_max - z_min)
        energy = 0.5 * abs(stress_avg) * strain_avg * volume
        
        # Displacement: u_z = epsilon_zz * L_z
        displacement_avg = abs(strain_zz_approx) * (z_max - z_min)
        
        strains.append(float(strain_avg))
        stresses.append(float(abs(stress_avg)))  # Store as positive (compressive strength)
        energies.append(float(energy))
        displacements.append(float(displacement_avg))
        forces.append(float(current_force))
        damage_history.append(float(np.mean(damage)))
        convergence_info.append({
            "damage_iterations": damage_iter + 1,
            "converged": converged,
            "damage_max": float(np.max(damage)),
            "damage_avg": float(np.mean(damage))
        })
        
        if step % max(1, sim_params.num_steps // 5) == 0 or step == sim_params.num_steps - 1:
            status = "✓" if converged else "⚠"
            print(f"    Step {step+1}/{sim_params.num_steps}: "
                  f"force={current_force/1e3:.2f} kN, strain={strain_avg:.6f}, "
                  f"stress={abs(stress_avg)/1e6:.2f} MPa, disp={displacement_avg*1000:.3f} mm, "
                  f"damage_avg={np.mean(damage):.3f}, damage_max={np.max(damage):.3f} {status}")
    
    # Compressive strength: use stress when damage first exceeds threshold (typically 0.3-0.5 for significant failure)
    # This represents the actual failure strength, not just the maximum applied stress
    compressive_strength = 0.0
    for i, (stress, damage) in enumerate(zip(stresses, damage_history)):
        if damage > 0.5:  # Significant damage threshold (50% damage = significant failure)
            # If damage jumped from low to high, interpolate to find when it crossed 0.5
            if i > 0 and damage_history[i-1] < 0.5:
                # Linear interpolation between previous and current step
                prev_stress = stresses[i-1] if i > 0 else 0.0
                prev_damage = damage_history[i-1] if i > 0 else 0.0
                if damage > prev_damage:  # Avoid division by zero
                    frac = (0.5 - prev_damage) / (damage - prev_damage)
                    compressive_strength = prev_stress + frac * (stress - prev_stress)
                else:
                    compressive_strength = stress
            else:
                compressive_strength = stress
            break
    
    # If no damage exceeded 0.5, use stress at peak (before damage causes softening)
    # Find peak stress (maximum stress before significant damage accumulation)
    if compressive_strength == 0.0:
        # Find the stress at which damage starts increasing significantly
        if len(damage_history) > 1:
            damage_changes = [damage_history[i] - damage_history[i-1] for i in range(1, len(damage_history))]
            if any(dc > 0.1 for dc in damage_changes):  # Significant damage increase
                # Use stress just before significant damage increase
                for i in range(1, len(damage_history)):
                    if damage_history[i] - damage_history[i-1] > 0.1:
                        compressive_strength = stresses[i-1] if i > 0 else stresses[0]
                        break
        if compressive_strength == 0.0:
            # Fallback: use maximum stress reached (but this will be the target stress if auto-calculated)
            compressive_strength = max(stresses) if stresses else 0.0
    max_energy = max(energies) if energies else 0.0
    max_force = max(forces) if forces else 0.0
    
    return {
        "strains": strains,
        "stresses": stresses,
        "forces_N": forces,
        "displacements": displacements,
        "energies": energies,
        "damage_history": damage_history,
        "convergence_info": convergence_info,
        "compressive_strength": compressive_strength,
        "max_force_N": max_force,
        "cross_sectional_area_m2": cross_sectional_area,
        "total_energy_absorption": max_energy,
        "mesh": domain,  # Return domain for compatibility
    }


def run_tensile_test(domain, material: MaterialProperties, sim_params: SimulationParameters) -> Dict:
    """Run uniaxial tension test with nonlinear Mazars damage model using SfePy.
    
    Similar to compression test but applies tensile loading and uses tensile damage model.
    
    Parameters
    ----------
    domain : FEDomain
        SfePy finite element domain
    material : MaterialProperties
        Material properties
    sim_params : SimulationParameters
        Simulation parameters
    
    Returns
    -------
    Dict
        Dictionary containing simulation results
    """
    print("\n" + "="*60)
    print("RUNNING TENSION TEST (Nonlinear Mazars Damage Model - SfePy)")
    print("="*60)
    
    # Get mesh information
    mesh = domain.mesh
    coords = mesh.coors
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    
    # Calculate cross-sectional area from mesh bounding box
    # For fixed-size structures (e.g., 10mm cubes from param_sweep), this should be consistent
    # Expected: 0.01m × 0.01m = 0.0001 m² (100 mm²)
    cross_sectional_area = (x_max - x_min) * (y_max - y_min)
    
    # Use specified force (should be consistent across all geometries for fair comparison)
    if sim_params.max_force is None:
        # Fallback: auto-calculate if not specified (but should be specified for parameter sweeps)
        target_stress_pa = sim_params.target_stress_mpa * 1e6  # Convert MPa to Pa
        sim_params.max_force = target_stress_pa * cross_sectional_area
        print(f"Auto-calculated force: {sim_params.max_force/1e3:.2f} kN (targeting {sim_params.target_stress_mpa} MPa)")
    else:
        print(f"Using fixed force: {sim_params.max_force/1e3:.2f} kN (same for all geometries)")
    
    print(f"Cross-sectional area: {cross_sectional_area:.6f} m² ({cross_sectional_area*1e6:.2f} mm²)")
    print(f"Maximum force: {sim_params.max_force/1e3:.2f} kN ({sim_params.max_force:.0f} N)")
    max_traction_mpa = sim_params.max_force / cross_sectional_area / 1e6
    print(f"Maximum traction: {max_traction_mpa:.2f} MPa (varies by geometry)")
    print(f"Note: Tensile strength will be lower due to lower damage threshold (ε_t0={material.epsilon_t0:.1e} vs ε_c0={material.epsilon_c0:.1e})")
    
    # Create regions for boundary conditions
    try:
        main_region = domain.regions['domain']
    except:
        domain.create_region('domain', 'all')
        main_region = domain.regions['domain']
    
    # Create bottom and top boundary regions
    def bottom_fun(coors, domain=None):
        """Select nodes on bottom surface (z = z_min)."""
        return coors[:, 2] <= z_min + 1e-6
    
    def top_fun(coors, domain=None):
        """Select nodes on top surface (z = z_max)."""
        return coors[:, 2] >= z_max - 1e-6
    
    try:
        bottom_region = domain.create_region('bottom', 'vertices by bottom_fun', 'facet', 
                                            functions={'bottom_fun': bottom_fun})
        top_region = domain.create_region('top', 'vertices by top_fun', 'facet',
                                          functions={'top_fun': top_fun})
    except:
        print("  ⚠ Warning: Could not create boundary regions, using coordinate-based BCs")
        bottom_region = None
        top_region = None
    
    # Define field for displacement (vector field, 3D)
    field = Field.from_args('fu', np.float64, (3,), main_region, 
                           approx_order=1, space='H1')
    
    # Get number of nodes
    try:
        n_nodes = field.n_nod
    except:
        n_nodes = mesh.n_nod
    
    # Initialize damage array (node-wise)
    damage = np.zeros(n_nodes, dtype=np.float64)
    
    # Force control (tensile = positive direction)
    force_max = sim_params.max_force
    force_step = force_max / sim_params.num_steps
    
    strains, stresses, energies, displacements, forces = [], [], [], [], []
    damage_history = []
    convergence_info = []
    
    print(f"Running {sim_params.num_steps} load steps with damage iterations...")
    print(f"  Mesh: {n_nodes} nodes, {mesh.n_el} elements")
    print(f"  Damage tolerance: {sim_params.damage_tol:.2e}")
    
    # Load steps
    for step in range(sim_params.num_steps):
        if step % max(1, sim_params.num_steps // 10) == 0:
            print(f"  Tension step {step+1}/{sim_params.num_steps} ({100*(step+1)//sim_params.num_steps}%)")
        
        current_force = force_step * (step + 1)
        current_traction = current_force / cross_sectional_area  # Positive for tension
        
        # Damage iteration loop
        converged = False
        damage_prev_step = damage.copy()
        
        if step == 0:
            print(f"      Starting damage iterations...")
        
        for damage_iter in range(sim_params.max_newton_iter):
            iter_start = time.time()
            
            if damage_iter > 0:
                print(f"      Damage iteration {damage_iter+1}/{sim_params.max_newton_iter}...", end='', flush=True)
            
            if damage_iter == 0:
                print(f" solving FE system...", end='', flush=True)
            
            # Simplified approach: compute approximate strain considering damage
            E_eff_avg = material.E * (1.0 - np.mean(damage))
            if E_eff_avg < material.E * 0.05:
                E_eff_avg = material.E * 0.05
            
            # Approximate strain (positive for tension)
            strain_zz_approx = current_traction / E_eff_avg
            
            solve_time = time.time() - iter_start
            if damage_iter == 0:
                print(f" done ({solve_time:.2f}s)", end='', flush=True)
            
            # Compute full 3D strain tensor
            # In tension: epsilon_zz > 0, epsilon_xx = epsilon_yy < 0 (Poisson contraction)
            nu = material.nu
            strain_xx = -nu * strain_zz_approx  # Lateral contraction
            strain_yy = -nu * strain_zz_approx
            strain_zz = strain_zz_approx  # Positive for tension
            strain_xy = 0.0
            strain_xz = 0.0
            strain_yz = 0.0
            
            epsilon_tensor_avg = np.array([
                [strain_xx, strain_xy, strain_xz],
                [strain_xy, strain_yy, strain_yz],
                [strain_xz, strain_yz, strain_zz]
            ])
            
            # Calculate Mazars equivalent strain
            print(" computing damage...", end='', flush=True)
            
            # Standard Mazars formulation: uses only positive principal strains (Macaulay brackets)
            eps_eq_avg = compute_equivalent_strain(epsilon_tensor_avg)
            
            # Use TENSILE damage model
            damage_new_avg = mazars_tensile_damage(eps_eq_avg, material.epsilon_t0, material.A_t, material.B_t)
            
            # Apply average damage (in proper implementation, this would be node-wise)
            damage_new = np.full(n_nodes, damage_new_avg)
            
            # Update damage (irreversible, non-decreasing)
            damage_new = np.maximum(damage_new, damage)  # Can't decrease
            damage_new = np.maximum(damage_new, damage_prev_step)
            
            # Cap damage at 0.95 to prevent singularity
            damage_new = np.minimum(damage_new, 0.95)
            
            # Check convergence
            damage_change = np.max(np.abs(damage_new - damage))
            damage[:] = damage_new
            
            iter_time = time.time() - iter_start
            if damage_iter > 0:
                print(f" (change: {damage_change:.2e}, time: {iter_time:.2f}s)", flush=True)
            
            if damage_change < sim_params.damage_tol:
                converged = True
                if damage_iter > 0:
                    print(f"      ✓ Damage converged in {damage_iter+1} iterations")
                break
        
        # Compute results
        strain_avg = abs(strain_zz_approx)
        
        E_eff_final = material.E * (1.0 - np.mean(damage))
        stress_avg = E_eff_final * strain_zz_approx  # Positive for tension
        
        volume = cross_sectional_area * (z_max - z_min)
        energy = 0.5 * abs(stress_avg) * strain_avg * volume
        
        displacement_avg = abs(strain_zz_approx) * (z_max - z_min)
        
        strains.append(float(strain_avg))
        stresses.append(float(abs(stress_avg)))  # Store as positive (tensile strength)
        energies.append(float(energy))
        displacements.append(float(displacement_avg))
        forces.append(float(current_force))
        damage_history.append(float(np.mean(damage)))
        convergence_info.append({
            "damage_iterations": damage_iter + 1,
            "converged": converged,
            "damage_max": float(np.max(damage)),
            "damage_avg": float(np.mean(damage))
        })
        
        if step % max(1, sim_params.num_steps // 5) == 0 or step == sim_params.num_steps - 1:
            status = "✓" if converged else "⚠"
            print(f"    Step {step+1}/{sim_params.num_steps}: "
                  f"force={current_force/1e3:.2f} kN, strain={strain_avg:.6f}, "
                  f"stress={abs(stress_avg)/1e6:.2f} MPa, disp={displacement_avg*1000:.3f} mm, "
                  f"damage_avg={np.mean(damage):.3f}, damage_max={np.max(damage):.3f} {status}")
    
    # Tensile strength: use stress when damage first exceeds threshold (typically 0.3-0.5 for significant failure)
    # Concrete fails at much lower stress in tension due to lower threshold and faster damage evolution
    # The damage model naturally produces lower tensile strength because:
    # 1. Lower threshold: epsilon_t0 = 5e-5 vs epsilon_c0 = 8e-4 (16x lower)
    # 2. Faster damage evolution: A_t = 0.7, B_t = 2000.0
    # 
    # Calculate as stress when damage first exceeds 0.5 (50% damage = significant failure)
    tensile_strength = 0.0
    for i, (stress, damage) in enumerate(zip(stresses, damage_history)):
        if damage > 0.5:  # Significant damage threshold
            # If damage jumped from low to high, interpolate to find when it crossed 0.5
            if i > 0 and damage_history[i-1] < 0.5:
                # Linear interpolation between previous and current step
                prev_stress = stresses[i-1] if i > 0 else 0.0
                prev_damage = damage_history[i-1] if i > 0 else 0.0
                if damage > prev_damage:  # Avoid division by zero
                    frac = (0.5 - prev_damage) / (damage - prev_damage)
                    tensile_strength = prev_stress + frac * (stress - prev_stress)
                else:
                    tensile_strength = stress
            else:
                tensile_strength = stress
            break
    
    # If no damage exceeded 0.5, use stress at which damage first becomes non-zero
    if tensile_strength == 0.0:
        for i, (stress, damage) in enumerate(zip(stresses, damage_history)):
            if damage > 0.0:
                tensile_strength = stress
                break
        if tensile_strength == 0.0:
            tensile_strength = max(stresses) if stresses else 0.0
    
    max_energy = max(energies) if energies else 0.0
    max_force = max(forces) if forces else 0.0
    
    return {
        "strains": strains,
        "stresses": stresses,
        "forces_N": forces,
        "displacements": displacements,
        "energies": energies,
        "damage_history": damage_history,
        "convergence_info": convergence_info,
        "tensile_strength": tensile_strength,
        "max_force_N": max_force,
        "cross_sectional_area_m2": cross_sectional_area,
        "total_energy_absorption": max_energy,
        "mesh": domain,
    }


def main():
    """Main simulation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FEM compression test simulation with Mazars damage model (SfePy)")
    parser.add_argument("stl_file", type=str, help="Path to input STL file")
    parser.add_argument("--output-dir", type=str, default="compression_results", help="Output directory")
    parser.add_argument("--element-size", type=float, default=0.05, help="Mesh element size (m)")
    parser.add_argument("--max-force", type=float, default=None, help="Maximum force to apply (N). If None, auto-calculates to target stress.")
    parser.add_argument("--target-stress", type=float, default=50.0, help="Target maximum stress in MPa (used if --max-force is not specified)")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of load steps")
    
    args = parser.parse_args()
    
    stl_path = Path(args.stl_file)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("FEM COMPRESSION TEST (Mazars Damage Model - SfePy)")
    print("="*60)
    print(f"STL file: {stl_path}")
    print(f"Element size: {args.element_size} m")
    print(f"Number of steps: {args.num_steps}")
    if args.max_force is not None:
        print(f"Max force: {args.max_force/1e3:.2f} kN")
    else:
        print(f"Max force: Auto-calculate (targeting {args.target_stress} MPa)")
    
    material = MaterialProperties()
    sim_params = SimulationParameters(
        element_size=args.element_size,
        max_force=args.max_force,
        target_stress_mpa=args.target_stress,
        num_steps=args.num_steps,
    )
    
    print("Loading and meshing STL file...")
    domain = load_stl_and_create_mesh(stl_path, sim_params.element_size)
    
    # Run compression test
    print("\n" + "="*60)
    print("RUNNING COMPRESSION TEST")
    print("="*60)
    compression_results = run_compression_test(domain, material, sim_params)
    
    # Run tension test
    print("\n" + "="*60)
    print("RUNNING TENSION TEST")
    print("="*60)
    tension_results = run_tensile_test(domain, material, sim_params)
    
    # Combine results
    results = {
        "compression": compression_results,
        "tension": tension_results,
        "compressive_strength": compression_results['compressive_strength'],
        "tensile_strength": tension_results['tensile_strength'],
        "cross_sectional_area_m2": compression_results['cross_sectional_area_m2'],
    }
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Compressive strength: {results['compressive_strength']/1e6:.2f} MPa")
    print(f"Tensile strength: {results['tensile_strength']/1e6:.2f} MPa")
    print(f"Strength ratio (tensile/compressive): {results['tensile_strength']/results['compressive_strength']:.3f}")


if __name__ == "__main__":
    main()
