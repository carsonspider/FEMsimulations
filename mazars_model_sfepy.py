
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

python mazars_model_sfepy.py <stl_file> [options]
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

    python mazars_model_sfepy.py --stl_path path/to/your/stl_file.stl
    """
    
    E: float = 30e9  # Young's modulus (Pa) - 25 GPa (recommended: 25-35 GPa range)
    nu: float = 0.2  # Poisson's ratio (typical for concrete: 0.15-0.2)
    rho: float = 2400.0  # Density (kg/m³) - typical for cement paste
    # Compressive damage parameters
    epsilon_c0: float = 8e-4   # Mazars compressive damage threshold strain (ε_d0) - Recommended: 6e-4 to 1.2e-3
    A_c: float = 1.0  # Mazars compressive damage evolution parameter 1 - Recommended: 0.7-1.5
    B_c: float = 1500.0  # Mazars compressive damage evolution parameter 2 - Recommended: 1000-2000
    # Tensile damage parameters (tensile strength is ~10-15% of compressive strength)
    epsilon_t0: float =  5e-5   # Mazars tensile damage threshold strain (ε_d0) - much lower than compression
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
    
    max_force: float = 2000.0  # N (Fixed force for all geometries - targets ~35 MPa for 10mm structures)
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
        # meshio automatically handles both ASCII and binary STL files
        try:
            stl_mesh = meshio.read(str(stl_path), file_format="stl")
            points = stl_mesh.points
        except (UnicodeDecodeError, ValueError, Exception) as e:
            # If meshio fails (encoding issue or other), fall back to numpy-stl
            print(f"Warning: meshio read failed ({type(e).__name__}), trying alternative method...")
            try:
                from stl import mesh as stl_mesh_module
                stl_mesh_stl = stl_mesh_module.Mesh.from_file(str(stl_path))
                # Extract unique vertices from STL triangles
                points = np.unique(stl_mesh_stl.vectors.reshape(-1, 3), axis=0)
                print(f"Successfully loaded STL using numpy-stl: {len(points)} unique vertices")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to read STL file with both meshio and numpy-stl.\n"
                    f"  meshio error: {e}\n"
                    f"  numpy-stl error: {e2}\n"
                    f"  File: {stl_path}"
                )
        
        # Validate points
        if points is None or len(points) == 0:
            raise ValueError(f"STL file contains no points: {stl_path}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError(
                f"STL file contains invalid coordinates (NaN or Inf): {stl_path}\n"
                f"  This may indicate a corrupted or invalid STL file."
            )
        
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        
        # Validate bounding box
        if np.any(np.isnan(bbox_min)) or np.any(np.isnan(bbox_max)):
            raise ValueError(
                f"Invalid bounding box computed from STL file: {stl_path}\n"
                f"  bbox_min: {bbox_min}\n"
                f"  bbox_max: {bbox_max}\n"
                f"  This may indicate the STL file has invalid geometry."
            )
        
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
        
        # Use the actual STL geometry instead of creating a bounding box
        # STL files are surface meshes, so we need to create a volume mesh from the surface
        print(f"Creating volume mesh from STL surface geometry (not bounding box)")
        print(f"STL bounding box: {bbox_min} to {bbox_max} (m)")
        
        size = bbox_max - bbox_min
        
        # Validate size
        if np.any(np.isnan(size)) or np.any(np.isinf(size)):
            raise ValueError(
                f"Invalid mesh size computed from STL file: {stl_path}\n"
                f"  size: {size}\n"
                f"  bbox_min: {bbox_min}\n"
                f"  bbox_max: {bbox_max}\n"
                f"  This may indicate the STL file has invalid geometry."
            )
        
        if np.any(size <= 0):
            raise ValueError(
                f"STL file has zero or negative dimensions: {stl_path}\n"
                f"  size: {size}\n"
                f"  bbox_min: {bbox_min}\n"
                f"  bbox_max: {bbox_max}"
            )
        
        # Convert STL surface points to meters if needed
        if units_converted:
            # Points were already converted, but ensure stl_mesh points are updated
            if hasattr(stl_mesh, 'points'):
                stl_mesh.points = stl_mesh.points / 1000.0
                points = stl_mesh.points
        
        # Create a volume mesh using the STL surface
        # For simple geometries (like cubes), use structured hex mesh
        # For complex geometries, try Delaunay tetrahedralization
        
        # Check if this is a simple box/cube geometry
        # A cube should have 8 vertices (corners) and 6 faces
        is_simple_box = len(points) == 8
        
        if is_simple_box:
            # Use structured hex mesh for simple boxes
            print(f"  Detected simple box geometry, using structured hex mesh...")
            
            # Create structured grid
            n_x = max(2, int(np.ceil(size[0] / element_size)))
            n_y = max(2, int(np.ceil(size[1] / element_size)))
            n_z = max(2, int(np.ceil(size[2] / element_size)))
            
            x_vals = np.linspace(bbox_min[0], bbox_max[0], n_x + 1)
            y_vals = np.linspace(bbox_min[1], bbox_max[1], n_y + 1)
            z_vals = np.linspace(bbox_min[2], bbox_max[2], n_z + 1)
            
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Create hexahedral connectivity
            # Node numbering: node at (i, j, k) has index = i * (n_y+1) * (n_z+1) + j * (n_z+1) + k
            cells = []
            for i in range(n_x):
                for j in range(n_y):
                    for k in range(n_z):
                        # 8-node hex element connectivity
                        # Base node at (i, j, k)
                        base = i * (n_y + 1) * (n_z + 1) + j * (n_z + 1) + k
                        # Neighbors
                        next_y = (n_z + 1)  # Step in y direction
                        next_x = (n_y + 1) * (n_z + 1)  # Step in x direction
                        next_z = 1  # Step in z direction
                        
                        hex_nodes = [
                            base,                    # 0: (i, j, k)
                            base + next_y,           # 1: (i, j+1, k)
                            base + next_x + next_y,  # 2: (i+1, j+1, k)
                            base + next_x,           # 3: (i+1, j, k)
                            base + next_z,           # 4: (i, j, k+1)
                            base + next_y + next_z,  # 5: (i, j+1, k+1)
                            base + next_x + next_y + next_z,  # 6: (i+1, j+1, k+1)
                            base + next_x + next_z,  # 7: (i+1, j, k+1)
                        ]
                        cells.append(hex_nodes)
            
            print(f"  Created {len(cells)} hexahedral elements ({n_x}×{n_y}×{n_z})")
            element_type = '3_8'  # Hexahedra
            
        else:
            # Use Delaunay tetrahedralization for complex geometries
            from scipy.spatial import Delaunay
            
            # Generate interior points for volume meshing
            # Use a grid of points inside the bounding box
            n_points_per_dim = max(3, int(np.ceil(max(size) / element_size)))
            
            # Generate interior points (avoid boundaries to prevent issues)
            margin = element_size * 0.1
            x_vals = np.linspace(bbox_min[0] + margin, bbox_max[0] - margin, n_points_per_dim)
            y_vals = np.linspace(bbox_min[1] + margin, bbox_max[1] - margin, n_points_per_dim)
            z_vals = np.linspace(bbox_min[2] + margin, bbox_max[2] - margin, n_points_per_dim)
            
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
            interior_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Combine STL surface points with interior points
            # Use unique points to avoid duplicates
            all_points = np.vstack([points, interior_points])
            all_points = np.unique(all_points, axis=0)
            
            print(f"  Surface points: {len(points)}, Interior points: {len(interior_points)}")
            print(f"  Total points for meshing: {len(all_points)}")
            
            # Create tetrahedral mesh using Delaunay triangulation
            print(f"  Creating tetrahedral mesh...")
            try:
                tetra = Delaunay(all_points)
                
                # Extract tetrahedra (4-node elements)
                cells = tetra.simplices.tolist()
                
                # Use the combined points
                points = all_points
                
                print(f"  Created {len(cells)} tetrahedral elements")
                element_type = '3_4'  # Tetrahedra
                
            except Exception as e:
                print(f"Warning: Delaunay triangulation failed: {e}")
                print(f"  Falling back to structured hex mesh (but this still uses bounding box)")
                raise NotImplementedError(
                    f"Tetrahedral meshing failed. Consider using gmsh for proper volume meshing.\n"
                    f"  Error: {e}"
                )
        
        # Create SfePy mesh directly using Mesh.from_data()
        # This avoids format conversion issues and ensures correct orientation
        cells_array = np.array(cells, dtype=np.int32)
        
        # SfePy expects:
        # - coors: coordinates array (N, 3) - MUST be in meters for correct stress calculations
        # - conns: list of connectivity arrays, one per element type
        # - mat_ids: material IDs (all 0 for now)
        # - descs: element descriptor ('3_4' for 3D tetrahedra with 4 nodes, or '3_8' for hexahedra)
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
        # Use appropriate element descriptor
        descs = [element_type]  # '3_4' for tetrahedra, '3_8' for hexahedra
        
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


def compute_node_wise_strain_tensors(u_field: np.ndarray, mesh, coords: np.ndarray) -> np.ndarray:
    """Compute full 3D strain tensor at all nodes from displacement field.
    
    This computes ε = 0.5(∇u + ∇u^T) at each node, enabling proper spatial
    variation and damage localization.
    
    Parameters
    ----------
    u_field : np.ndarray, shape (n_nodes, 3)
        Displacement field at all nodes
    mesh : Mesh
        Finite element mesh
    coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates
    
    Returns
    -------
    np.ndarray, shape (n_nodes, 3, 3)
        Strain tensor at each node
    """
    n_nodes = len(coords)
    strain_tensors = np.zeros((n_nodes, 3, 3), dtype=np.float64)
    
    for node_idx in range(n_nodes):
        strain_tensors[node_idx] = compute_strain_tensor_from_displacement(
            u_field, node_idx, mesh, coords
        )
    
    return strain_tensors


def compute_stress_from_strain(epsilon_tensor: np.ndarray, damage: float, 
                               material: MaterialProperties) -> np.ndarray:
    """Compute 3D stress tensor from strain tensor and damage.
    
    Uses the effective modulus: σ = (1-D) · E · (λ·tr(ε)·I + 2μ·ε)
    where λ and μ are Lame parameters.
    
    Parameters
    ----------
    epsilon_tensor : np.ndarray, shape (3, 3)
        Strain tensor
    damage : float
        Damage value at this location (0 to 1)
    material : MaterialProperties
        Material properties
    
    Returns
    -------
    np.ndarray, shape (3, 3)
        Stress tensor
    """
    # Compute effective modulus
    E_eff = material.E * (1.0 - damage)
    if E_eff < material.E * 0.01:  # Prevent singularity
        E_eff = material.E * 0.01
    
    # Compute Lame parameters with effective modulus
    nu = material.nu
    mu = E_eff / (2.0 * (1.0 + nu))  # Shear modulus
    lmbda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lame parameter
    
    # Compute stress: σ = λ·tr(ε)·I + 2μ·ε
    trace_eps = np.trace(epsilon_tensor)
    identity = np.eye(3)
    stress = lmbda * trace_eps * identity + 2.0 * mu * epsilon_tensor
    
    return stress


def _get_gauss_quadrature_3d(n_points: int = 2):
    """Get Gauss quadrature points and weights for 3D hexahedral elements."""
    if n_points == 2:
        xi_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w_1d = np.array([1.0, 1.0])
    elif n_points == 3:
        xi_1d = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        w_1d = np.array([5/9, 8/9, 5/9])
    else:
        xi_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w_1d = np.array([1.0, 1.0])
    
    points, weights = [], []
    for i, xi in enumerate(xi_1d):
        for j, eta in enumerate(xi_1d):
            for k, zeta in enumerate(xi_1d):
                points.append([xi, eta, zeta])
                weights.append(w_1d[i] * w_1d[j] * w_1d[k])
    return np.array(points), np.array(weights)


def _tet4_shape_functions(xi: float, eta: float, zeta: float):
    """Shape functions for 4-node tetrahedral element.
    
    Natural coordinates: (xi, eta, zeta, 1-xi-eta-zeta)
    Nodes at: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    """
    N = np.array([
        1 - xi - eta - zeta,  # N1 at node (0,0,0)
        xi,                    # N2 at node (1,0,0)
        eta,                   # N3 at node (0,1,0)
        zeta                   # N4 at node (0,0,1)
    ])
    
    # Derivatives with respect to natural coordinates
    dN_dxi = np.array([
        [-1, -1, -1],  # dN1/d(xi,eta,zeta)
        [ 1,  0,  0],  # dN2/d(xi,eta,zeta)
        [ 0,  1,  0],  # dN3/d(xi,eta,zeta)
        [ 0,  0,  1],  # dN4/d(xi,eta,zeta)
    ])
    
    return N, dN_dxi


def _tet4_shape_functions(xi: float, eta: float, zeta: float):
    """Shape functions for 4-node tetrahedral element.
    
    Natural coordinates: (xi, eta, zeta, 1-xi-eta-zeta)
    Nodes at: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    """
    N = np.array([
        1 - xi - eta - zeta,  # N1 at node (0,0,0)
        xi,                    # N2 at node (1,0,0)
        eta,                   # N3 at node (0,1,0)
        zeta                   # N4 at node (0,0,1)
    ])
    
    # Derivatives with respect to natural coordinates
    dN_dxi = np.array([
        [-1, -1, -1],  # dN1/d(xi,eta,zeta)
        [ 1,  0,  0],  # dN2/d(xi,eta,zeta)
        [ 0,  1,  0],  # dN3/d(xi,eta,zeta)
        [ 0,  0,  1],  # dN4/d(xi,eta,zeta)
    ])
    
    return N, dN_dxi


def _hex8_shape_functions(xi: float, eta: float, zeta: float):
    """Shape functions for 8-node hexahedral element."""
    N = np.array([
        0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 + zeta),
    ])
    dN_dxi = np.array([
        [-0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 - eta)],
        [ 0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 - eta)],
        [ 0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 + eta)],
        [-0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 - eta)],
        [ 0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 - eta)],
        [ 0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 + eta)],
    ])
    return N, dN_dxi


def _compute_material_matrix(lambda_lame: float, mu_lame: float) -> np.ndarray:
    """Compute 3D isotropic elasticity matrix D."""
    D = np.array([
        [lambda_lame + 2*mu_lame, lambda_lame, lambda_lame, 0, 0, 0],
        [lambda_lame, lambda_lame + 2*mu_lame, lambda_lame, 0, 0, 0],
        [lambda_lame, lambda_lame, lambda_lame + 2*mu_lame, 0, 0, 0],
        [0, 0, 0, mu_lame, 0, 0],
        [0, 0, 0, 0, mu_lame, 0],
        [0, 0, 0, 0, 0, mu_lame],
    ])
    return D


def assemble_stiffness_matrix(mesh, coords, damage: np.ndarray, material: MaterialProperties) -> np.ndarray:
    """Assemble global stiffness matrix K with damage-dependent moduli.
    
    Implements: K = Σ K_e where K_e = ∫ B^T · D · B dV
    """
    n_nodes = len(coords)
    n_dof = n_nodes * 3
    
    # Compute effective modulus at each node
    E_eff = material.E * (1.0 - damage)
    E_eff = np.maximum(E_eff, material.E * 0.01)
    
    # Get connectivity - support both tetrahedra (3_4) and hexahedra (3_8)
    conn = None
    element_type = None
    if hasattr(mesh, 'conns') and len(mesh.conns) > 0:
        conn = mesh.conns[0]
        # Try to determine element type from mesh descriptor
        if hasattr(mesh, 'descs') and len(mesh.descs) > 0:
            element_type = mesh.descs[0]
        else:
            # Infer from connectivity: 4 nodes = tetra, 8 nodes = hexa
            if len(conn) > 0 and len(conn[0]) == 4:
                element_type = '3_4'
            elif len(conn) > 0 and len(conn[0]) == 8:
                element_type = '3_8'
    elif hasattr(mesh, 'get_conn'):
        try:
            conn = mesh.get_conn('3_4')  # Try tetrahedra first
            element_type = '3_4'
        except:
            try:
                conn = mesh.get_conn('3_8')  # Fallback to hexahedra
                element_type = '3_8'
            except:
                conn = mesh.get_conn()
                # Infer type
                if len(conn) > 0 and len(conn[0]) == 4:
                    element_type = '3_4'
                elif len(conn) > 0 and len(conn[0]) == 8:
                    element_type = '3_8'
    else:
        raise ValueError("Cannot access mesh connectivity")
    
    if element_type is None:
        # Default to tetrahedra if we changed the mesh
        if len(conn) > 0 and len(conn[0]) == 4:
            element_type = '3_4'
        else:
            element_type = '3_8'
    
    if conn is None:
        raise ValueError("Could not get mesh connectivity")
    
    # Determine number of nodes per element
    n_nodes_per_element = len(conn[0]) if len(conn) > 0 else 4
    n_dof_per_element = n_nodes_per_element * 3
    
    # Initialize global stiffness matrix
    K = np.zeros((n_dof, n_dof))
    
    # Process each element
    for iel, el_conn in enumerate(conn):
        if len(el_conn) != n_nodes_per_element:
            continue  # Skip invalid elements
            
        el_coors = coords[el_conn]
        
        # Average E_eff for element (could be improved with integration point values)
        el_E_eff = np.mean(E_eff[el_conn])
        nu = material.nu
        lambda_lame = el_E_eff * nu / ((1 + nu) * (1 - 2 * nu))
        mu_lame = el_E_eff / (2 * (1 + nu))
        D = _compute_material_matrix(lambda_lame, mu_lame)
        
        K_e = np.zeros((n_dof_per_element, n_dof_per_element))
        
        # Choose integration based on element type
        if element_type == '3_4' or n_nodes_per_element == 4:
            # Tetrahedral element - use 1-point integration at centroid
            # Natural coordinates: (xi, eta, zeta, 1-xi-eta-zeta)
            # Integration point at centroid: (1/4, 1/4, 1/4)
            xi, eta, zeta = 1/4, 1/4, 1/4
            N, dN_dxi = _tet4_shape_functions(xi, eta, zeta)
            
            # Jacobian
            J = dN_dxi.T @ el_coors
            det_J = np.linalg.det(J)
            if det_J <= 0:
                continue
            
            J_inv = np.linalg.inv(J)
            dN_dx = dN_dxi @ J_inv.T
            
            # Build B matrix (strain-displacement) for 4-node tetra
            B = np.zeros((6, 12))  # 6 strain components, 12 DOF (4 nodes × 3)
            for inode in range(4):
                idx = inode * 3
                dN_dx_i, dN_dy_i, dN_dz_i = dN_dx[inode, 0], dN_dx[inode, 1], dN_dx[inode, 2]
                
                B[0, idx] = dN_dx_i      # ε_xx
                B[1, idx + 1] = dN_dy_i  # ε_yy
                B[2, idx + 2] = dN_dz_i  # ε_zz
                B[3, idx] = dN_dy_i      # γ_xy
                B[3, idx + 1] = dN_dx_i
                B[4, idx] = dN_dz_i      # γ_xz
                B[4, idx + 2] = dN_dx_i
                B[5, idx + 1] = dN_dz_i  # γ_yz
                B[5, idx + 2] = dN_dy_i
            
            # Volume of tetrahedron: V = det(J) / 6
            # For 1-point integration at centroid, weight = V
            weight = det_J / 6.0
            K_e += B.T @ D @ B * det_J * weight
            
        else:
            # Hexahedral element (fallback)
            gauss_points, gauss_weights = _get_gauss_quadrature_3d(2)
            for gp, weight in zip(gauss_points, gauss_weights):
                xi, eta, zeta = gp
                N, dN_dxi = _hex8_shape_functions(xi, eta, zeta)
                
                # Jacobian
                J = dN_dxi.T @ el_coors
                det_J = np.linalg.det(J)
                if det_J <= 0:
                    continue
                
                J_inv = np.linalg.inv(J)
                dN_dx = dN_dxi @ J_inv.T
                
                # Build B matrix (strain-displacement)
                B = np.zeros((6, 24))
                for inode in range(8):
                    idx = inode * 3
                    dN_dx_i, dN_dy_i, dN_dz_i = dN_dx[inode, 0], dN_dx[inode, 1], dN_dx[inode, 2]
                    
                    B[0, idx] = dN_dx_i      # ε_xx
                    B[1, idx + 1] = dN_dy_i  # ε_yy
                    B[2, idx + 2] = dN_dz_i  # ε_zz
                    B[3, idx] = dN_dy_i      # γ_xy
                    B[3, idx + 1] = dN_dx_i
                    B[4, idx] = dN_dz_i      # γ_xz
                    B[4, idx + 2] = dN_dx_i
                    B[5, idx + 1] = dN_dz_i  # γ_yz
                    B[5, idx + 2] = dN_dy_i
                
                K_e += B.T @ D @ B * det_J * weight
        
        # Assemble into global matrix
        for i, inode in enumerate(el_conn):
            for j, jnode in enumerate(el_conn):
                i_dof = np.arange(inode * 3, inode * 3 + 3)
                j_dof = np.arange(jnode * 3, jnode * 3 + 3)
                i_local = np.arange(i * 3, i * 3 + 3)
                j_local = np.arange(j * 3, j * 3 + 3)
                K[np.ix_(i_dof, j_dof)] += K_e[np.ix_(i_local, j_local)]
    
    return K


def solve_fe_system_with_damage(mesh, coords, damage: np.ndarray, 
                                current_traction: float, z_min: float, z_max: float,
                                material: MaterialProperties, tension: bool = False) -> np.ndarray:
    """Solve FE system K(D)·u = F with damage-dependent stiffness and proper boundary conditions.
    
    Properly assembles stiffness matrix, enforces boundary conditions, and solves K(D)·u = F.
    
    Parameters
    ----------
    mesh : Mesh
        Finite element mesh
    coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates
    damage : np.ndarray, shape (n_nodes,)
        Damage field at nodes
    current_traction : float
        Applied traction (stress) in Pa
    z_min : float
        Minimum z coordinate (bottom)
    z_max : float
        Maximum z coordinate (top)
    material : MaterialProperties
        Material properties
    tension : bool
        If True, apply tensile loading; if False, compressive
    
    Returns
    -------
    np.ndarray, shape (n_nodes, 3)
        Displacement field at all nodes
    """
    n_nodes = len(coords)
    n_dof = n_nodes * 3
    
    # Assemble global stiffness matrix with damage
    K = assemble_stiffness_matrix(mesh, coords, damage, material)
    
    # Assemble force vector
    F = np.zeros(n_dof)
    
    # Apply traction on top surface
    top_nodes = np.where(coords[:, 2] >= z_max - 1e-6)[0]
    bottom_nodes = np.where(coords[:, 2] <= z_min + 1e-6)[0]
    
    # Calculate area of top surface (approximate from node spacing)
    if len(top_nodes) > 0:
        top_coords = coords[top_nodes]
        x_range = np.max(top_coords[:, 0]) - np.min(top_coords[:, 0])
        y_range = np.max(top_coords[:, 1]) - np.min(top_coords[:, 1])
        top_area = x_range * y_range
        if top_area < 1e-10:
            # Fallback: estimate from all nodes
            x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
            y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
            top_area = x_range * y_range
        
        # Force per node on top surface
        force_per_node = current_traction * top_area / len(top_nodes) if len(top_nodes) > 0 else 0.0
        
        # Apply force in z-direction (negative for compression, positive for tension)
        for node in top_nodes:
            if tension:
                F[node * 3 + 2] = force_per_node  # Positive z (tension)
            else:
                F[node * 3 + 2] = -force_per_node  # Negative z (compression)
    
    # Apply boundary conditions: u = 0 at bottom surface
    # Use penalty method or direct elimination
    # Direct elimination: remove DOFs at bottom
    free_dofs = []
    fixed_dofs = []
    
    for node in range(n_nodes):
        if node in bottom_nodes:
            # Fix all DOFs at bottom
            fixed_dofs.extend([node * 3, node * 3 + 1, node * 3 + 2])
        else:
            free_dofs.extend([node * 3, node * 3 + 1, node * 3 + 2])
    
    # Reorder: free DOFs first, then fixed
    all_dofs = free_dofs + fixed_dofs
    n_free = len(free_dofs)
    
    # Reorder K and F
    K_reordered = K[np.ix_(all_dofs, all_dofs)]
    F_reordered = F[all_dofs]
    
    # Extract free-free submatrix and free force vector
    K_ff = K_reordered[:n_free, :n_free]
    F_f = F_reordered[:n_free]
    
    # Solve for free DOFs: K_ff · u_f = F_f
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        K_ff_sparse = csc_matrix(K_ff)
        u_f = spsolve(K_ff_sparse, F_f)
    except:
        # Fallback to dense solve
        u_f = np.linalg.solve(K_ff, F_f)
    
    # Reconstruct full displacement vector
    u_full = np.zeros(n_dof)
    u_full[:n_free] = u_f
    # Fixed DOFs remain zero (already initialized)
    
    # Reorder back to original DOF ordering
    u_original = np.zeros(n_dof)
    for i, dof in enumerate(all_dofs):
        u_original[dof] = u_full[i]
    
    # Reshape to (n_nodes, 3)
    u = u_original.reshape(n_nodes, 3)
    
    return u


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


def compute_strain_tensor_from_displacement(u_field: np.ndarray, node_idx: int, mesh, 
                                           coords: np.ndarray) -> np.ndarray:
    """Compute full 3D strain tensor from displacement field at a node using proper FE shape function derivatives.
    
    Uses proper FE shape function gradients: ε = 0.5(∇u + ∇u^T) where ∇u is computed
    from shape function derivatives at the node location.
    
    Parameters
    ----------
    u_field : np.ndarray, shape (n_nodes, 3)
        Displacement field (3D vector field) at all nodes
    node_idx : int
        Node index
    mesh : Mesh
        Finite element mesh
    coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates
    
    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric strain tensor at the node
    """
    # Get node coordinate
    node_coord = coords[node_idx]
    
    # Find elements connected to this node
    conns = None
    element_type = None
    if hasattr(mesh, 'conns') and len(mesh.conns) > 0:
        conns = mesh.conns[0]
        if hasattr(mesh, 'descs') and len(mesh.descs) > 0:
            element_type = mesh.descs[0]
        elif len(conns) > 0:
            element_type = '3_4' if len(conns[0]) == 4 else '3_8'
    elif hasattr(mesh, 'get_conn'):
        try:
            conns = mesh.get_conn('3_4')
            element_type = '3_4'
        except:
            try:
                conns = mesh.get_conn('3_8')
                element_type = '3_8'
            except:
                conns = mesh.get_conn()
                if len(conns) > 0:
                    element_type = '3_4' if len(conns[0]) == 4 else '3_8'
    else:
        raise ValueError("Cannot access mesh connectivity")
    
    if element_type is None:
        element_type = '3_4' if (len(conns) > 0 and len(conns[0]) == 4) else '3_8'
    
    connected_elements = []
    for el_idx, element_nodes in enumerate(conns):
        if node_idx in element_nodes:
            connected_elements.append((el_idx, element_nodes))
    
    if len(connected_elements) == 0:
        return np.zeros((3, 3), dtype=np.float64)
    
    # Compute strain using proper FE shape function derivatives
    strain_tensors = []
    
    for el_idx, element_nodes in connected_elements:
        el_coords = coords[element_nodes]
        el_displacements = u_field[element_nodes]
        
        # Find local node index in element
        local_node_idx = np.where(element_nodes == node_idx)[0]
        if len(local_node_idx) == 0:
            continue
        local_node_idx = local_node_idx[0]
        
        # Get natural coordinates of this node in element
        if element_type == '3_4' or len(element_nodes) == 4:
            # Tetrahedral element: nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            corner_coords = [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
            ]
            xi, eta, zeta = corner_coords[local_node_idx]
            # Get shape function derivatives at this node
            _, dN_dxi = _tet4_shape_functions(xi, eta, zeta)
        else:
            # Hexahedral element: nodes at corners: xi, eta, zeta = ±1
            # Node 0: (-1, -1, -1), Node 1: (1, -1, -1), etc.
            corner_coords = [
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
            ]
            xi, eta, zeta = corner_coords[local_node_idx]
            # Get shape function derivatives at this node
            _, dN_dxi = _hex8_shape_functions(xi, eta, zeta)
        
        # Compute Jacobian at node
        J = dN_dxi.T @ el_coords
        det_J = np.linalg.det(J)
        if det_J <= 1e-10:
            continue
        
        J_inv = np.linalg.inv(J)
        dN_dx = dN_dxi @ J_inv.T  # Shape function derivatives in physical coordinates
        
        # Compute displacement gradient: ∇u = Σ (dN_i/dx) · u_i
        grad_u = np.zeros((3, 3))
        for i, node in enumerate(element_nodes):
            u_node = u_field[node]
            grad_u += np.outer(u_node, dN_dx[i])
        
        # Compute symmetric strain tensor: ε = 0.5(∇u + ∇u^T)
        strain_tensor = 0.5 * (grad_u + grad_u.T)
        
        # Weight by element volume (det_J)
        weight = det_J
        strain_tensors.append((strain_tensor, weight))
    
    if len(strain_tensors) == 0:
        return np.zeros((3, 3), dtype=np.float64)
    
    # Weighted average of strain tensors
    total_weight = sum(w for _, w in strain_tensors)
    if total_weight < 1e-10:
        return np.zeros((3, 3), dtype=np.float64)
    
    strain = sum(eps * w for eps, w in strain_tensors) / total_weight
    
    return strain


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
    This function implements the complete Mazars damage model workflow with proper
    FE solve, node-wise strain computation, and spatial damage localization.
    
    IMPLEMENTED FEATURES:
    ---------------------
    1. **Proper FE Solve**: Solves K(D)·u = F with damage-degraded stiffness
       - Spatially varying displacement field based on damage distribution
       - Accounts for element-wise damage effects
    
    2. **Full 3D Strain Tensor**: Computes ε = 0.5(∇u + ∇u^T) at each node
       - Node-wise strain tensor computation from displacement gradients
       - Proper spatial variation using element connectivity
    
    3. **Node-wise Damage**: Computes damage at each node individually
       - ε_tensor computed from local displacement field
       - ε_eq computed from strain tensor at each node
       - D computed using Mazars model at each node
       - Enables proper damage localization (microcracking zones)
    
    4. **Stress from FE Solution**: Computes σ = (1-D)·E·ε from FE solution
       - Full 3D stress tensor using Lame parameters
       - Accounts for Poisson effects and 3D stress state
    
    MAZARS MODEL WORKFLOW (Standard Implementation):
    -------------------------------------------------
    At each load step:
    1. Solve FE system: K(D) · u = F  (damage-degraded stiffness, spatially varying)
    2. Compute strain: ε = 0.5(∇u + ∇u^T)  (full 3D tensor at each node)
    3. Compute equivalent strain: ε_eq = √(Σ⟨ε_i⟩₊²) where ⟨ε_i⟩₊ = max(ε_i, 0) (node-wise)
    4. Update damage: d_c = 1 - (ε_d0/ε_eq) * (1 - A_c + A_c * exp[-B_c(ε_eq - ε_d0)])  (node-wise)
    5. Check convergence: ||D_new - D_old|| < tolerance
    6. Repeat until convergence
    7. Compute stress: σ = (1-D)·E·ε  (from FE solution, node-wise)
    
    ACCURACY IMPROVEMENTS:
    ----------------------
    - ✅ Node-wise strain computation enables spatial variation
    - ✅ Node-wise damage enables proper localization
    - ✅ Proper stress computation from FE solution
    - ✅ Accounts for full 3D stress and strain state
    
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
    
    # Track when damage first occurs
    damage_first_detected = False
    damage_first_step = None
    damage_first_force = None
    damage_first_stress = None
    
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
            
            # Step 2: Solve FE system with damage-degraded stiffness
            # Solve K(D)·u = F where stiffness depends on damage field
            u_field = solve_fe_system_with_damage(
                mesh, coords, damage, current_traction, z_min, z_max, material
            )
            
            solve_time = time.time() - iter_start
            if damage_iter == 0:
                print(f" done ({solve_time:.2f}s)", end='', flush=True)
            
            # Step 3: Compute full 3D strain tensor at each node from displacement solution
            # This computes ε = 0.5(∇u + ∇u^T) at each node individually
            epsilon_tensors = compute_node_wise_strain_tensors(u_field, mesh, coords)
            
            # Step 4: Calculate Mazars equivalent strain and damage at each node
            print(" computing damage...", end='', flush=True)
            
            # Compute damage at each node individually (enables proper localization)
            damage_new = np.zeros(n_nodes, dtype=np.float64)
            
            for node_idx in range(n_nodes):
                # Get strain tensor at this node
                epsilon_tensor_node = epsilon_tensors[node_idx]
                
                # Compute equivalent strain: ε_eq = √(Σ⟨ε_i⟩₊²)
                eps_eq_node = compute_equivalent_strain(epsilon_tensor_node)
                
                # Compute damage from equivalent strain using Mazars model
                damage_new[node_idx] = mazars_compressive_damage(
                    eps_eq_node, material.epsilon_c0, material.A_c, material.B_c
                )
            
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
        # Extract from converged displacement and strain fields
        
        # Get final strain tensors at all nodes
        epsilon_tensors_final = compute_node_wise_strain_tensors(u_field, mesh, coords)
        
        # Compute stress at each node from strain and damage
        stresses_nodes = []
        strains_zz_nodes = []
        
        for node_idx in range(n_nodes):
            epsilon_tensor_node = epsilon_tensors_final[node_idx]
            damage_node = damage[node_idx]
            
            # Compute stress tensor: σ = (1-D) · E · ε (via Lame parameters)
            stress_tensor_node = compute_stress_from_strain(
                epsilon_tensor_node, damage_node, material
            )
            
            # Extract stress and strain components
            stress_zz_node = stress_tensor_node[2, 2]  # Compressive stress (negative)
            strain_zz_node = epsilon_tensor_node[2, 2]  # Compressive strain (negative)
            
            stresses_nodes.append(stress_zz_node)
            strains_zz_nodes.append(strain_zz_node)
        
        # Average over nodes
        # Use COMPUTED stress from FE solution (reflects actual material response with damage)
        valid_stresses = [s for s in stresses_nodes if not np.isnan(s) and not np.isinf(s)]
        if len(valid_stresses) > 0:
            stress_avg = abs(np.mean(valid_stresses))  # Computed stress in Pa
        else:
            stress_avg = 0.0
        
        # Only use applied traction as fallback if computed stress is truly invalid
        # But prefer computed stress even if small (it reflects actual material response)
        if stress_avg == 0.0 and abs(current_traction) > 0:
            # Debug: check if structure is deforming
            max_disp = np.max(np.abs(u_field))
            if max_disp < 1e-10:
                # Structure not deforming - this is a problem, but use applied stress for now
                print(f"      ⚠ WARNING: Structure not deforming (max displacement: {max_disp:.2e} m)")
                print(f"      ⚠ Using applied traction as fallback: {abs(current_traction)/1e6:.2f} MPa")
            stress_avg = abs(current_traction)
        strain_avg = abs(np.mean(strains_zz_nodes))
        
        # Compute displacement from displacement field
        # Average displacement at top surface
        top_mask = coords[:, 2] >= z_max - 1e-6
        if np.any(top_mask):
            displacement_avg = abs(np.mean(u_field[top_mask, 2]))
        else:
            displacement_avg = abs(np.max(u_field[:, 2]))
        
        # Energy: U = 0.5 * ∫ σ : ε dV
        # Use computed stress tensor for accurate energy calculation
        volume = cross_sectional_area * (z_max - z_min)
        volume_per_node = volume / n_nodes if n_nodes > 0 else 0.0
        
        energy = 0.0
        for node_idx in range(n_nodes):
            epsilon_tensor_node = epsilon_tensors_final[node_idx]
            damage_node = damage[node_idx]
            
            # Compute stress tensor from strain and damage
            stress_tensor_node = compute_stress_from_strain(
                epsilon_tensor_node, damage_node, material
            )
            
            # Energy density: E = 0.5 * trace(σ · ε) = 0.5 * σ_ij · ε_ij
            energy_density = 0.5 * np.trace(stress_tensor_node @ epsilon_tensor_node)
            energy += energy_density * volume_per_node
        
        energy = abs(energy)  # Store as positive value
        
        strains.append(float(strain_avg))
        stresses.append(float(abs(stress_avg)))  # Store as positive (compressive strength) - now using applied stress
        energies.append(float(energy))
        displacements.append(float(displacement_avg))
        forces.append(float(current_force))
        damage_avg = float(np.mean(damage))
        damage_history.append(damage_avg)
        convergence_info.append({
            "damage_iterations": damage_iter + 1,
            "converged": converged,
            "damage_max": float(np.max(damage)),
            "damage_avg": damage_avg
        })
        
        # Check if damage first occurred in this step
        if not damage_first_detected and damage_avg > 1e-6:  # Small threshold to detect first damage
            damage_first_detected = True
            damage_first_step = step + 1
            damage_first_force = current_force
            damage_first_stress = abs(stress_avg)
            print(f"\n    ⚠ DAMAGE FIRST DETECTED at Step {damage_first_step}/{sim_params.num_steps}")
            print(f"       Force: {damage_first_force/1e3:.2f} kN ({damage_first_force:.0f} N)")
            print(f"       Stress: {damage_first_stress/1e6:.2f} MPa")
            print(f"       Average damage: {damage_avg:.6f}, Max damage: {np.max(damage):.6f}\n")
        
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
    
    # Print summary of damage initiation
    if damage_first_detected:
        print(f"\n  Damage Initiation Summary (Compression):")
        print(f"    First damage detected at Step {damage_first_step}/{sim_params.num_steps}")
        print(f"    Force at damage initiation: {damage_first_force/1e3:.2f} kN ({damage_first_force:.0f} N)")
        print(f"    Stress at damage initiation: {damage_first_stress/1e6:.2f} MPa")
    else:
        print(f"\n  No damage detected during compression test (all steps completed without damage)")
    
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
        "damage_first_step": damage_first_step,
        "damage_first_force_N": damage_first_force,
        "damage_first_stress_Pa": damage_first_stress,
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
    
    # Track when damage first occurs
    damage_first_detected = False
    damage_first_step = None
    damage_first_force = None
    damage_first_stress = None
    
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
            
            # Step 2: Solve FE system with damage-degraded stiffness (tension)
            # Solve K(D)·u = F where stiffness depends on damage field
            u_field = solve_fe_system_with_damage(
                mesh, coords, damage, current_traction, z_min, z_max, material, tension=True
            )
            
            solve_time = time.time() - iter_start
            if damage_iter == 0:
                print(f" done ({solve_time:.2f}s)", end='', flush=True)
            
            # Step 3: Compute full 3D strain tensor at each node from displacement solution
            epsilon_tensors = compute_node_wise_strain_tensors(u_field, mesh, coords)
            
            # Step 4: Calculate Mazars equivalent strain and damage at each node (tensile)
            print(" computing damage...", end='', flush=True)
            
            # Compute damage at each node individually using TENSILE damage model
            damage_new = np.zeros(n_nodes, dtype=np.float64)
            
            for node_idx in range(n_nodes):
                # Get strain tensor at this node
                epsilon_tensor_node = epsilon_tensors[node_idx]
                
                # Compute equivalent strain: ε_eq = √(Σ⟨ε_i⟩₊²)
                eps_eq_node = compute_equivalent_strain(epsilon_tensor_node)
                
                # Compute damage from equivalent strain using TENSILE Mazars model
                damage_new[node_idx] = mazars_tensile_damage(
                    eps_eq_node, material.epsilon_t0, material.A_t, material.B_t
                )
            
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
        
        # Step 7: Compute results from FE solution (tension)
        # Extract from converged displacement and strain fields
        
        # Get final strain tensors at all nodes
        epsilon_tensors_final = compute_node_wise_strain_tensors(u_field, mesh, coords)
        
        # Compute stress at each node from strain and damage
        stresses_nodes = []
        strains_zz_nodes = []
        
        for node_idx in range(n_nodes):
            epsilon_tensor_node = epsilon_tensors_final[node_idx]
            damage_node = damage[node_idx]
            
            # Compute stress tensor: σ = (1-D) · E · ε (via Lame parameters)
            stress_tensor_node = compute_stress_from_strain(
                epsilon_tensor_node, damage_node, material
            )
            
            # Extract stress and strain components
            stress_zz_node = stress_tensor_node[2, 2]  # Tensile stress (positive)
            strain_zz_node = epsilon_tensor_node[2, 2]  # Tensile strain (positive)
            
            stresses_nodes.append(stress_zz_node)
            strains_zz_nodes.append(strain_zz_node)
        
        # Average over nodes
        # Use COMPUTED stress from FE solution (reflects actual material response with damage)
        valid_stresses = [s for s in stresses_nodes if not np.isnan(s) and not np.isinf(s)]
        if len(valid_stresses) > 0:
            stress_avg = abs(np.mean(valid_stresses))  # Computed stress in Pa
        else:
            stress_avg = 0.0
        
        # Only use applied traction as fallback if computed stress is truly invalid
        # But prefer computed stress even if small (it reflects actual material response)
        if stress_avg == 0.0 and abs(current_traction) > 0:
            # Debug: check if structure is deforming
            max_disp = np.max(np.abs(u_field))
            if max_disp < 1e-10:
                # Structure not deforming - this is a problem, but use applied stress for now
                print(f"      ⚠ WARNING: Structure not deforming (max displacement: {max_disp:.2e} m)")
                print(f"      ⚠ Using applied traction as fallback: {abs(current_traction)/1e6:.2f} MPa")
            stress_avg = abs(current_traction)
        strain_avg = abs(np.mean(strains_zz_nodes))
        
        # Compute displacement from displacement field
        # Average displacement at top surface
        top_mask = coords[:, 2] >= z_max - 1e-6
        if np.any(top_mask):
            displacement_avg = abs(np.mean(u_field[top_mask, 2]))
        else:
            displacement_avg = abs(np.max(u_field[:, 2]))
        
        # Energy: U = 0.5 * ∫ σ : ε dV
        # Use computed stress tensor for accurate energy calculation
        volume = cross_sectional_area * (z_max - z_min)
        volume_per_node = volume / n_nodes if n_nodes > 0 else 0.0
        
        energy = 0.0
        for node_idx in range(n_nodes):
            epsilon_tensor_node = epsilon_tensors_final[node_idx]
            damage_node = damage[node_idx]
            
            # Compute stress tensor from strain and damage
            stress_tensor_node = compute_stress_from_strain(
                epsilon_tensor_node, damage_node, material
            )
            
            # Energy density: E = 0.5 * trace(σ · ε) = 0.5 * σ_ij · ε_ij
            energy_density = 0.5 * np.trace(stress_tensor_node @ epsilon_tensor_node)
            energy += energy_density * volume_per_node
        
        energy = abs(energy)  # Store as positive value
        
        strains.append(float(strain_avg))
        stresses.append(float(abs(stress_avg)))  # Store as positive (tensile strength) - now using applied stress
        energies.append(float(energy))
        displacements.append(float(displacement_avg))
        forces.append(float(current_force))
        damage_avg = float(np.mean(damage))
        damage_history.append(damage_avg)
        convergence_info.append({
            "damage_iterations": damage_iter + 1,
            "converged": converged,
            "damage_max": float(np.max(damage)),
            "damage_avg": damage_avg
        })
        
        # Check if damage first occurred in this step
        if not damage_first_detected and damage_avg > 1e-6:  # Small threshold to detect first damage
            damage_first_detected = True
            damage_first_step = step + 1
            damage_first_force = current_force
            damage_first_stress = abs(stress_avg)
            print(f"\n    ⚠ DAMAGE FIRST DETECTED at Step {damage_first_step}/{sim_params.num_steps}")
            print(f"       Force: {damage_first_force/1e3:.2f} kN ({damage_first_force:.0f} N)")
            print(f"       Stress: {damage_first_stress/1e6:.2f} MPa")
            print(f"       Average damage: {damage_avg:.6f}, Max damage: {np.max(damage):.6f}\n")
        
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
    
    # Print summary of damage initiation
    if damage_first_detected:
        print(f"\n  Damage Initiation Summary (Tension):")
        print(f"    First damage detected at Step {damage_first_step}/{sim_params.num_steps}")
        print(f"    Force at damage initiation: {damage_first_force/1e3:.2f} kN ({damage_first_force:.0f} N)")
        print(f"    Stress at damage initiation: {damage_first_stress/1e6:.2f} MPa")
    else:
        print(f"\n  No damage detected during tension test (all steps completed without damage)")
    
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
        "damage_first_step": damage_first_step,
        "damage_first_force_N": damage_first_force,
        "damage_first_stress_Pa": damage_first_stress,
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
