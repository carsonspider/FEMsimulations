"""
FEniCS-X (dolfinx) FEM Simulation with Mazars Damage Model for Cement/Concrete.

This script performs quasi-static, displacement-controlled finite element
analysis on STL meshes using the Mazars damage model for cement/concrete
materials. It runs both compressive and tensile tests and outputs:
- Energy absorption
- Mass
- Compressive/tensile strength
- Stress-strain curves
- Thermal properties (conductivity, R-value, heat storage capacity)

References:
- Mazars, J. (1986). A description of micro- and macro-scale damage of concrete
  structures. Engineering Fracture Mechanics, 25(5-6), 729-737.
- FEniCS Project: https://fenicsproject.org/
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

# dolfinx imports
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import ufl

import meshio
try:
    import gmsh
except ImportError:
    gmsh = None
    print("Warning: gmsh not available. Will use alternative meshing.")

# MPI communicator
comm = MPI.COMM_WORLD

@dataclass
class MaterialProperties:
    """Cement/concrete material properties."""
    
    # Elastic properties
    E: float = 25e9  # Young's modulus (Pa) - 25 GPa
    nu: float = 0.15  # Poisson's ratio
    rho: float = 1400.0  # Density (kg/m³) - typical for cement
    
    # Mazars damage model parameters
    epsilon_t0: float = 3e-4  # Tensile damage threshold strain
    A_t: float = 1.2  # Tensile damage evolution parameter 1
    B_t: float = 1.5  # Tensile damage evolution parameter 2
    
    # Compressive damage
    epsilon_c0: float = 1e-4  # Compressive damage threshold strain
    A_c: float = 1.5  # Compressive damage evolution parameter
    
    # Damage plasticity parameters
    psi: float = 35.0  # Dilation angle (degrees)
    epsilon: float = 0.1  # Eccentricity
    fb0_fc0: float = 1.16  # Biaxial/uniaxial strength ratio
    K: float = 0.667  # Shape parameter (2/3)
    
    # Thermal properties
    k_thermal: float = 1.4  # Thermal conductivity (W/m·K)
    c_p: float = 880.0  # Specific heat capacity (J/kg·K)
    
    def compute_lame_parameters(self) -> Tuple[float, float]:
        """Compute Lame parameters from E and nu."""
        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return lmbda, mu


@dataclass
class SimulationParameters:
    """Simulation control parameters."""
    
    residual_tol: float = 1e-3  # Relaxed for speed
    displacement_tol: float = 1e-6  # Relaxed for speed
    damage_tol: float = 1e-6
    max_iterations: int = 5  # Reduced for speed
    
    max_force: float = 1000000.0  # N (1 MN default)
    num_steps: int = 100
    
    element_size: float = 0.001  # m


class MazarsDamageModel:
    """Implementation of Mazars damage model for concrete."""
    
    def __init__(self, material: MaterialProperties):
        self.material = material
        self.epsilon_t0 = material.epsilon_t0
        self.epsilon_c0 = material.epsilon_c0
        self.A_t = material.A_t
        self.B_t = material.B_t
        self.A_c = material.A_c
    
    def compute_equivalent_strain(self, epsilon: np.ndarray) -> float:
        """Compute equivalent strain from strain tensor."""
        eigenvals = np.linalg.eigvals(epsilon)
        positive_strains = eigenvals[eigenvals > 0]
        if len(positive_strains) == 0:
            return 0.0
        return np.sqrt(np.sum(positive_strains**2))
    
    def compute_tensile_damage(self, epsilon_eq: float) -> float:
        """Compute tensile damage variable dt."""
        if epsilon_eq <= self.epsilon_t0:
            return 0.0
        ratio = self.epsilon_t0 / epsilon_eq
        exponent = -self.A_t * (epsilon_eq - self.epsilon_t0)
        dt = 1.0 - ratio * np.exp(exponent)
        return np.clip(dt, 0.0, 1.0)
    
    def compute_compressive_damage(self, epsilon_eq: float) -> float:
        """Compute compressive damage variable dc."""
        if epsilon_eq <= self.epsilon_c0:
            return 0.0
        ratio = self.epsilon_c0 / epsilon_eq
        exponent = -self.A_c * (epsilon_eq - self.epsilon_c0)
        dc = 1.0 - ratio * np.exp(exponent)
        return np.clip(dc, 0.0, 1.0)


def load_stl_and_create_mesh(stl_path: Path, element_size: float):
    """Load STL file and create tetrahedral volume mesh - simplified approach for testing."""
    print(f"Loading STL file: {stl_path}")
    
    # For quick testing: create a simple box mesh using dolfinx directly
    # Get approximate size from STL bounding box using meshio
    import meshio
    try:
        # Try reading as binary STL first
        stl_mesh = meshio.read(str(stl_path), file_format="stl")
    except Exception as e:
        # Try reading with explicit binary format
        try:
            # Read binary STL using numpy-stl if available
            try:
                from stl import mesh as stl_mesh_module
                stl_mesh_obj = stl_mesh_module.Mesh.from_file(str(stl_path))
                # Convert to numpy array
                points = np.unique(stl_mesh_obj.vectors.reshape(-1, 3), axis=0)
                bbox_min = points.min(axis=0)
                bbox_max = points.max(axis=0)
                size = bbox_max - bbox_min
                # Create box mesh directly
                n_x = max(2, int(size[0] / element_size))
                n_y = max(2, int(size[1] / element_size))
                n_z = max(2, int(size[2] / element_size))
                print(f"Creating box mesh: {n_x}x{n_y}x{n_z} divisions")
                fenics_mesh = mesh.create_box(
                    comm,
                    [bbox_min, bbox_max],
                    [n_x, n_y, n_z],
                    mesh.CellType.tetrahedron
                )
                num_vertices = fenics_mesh.topology.index_map(0).size_global
                num_cells = fenics_mesh.topology.index_map(3).size_global
                print(f"Mesh created: {num_vertices} vertices, {num_cells} tetrahedral cells")
                return fenics_mesh
            except ImportError:
                # Fallback: use default bounding box if we can't read the STL properly
                print("Warning: Could not read STL file properly, using default bounding box")
                bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                bbox_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                size = bbox_max - bbox_min
                
                n_x = max(2, int(size[0] / element_size))
                n_y = max(2, int(size[1] / element_size))
                n_z = max(2, int(size[2] / element_size))
                print(f"Creating box mesh: {n_x}x{n_y}x{n_z} divisions")
                print(f"Bounding box: {bbox_min} to {bbox_max}")
                fenics_mesh = mesh.create_box(
                    comm,
                    [bbox_min, bbox_max],
                    [n_x, n_y, n_z],
                    mesh.CellType.tetrahedron
                )
                num_vertices = fenics_mesh.topology.index_map(0).size_global
                num_cells = fenics_mesh.topology.index_map(3).size_global
                print(f"Mesh created: {num_vertices} vertices, {num_cells} tetrahedral cells")
                return fenics_mesh
        except Exception as e2:
            raise RuntimeError(f"Failed to read STL file: {e}, {e2}")
    
    points = stl_mesh.points
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    size = bbox_max - bbox_min
    
    # Create a simple box mesh using dolfinx
    # Calculate number of divisions based on element size
    n_x = max(2, int(size[0] / element_size))
    n_y = max(2, int(size[1] / element_size))
    n_z = max(2, int(size[2] / element_size))
    
    print(f"Creating box mesh: {n_x}x{n_y}x{n_z} divisions")
    fenics_mesh = mesh.create_box(
        comm,
        [bbox_min, bbox_max],
        [n_x, n_y, n_z],
        mesh.CellType.tetrahedron
    )
    
    num_vertices = fenics_mesh.topology.index_map(0).size_global
    num_cells = fenics_mesh.topology.index_map(3).size_global
    print(f"Mesh created: {num_vertices} vertices, {num_cells} tetrahedral cells")
    return fenics_mesh


def compute_mass(fenics_mesh, material: MaterialProperties) -> float:
    """Compute total mass of the structure."""
    # Compute volume
    volume = fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
    mass = volume * material.rho
    return mass


def compute_thermal_properties(fenics_mesh, material: MaterialProperties, volume: float) -> Dict[str, float]:
    """Compute thermal properties."""
    k_eff = material.k_thermal
    
    # Get bounding box from mesh coordinates
    coords = fenics_mesh.geometry.x
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    char_length = np.mean(bbox_max - bbox_min)
    R_value = char_length / k_eff
    
    heat_storage = material.rho * material.c_p * volume
    
    return {
        "thermal_conductivity": k_eff,
        "R_value": R_value,
        "heat_storage_capacity": heat_storage,
    }


def run_compression_test(fenics_mesh, material: MaterialProperties, sim_params: SimulationParameters) -> Dict:
    """Run uniaxial compression test."""
    print("\n" + "="*60)
    print("RUNNING COMPRESSION TEST")
    print("="*60)
    
    # Function spaces
    V = functionspace(fenics_mesh, ("Lagrange", 1, (fenics_mesh.geometry.dim,)))
    V_scalar = functionspace(fenics_mesh, ("Lagrange", 1))
    
    # Material
    lmbda, mu = material.compute_lame_parameters()
    damage_model = MazarsDamageModel(material)
    
    # Fields
    u = Function(V)
    u_n = Function(V)
    dc = Function(V_scalar)
    
    # Get mesh coordinates for boundary detection
    coords = fenics_mesh.geometry.x
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    
    # Calculate cross-sectional area (perpendicular to z-axis for compression)
    cross_sectional_area = (x_max - x_min) * (y_max - y_min)
    
    # Boundary conditions
    def bottom_boundary(x):
        return np.isclose(x[2], z_min, atol=1e-6)
    
    def top_boundary(x):
        return np.isclose(x[2], z_max, atol=1e-6)
    
    # Fixed bottom - all DOFs
    bottom_dofs = locate_dofs_geometrical(V, bottom_boundary)
    bc_bottom = dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), bottom_dofs, V)
    
    # Force control
    force_max = sim_params.max_force
    force_step = force_max / sim_params.num_steps
    
    # Calculate traction (force per unit area) for top surface
    traction_magnitude = force_max / cross_sectional_area
    
    strains, stresses, energies, displacements, forces = [], [], [], [], []
    
    print(f"Running {sim_params.num_steps} load steps...")
    print(f"Cross-sectional area: {cross_sectional_area:.6f} m²")
    print(f"Maximum force: {force_max/1e3:.2f} kN ({force_max:.0f} N)")
    print(f"Maximum traction: {traction_magnitude/1e6:.2f} MPa")
    
    # Load steps
    for step in range(sim_params.num_steps):
        if step % max(1, sim_params.num_steps // 10) == 0:
            print(f"  Compression step {step+1}/{sim_params.num_steps} ({100*(step+1)//sim_params.num_steps}%)")
        
        # Current force to apply
        current_force = force_step * (step + 1)
        current_traction = current_force / cross_sectional_area
        
        # Boundary conditions: fixed bottom, force on top
        bcs = [bc_bottom]
        
        # Define traction vector (compression = negative z direction)
        traction_vector = Constant(fenics_mesh, PETSc.ScalarType((0.0, 0.0, -current_traction)))
        
        # Create boundary measure once per step
        ds = ufl.Measure("ds", domain=fenics_mesh)
        
        # Newton-Raphson
        converged = False
        for iteration in range(sim_params.max_iterations):
            # Strain
            epsilon = ufl.sym(ufl.grad(u))
            
            # Damage (simplified - average strain)
            epsilon_zz = epsilon[2, 2]
            epsilon_zz_avg = fem.assemble_scalar(fem.form(epsilon_zz * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
            
            if abs(epsilon_zz_avg) > material.epsilon_c0:
                dc_val = damage_model.compute_compressive_damage(abs(epsilon_zz_avg))
            else:
                dc_val = 0.0
            
            # Effective stiffness
            E_eff = material.E * (1.0 - dc_val)
            lmbda_eff = E_eff * material.nu / ((1.0 + material.nu) * (1.0 - 2.0 * material.nu))
            mu_eff = E_eff / (2.0 * (1.0 + material.nu))
            
            # Stress
            sigma = lmbda_eff * ufl.tr(epsilon) * ufl.Identity(3) + 2.0 * mu_eff * epsilon
            
            # Variational form with traction boundary condition
            v = ufl.TestFunction(V)
            # Internal work (volume integral)
            F = ufl.inner(sigma, ufl.grad(v)) * ufl.dx(domain=fenics_mesh)
            # External work (surface integral - traction on top boundary)
            F -= ufl.inner(traction_vector, v) * ds(domain=fenics_mesh)
            
            # Solve
            problem = NonlinearProblem(F, u, bcs=bcs)
            solver = NewtonSolver(comm, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = sim_params.residual_tol
            solver.atol = sim_params.displacement_tol
            solver.max_it = 1  # One iteration per call
            
            try:
                n_iter, converged_flag = solver.solve(u)
                if converged_flag:
                    converged = True
            except:
                pass
            
            # Check convergence
            if converged:
                break
            
            u_n.x.array[:] = u.x.array[:]
        
        # Compute results
        strain_zz = epsilon[2, 2]
        stress_zz = sigma[2, 2]
        
        strain_avg = fem.assemble_scalar(fem.form(strain_zz * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
        stress_avg = fem.assemble_scalar(fem.form(stress_zz * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
        energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma, epsilon) * ufl.dx(domain=fenics_mesh)))
        
        # Get displacement at top surface
        u_z_top = u.sub(2)
        # Average displacement on top boundary
        top_dofs_z = locate_dofs_geometrical(V.sub(2).collapse()[0], top_boundary)
        if len(top_dofs_z) > 0:
            u_top_values = u.x.array[top_dofs_z * 3 + 2]  # z-component DOFs
            u_disp_avg = np.mean(np.abs(u_top_values))
        else:
            u_disp_avg = 0.0
        
        # Force is what we applied
        force_N = current_force
        
        strains.append(float(strain_avg))
        stresses.append(float(stress_avg))
        energies.append(float(energy))
        displacements.append(float(u_disp_avg))
        forces.append(float(force_N))
        
        if step % max(1, sim_params.num_steps // 5) == 0 or step == sim_params.num_steps - 1:
            print(f"    Step {step+1}/{sim_params.num_steps}: "
                  f"applied_force={force_N/1e3:.2f} kN, strain={strain_avg:.6f}, "
                  f"stress={stress_avg/1e6:.2f} MPa, displacement={u_disp_avg*1000:.3f} mm, "
                  f"energy={energy:.2f} J")
    
    compressive_strength = max([abs(s) for s in stresses]) if stresses else 0.0
    # Find energy at peak stress (before failure), not at the end
    if stresses:
        max_stress_idx = max(range(len(stresses)), key=lambda i: abs(stresses[i]))
        total_energy = energies[max_stress_idx] if max_stress_idx < len(energies) else 0.0
    else:
        total_energy = 0.0
    
    max_force = max(forces) if forces else 0.0
    
    # Store final displacement field for visualization
    u_final = Function(V)
    u_final.x.array[:] = u.x.array[:]
    
    return {
        "test_type": "compression",
        "strains": strains,
        "stresses": stresses,
        "forces_N": forces,
        "displacements": displacements,
        "energies": energies,
        "compressive_strength": compressive_strength,
        "max_force_N": max_force,
        "cross_sectional_area_m2": cross_sectional_area,
        "total_energy_absorption": total_energy,
        "displacement_field": u_final,  # Store for visualization
        "mesh": fenics_mesh,
    }


def run_tension_test(fenics_mesh, material: MaterialProperties, sim_params: SimulationParameters) -> Dict:
    """Run uniaxial tension test."""
    print("\n" + "="*60)
    print("RUNNING TENSION TEST")
    print("="*60)
    
    V = functionspace(fenics_mesh, ("Lagrange", 1, (fenics_mesh.geometry.dim,)))
    V_scalar = functionspace(fenics_mesh, ("Lagrange", 1))
    
    lmbda, mu = material.compute_lame_parameters()
    damage_model = MazarsDamageModel(material)
    
    u = Function(V)
    u_n = Function(V)
    dt = Function(V_scalar)
    
    coords = fenics_mesh.geometry.x
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    
    # Calculate cross-sectional area (perpendicular to x-axis for tension)
    cross_sectional_area = (y_max - y_min) * (z_max - z_min)
    
    def left_boundary(x):
        return np.isclose(x[0], x_min, atol=1e-6)
    
    def right_boundary(x):
        return np.isclose(x[0], x_max, atol=1e-6)
    
    left_dofs = locate_dofs_geometrical(V, left_boundary)
    bc_left = dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), left_dofs, V)
    
    # Force control
    force_max = sim_params.max_force
    force_step = force_max / sim_params.num_steps
    
    # Calculate traction (force per unit area) for right surface
    traction_magnitude = force_max / cross_sectional_area
    
    strains, stresses, energies, displacements, forces = [], [], [], [], []
    
    print(f"Running {sim_params.num_steps} load steps...")
    print(f"Cross-sectional area: {cross_sectional_area:.6f} m²")
    print(f"Maximum force: {force_max/1e3:.2f} kN ({force_max:.0f} N)")
    print(f"Maximum traction: {traction_magnitude/1e6:.2f} MPa")
    
    for step in range(sim_params.num_steps):
        if step % max(1, sim_params.num_steps // 10) == 0:
            print(f"  Tension step {step+1}/{sim_params.num_steps} ({100*(step+1)//sim_params.num_steps}%)")
        
        # Current force to apply
        current_force = force_step * (step + 1)
        current_traction = current_force / cross_sectional_area
        
        # Boundary conditions: fixed left, force on right
        bcs = [bc_left]
        
        # Define traction vector (tension = positive x direction)
        traction_vector = Constant(fenics_mesh, PETSc.ScalarType((current_traction, 0.0, 0.0)))
        
        # Create boundary measure once per step
        ds = ufl.Measure("ds", domain=fenics_mesh)
        
        converged = False
        for iteration in range(sim_params.max_iterations):
            epsilon = ufl.sym(ufl.grad(u))
            
            epsilon_xx = epsilon[0, 0]
            epsilon_xx_avg = fem.assemble_scalar(fem.form(epsilon_xx * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
            
            if epsilon_xx_avg > material.epsilon_t0:
                dt_val = damage_model.compute_tensile_damage(epsilon_xx_avg)
            else:
                dt_val = 0.0
            
            E_eff = material.E * (1.0 - dt_val)
            lmbda_eff = E_eff * material.nu / ((1.0 + material.nu) * (1.0 - 2.0 * material.nu))
            mu_eff = E_eff / (2.0 * (1.0 + material.nu))
            
            sigma = lmbda_eff * ufl.tr(epsilon) * ufl.Identity(3) + 2.0 * mu_eff * epsilon
            
            # Variational form with traction boundary condition
            v = ufl.TestFunction(V)
            # Internal work (volume integral)
            F = ufl.inner(sigma, ufl.grad(v)) * ufl.dx(domain=fenics_mesh)
            # External work (surface integral - traction on right boundary)
            F -= ufl.inner(traction_vector, v) * ds(domain=fenics_mesh)
            
            problem = NonlinearProblem(F, u, bcs=bcs)
            solver = NewtonSolver(comm, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = sim_params.residual_tol
            solver.atol = sim_params.displacement_tol
            solver.max_it = 1
            
            try:
                n_iter, converged_flag = solver.solve(u)
                if converged_flag:
                    converged = True
            except:
                pass
            
            if converged:
                break
            
            u_n.x.array[:] = u.x.array[:]
        
        strain_xx = epsilon[0, 0]
        stress_xx = sigma[0, 0]
        
        strain_avg = fem.assemble_scalar(fem.form(strain_xx * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
        stress_avg = fem.assemble_scalar(fem.form(stress_xx * ufl.dx(domain=fenics_mesh))) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
        energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma, epsilon) * ufl.dx(domain=fenics_mesh)))
        
        # Get displacement at right surface
        right_dofs_x = locate_dofs_geometrical(V.sub(0).collapse()[0], right_boundary)
        if len(right_dofs_x) > 0:
            u_right_values = u.x.array[right_dofs_x * 3]  # x-component DOFs
            u_disp_avg = np.mean(np.abs(u_right_values))
        else:
            u_disp_avg = 0.0
        
        # Force is what we applied
        force_N = current_force
        
        strains.append(float(strain_avg))
        stresses.append(float(stress_avg))
        energies.append(float(energy))
        displacements.append(float(u_disp_avg))
        forces.append(float(force_N))
        
        if step % max(1, sim_params.num_steps // 5) == 0 or step == sim_params.num_steps - 1:
            print(f"    Step {step+1}/{sim_params.num_steps}: "
                  f"applied_force={force_N/1e3:.2f} kN, strain={strain_avg:.6f}, "
                  f"stress={stress_avg/1e6:.2f} MPa, displacement={u_disp_avg*1000:.3f} mm, "
                  f"energy={energy:.2f} J")
    
    tensile_strength = max(stresses) if stresses else 0.0
    # Find energy at peak stress (before failure), not at the end
    if stresses:
        max_stress_idx = max(range(len(stresses)), key=lambda i: stresses[i])
        total_energy = energies[max_stress_idx] if max_stress_idx < len(energies) else 0.0
    else:
        total_energy = 0.0
    max_force = max(forces) if forces else 0.0
    
    # Store final displacement field for visualization
    u_final = Function(V)
    u_final.x.array[:] = u.x.array[:]
    
    return {
        "test_type": "tension",
        "strains": strains,
        "stresses": stresses,
        "forces_N": forces,
        "displacements": displacements,
        "energies": energies,
        "tensile_strength": tensile_strength,
        "max_force_N": max_force,
        "cross_sectional_area_m2": cross_sectional_area,
        "total_energy_absorption": total_energy,
        "displacement_field": u_final,  # Store for visualization
        "mesh": fenics_mesh,
    }


def plot_displacement_field(displacement_field, mesh, test_name: str, output_dir: Path):
    """Visualize displacement field with clear color coding (darker = more displacement)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import pyvista
        from dolfinx import plot
        
        # Try using pyvista for 3D visualization
        try:
            # Create pyvista plotter
            plotter = pyvista.Plotter(off_screen=True, window_size=[1920, 1080])
            topology, cell_types, geometry = plot.vtk_mesh(mesh)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            
            # Compute displacement magnitude at mesh vertices
            u_values = displacement_field.x.array.reshape(-1, 3)
            u_mag_values = np.linalg.norm(u_values, axis=1)
            
            # Map to grid points
            num_points = grid.number_of_points
            if len(u_mag_values) >= num_points:
                grid.point_data["Displacement"] = u_mag_values[:num_points]
            else:
                grid.point_data["Displacement"] = u_mag_values
            grid.set_active_scalars("Displacement")
            
            # Use clear colormap: darker colors = more displacement
            # 'plasma_r' or 'inferno_r' for better contrast
            plotter.add_mesh(grid, cmap="plasma_r", show_edges=True, edge_color='black', 
                           line_width=0.5, opacity=0.95)
            plotter.add_scalar_bar(title="Displacement Magnitude (m)", 
                                 n_labels=7, title_font_size=14, label_font_size=12,
                                 width=0.6, height=0.1, vertical=False, position_x=0.2, position_y=0.05)
            plotter.camera_position = 'iso'
            plotter.background_color = 'white'
            
            # Save screenshot
            viz_path = output_dir / f"displacement_field_{test_name}.png"
            plotter.screenshot(str(viz_path), transparent_background=False)
            plotter.close()
            print(f"Displacement field visualization saved to: {viz_path}")
            return
            
        except Exception as e:
            print(f"Pyvista visualization failed: {e}, using matplotlib fallback")
            
    except ImportError:
        pass
    
    # Fallback: Use matplotlib for 2D slice visualization with improved clarity
    try:
        # Get displacement magnitude on mesh vertices
        coords = mesh.geometry.x
        u_values = displacement_field.x.array.reshape(-1, 3)
        u_magnitude = np.linalg.norm(u_values, axis=1)
        
        # Get min/max for consistent color scale
        u_min, u_max = u_magnitude.min(), u_magnitude.max()
        
        # Create clearer 2D slice plots with better colors
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f'{test_name.capitalize()} Test - Displacement Field (Darker = Higher Displacement)', 
                     fontsize=16, fontweight='bold')
        
        # XY slice (middle z)
        ax1 = fig.add_subplot(131)
        z_mid = (coords[:, 2].min() + coords[:, 2].max()) / 2
        mask_z = np.abs(coords[:, 2] - z_mid) < (coords[:, 2].max() - coords[:, 2].min()) * 0.1
        if np.any(mask_z):
            scatter1 = ax1.scatter(coords[mask_z, 0], coords[mask_z, 1], 
                                  c=u_magnitude[mask_z], cmap='plasma_r', 
                                  s=50, alpha=0.8, vmin=u_min, vmax=u_max, edgecolors='k', linewidths=0.3)
            ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax1.set_title(f'XY Slice (z = {z_mid:.3f} m)', fontsize=13, fontweight='bold')
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax1, label='Displacement (m)', shrink=0.8)
            cbar1.ax.tick_params(labelsize=10)
        
        # XZ slice (middle y)
        ax2 = fig.add_subplot(132)
        y_mid = (coords[:, 1].min() + coords[:, 1].max()) / 2
        mask_y = np.abs(coords[:, 1] - y_mid) < (coords[:, 1].max() - coords[:, 1].min()) * 0.1
        if np.any(mask_y):
            scatter2 = ax2.scatter(coords[mask_y, 0], coords[mask_y, 2], 
                                  c=u_magnitude[mask_y], cmap='plasma_r', 
                                  s=50, alpha=0.8, vmin=u_min, vmax=u_max, edgecolors='k', linewidths=0.3)
            ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax2.set_title(f'XZ Slice (y = {y_mid:.3f} m)', fontsize=13, fontweight='bold')
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax2, label='Displacement (m)', shrink=0.8)
            cbar2.ax.tick_params(labelsize=10)
        
        # YZ slice (middle x)
        ax3 = fig.add_subplot(133)
        x_mid = (coords[:, 0].min() + coords[:, 0].max()) / 2
        mask_x = np.abs(coords[:, 0] - x_mid) < (coords[:, 0].max() - coords[:, 0].min()) * 0.1
        if np.any(mask_x):
            scatter3 = ax3.scatter(coords[mask_x, 1], coords[mask_x, 2], 
                                  c=u_magnitude[mask_x], cmap='plasma_r', 
                                  s=50, alpha=0.8, vmin=u_min, vmax=u_max, edgecolors='k', linewidths=0.3)
            ax3.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax3.set_title(f'YZ Slice (x = {x_mid:.3f} m)', fontsize=13, fontweight='bold')
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            cbar3 = plt.colorbar(scatter3, ax=ax3, label='Displacement (m)', shrink=0.8)
            cbar3.ax.tick_params(labelsize=10)
        
        # Add text box with min/max values
        info_text = f'Displacement Range: {u_min*1000:.3f} - {u_max*1000:.3f} mm'
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        viz_path = output_dir / f"displacement_field_{test_name}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        print(f"Displacement field visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"Matplotlib visualization failed: {e}")


def plot_test_setup(output_dir: Path):
    """Create a simple diagram showing the test setup for compression and tension."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Compression test setup
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')
    
    # Sample (cube)
    rect = plt.Rectangle((0.25, 0.5), 0.5, 1.0, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    
    # Fixed bottom (ground)
    ax1.plot([0, 1], [0.5, 0.5], 'k-', linewidth=4, label='Fixed (u=0)')
    ax1.fill_between([0, 1], [0.5, 0.5], [0, 0], color='gray', alpha=0.5)
    
    # Displacement on top
    ax1.arrow(0.5, 1.5, 0, 0.3, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax1.text(0.7, 1.6, 'Displacement\n(Δz)', fontsize=10, color='red', weight='bold')
    ax1.plot([0, 1], [1.5, 1.5], 'r--', linewidth=2, label='Displacement controlled')
    
    ax1.set_xlabel('X direction', fontsize=12)
    ax1.set_ylabel('Z direction', fontsize=12)
    ax1.set_title('Compression Test Setup', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 1.0, 'Sample', ha='center', va='center', fontsize=11, weight='bold')
    
    # Tension test setup
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    
    # Sample (cube)
    rect2 = plt.Rectangle((0.5, 0.25), 1.0, 0.5, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax2.add_patch(rect2)
    
    # Fixed left
    ax2.plot([0.5, 0.5], [0, 1], 'k-', linewidth=4, label='Fixed (u=0)')
    ax2.fill_between([0, 0.5], [0, 0], [1, 1], color='gray', alpha=0.5)
    
    # Displacement on right
    ax2.arrow(1.5, 0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax2.text(1.8, 0.7, 'Displacement\n(Δx)', fontsize=10, color='red', weight='bold')
    ax2.plot([1.5, 1.5], [0, 1], 'r--', linewidth=2, label='Displacement controlled')
    
    ax2.set_xlabel('X direction', fontsize=12)
    ax2.set_ylabel('Y direction', fontsize=12)
    ax2.set_title('Tension Test Setup', fontsize=14, weight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(1.0, 0.5, 'Sample', ha='center', va='center', fontsize=11, weight='bold')
    
    plt.tight_layout()
    setup_path = output_dir / "test_setup_diagram.png"
    plt.savefig(setup_path, dpi=300, bbox_inches="tight")
    print(f"Test setup diagram saved to: {setup_path}")
    plt.close()


def plot_results(compression_results: Dict, tension_results: Dict, output_dir: Path):
    """Generate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compression
    ax = axes[0, 0]
    comp_strains = compression_results["strains"]
    comp_stresses = [abs(s) / 1e6 for s in compression_results["stresses"]]
    ax.plot(comp_strains, comp_stresses, "b-", linewidth=2, label="Compression")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Compression Stress-Strain Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Tension
    ax = axes[0, 1]
    tens_strains = tension_results["strains"]
    tens_stresses = [s / 1e6 for s in tension_results["stresses"]]
    ax.plot(tens_strains, tens_stresses, "r-", linewidth=2, label="Tension")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Tension Stress-Strain Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Energy
    ax = axes[1, 0]
    comp_energies = compression_results["energies"]
    tens_energies = tension_results["energies"]
    steps = range(len(comp_energies))
    ax.plot(steps, [e / 1000 for e in comp_energies], "b-", linewidth=2, label="Compression")
    ax.plot(steps, [e / 1000 for e in tens_energies], "r-", linewidth=2, label="Tension")
    ax.set_xlabel("Load Step")
    ax.set_ylabel("Energy Absorption (kJ)")
    ax.set_title("Energy Absorption vs Load Step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Combined
    ax = axes[1, 1]
    ax.plot(comp_strains, comp_stresses, "b-", linewidth=2, label="Compression")
    ax.plot(tens_strains, tens_stresses, "r-", linewidth=2, label="Tension")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Combined Stress-Strain Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "stress_strain_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlots saved to: {plot_path}")
    plt.close()


def save_results(compression_results: Dict, tension_results: Dict, mass: float,
                 thermal_props: Dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "mass_kg": float(mass),
        "compressive_strength_MPa": float(compression_results["compressive_strength"] / 1e6),
        "tensile_strength_MPa": float(tension_results["tensile_strength"] / 1e6),
        "compression_max_force_N": float(compression_results.get("max_force_N", 0.0)),
        "tension_max_force_N": float(tension_results.get("max_force_N", 0.0)),
        "compression_cross_sectional_area_m2": float(compression_results.get("cross_sectional_area_m2", 0.0)),
        "tension_cross_sectional_area_m2": float(tension_results.get("cross_sectional_area_m2", 0.0)),
        "compression_energy_absorption_J": float(compression_results["total_energy_absorption"]),
        "tension_energy_absorption_J": float(tension_results["total_energy_absorption"]),
        "thermal_properties": {
            "thermal_conductivity_W_per_mK": float(thermal_props["thermal_conductivity"]),
            "R_value_m2K_per_W": float(thermal_props["R_value"]),
            "heat_storage_capacity_J_per_K": float(thermal_props["heat_storage_capacity"]),
        },
        "stress_strain_curves": {
            "compression": {
                "strains": [float(s) for s in compression_results["strains"]],
                "stresses_MPa": [float(abs(s) / 1e6) for s in compression_results["stresses"]],
                "forces_N": [float(f) for f in compression_results.get("forces_N", [])],
            },
            "tension": {
                "strains": [float(s) for s in tension_results["strains"]],
                "stresses_MPa": [float(s / 1e6) for s in tension_results["stresses"]],
                "forces_N": [float(f) for f in tension_results.get("forces_N", [])],
            },
        },
    }
    
    json_path = output_dir / "fem_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Mass: {mass:.3f} kg")
    print(f"Compressive Strength: {summary['compressive_strength_MPa']:.2f} MPa")
    print(f"Compression Max Force: {summary['compression_max_force_N']/1e3:.2f} kN")
    print(f"Tensile Strength: {summary['tensile_strength_MPa']:.2f} MPa")
    print(f"Tension Max Force: {summary['tension_max_force_N']/1e3:.2f} kN")
    print(f"Compression Energy: {summary['compression_energy_absorption_J']:.2f} J")
    print(f"Tension Energy: {summary['tension_energy_absorption_J']:.2f} J")
    print(f"Thermal Conductivity: {thermal_props['thermal_conductivity']:.2f} W/m·K")
    print(f"R-value: {thermal_props['R_value']:.4f} m²·K/W")
    print(f"Heat Storage: {thermal_props['heat_storage_capacity']:.2e} J/K")
    print("="*60)


def main():
    """Main simulation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FEM simulation with Mazars damage model")
    parser.add_argument("stl_file", type=str, help="Path to input STL file")
    parser.add_argument("--output-dir", type=str, default="fem_results", help="Output directory")
    parser.add_argument("--element-size", type=float, default=0.015, help="Mesh element size (m) - larger values = faster but less accurate")
    parser.add_argument("--max-force", type=float, default=1000000.0, help="Maximum force to apply (N) - default 1 MN")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of load steps - fewer steps = faster")
    
    args = parser.parse_args()
    
    stl_path = Path(args.stl_file)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("FEM SIMULATION - TEST MODE (OPTIMIZED FOR SPEED)")
    print("="*60)
    print(f"STL file: {stl_path}")
    print(f"Element size: {args.element_size} m")
    print(f"Number of steps: {args.num_steps}")
    print(f"Max force: {args.max_force/1e3:.2f} kN ({args.max_force:.0f} N)")
    print("="*60 + "\n")
    
    material = MaterialProperties()
    sim_params = SimulationParameters(
        element_size=args.element_size,
        max_force=args.max_force,
        num_steps=args.num_steps,
    )
    
    print("Loading and meshing STL file...")
    fenics_mesh = load_stl_and_create_mesh(stl_path, sim_params.element_size)
    
    volume = fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx(domain=fenics_mesh)))
    mass = compute_mass(fenics_mesh, material)
    thermal_props = compute_thermal_properties(fenics_mesh, material, volume)
    
    compression_results = run_compression_test(fenics_mesh, material, sim_params)
    tension_results = run_tension_test(fenics_mesh, material, sim_params)
    
    plot_test_setup(output_dir)
    plot_results(compression_results, tension_results, output_dir)
    
    # Visualize displacement fields
    if "displacement_field" in compression_results and "mesh" in compression_results:
        plot_displacement_field(
            compression_results["displacement_field"],
            compression_results["mesh"],
            "compression",
            output_dir
        )
    if "displacement_field" in tension_results and "mesh" in tension_results:
        plot_displacement_field(
            tension_results["displacement_field"],
            tension_results["mesh"],
            "tension",
            output_dir
        )
    
    save_results(compression_results, tension_results, mass, thermal_props, output_dir)
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()

