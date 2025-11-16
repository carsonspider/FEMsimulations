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
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import NonlinearProblem, NewtonSolver
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
    rho: float = 2400.0  # Density (kg/m³) - typical for cement
    
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
    
    residual_tol: float = 1e-4
    displacement_tol: float = 1e-7
    damage_tol: float = 1e-6
    max_iterations: int = 12
    
    max_displacement: float = 0.01  # m
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
    """Load STL file and create tetrahedral volume mesh using gmsh."""
    print(f"Loading STL file: {stl_path}")
    
    if gmsh is None:
        raise RuntimeError("gmsh is required for STL to volume mesh conversion")
    
    gmsh.initialize()
    gmsh.model.add("gyroid")
    gmsh.merge(str(stl_path))
    
    surface_entities = gmsh.model.getEntities(2)
    if len(surface_entities) == 0:
        gmsh.finalize()
        raise ValueError("No surface entities found in STL file")
    
    surface_tags = [s[1] for s in surface_entities]
    try:
        volumes = gmsh.model.occ.addVolume([(2, tag) for tag in surface_tags])
        gmsh.model.occ.synchronize()
    except Exception as e:
        gmsh.finalize()
        raise RuntimeError(f"Failed to create volume from STL: {e}. Ensure STL is watertight.")
    
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), element_size)
    gmsh.model.mesh.generate(3)
    
    # Convert to dolfinx mesh
    try:
        fenics_mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=3
        )
    except Exception as e:
        # Fallback: save as msh and read with meshio
        msh_file = str(stl_path).replace('.stl', '.msh')
        gmsh.write(msh_file)
        gmsh.finalize()
        
        msh = meshio.read(msh_file)
        cells = msh.cells_dict.get("tetra", None)
        if cells is None:
            raise ValueError("No tetrahedral cells found")
        
        # Create dolfinx mesh from meshio
        from dolfinx import geometry
        fenics_mesh = mesh.create_mesh(comm, msh.points, cells, mesh.CellType.tetrahedron)
    
    gmsh.finalize()
    
    num_vertices = fenics_mesh.topology.index_map(0).size_global
    num_cells = fenics_mesh.topology.index_map(3).size_global
    print(f"Mesh created: {num_vertices} vertices, {num_cells} tetrahedral cells")
    return fenics_mesh


def compute_mass(fenics_mesh, material: MaterialProperties) -> float:
    """Compute total mass of the structure."""
    # Compute volume
    volume = fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
    mass = volume * material.rho
    return mass


def compute_thermal_properties(fenics_mesh, material: MaterialProperties, volume: float) -> Dict[str, float]:
    """Compute thermal properties."""
    k_eff = material.k_thermal
    
    # Get bounding box
    bb_tree = geometry.bb_tree(fenics_mesh, fenics_mesh.topology.dim)
    bbox_min, bbox_max = geometry.compute_bounding_box(bb_tree)
    char_length = np.mean(np.array(bbox_max) - np.array(bbox_min))
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
    element = ("Lagrange", 1)
    V = functionspace(fenics_mesh, (element, (fenics_mesh.geometry.dim,)))
    V_scalar = functionspace(fenics_mesh, element)
    
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
    
    # Boundary conditions
    def bottom_boundary(x):
        return np.isclose(x[2], z_min, atol=1e-6)
    
    def top_boundary(x):
        return np.isclose(x[2], z_max, atol=1e-6)
    
    # Fixed bottom - all DOFs
    bottom_dofs = locate_dofs_geometrical(V, bottom_boundary)
    bc_bottom = dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), bottom_dofs, V)
    
    # Displacement control
    du_max = sim_params.max_displacement
    du_step = du_max / sim_params.num_steps
    
    strains, stresses, energies, displacements = [], [], [], []
    
    # Load steps
    for step in range(sim_params.num_steps):
        u_disp = -du_step * (step + 1)
        
        # Top boundary condition - z component only
        V_z = V.sub(2).collapse()[0]
        top_dofs_z = locate_dofs_geometrical(V_z, top_boundary)
        bc_top = dirichletbc(PETSc.ScalarType(u_disp), top_dofs_z, V.sub(2))
        bcs = [bc_bottom, bc_top]
        
        # Newton-Raphson
        converged = False
        for iteration in range(sim_params.max_iterations):
            # Strain
            epsilon = ufl.sym(ufl.grad(u))
            
            # Damage (simplified - average strain)
            epsilon_zz = epsilon[2, 2]
            epsilon_zz_avg = fem.assemble_scalar(fem.form(epsilon_zz * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
            
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
            
            # Variational form
            v = ufl.TestFunction(V)
            F = ufl.inner(sigma, ufl.grad(v)) * ufl.dx
            
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
        
        strain_avg = fem.assemble_scalar(fem.form(strain_zz * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
        stress_avg = fem.assemble_scalar(fem.form(stress_zz * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
        energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma, epsilon) * ufl.dx))
        
        strains.append(float(strain_avg))
        stresses.append(float(stress_avg))
        energies.append(float(energy))
        displacements.append(abs(u_disp))
        
        if step % 10 == 0:
            print(f"Step {step+1}/{sim_params.num_steps}: "
                  f"strain={strain_avg:.6f}, stress={stress_avg/1e6:.2f} MPa, "
                  f"energy={energy:.2f} J")
    
    compressive_strength = max([abs(s) for s in stresses]) if stresses else 0.0
    total_energy = energies[-1] if energies else 0.0
    
    return {
        "test_type": "compression",
        "strains": strains,
        "stresses": stresses,
        "displacements": displacements,
        "energies": energies,
        "compressive_strength": compressive_strength,
        "total_energy_absorption": total_energy,
    }


def run_tension_test(fenics_mesh, material: MaterialProperties, sim_params: SimulationParameters) -> Dict:
    """Run uniaxial tension test."""
    print("\n" + "="*60)
    print("RUNNING TENSION TEST")
    print("="*60)
    
    element = ("Lagrange", 1)
    V = functionspace(fenics_mesh, (element, (fenics_mesh.geometry.dim,)))
    V_scalar = functionspace(fenics_mesh, element)
    
    lmbda, mu = material.compute_lame_parameters()
    damage_model = MazarsDamageModel(material)
    
    u = Function(V)
    u_n = Function(V)
    dt = Function(V_scalar)
    
    coords = fenics_mesh.geometry.x
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    
    def left_boundary(x):
        return np.isclose(x[0], x_min, atol=1e-6)
    
    def right_boundary(x):
        return np.isclose(x[0], x_max, atol=1e-6)
    
    left_dofs = locate_dofs_geometrical(V, left_boundary)
    bc_left = dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), left_dofs, V)
    
    du_max = sim_params.max_displacement
    du_step = du_max / sim_params.num_steps
    
    strains, stresses, energies, displacements = [], [], [], []
    
    for step in range(sim_params.num_steps):
        u_disp = du_step * (step + 1)
        
        # Right boundary condition - x component only
        V_x = V.sub(0).collapse()[0]
        right_dofs_x = locate_dofs_geometrical(V_x, right_boundary)
        bc_right = dirichletbc(PETSc.ScalarType(u_disp), right_dofs_x, V.sub(0))
        bcs = [bc_left, bc_right]
        
        converged = False
        for iteration in range(sim_params.max_iterations):
            epsilon = ufl.sym(ufl.grad(u))
            
            epsilon_xx = epsilon[0, 0]
            epsilon_xx_avg = fem.assemble_scalar(fem.form(epsilon_xx * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
            
            if epsilon_xx_avg > material.epsilon_t0:
                dt_val = damage_model.compute_tensile_damage(epsilon_xx_avg)
            else:
                dt_val = 0.0
            
            E_eff = material.E * (1.0 - dt_val)
            lmbda_eff = E_eff * material.nu / ((1.0 + material.nu) * (1.0 - 2.0 * material.nu))
            mu_eff = E_eff / (2.0 * (1.0 + material.nu))
            
            sigma = lmbda_eff * ufl.tr(epsilon) * ufl.Identity(3) + 2.0 * mu_eff * epsilon
            
            v = ufl.TestFunction(V)
            F = ufl.inner(sigma, ufl.grad(v)) * ufl.dx
            
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
        
        strain_avg = fem.assemble_scalar(fem.form(strain_xx * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
        stress_avg = fem.assemble_scalar(fem.form(stress_xx * ufl.dx)) / fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
        energy = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma, epsilon) * ufl.dx))
        
        strains.append(float(strain_avg))
        stresses.append(float(stress_avg))
        energies.append(float(energy))
        displacements.append(u_disp)
        
        if step % 10 == 0:
            print(f"Step {step+1}/{sim_params.num_steps}: "
                  f"strain={strain_avg:.6f}, stress={stress_avg/1e6:.2f} MPa, "
                  f"energy={energy:.2f} J")
    
    tensile_strength = max(stresses) if stresses else 0.0
    total_energy = energies[-1] if energies else 0.0
    
    return {
        "test_type": "tension",
        "strains": strains,
        "stresses": stresses,
        "displacements": displacements,
        "energies": energies,
        "tensile_strength": tensile_strength,
        "total_energy_absorption": total_energy,
    }


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
            },
            "tension": {
                "strains": [float(s) for s in tension_results["strains"]],
                "stresses_MPa": [float(s / 1e6) for s in tension_results["stresses"]],
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
    print(f"Tensile Strength: {summary['tensile_strength_MPa']:.2f} MPa")
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
    parser.add_argument("--element-size", type=float, default=0.001, help="Mesh element size (m)")
    parser.add_argument("--max-displacement", type=float, default=0.01, help="Maximum displacement (m)")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of load steps")
    
    args = parser.parse_args()
    
    stl_path = Path(args.stl_file)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    material = MaterialProperties()
    sim_params = SimulationParameters(
        element_size=args.element_size,
        max_displacement=args.max_displacement,
        num_steps=args.num_steps,
    )
    
    fenics_mesh = load_stl_and_create_mesh(stl_path, sim_params.element_size)
    
    volume = fem.assemble_scalar(fem.form(Constant(fenics_mesh, PETSc.ScalarType(1.0)) * ufl.dx))
    mass = compute_mass(fenics_mesh, material)
    thermal_props = compute_thermal_properties(fenics_mesh, material, volume)
    
    compression_results = run_compression_test(fenics_mesh, material, sim_params)
    tension_results = run_tension_test(fenics_mesh, material, sim_params)
    
    plot_results(compression_results, tension_results, output_dir)
    save_results(compression_results, tension_results, mass, thermal_props, output_dir)
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()

