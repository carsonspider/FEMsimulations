#!/usr/bin/env python3
"""
2D Finite Element Method Example using ngSolve

This example demonstrates how to solve a 2D Poisson equation using ngSolve.
We solve: -Δu = f in Ω with u = 0 on ∂Ω
where Ω is a unit square [0,1]×[0,1] and f is a given source function.

This is a classic example in FEM that shows:
- Mesh generation for a 2D domain
- Function space definition
- Bilinear and linear forms
- Boundary condition application
- System solving
- Visualization of results

Author: Generated for prospectory-api examples
Date: September 2025
"""

from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.geom2d import unit_square

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_2d_mesh(max_element_size: float = 0.1) -> ngs.Mesh:
    """
    Create a 2D mesh for the unit square domain [0,1]×[0,1].

    Args:
        max_element_size: Maximum size of mesh elements (smaller = finer mesh)

    Returns:
        ngSolve mesh object
    """
    # Create unit square geometry using netgen
    geo = unit_square

    # Generate mesh with specified maximum element size
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=max_element_size))

    print(f"Mesh created with {mesh.ne} elements and {mesh.nv} vertices")
    return mesh


def setup_function_space(mesh: ngs.Mesh, order: int = 1) -> ngs.FESpace:
    """
    Set up the finite element function space.

    Args:
        mesh: The mesh object
        order: Polynomial order of the finite elements (1 = linear, 2 = quadratic, etc.)

    Returns:
        H1 finite element space with homogeneous Dirichlet boundary conditions
    """
    # Create H1 finite element space (continuous piecewise polynomials)
    # dirichlet=".*" means homogeneous Dirichlet BCs on all boundaries
    fes = ngs.H1(mesh, order=order, dirichlet=".*")

    print(f"Function space created with {fes.ndof} degrees of freedom")
    return fes


def define_source_function() -> ngs.CoefficientFunction:
    """
    Define the right-hand side source function f(x,y).

    For this example, we use f(x,y) = 32 * (x*(1-x) + y*(1-y))
    This creates a smooth source that's zero at the boundaries.

    Returns:
        ngSolve coefficient function representing the source term
    """
    # Get coordinate functions
    x, y = ngs.x, ngs.y

    # Define source function: creates a "bowl" shape
    f = 32 * (x * (1 - x) + y * (1 - y))

    return f


def solve_poisson_equation(mesh: ngs.Mesh, order: int = 1) -> tuple[ngs.GridFunction, ngs.FESpace]:
    """
    Solve the 2D Poisson equation -Δu = f with homogeneous Dirichlet BCs.

    Args:
        mesh: The computational mesh
        order: Polynomial order for finite elements

    Returns:
        Tuple of (solution, function_space)
    """
    print("Setting up the finite element problem...")

    # 1. Create function space
    fes = setup_function_space(mesh, order)

    # 2. Create trial and test functions
    u = fes.TrialFunction()  # Unknown function
    v = fes.TestFunction()  # Test function for weak formulation

    # 3. Define source function
    f = define_source_function()

    # 4. Set up bilinear form (left-hand side): ∫∇u·∇v dx
    a = ngs.BilinearForm(fes)
    a += ngs.grad(u) * ngs.grad(v) * ngs.dx

    # 5. Set up linear form (right-hand side): ∫f*v dx
    L = ngs.LinearForm(fes)
    L += f * v * ngs.dx

    # 6. Assemble the system matrices
    print("Assembling system matrices...")
    a.Assemble()
    L.Assemble()

    # 7. Create solution grid function
    gfu = ngs.GridFunction(fes)

    # 8. Solve the linear system
    print("Solving linear system...")
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * L.vec

    print("Solution computed successfully!")
    return gfu, fes


def analyze_solution(gfu: ngs.GridFunction) -> dict[str, float]:
    """
    Analyze the computed solution and extract key metrics.

    Args:
        gfu: The computed solution

    Returns:
        Dictionary with solution statistics
    """
    # Compute L2 norm of the solution
    l2_norm = ngs.sqrt(ngs.Integrate(gfu * gfu, gfu.space.mesh))

    # Find maximum and minimum values
    max_val = max(gfu.vec)
    min_val = min(gfu.vec)

    # Compute H1 norm (includes gradient)
    h1_norm = ngs.sqrt(ngs.Integrate(gfu * gfu + ngs.grad(gfu) * ngs.grad(gfu), gfu.space.mesh))

    stats = {
        "l2_norm": float(l2_norm),
        "h1_norm": float(h1_norm),
        "max_value": float(max_val),
        "min_value": float(min_val),
        "ndof": gfu.space.ndof,
    }

    print("\nSolution Analysis:")
    print(f"  L2 norm: {stats['l2_norm']:.6f}")
    print(f"  H1 norm: {stats['h1_norm']:.6f}")
    print(f"  Max value: {stats['max_value']:.6f}")
    print(f"  Min value: {stats['min_value']:.6f}")
    print(f"  Degrees of freedom: {stats['ndof']}")

    return stats


def visualize_solution(gfu: ngs.GridFunction, save_plot: bool = True) -> None:
    """
    Create visualization of the computed solution.

    Args:
        gfu: The computed solution
        save_plot: Whether to save the plot to file
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting. Install with: uv pip install matplotlib")
        return

    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Contour plot of the solution
        mesh = gfu.space.mesh

        # Extract mesh coordinates and solution values for plotting
        # Note: This is a simplified visualization approach
        print("Creating visualization...")

        # For now, we'll create a simple text-based output since matplotlib
        # integration with ngSolve can be complex
        ax1.text(
            0.5,
            0.5,
            'NGSolve Solution\n(Use NGSolve GUI\nfor full visualization)',
            ha='center',
            va='center',
            transform=ax1.transAxes,
            fontsize=12,
        )
        ax1.set_title('2D FEM Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Plot 2: Show mesh information
        stats_text = f"""Mesh Statistics:
Elements: {mesh.ne}
Vertices: {mesh.nv}
DOFs: {gfu.space.ndof}

Solution Stats:
Max: {max(gfu.vec):.4f}
Min: {min(gfu.vec):.4f}"""

        ax2.text(0.1, 0.5, stats_text, ha='left', va='center', transform=ax2.transAxes, fontsize=10, family='monospace')
        ax2.set_title('Problem Information')
        ax2.axis('off')

        plt.tight_layout()

        if save_plot:
            output_path = Path(__file__).parent / "ngsolve_solution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")

        plt.show()

    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Visualization error: {e}")


def demonstrate_convergence(max_sizes: list[float] = [0.2, 0.1, 0.05]) -> None:
    """
    Demonstrate mesh convergence by solving with different mesh sizes.

    Args:
        max_sizes: List of maximum element sizes to test
    """
    print("\n" + "=" * 50)
    print("CONVERGENCE STUDY")
    print("=" * 50)

    results = []

    for h in max_sizes:
        print(f"\nSolving with max element size h = {h}")
        mesh = create_2d_mesh(h)
        gfu, fes = solve_poisson_equation(mesh, order=1)
        stats = analyze_solution(gfu)

        results.append({'h': h, 'ndof': stats['ndof'], 'l2_norm': stats['l2_norm'], 'max_value': stats['max_value']})

    print("\nConvergence Results:")
    print(f"{'h':>8} {'DOFs':>8} {'L2 Norm':>12} {'Max Value':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['h']:>8.3f} {r['ndof']:>8} {r['l2_norm']:>12.6f} {r['max_value']:>12.6f}")


def main() -> None:
    """
    Main function demonstrating 2D FEM with ngSolve.
    """
    print("2D Finite Element Method Example with ngSolve")
    print("=" * 50)

    # 1. Create mesh
    print("\n1. Creating 2D mesh...")
    mesh = create_2d_mesh(max_element_size=0.1)

    # 2. Solve the problem
    print("\n2. Solving 2D Poisson equation...")
    solution, fes = solve_poisson_equation(mesh, order=1)

    # 3. Analyze results
    print("\n3. Analyzing solution...")
    stats = analyze_solution(solution)

    # 4. Visualize (basic)
    print("\n4. Creating visualization...")
    visualize_solution(solution, save_plot=True)

    # 5. Optional: Convergence study
    print("\n5. Running convergence study...")
    demonstrate_convergence()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("For advanced visualization, use NGSolve's built-in GUI:")
    print("  from ngsolve.webgui import Draw")
    print("  Draw(solution)")
    print("=" * 50)


if __name__ == "__main__":  # pragma: no cover
    # Check if running as script or test
    import sys

    from commons.utils import pytest_this_file

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        pytest_this_file()
    else:
        main()
