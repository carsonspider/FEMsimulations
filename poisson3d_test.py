from pathlib import Path

# Use Netgen/NGSolve to solve a 3D Poisson problem on an STL geometry
# -Delta u = 1 in Omega, u = 0 on boundary

from ngsolve import *
from ngsolve.webgui import Draw

# Netgen STL geometry import
from netgen.stl import STLGeometry


def load_mesh_from_stl(stl_path: Path, maxh: float = 0.2):
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")
    geo = STLGeometry(str(stl_path))
    ngmesh = geo.GenerateMesh(maxh=maxh)
    # optional: ensure boundary info is prepared
    try:
        ngmesh.CalcSurfaces()
    except Exception:
        pass
    return Mesh(ngmesh)


def solve_poisson(mesh, order: int = 2):
    fes = H1(mesh, order=order, dirichlet=".*")  # Dirichlet on all boundaries
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes, symmetric=True)
    a += grad(u) * grad(v) * dx

    f = LinearForm(fes)
    f += 1.0 * v * dx

    a.Assemble()
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

    return fes, gfu, a, f


def report_numerics(mesh, fes, gfu):
    volume = Integrate(1.0, mesh)
    l2_u = sqrt(Integrate(gfu * gfu, mesh))
    h1_semi_u = sqrt(Integrate(grad(gfu) * grad(gfu), mesh))

    print("=== FEM Results (3D Poisson on STL) ===")
    print(f"ndof: {fes.ndof}")
    print(f"domain volume: {volume}")
    print(f"L2 norm of u: {l2_u}")
    print(f"H1 seminorm of u: {h1_semi_u}")


def visualize(mesh, gfu):
    print("\nGenerating matplotlib visualization...")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from datetime import datetime
    
    # Extract mesh data
    vertices = []
    for i in range(mesh.nv):
        p = mesh.vertices[i].point
        vertices.append([p[0], p[1], p[2]])
    vertices = np.array(vertices)
    
    # Get solution values at vertices
    solution_values = np.array([gfu.vec[i] for i in range(mesh.nv)])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot of solution
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         c=solution_values, cmap='viridis', s=2, alpha=0.6)
    ax1.set_title('Solution u (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # XY cross-section (z=0)
    ax2 = fig.add_subplot(222)
    z_mask = np.abs(vertices[:, 2]) < 0.1
    if np.any(z_mask):
        scatter2 = ax2.scatter(vertices[z_mask, 0], vertices[z_mask, 1], 
                              c=solution_values[z_mask], cmap='viridis', s=1)
        ax2.set_title('Solution u (XY plane)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(scatter2, ax=ax2)
    
    # XZ cross-section (y=0)
    ax3 = fig.add_subplot(223)
    y_mask = np.abs(vertices[:, 1]) < 0.1
    if np.any(y_mask):
        scatter3 = ax3.scatter(vertices[y_mask, 0], vertices[y_mask, 2], 
                              c=solution_values[y_mask], cmap='viridis', s=1)
        ax3.set_title('Solution u (XZ plane)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        plt.colorbar(scatter3, ax=ax3)
    
    # YZ cross-section (x=0)
    ax4 = fig.add_subplot(224)
    x_mask = np.abs(vertices[:, 0]) < 0.1
    if np.any(x_mask):
        scatter4 = ax4.scatter(vertices[x_mask, 1], vertices[x_mask, 2], 
                              c=solution_values[x_mask], cmap='viridis', s=1)
        ax4.set_title('Solution u (YZ plane)')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        plt.colorbar(scatter4, ax=ax4)
    
    plt.tight_layout()
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_cube_ngsolve_poisson_{timestamp}.png"
    filepath = f"/Users/juliettecarson/Desktop/ngsolve/{filename}"
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.show()
    print("Matplotlib visualization displayed.")


def main():
    # Absolute path to your STL on Desktop/ngsolve
    stl_path = Path("/Users/juliettecarson/Desktop/ngsolve/test_cube.stl")

    # Generate a coarse mesh for quick testing. Lower maxh for finer mesh.
    mesh = load_mesh_from_stl(stl_path, maxh=0.5)

    fes, gfu, a, f = solve_poisson(mesh, order=2)

    report_numerics(mesh, fes, gfu)

    # Interactive visualization in browser
    visualize(mesh, gfu)


if __name__ == "__main__":
    main()


