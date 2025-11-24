#!/usr/bin/env python3
"""
Tile a single gyroid STL file into a grid of repeating patterns.

This script takes a single gyroid unit cell STL file and creates a larger
structure by tiling it in a grid pattern (nx × ny × nz). The gyroids are
positioned to connect seamlessly, creating an interconnected lattice structure.

Usage:
    python tile_gyroid_grid.py input.stl --nx 3 --ny 3 --nz 3 --output grid_gyroid.stl
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from stl import mesh


def load_stl(filepath: Path) -> mesh.Mesh:
    """Load an STL file and return the mesh object."""
    if not filepath.exists():
        raise FileNotFoundError(f"STL file not found: {filepath}")
    
    print(f"Loading STL file: {filepath}")
    stl_mesh = mesh.Mesh.from_file(str(filepath))
    print(f"  Loaded {len(stl_mesh)} triangles")
    return stl_mesh


def get_bounding_box(stl_mesh: mesh.Mesh) -> tuple:
    """
    Calculate the bounding box of the STL mesh.
    
    Returns:
        (min_x, max_x, min_y, max_y, min_z, max_z, width, height, depth)
    """
    # Get all vertices from all triangles
    vertices = stl_mesh.vectors.reshape(-1, 3)
    
    min_x, min_y, min_z = vertices.min(axis=0)
    max_x, max_y, max_z = vertices.max(axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    return (min_x, max_x, min_y, max_y, min_z, max_z, width, height, depth)


def translate_mesh(stl_mesh: mesh.Mesh, translation: np.ndarray) -> mesh.Mesh:
    """
    Create a translated copy of the mesh.
    
    Args:
        stl_mesh: Original mesh
        translation: Translation vector [dx, dy, dz]
    
    Returns:
        New mesh object with translated vertices
    """
    # Create a copy of the mesh
    translated_mesh = mesh.Mesh(np.zeros(stl_mesh.data.shape[0], dtype=mesh.Mesh.dtype))
    
    # Translate all vertices
    for i, triangle in enumerate(stl_mesh.vectors):
        translated_mesh.vectors[i] = triangle + translation
    
    return translated_mesh


def create_grid(stl_mesh: mesh.Mesh, nx: int, ny: int, nz: int, 
                width: float, height: float, depth: float,
                min_x: float, min_y: float, min_z: float) -> mesh.Mesh:
    """
    Create a grid of meshes by tiling the original mesh.
    Ensures watertight connection by translating each copy by the unit cell size.
    
    Args:
        stl_mesh: Original single gyroid mesh
        nx, ny, nz: Number of copies in x, y, z directions
        width, height, depth: Dimensions of a single gyroid unit (unit cell size)
        min_x, min_y, min_z: Minimum coordinates of the original mesh
    
    Returns:
        Combined mesh containing all tiled copies
    """
    print(f"\nCreating grid: {nx} × {ny} × {nz} = {nx * ny * nz} total gyroids")
    print("  Ensuring watertight connections...")
    
    # Collect all meshes
    all_meshes = []
    
    # Create copies in a 3D grid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate translation for this position
                # Translate relative to the original mesh's position
                # Each copy is offset by (i*width, j*height, k*depth) from the first
                dx = i * width
                dy = j * height
                dz = k * depth
                translation = np.array([dx, dy, dz])
                
                # Create translated copy
                translated = translate_mesh(stl_mesh, translation)
                all_meshes.append(translated)
    
    # Combine all meshes into one
    print("  Combining meshes...")
    combined_data = np.concatenate([m.data for m in all_meshes])
    combined_mesh = mesh.Mesh(combined_data)
    
    print(f"  Total triangles: {len(combined_mesh)}")
    return combined_mesh


def save_stl(mesh_obj: mesh.Mesh, output_path: Path):
    """Save the mesh to an STL file."""
    print(f"\nSaving grid STL to: {output_path}")
    mesh_obj.save(str(output_path))
    
    # Get final bounding box
    vertices = mesh_obj.vectors.reshape(-1, 3)
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    dimensions = max_coords - min_coords
    
    print(f"  Final structure dimensions:")
    print(f"    X: {dimensions[0]:.2f} mm")
    print(f"    Y: {dimensions[1]:.2f} mm")
    print(f"    Z: {dimensions[2]:.2f} mm")
    print(f"  Total volume: {dimensions[0] * dimensions[1] * dimensions[2]:.2f} mm³")


def main():
    parser = argparse.ArgumentParser(
        description="Tile a single gyroid STL into a grid of repeating patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a 3×3×3 grid
  python tile_gyroid_grid.py gyroid_20251123_191409.stl --nx 3 --ny 3 --nz 3
  
  # Create a 5×5×2 grid with custom output name
  python tile_gyroid_grid.py input.stl --nx 5 --ny 5 --nz 2 -o large_grid.stl
        """
    )
    
    parser.add_argument(
        "input_stl",
        type=str,
        help="Path to input STL file (single gyroid unit cell)"
    )
    
    parser.add_argument(
        "--nx", "--x",
        type=int,
        default=3,
        help="Number of gyroids in X direction (default: 3)"
    )
    
    parser.add_argument(
        "--ny", "--y",
        type=int,
        default=3,
        help="Number of gyroids in Y direction (default: 3)"
    )
    
    parser.add_argument(
        "--nz", "--z",
        type=int,
        default=3,
        help="Number of gyroids in Z direction (default: 3)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output STL file path (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gyroid_outputs",
        help="Output directory for generated STL files (default: gyroid_outputs)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_path = Path(args.input_stl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = input_path.stem
        output_path = output_dir / f"gyroid_grid_{input_name}_{args.nx}x{args.ny}x{args.nz}_{timestamp}.stl"
    
    print("=" * 60)
    print("Gyroid Grid Tiler")
    print("=" * 60)
    
    # Load the input STL
    input_mesh = load_stl(input_path)
    
    # Get bounding box dimensions
    min_x, max_x, min_y, max_y, min_z, max_z, width, height, depth = get_bounding_box(input_mesh)
    
    print(f"\nSingle gyroid dimensions:")
    print(f"  X: {width:.2f} mm (range: {min_x:.2f} to {max_x:.2f})")
    print(f"  Y: {height:.2f} mm (range: {min_y:.2f} to {max_y:.2f})")
    print(f"  Z: {depth:.2f} mm (range: {min_z:.2f} to {max_z:.2f})")
    print(f"\nNote: For watertight connection, gyroids will be tiled using these dimensions")
    print(f"      as the unit cell size. Ensure your input gyroid is a single periodic unit.")
    
    # Create the grid
    grid_mesh = create_grid(input_mesh, args.nx, args.ny, args.nz, 
                            width, height, depth, min_x, min_y, min_z)
    
    # Save the result
    save_stl(grid_mesh, output_path)
    
    print("\n" + "=" * 60)
    print("Grid creation complete!")
    print("=" * 60)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

