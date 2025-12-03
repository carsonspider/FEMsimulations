import os
import sys
import csv
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Set matplotlib to non-interactive backend before importing (prevents plot windows)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np

# Import
#  gyroid generation functions

from active_gyroid_gen import (
    GyroidParameters,
    
    validate_params,
    create_gyroid
)

# 
#Import simulation functions from mazers_model_active
from mazars_model_sfepy import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test,
    run_tensile_test
)


# Configuration - modify these to change the parameter sweep
# Using active_gyroid_gen parameters

# TPMS structure types to test
TPMS_TYPES = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']

# Unit cell sizes to test (mm) - comprehensive range for full dataset
# Note: Very small cells with thin walls and high porosity may fail
UNIT_CELL_SIZES = [0.5, 0.6, 0.7, 0.8, 1.0]  # mm (increased minimum to 0.5mm for reliability)

# Wall thickness values to test (mm) - comprehensive range for full dataset
# Note: Wall thickness should be at least 30% of unit cell size for valid geometry
WALL_THICKNESSES = [0.3, 0.4, 0.5, 0.6, 0.7]  # mm (increased minimum to 0.3mm for reliability)

# Porosity ranges to test - comprehensive range for full dataset
POROSITY_MIN_VALUES = [0.2, 0.3, 0.4, 0.5]  # Minimum porosity
POROSITY_MAX_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]  # Maximum porosity

# Function degree values to test (0=constant, 1=linear, 2=quadratic)
FUNC_DEGREE_VALUES = [1, 2]  # Linear, quadratic gradient (cubic not supported)

# Fixed parameters for all structures - balanced for quality and speed
# FIXED VOLUME: All structures will have the same overall dimensions
FIXED_SIZE_MM = 10.0  # Fixed size in mm (all structures will be 10mm x 10mm x 10mm cubes)
NSTEPS = 25  # Voxel resolution per unit cell (increased for better geometry capture)
GRAD = 1  # Graded porosity (1) or constant (0)
DELTA = 0.2  # Porosity tolerance
SMOOTHNESS = 0.8  # Gaussian smoothing
MARCHING_STEP = 1  # Marching cubes resolution (reduced from 2 to 1 for finer meshes)

# Simulation parameters - FULL simulations
SIM_ELEMENT_SIZE = 0.05  # m (balanced for accuracy)
# Fixed force: For 10mm cube (0.01m × 0.01m = 0.0001 m²), targeting ~35 MPa stress
# Force = Stress × Area = 35e6 Pa × 0.0001 m² = 3500 N = 3.5 kN
SIM_MAX_FORCE = 3500.0  # Fixed force in N (3.5 kN) - targets ~35 MPa for 10mm structures
SIM_NUM_STEPS = 10  # Full simulation with 10 steps

# Test limit - set to None to test all combinations
MAX_COMBINATIONS = None  # Test all combinations for full dataset

# Output settings
OUTPUT_CSV = 'dataset_full.csv'
TEMP_DIR = Path('temp_sweep_files')
TEMP_DIR.mkdir(exist_ok=True)
GYROID_OUTPUT_DIR = Path('gyroid_outputs')
GYROID_OUTPUT_DIR.mkdir(exist_ok=True)


def generate_gyroid_structure(unit_cell_size: float, wall_thickness: float,
                               porosity_min: float, porosity_max: float,
                               output_stl_path: Path, tpms_type: str = 'gyroid', func_degree: int = 1) -> Tuple[bool, Path]:
    """Generate TPMS STL file with given parameters using active_gyroid_gen.
    
    All structures will have the same overall dimensions (FIXED_SIZE_MM x FIXED_SIZE_MM x FIXED_SIZE_MM).
    The number of unit cells is calculated to fill this fixed volume based on unit_cell_size.
    """
    try:
        # Calculate number of unit cells needed to fill the fixed volume
        # Round down to ensure we don't exceed the fixed size
        numx = int(FIXED_SIZE_MM / unit_cell_size)
        numy = int(FIXED_SIZE_MM / unit_cell_size)
        numz = int(FIXED_SIZE_MM / unit_cell_size)
        
        # Ensure at least 1 unit cell in each direction
        numx = max(1, numx)
        numy = max(1, numy)
        numz = max(1, numz)
        
        actual_size_x = numx * unit_cell_size
        actual_size_y = numy * unit_cell_size
        actual_size_z = numz * unit_cell_size
        
        print(f"Generating {tpms_type} TPMS: cell_size={unit_cell_size}mm, wall={wall_thickness}mm, "
              f"porosity=[{porosity_min:.2f}, {porosity_max:.2f}], func_degree={func_degree}")
        print(f"  Fixed volume: {FIXED_SIZE_MM}mm³ → {numx}x{numy}x{numz} cells = {actual_size_x:.2f}x{actual_size_y:.2f}x{actual_size_z:.2f}mm")

        # Create GyroidParameters using the new API
        params = GyroidParameters(
            numx=numx,
            numy=numy,
            numz=numz,
            unit_cell_size=unit_cell_size,
            nsteps=NSTEPS,
            porosity_min=porosity_min,
            porosity_max=porosity_max,
            grad=GRAD,
            func_degree=func_degree,
            delta=DELTA,
            smoothness=SMOOTHNESS,
            marching_step=MARCHING_STEP,
            wall_thickness=wall_thickness,
            tpms_type=tpms_type
        )

        # Validate parameters
        params = validate_params(params)

        # Generate gyroid and create STL using the new API (disable visualization for batch processing)
        stl_dir = output_stl_path.parent
        stl_path = create_gyroid(params, stl_dir)

        if stl_path.exists():
            print(f"✓ Generated STL: {stl_path.name}")
            return True, stl_path
        else:
            print(f"✗ STL file not created at {stl_path}")
            return False, output_stl_path

    except ValueError as e:
        # Handle specific errors like "Surface level must be within volume data range"
        if "Surface level must be within volume data range" in str(e):
            print(f"✗ Invalid geometry: parameters produce empty/invalid volume (likely wall too thin or cell too small)")
        else:
            print(f"✗ Validation error: {e}")
        return False, output_stl_path
    except Exception as e:
        print(f"✗ Error generating gyroid: {e}")
        import traceback
        traceback.print_exc()
        return False, output_stl_path


def run_simulation(stl_path: Path) -> Dict:
    try:
        print(f"Running simulation on: {stl_path.name}")

        # Material properties
        material = MaterialProperties()

        # Simulation parameters
        # Use fixed force for all geometries to enable fair comparison
        # All structures have the same fixed size (10mm × 10mm × 10mm), so cross-sectional area is the same
        sim_params = SimulationParameters(
            element_size=SIM_ELEMENT_SIZE,
            max_force=SIM_MAX_FORCE,  # Fixed force (same for all geometries)
            target_stress_mpa=35.0,  # Not used when max_force is specified, but updated to match default
            num_steps=SIM_NUM_STEPS,
        )

        # Load and mesh STL
        fenics_mesh = load_stl_and_create_mesh(stl_path, sim_params.element_size)

        # Run compression test
        compression_results = run_compression_test(fenics_mesh, material, sim_params)
        
        # Run tension test
        tension_results = run_tensile_test(fenics_mesh, material, sim_params)
        
        # Combine results (similar to main() function in mazars_model_sfepy.py)
        results = {
            "compression": compression_results,
            "tension": tension_results,
            "compressive_strength": compression_results['compressive_strength'],
            "tensile_strength": tension_results['tensile_strength'],
            "cross_sectional_area_m2": compression_results['cross_sectional_area_m2'],
        }

        print(f"✓ Simulation completed (compression + tension)")
        return results

    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_results_summary(results: Dict) -> Dict:
    """Extract summary results from simulation, handling both old and new result formats."""
    # Handle new format with compression/tension split
    if 'compression' in results and 'tension' in results:
        comp = results['compression']
        tens = results['tension']
        return {
            'compressive_strength_MPa': results.get('compressive_strength', comp.get('compressive_strength', 0.0)) / 1e6,
            'tensile_strength_MPa': results.get('tensile_strength', tens.get('tensile_strength', 0.0)) / 1e6,
            'max_force_N': comp.get('max_force_N', 0.0),
            'cross_sectional_area_m2': results.get('cross_sectional_area_m2', comp.get('cross_sectional_area_m2', 0.0)),
            'energy_absorption_J': comp.get('total_energy_absorption', 0.0),
            'max_displacement_mm': max([abs(d) for d in comp.get('displacements', [])]) * 1000 if comp.get('displacements') else 0.0,
            'max_strain': max([abs(s) for s in comp.get('strains', [])]) if comp.get('strains') else 0.0,
        }
    else:
        # Handle old format (compression only)
        return {
            'compressive_strength_MPa': results.get('compressive_strength', 0.0) / 1e6,
            'tensile_strength_MPa': 0.0,  # Not available in old format
            'max_force_N': results.get('max_force_N', 0.0),
            'cross_sectional_area_m2': results.get('cross_sectional_area_m2', 0.0),
            'energy_absorption_J': results.get('total_energy_absorption', 0.0),
            'max_displacement_mm': max([abs(d) for d in results.get('displacements', [])]) * 1000 if results.get('displacements') else 0.0,
            'max_strain': max([abs(s) for s in results.get('strains', [])]) if results.get('strains') else 0.0,
        }


def main():
    print("\n" + "="*60)
    print("GYROID PARAMETER SWEEP - FULL SIMULATION MODE")
    print("(10-step FEM simulations with Mazars damage model)")
    print("="*60)
    print(f"TPMS types: {TPMS_TYPES}")
    print(f"Unit cell sizes: {UNIT_CELL_SIZES} mm")
    print(f"Wall thicknesses: {WALL_THICKNESSES} mm")
    print(f"Porosity min values: {POROSITY_MIN_VALUES}")
    print(f"Porosity max values: {POROSITY_MAX_VALUES}")
    print(f"Function degree values: {FUNC_DEGREE_VALUES}")
    print(f"FIXED VOLUME: All structures = {FIXED_SIZE_MM}mm × {FIXED_SIZE_MM}mm × {FIXED_SIZE_MM}mm")
    print(f"  (Number of unit cells adjusts automatically based on unit_cell_size)")
    print(f"Mesh resolution: {NSTEPS} voxels/unit cell")
    print(f"Marching cubes step: {MARCHING_STEP}")
    print(f"Simulation steps: {SIM_NUM_STEPS} (full simulation)")
    total_combinations = len(TPMS_TYPES) * len(UNIT_CELL_SIZES) * len(WALL_THICKNESSES) * len(POROSITY_MIN_VALUES) * len(POROSITY_MAX_VALUES) * len(FUNC_DEGREE_VALUES)
    print(f"Total possible combinations: {total_combinations}")
    if MAX_COMBINATIONS:
        print(f"Testing limit: {MAX_COMBINATIONS} combinations")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("="*60 + "\n")

    # Prepare CSV output - open file for incremental writing
    fieldnames = [
        'tpms_type',
        'unit_cell_size_mm',
        'wall_thickness_mm',
        'porosity_min',
        'porosity_max',
        'func_degree',
        'stl_path',
        'compressive_strength_MPa',
        'tensile_strength_MPa',
        'max_force_N',
        'cross_sectional_area_m2',
        'energy_absorption_J',
        'max_displacement_mm',
        'max_strain',
        'status',
    ]

    # Open CSV file for writing (write mode - creates new file or overwrites)
    csvfile = open(OUTPUT_CSV, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header immediately
    writer.writeheader()
    csvfile.flush()  # Ensure header is written immediately
    print(f"Opened CSV file for incremental writing: {OUTPUT_CSV}")

    # Iterate through all parameter combinations
    current_combination = 0
    successful_count = 0

    for tpms_type in TPMS_TYPES:
        for unit_cell_size in UNIT_CELL_SIZES:
            for wall_thickness in WALL_THICKNESSES:
                for porosity_min in POROSITY_MIN_VALUES:
                    for porosity_max in POROSITY_MAX_VALUES:
                        for func_degree in FUNC_DEGREE_VALUES:
                            # Skip if min > max
                            if porosity_min > porosity_max:
                                continue
                            
                            # Skip if wall thickness is too small relative to unit cell size
                            # Wall thickness should be at least 30% of unit cell size for valid geometry
                            if wall_thickness < 0.3 * unit_cell_size:
                                print(f"  Skipping: wall_thickness ({wall_thickness}mm) too small for unit_cell_size ({unit_cell_size}mm)")
                                continue
                            
                            # Skip if high porosity with thin walls (likely to fail)
                            # High porosity (>0.7) with thin walls (<0.4mm) often produces invalid geometries
                            if porosity_max > 0.7 and wall_thickness < 0.4:
                                print(f"  Skipping: high porosity ({porosity_max:.2f}) with thin wall ({wall_thickness}mm) likely to fail")
                                continue

                            # Check if we've reached the limit (before processing)
                            if MAX_COMBINATIONS and successful_count >= MAX_COMBINATIONS:
                                print(f"\n{'='*60}")
                                print(f"Reached test limit of {MAX_COMBINATIONS} successful combinations")
                                print(f"{'='*60}")
                                csvfile.close()
                                return

                            current_combination += 1
                            print(f"\n[{current_combination}/{total_combinations if not MAX_COMBINATIONS else MAX_COMBINATIONS}] Processing combination...")
                            print(f"  TPMS type: {tpms_type}, func_degree: {func_degree}")

                            # Create unique filename for this combination
                            stl_filename_base = f"{tpms_type}_cs{unit_cell_size:.1f}_wt{wall_thickness:.2f}_p{porosity_min:.2f}_{porosity_max:.2f}_fd{func_degree}"
                            stl_path = GYROID_OUTPUT_DIR / f"{stl_filename_base}.stl"

                            # Generate TPMS structure using active_gyroid_gen
                            success, actual_stl_path = generate_gyroid_structure(
                                unit_cell_size, wall_thickness, porosity_min, porosity_max, stl_path, tpms_type=tpms_type, func_degree=func_degree
                            )
                            if not success:
                                print(f"Skipping simulation due to generation failure")
                                # Still record in CSV with error flag
                                row = {
                                    'tpms_type': tpms_type,
                                    'unit_cell_size_mm': unit_cell_size,
                                    'wall_thickness_mm': wall_thickness,
                                    'porosity_min': porosity_min,
                                    'porosity_max': porosity_max,
                                    'func_degree': func_degree,
                                    'stl_path': None,
                                    'compressive_strength_MPa': None,
                                    'tensile_strength_MPa': None,
                                    'max_force_N': None,
                                    'cross_sectional_area_m2': None,
                                    'energy_absorption_J': None,
                                    'max_displacement_mm': None,
                                    'max_strain': None,
                                    'status': 'generation_failed',
                                }
                                writer.writerow(row)
                                csvfile.flush()  # Write immediately
                                print(f"  → Written to CSV: generation_failed")
                                continue

                            # Use the actual path that was created
                            stl_path = actual_stl_path

                            # Run simulation using mazers_model_active
                            results = run_simulation(stl_path)

                            if results is None:
                                print(f"⚠ Skipping CSV entry due to simulation failure")
                                # Still record in CSV with error flag
                                row = {
                                    'tpms_type': tpms_type,
                                    'unit_cell_size_mm': unit_cell_size,
                                    'wall_thickness_mm': wall_thickness,
                                    'porosity_min': porosity_min,
                                    'porosity_max': porosity_max,
                                    'func_degree': func_degree,
                                    'stl_path': str(stl_path) if stl_path.exists() else None,
                                    'compressive_strength_MPa': None,
                                    'tensile_strength_MPa': None,
                                    'max_force_N': None,
                                    'cross_sectional_area_m2': None,
                                    'energy_absorption_J': None,
                                    'max_displacement_mm': None,
                                    'max_strain': None,
                                    'status': 'simulation_failed',
                                }
                                writer.writerow(row)
                                csvfile.flush()  # Write immediately
                                print(f"  → Written to CSV: simulation_failed")
                                continue

                            # Extract summary results
                            summary = extract_results_summary(results)

                            # Add input parameters
                            row = {
                                'tpms_type': tpms_type,
                                'unit_cell_size_mm': unit_cell_size,
                                'wall_thickness_mm': wall_thickness,
                                'porosity_min': porosity_min,
                                'porosity_max': porosity_max,
                                'func_degree': func_degree,
                                'stl_path': str(stl_path),  # STL file path
                                **summary,
                                'status': 'success',
                            }

                            # Write to CSV immediately
                            writer.writerow(row)
                            csvfile.flush()  # Ensure data is written to disk

                            successful_count += 1

                            # Print summary
                            print(f"Results: Compressive strength = {summary['compressive_strength_MPa']:.2f} MPa, "
                                  f"Tensile strength = {summary['tensile_strength_MPa']:.2f} MPa")
                            print(f"  → Written to CSV: success ({successful_count}/{MAX_COMBINATIONS if MAX_COMBINATIONS else 'all'})")

                            # Clean up STL file to save space (optional - comment out if you want to keep them)
                            # stl_path.unlink()

    # Close CSV file
    csvfile.close()

    # Read back CSV to get statistics
    print(f"\n{'='*60}")
    print(f"CSV file: {OUTPUT_CSV}")
    print(f"{'='*60}")

    # Read CSV to get statistics
    csv_rows: List[Dict] = []
    if Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        print(f"✓ Total rows in CSV: {len(csv_rows)}")

        # Print summary statistics
        successful_rows = [r for r in csv_rows if r.get('status') == 'success']
        if successful_rows:
            print(f"\nSummary Statistics:")
            print(f"  Successful simulations: {len(successful_rows)}/{len(csv_rows)}")
            comp_strengths = [float(r['compressive_strength_MPa']) for r in successful_rows if r.get('compressive_strength_MPa') and r['compressive_strength_MPa'] != 'None']
            if comp_strengths:
                print(f"  Compressive strength range: {min(comp_strengths):.2f} - {max(comp_strengths):.2f} MPa")
                print(f"  Average compressive strength: {np.mean(comp_strengths):.2f} MPa")
            tens_strengths = [float(r['tensile_strength_MPa']) for r in successful_rows if r.get('tensile_strength_MPa') and r['tensile_strength_MPa'] != 'None' and r['tensile_strength_MPa'] != '0.0']
            if tens_strengths:
                print(f"  Tensile strength range: {min(tens_strengths):.2f} - {max(tens_strengths):.2f} MPa")
                print(f"  Average tensile strength: {np.mean(tens_strengths):.2f} MPa")
                if comp_strengths and len(comp_strengths) == len(tens_strengths):
                    ratios = [t/c for t, c in zip(tens_strengths, comp_strengths)]
                    print(f"  Tensile/Compressive ratio: {np.mean(ratios):.3f} (range: {min(ratios):.3f} - {max(ratios):.3f})")
    else:
        print("No CSV file found!")

    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()