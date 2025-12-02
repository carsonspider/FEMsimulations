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

# Import gyroid generation functions
from active_gyroid_gen import (
    GyroidParameters,
    validate_params,
    create_gyroid
)

# Import simulation functions from mazers_model_active
from mazars_model_sfepy import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test
)


# Configuration - modify these to change the parameter sweep
# Using active_gyroid_gen parameters

# TPMS structure types to test
TPMS_TYPES = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']

# Unit cell sizes to test (mm) - comprehensive range for full dataset
UNIT_CELL_SIZES = [0.2, 0.3, 0.4, 0.5, 0.6]  # mm

# Wall thickness values to test (mm) - comprehensive range for full dataset
WALL_THICKNESSES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # mm

# Porosity ranges to test - comprehensive range for full dataset
POROSITY_MIN_VALUES = [0.2, 0.3, 0.4, 0.5]  # Minimum porosity
POROSITY_MAX_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]  # Maximum porosity

# Function degree values to test
FUNC_DEGREE_VALUES = [1, 2, 3]  # Linear, quadratic, cubic gradient

# Fixed parameters for all structures - balanced for quality and speed
NUMX = 1  # Number of unit cells in x (single cell)
NUMY = 1  # Number of unit cells in y (single cell)
NUMZ = 1  # Number of unit cells in z (single cell)
NSTEPS = 20  # Voxel resolution per unit cell (balanced quality/speed)
GRAD = 1  # Graded porosity (1) or constant (0)
DELTA = 0.2  # Porosity tolerance
SMOOTHNESS = 0.8  # Gaussian smoothing
MARCHING_STEP = 2  # Marching cubes resolution (balanced quality/speed)

# Simulation parameters - FULL simulations
SIM_ELEMENT_SIZE = 0.05  # m (balanced for accuracy)
SIM_MAX_FORCE = 30.0  # N (targets 10-50 MPa stress range for 0.2-0.6 mm unit cells)
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
    """Generate TPMS STL file with given parameters using active_gyroid_gen."""
    try:
        print(f"Generating {tpms_type} TPMS: cell_size={unit_cell_size}mm, wall={wall_thickness}mm, "
              f"porosity=[{porosity_min:.2f}, {porosity_max:.2f}], func_degree={func_degree}")

        # Create GyroidParameters using the new API
        params = GyroidParameters(
            numx=NUMX,
            numy=NUMY,
            numz=NUMZ,
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
        sim_params = SimulationParameters(
            element_size=SIM_ELEMENT_SIZE,
            max_force=SIM_MAX_FORCE,
            num_steps=SIM_NUM_STEPS,
        )

        # Load and mesh STL
        fenics_mesh = load_stl_and_create_mesh(stl_path, sim_params.element_size)

        # Run compression test
        results = run_compression_test(fenics_mesh, material, sim_params)

        print(f"✓ Simulation completed")
        return results

    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_results_summary(results: Dict) -> Dict:
    return {
        'compressive_strength_MPa': results['compressive_strength'] / 1e6,
        'max_force_N': results['max_force_N'],
        'cross_sectional_area_m2': results['cross_sectional_area_m2'],
        'energy_absorption_J': results['total_energy_absorption'],
        'max_displacement_mm': max([abs(d) for d in results['displacements']]) * 1000 if results['displacements'] else 0.0,
        'max_strain': max([abs(s) for s in results['strains']]) if results['strains'] else 0.0,
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
    print(f"Mesh resolution: {NSTEPS} voxels/unit cell (reduced for speed)")
    print(f"Marching cubes step: {MARCHING_STEP} (lower = faster)")
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
                            print(f"Results: Compressive strength = {summary['compressive_strength_MPa']:.2f} MPa")
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
            strengths = [float(r['compressive_strength_MPa']) for r in successful_rows if r.get('compressive_strength_MPa') and r['compressive_strength_MPa'] != 'None']
            if strengths:
                print(f"  Compressive strength range: {min(strengths):.2f} - {max(strengths):.2f} MPa")
                print(f"  Average compressive strength: {np.mean(strengths):.2f} MPa")
    else:
        print("No CSV file found!")

    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()