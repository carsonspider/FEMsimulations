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
from mazers_model_active import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test
)


# Configuration - modify these to change the parameter sweep
# Using active_gyroid_gen parameters

# Unit cell sizes to test (mm) - comprehensive range for full dataset
UNIT_CELL_SIZES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # mm

# Wall thickness values to test (mm) - comprehensive range for full dataset
WALL_THICKNESSES = [0.2, 0.3, 0.4, 0.5, 0.6]  # mm

# Porosity ranges to test - comprehensive range for full dataset
POROSITY_MIN_VALUES = [0.2, 0.3, 0.4]  # Minimum porosity
POROSITY_MAX_VALUES = [0.5, 0.6, 0.7, 0.8]  # Maximum porosity

# Fixed parameters for all structures - balanced for quality and speed
NUMX = 1  # Number of unit cells in x (single cell)
NUMY = 1  # Number of unit cells in y (single cell)
NUMZ = 1  # Number of unit cells in z (single cell)
NSTEPS = 20  # Voxel resolution per unit cell (balanced quality/speed)
GRAD = 1  # Graded porosity (1) or constant (0)
FUNC_DEGREE = 1  # Linear gradient
DELTA = 0.2  # Porosity tolerance
SMOOTHNESS = 0.8  # Gaussian smoothing
MARCHING_STEP = 2  # Marching cubes resolution (balanced quality/speed)

# Simulation parameters - FULL simulations
SIM_ELEMENT_SIZE = 0.05  # m (balanced for accuracy)
SIM_MAX_FORCE = 20000000.0  # N (20 MN - realistic stress levels)
SIM_NUM_STEPS = 10  # Full simulation with 10 steps

# Output settings
OUTPUT_CSV = 'dataset_full.csv'
TEMP_DIR = Path('temp_sweep_files')
TEMP_DIR.mkdir(exist_ok=True)
GYROID_OUTPUT_DIR = Path('gyroid_outputs')
GYROID_OUTPUT_DIR.mkdir(exist_ok=True)


def generate_gyroid_structure(unit_cell_size: float, wall_thickness: float, 
                               porosity_min: float, porosity_max: float, 
                               output_stl_path: Path) -> Tuple[bool, Path]:
    """Generate gyroid STL file with given parameters using active_gyroid_gen."""
    try:
        print(f"Generating gyroid: cell_size={unit_cell_size}mm, wall={wall_thickness}mm, "
              f"porosity=[{porosity_min:.2f}, {porosity_max:.2f}]")
        
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
            func_degree=FUNC_DEGREE,
            delta=DELTA,
            smoothness=SMOOTHNESS,
            marching_step=MARCHING_STEP,
            wall_thickness=wall_thickness
        )
        
        # Validate parameters
        params = validate_params(params)
        
        # Generate gyroid and create STL using the new API (disable visualization for batch processing)
        stl_dir = output_stl_path.parent
        stl_path = create_gyroid(params, stl_dir, show_plot=False)
        
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
    print(f"Unit cell sizes: {UNIT_CELL_SIZES} mm")
    print(f"Wall thicknesses: {WALL_THICKNESSES} mm")
    print(f"Porosity min values: {POROSITY_MIN_VALUES}")
    print(f"Porosity max values: {POROSITY_MAX_VALUES}")
    print(f"Mesh resolution: {NSTEPS} voxels/unit cell (reduced for speed)")
    print(f"Marching cubes step: {MARCHING_STEP} (lower = faster)")
    print(f"Simulation steps: {SIM_NUM_STEPS} (full simulation)")
    total_combinations = len(UNIT_CELL_SIZES) * len(WALL_THICKNESSES) * len(POROSITY_MIN_VALUES) * len(POROSITY_MAX_VALUES)
    print(f"Total combinations: {total_combinations}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("="*60 + "\n")
    
    # Prepare CSV output
    csv_rows: List[Dict] = []
    
    # Iterate through all parameter combinations
    current_combination = 0
    
    for unit_cell_size in UNIT_CELL_SIZES:
        for wall_thickness in WALL_THICKNESSES:
            for porosity_min in POROSITY_MIN_VALUES:
                for porosity_max in POROSITY_MAX_VALUES:
                    # Skip if min > max
                    if porosity_min > porosity_max:
                        continue
                    
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Processing combination...")
                    
                    # Create unique filename for this combination
                    stl_filename_base = f"gyroid_cs{unit_cell_size:.1f}_wt{wall_thickness:.2f}_p{porosity_min:.2f}_{porosity_max:.2f}"
                    stl_path = GYROID_OUTPUT_DIR / f"{stl_filename_base}.stl"
                    
                    # Generate gyroid structure using active_gyroid_gen
                    success, actual_stl_path = generate_gyroid_structure(
                        unit_cell_size, wall_thickness, porosity_min, porosity_max, stl_path
                    )
                    if not success:
                        print(f"Skipping simulation due to generation failure")
                        # Still record in CSV with error flag
                        csv_rows.append({
                            'unit_cell_size_mm': unit_cell_size,
                            'wall_thickness_mm': wall_thickness,
                            'porosity_min': porosity_min,
                            'porosity_max': porosity_max,
                            'stl_path': None,
                            'compressive_strength_MPa': None,
                            'max_force_N': None,
                            'cross_sectional_area_m2': None,
                            'energy_absorption_J': None,
                            'max_displacement_mm': None,
                            'max_strain': None,
                            'status': 'generation_failed',
                        })
                        continue
                    
                    # Use the actual path that was created
                    stl_path = actual_stl_path
                    
                    # Run simulation using mazers_model_active
                    results = run_simulation(stl_path)
                    
                    if results is None:
                        print(f"⚠ Skipping CSV entry due to simulation failure")
                        # Still record in CSV with error flag
                        csv_rows.append({
                            'unit_cell_size_mm': unit_cell_size,
                            'wall_thickness_mm': wall_thickness,
                            'porosity_min': porosity_min,
                            'porosity_max': porosity_max,
                            'stl_path': str(stl_path) if stl_path.exists() else None,
                            'compressive_strength_MPa': None,
                            'max_force_N': None,
                            'cross_sectional_area_m2': None,
                            'energy_absorption_J': None,
                            'max_displacement_mm': None,
                            'max_strain': None,
                            'status': 'simulation_failed',
                        })
                        continue
                    
                    # Extract summary results
                    summary = extract_results_summary(results)
                    
                    # Add input parameters
                    row = {
                        'unit_cell_size_mm': unit_cell_size,
                        'wall_thickness_mm': wall_thickness,
                        'porosity_min': porosity_min,
                        'porosity_max': porosity_max,
                        'stl_path': str(stl_path),  # STL file path
                        **summary,
                        'status': 'success',
                    }
                    
                    csv_rows.append(row)
                    
                    # Print summary
                    print(f"Results: Compressive strength = {summary['compressive_strength_MPa']:.2f} MPa")
                    
                    # Clean up STL file to save space (optional - comment out if you want to keep them)
                    # stl_path.unlink()
    
    # Write CSV file
    print(f"\n{'='*60}")
    print(f"Writing results to CSV: {OUTPUT_CSV}")
    print(f"{'='*60}")
    
    if csv_rows:
        fieldnames = [
            'unit_cell_size_mm',
            'wall_thickness_mm',
            'porosity_min',
            'porosity_max',
            'stl_path',  # STL file path
            'compressive_strength_MPa',
            'max_force_N',
            'cross_sectional_area_m2',
            'energy_absorption_J',
            'max_displacement_mm',
            'max_strain',
            'status',
        ]
        
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"✓ CSV file written: {OUTPUT_CSV}")
        print(f"✓ Total rows: {len(csv_rows)}")
        
        # Print summary statistics
        successful_rows = [r for r in csv_rows if r['status'] == 'success']
        if successful_rows:
            print(f"\nSummary Statistics:")
            print(f"  Successful simulations: {len(successful_rows)}/{len(csv_rows)}")
            strengths = [r['compressive_strength_MPa'] for r in successful_rows if r['compressive_strength_MPa'] is not None]
            if strengths:
                print(f"  Compressive strength range: {min(strengths):.2f} - {max(strengths):.2f} MPa")
                print(f"  Average compressive strength: {np.mean(strengths):.2f} MPa")
    else:
        print("No results to write!")
    
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
