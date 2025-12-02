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

# Add _sfePy_active folder to path to import from there
import sys
_sfepy_path = Path(__file__).parent / '_sfePy_active'
sys.path.insert(0, str(_sfepy_path))

# Import gyroid generation functions from _sfePy_active folder
from active_gyroid_gen import (
    GyroidParameters,
    validate_params,
    create_gyroid
)

# Import simulation functions from _sfePy_active folder
from mazars_model_sfepy import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    run_compression_test
)


# Configuration - modify these to change the parameter sweep
# Using active_gyroid_gen parameters

# ============================================================================
# QUICK TEST MODE - Set TEST_MODE = True for fast testing (1 combination)
# ============================================================================
TEST_MODE = True  # Set to False for full parameter sweep

if TEST_MODE:
    # Quick test: Single combination for fast verification
    # Using parameters that ensure valid mesh generation
    UNIT_CELL_SIZES = [5.0]  # mm - slightly larger for better mesh
    WALL_THICKNESSES = [0.4]  # mm - thicker walls for more material
    POROSITY_MIN_VALUES = [0.2]  # Lower porosity = more material
    POROSITY_MAX_VALUES = [0.5]  # Lower max porosity = more material
    
    # Generation parameters - balanced for valid mesh but still fast
    NUMX = 1
    NUMY = 1
    NUMZ = 1
    NSTEPS = 15  # Higher resolution to ensure enough voxels for mesh
    GRAD = 1
    FUNC_DEGREE = 1
    DELTA = 0.2
    SMOOTHNESS = 0.8
    MARCHING_STEP = 2  # Lower step = more faces, better mesh quality
    
    # Fast simulation parameters
    SIM_ELEMENT_SIZE = 0.08  # m (balanced - not too large to lose detail)
    SIM_MAX_FORCE = 5000000.0  # N (5 MN - lower for faster test)
    SIM_NUM_STEPS = 2  # Just 2 steps for quick test
else:
    # Full parameter sweep - comprehensive range for full dataset
    UNIT_CELL_SIZES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # mm
    WALL_THICKNESSES = [0.2, 0.3, 0.4, 0.5, 0.6]  # mm
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
ERROR_LOG = 'simulation_errors.log'
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
            # Validate STL file size (should be at least 5KB for a valid mesh)
            file_size = stl_path.stat().st_size
            if file_size < 5000:  # Less than 5KB is suspicious
                print(f"✗ STL file too small ({file_size} bytes, {file_size/1024:.1f} KB), likely corrupted or empty mesh")
                return False, output_stl_path
            
            print(f"✓ Generated STL: {stl_path.name} ({file_size/1024:.1f} KB)")
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
        
        # Check if STL file exists
        if not stl_path.exists():
            print(f"✗ STL file not found: {stl_path}")
            return None
        
        # Check file size (very small files might be corrupted)
        file_size = stl_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            print(f"✗ STL file too small ({file_size} bytes), likely corrupted")
            return None
        
        # Material properties
        material = MaterialProperties()
        
        # Simulation parameters
        sim_params = SimulationParameters(
            element_size=SIM_ELEMENT_SIZE,
            max_force=SIM_MAX_FORCE,
            num_steps=SIM_NUM_STEPS,
        )
        
        # Load and mesh STL (returns SfePy domain, not FEniCS mesh)
        print(f"  → Loading STL and creating mesh (element_size={SIM_ELEMENT_SIZE}m)...")
        domain = load_stl_and_create_mesh(stl_path, sim_params.element_size)
        
        # Run compression test
        print(f"  → Running compression test ({SIM_NUM_STEPS} steps)...")
        results = run_compression_test(domain, material, sim_params)
        
        print(f"✓ Simulation completed")
        return results
        
    except Exception as e:
        error_msg = f"✗ Error running simulation: {e}"
        print(error_msg)
        import traceback
        tb_str = traceback.format_exc()
        print("Full traceback:")
        print(tb_str)
        
        # Log error to file for debugging
        try:
            with open(ERROR_LOG, 'a') as log_file:
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"STL: {stl_path}\n")
                log_file.write(f"Error: {e}\n")
                log_file.write(f"Traceback:\n{tb_str}\n")
                log_file.flush()  # Ensure it's written immediately
        except Exception as log_err:
            print(f"Warning: Could not write to error log: {log_err}")
        
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
    mode_str = "QUICK TEST MODE" if TEST_MODE else "FULL SIMULATION MODE"
    steps_str = f"{SIM_NUM_STEPS}-step" if not TEST_MODE else f"{SIM_NUM_STEPS}-step (TEST)"
    
    print("\n" + "="*60)
    print(f"GYROID PARAMETER SWEEP - {mode_str}")
    print(f"({steps_str} FEM simulations with Mazars damage model)")
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
    
    # Prepare CSV output - write incrementally
    csv_file_path = Path(OUTPUT_CSV)
    fieldnames = [
        'unit_cell_size_mm',
        'wall_thickness_mm',
        'porosity_min',
        'porosity_max',
        'stl_path',
        'compressive_strength_MPa',
        'max_force_N',
        'cross_sectional_area_m2',
        'energy_absorption_J',
        'max_displacement_mm',
        'max_strain',
        'status',
    ]
    
    # Initialize CSV file with header (overwrite if exists)
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"✓ CSV file initialized: {OUTPUT_CSV}")
    print(f"✓ Error log file: {ERROR_LOG}")
    print(f"  (Results will be appended after each combination)\n")
    
    # Clear error log at start
    if Path(ERROR_LOG).exists():
        Path(ERROR_LOG).unlink()
    
    # Iterate through all parameter combinations
    current_combination = 0
    successful_count = 0
    failed_count = 0
    
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
                    # Write to CSV immediately with error flag
                    row = {
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
                    }
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row)
                    failed_count += 1
                    continue
            
                # Use the actual path that was created
                stl_path = actual_stl_path
                
                # Run simulation using mazers_model_active
                results = run_simulation(stl_path)
                
                if results is None:
                    print(f"⚠ Skipping CSV entry due to simulation failure")
                    # Write to CSV immediately with error flag
                    row = {
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
                    }
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row)
                    failed_count += 1
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
                
                # Write to CSV immediately (incremental save)
                with open(csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row)
                
                successful_count += 1
                
                # Print summary
                print(f"Results: Compressive strength = {summary['compressive_strength_MPa']:.2f} MPa")
                print(f"✓ Saved to CSV ({successful_count} successful, {failed_count} failed so far)")
                
                # Clean up STL file to save space (optional - comment out if you want to keep them)
                # stl_path.unlink()
    
    # Final summary (CSV already written incrementally)
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"CSV file: {OUTPUT_CSV}")
    print(f"Total combinations processed: {current_combination}")
    print(f"  ✓ Successful: {successful_count}")
    print(f"  ✗ Failed: {failed_count}")
    
    # Read CSV to get statistics (optional - only if pandas available)
    if csv_file_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            successful_df = df[df['status'] == 'success']
            if len(successful_df) > 0:
                strengths = successful_df['compressive_strength_MPa'].dropna()
                if len(strengths) > 0:
                    print(f"\nSummary Statistics:")
                    print(f"  Compressive strength range: {strengths.min():.2f} - {strengths.max():.2f} MPa")
                    print(f"  Average compressive strength: {strengths.mean():.2f} MPa")
        except ImportError:
            # pandas not available, skip statistics
            pass
        except Exception as e:
            print(f"  (Could not compute statistics: {e})")
    
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
