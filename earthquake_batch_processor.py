#!/usr/bin/env python3
"""
Batch processing script for earthquake simulations on multiple STL files
"""

import sys
from pathlib import Path
import json
import csv
from typing import List, Dict, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from _sfePy_active.earthquake_simulator import (
    GroundMotion,
    EarthquakeIntensity,
    EarthquakeSimulationParameters,
    run_earthquake_test
)
from _sfePy_active.mazars_model_sfepy import MaterialProperties
from _sfePy_active.earthquake_visualization import plot_earthquake_results, plot_batch_comparison


def batch_earthquake_simulation(
    stl_files: List[Path],
    intensity: EarthquakeIntensity,
    output_dir: Path = Path("earthquake_batch_results"),
    element_size: float = 0.05,
    damping_ratio: float = 0.05,
    use_peer_nga: Optional[str] = None,
    scale_peer_pga: Optional[float] = None,
    create_visualizations: bool = True
) -> Dict:
    """
    Run earthquake simulations on multiple STL files
    
    Parameters:
    -----------
    stl_files : List[Path]
        List of STL file paths to simulate
    intensity : EarthquakeIntensity
        Earthquake intensity parameters
    output_dir : Path
        Directory to save results
    element_size : float
        Mesh element size (meters)
    damping_ratio : float
        Damping ratio
    use_peer_nga : str, optional
        Path to PEER NGA file (if None, uses synthetic)
    scale_peer_pga : float, optional
        Scale PEER record to this PGA (m/s²)
    create_visualizations : bool
        Whether to create visualization plots
        
    Returns:
    --------
    Dict with batch results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ground motion
    if use_peer_nga:
        print(f"Loading PEER NGA record: {use_peer_nga}")
        ground_motion = GroundMotion.from_peer_nga(use_peer_nga, scale_pga=scale_peer_pga)
    else:
        target_pga = intensity.to_pga()
        print(f"Generating synthetic ground motion with PGA: {target_pga:.3f} m/s² ({target_pga/9.81:.3f} g)")
        ground_motion = GroundMotion.synthetic(
            duration=20.0,
            pga=target_pga,
            name=f"synthetic_{target_pga/9.81:.2f}g"
        )
    
    # Material properties
    material = MaterialProperties()
    
    # Run simulations
    results_list = []
    failed_files = []
    
    print(f"\n{'='*60}")
    print(f"BATCH EARTHQUAKE SIMULATION")
    print(f"{'='*60}")
    print(f"STL files: {len(stl_files)}")
    print(f"Intensity: PGA = {ground_motion.pga/9.81:.3f} g ({ground_motion.pga:.2f} m/s²)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    for i, stl_file in enumerate(stl_files, 1):
        print(f"\n[{i}/{len(stl_files)}] Processing: {stl_file.name}")
        print("-" * 60)
        
        try:
            # Run simulation
            results = run_earthquake_test(
                stl_file,
                ground_motion,
                material=material,
                element_size=element_size,
                damping_ratio=damping_ratio
            )
            
            # Add file info
            results['stl_file'] = str(stl_file)
            results['stl_name'] = stl_file.stem
            
            # Save individual results
            result_file = output_dir / f"{stl_file.stem}_earthquake_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create visualization
            if create_visualizations:
                viz_file = output_dir / f"{stl_file.stem}_earthquake_plot.png"
                plot_earthquake_results(results, output_path=viz_file, show_plot=False)
            
            results_list.append(results)
            print(f"✓ Success: Max displacement = {results['max_displacement_mm']:.2f} mm, "
                  f"Max damage = {results['max_damage']:.3f}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed_files.append((stl_file, str(e)))
            import traceback
            traceback.print_exc()
    
    # Create summary CSV
    csv_file = output_dir / "batch_results_summary.csv"
    if results_list:
        fieldnames = [
            'stl_name', 'pga_g', 'pga_m_s2', 'max_displacement_mm', 'residual_displacement_mm',
            'max_damage', 'residual_damage', 'max_stress_MPa', 'residual_stress_MPa',
            'peak_acceleration_g', 'response_amplification', 'failure_occurred',
            'max_inter_story_drift', 'stress_concentration_factor'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_list:
                row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"\n✓ Saved summary CSV: {csv_file}")
    
    # Create batch comparison plot
    if len(results_list) > 1 and create_visualizations:
        comparison_file = output_dir / "batch_comparison.png"
        plot_batch_comparison(results_list, output_path=comparison_file)
    
    # Create summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(stl_files),
        'successful': len(results_list),
        'failed': len(failed_files),
        'intensity': {
            'pga_g': ground_motion.pga / 9.81,
            'pga_m_s2': ground_motion.pga,
            'duration': ground_motion.duration
        },
        'results_files': [str(output_dir / f"{r['stl_name']}_earthquake_results.json") 
                         for r in results_list],
        'failed_files': [{'file': str(f[0]), 'error': f[1]} for f in failed_files]
    }
    
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(results_list)}/{len(stl_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return {
        'results': results_list,
        'failed': failed_files,
        'summary': summary,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch earthquake simulation for STL files")
    parser.add_argument("stl_dir", type=str, help="Directory containing STL files")
    parser.add_argument("--pga", type=float, default=0.5, help="PGA in g units (e.g., 0.5 for 0.5g)")
    parser.add_argument("--peer-nga", type=str, help="Path to PEER NGA file")
    parser.add_argument("--scale-peer", type=float, help="Scale PEER record to this PGA (m/s²)")
    parser.add_argument("--output-dir", type=str, default="earthquake_batch_results", 
                       help="Output directory")
    parser.add_argument("--element-size", type=float, default=0.05, help="Mesh element size (m)")
    parser.add_argument("--damping", type=float, default=0.05, help="Damping ratio")
    parser.add_argument("--pattern", type=str, default="*.stl", help="STL file pattern")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    # Find STL files
    stl_dir = Path(args.stl_dir)
    stl_files = list(stl_dir.glob(args.pattern))
    
    if not stl_files:
        print(f"Error: No STL files found in {stl_dir} matching pattern {args.pattern}")
        sys.exit(1)
    
    # Create intensity
    intensity = EarthquakeIntensity.from_pga(args.pga, unit="g")
    
    # Run batch simulation
    batch_earthquake_simulation(
        stl_files=stl_files,
        intensity=intensity,
        output_dir=Path(args.output_dir),
        element_size=args.element_size,
        damping_ratio=args.damping,
        use_peer_nga=args.peer_nga,
        scale_peer_pga=args.scale_peer,
        create_visualizations=not args.no_viz
    )

