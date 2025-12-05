#!/usr/bin/env python3
"""
Visualization tools for earthquake simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import json


def plot_earthquake_results(results: Dict, output_path: Optional[Path] = None, 
                           show_plot: bool = True) -> Path:
    """
    Create comprehensive visualization of earthquake simulation results
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from earthquake simulation
    output_path : Path, optional
        Path to save figure (if None, auto-generates name)
    show_plot : bool
        Whether to display plot
        
    Returns:
    --------
    Path to saved figure
    """
    if output_path is None:
        output_path = Path(f"earthquake_results_{results['ground_motion_name']}.png")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    time = np.array(results['time_history'])
    
    # 1. Ground motion (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ground_acc = results.get('ground_motion_acceleration', None)
    if ground_acc is None:
        # Reconstruct from PGA if available
        ax1.text(0.5, 0.5, f"PGA: {results['pga_g']:.3f} g\n({results['pga']:.2f} m/s²)",
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    else:
        ax1.plot(time, ground_acc / 9.81, 'k-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Ground Acceleration (g)')
    ax1.set_title('Ground Motion')
    ax1.grid(True, alpha=0.3)
    
    # 2. Displacement time history (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    displacements = np.array(results['displacement_history'])
    if displacements.ndim == 2:
        # Plot maximum displacement
        max_disp = np.max(np.abs(displacements), axis=1)
        ax2.plot(time, max_disp * 1000, 'b-', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (mm)')
    ax2.set_title(f'Max Displacement: {results["max_displacement_mm"]:.2f} mm')
    ax2.grid(True, alpha=0.3)
    
    # 3. Damage evolution (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    damages = np.array(results['damage_history'])
    if damages.ndim == 2:
        max_damage = np.max(damages, axis=1)
        ax3.plot(time, max_damage, 'r-', linewidth=1.5)
    ax3.axhline(y=0.5, color='orange', linestyle='--', label='Failure threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Damage')
    ax3.set_title(f'Max Damage: {results["max_damage"]:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    # 4. Stress time history (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    stresses = results.get('stress_history', None)
    if stresses is not None:
        stress_array = np.array(stresses)
        if stress_array.ndim == 2:
            max_stress = np.max(np.abs(stress_array), axis=1)
        else:
            max_stress = np.abs(stress_array)
        ax4.plot(time, max_stress / 1e6, 'g-', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Stress (MPa)')
    ax4.set_title(f'Max Stress: {results["max_stress_MPa"]:.2f} MPa')
    ax4.grid(True, alpha=0.3)
    
    # 5. Acceleration response (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    accelerations = results.get('acceleration_history', None)
    if accelerations is not None:
        accel_array = np.array(accelerations)
        if accel_array.ndim == 2:
            max_accel = np.max(np.abs(accel_array), axis=1)
        else:
            max_accel = np.abs(accel_array)
        ax5.plot(time, max_accel / 9.81, 'm-', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Acceleration (g)')
    ax5.set_title(f'Peak Response: {results["peak_acceleration_g"]:.3f} g\n'
                  f'Amplification: {results["response_amplification"]:.2f}x')
    ax5.grid(True, alpha=0.3)
    
    # 6. Energy dissipation (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    # Plot kinetic and strain energy if available
    if 'max_kinetic_energy_J' in results and 'max_strain_energy_J' in results:
        energies = [results['max_kinetic_energy_J'], results['max_strain_energy_J']]
        labels = ['Kinetic', 'Strain']
        colors = ['blue', 'red']
        ax6.bar(labels, energies, color=colors, alpha=0.7)
    ax6.set_ylabel('Energy (J)')
    ax6.set_title('Energy Metrics')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Summary statistics (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    summary_text = f"""
EARTHQUAKE SIMULATION RESULTS
{'='*40}
Ground Motion: {results['ground_motion_name']}
PGA: {results['pga_g']:.3f} g ({results['pga']:.2f} m/s²)
Duration: {results['duration']:.2f} s

DISPLACEMENT:
  Max: {results['max_displacement_mm']:.2f} mm
  Residual: {results['residual_displacement_mm']:.2f} mm
  Max Drift: {results['max_inter_story_drift']:.4f}

DAMAGE:
  Max: {results['max_damage']:.3f}
  Residual: {results['residual_damage']:.3f}
  Failure: {'YES' if results['failure_occurred'] else 'NO'}
  {'Failure Time: ' + str(results['failure_time_s']) + ' s' if results['failure_time_s'] else ''}

STRESS:
  Max: {results['max_stress_MPa']:.2f} MPa
  Residual: {results['residual_stress_MPa']:.2f} MPa
  Concentration: {results['stress_concentration_factor']:.2f}x
"""
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')
    
    # 8. Response spectrum (bottom middle) - simplified
    ax8 = fig.add_subplot(gs[2, 1])
    # For full implementation, would compute response spectrum
    ax8.text(0.5, 0.5, 'Response Spectrum\n(Full implementation\nwould show Sa vs T)',
            ha='center', va='center', transform=ax8.transAxes, fontsize=10)
    ax8.set_xlabel('Period (s)')
    ax8.set_ylabel('Spectral Acceleration (g)')
    ax8.set_title('Response Spectrum')
    ax8.grid(True, alpha=0.3)
    
    # 9. Damage distribution histogram (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    if damages.ndim == 2:
        final_damage = damages[-1]
        ax9.hist(final_damage, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Damage')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Final Damage Distribution')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Earthquake Simulation: {results["ground_motion_name"]}', 
                 fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path


def plot_batch_comparison(results_list: list, output_path: Optional[Path] = None) -> Path:
    """
    Compare multiple earthquake simulation results
    
    Parameters:
    -----------
    results_list : list
        List of results dictionaries
    output_path : Path, optional
        Output path for figure
    """
    if output_path is None:
        output_path = Path("earthquake_batch_comparison.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    names = [r['ground_motion_name'] for r in results_list]
    pgas = [r['pga_g'] for r in results_list]
    max_disp = [r['max_displacement_mm'] for r in results_list]
    max_damage = [r['max_damage'] for r in results_list]
    max_stress = [r['max_stress_MPa'] for r in results_list]
    
    # 1. Max displacement vs PGA
    ax1 = axes[0, 0]
    ax1.scatter(pgas, max_disp, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax1.annotate(name[:15], (pgas[i], max_disp[i]), fontsize=8)
    ax1.set_xlabel('PGA (g)')
    ax1.set_ylabel('Max Displacement (mm)')
    ax1.set_title('Displacement vs Ground Motion Intensity')
    ax1.grid(True, alpha=0.3)
    
    # 2. Max damage vs PGA
    ax2 = axes[0, 1]
    ax2.scatter(pgas, max_damage, s=100, alpha=0.7, color='red')
    for i, name in enumerate(names):
        ax2.annotate(name[:15], (pgas[i], max_damage[i]), fontsize=8)
    ax2.set_xlabel('PGA (g)')
    ax2.set_ylabel('Max Damage')
    ax2.set_title('Damage vs Ground Motion Intensity')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Failure')
    ax2.legend()
    
    # 3. Max stress vs PGA
    ax3 = axes[1, 0]
    ax3.scatter(pgas, max_stress, s=100, alpha=0.7, color='green')
    for i, name in enumerate(names):
        ax3.annotate(name[:15], (pgas[i], max_stress[i]), fontsize=8)
    ax3.set_xlabel('PGA (g)')
    ax3.set_ylabel('Max Stress (MPa)')
    ax3.set_title('Stress vs Ground Motion Intensity')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = []
    for r in results_list:
        table_data.append([
            r['ground_motion_name'][:20],
            f"{r['pga_g']:.3f}",
            f"{r['max_displacement_mm']:.2f}",
            f"{r['max_damage']:.3f}",
            f"{r['max_stress_MPa']:.2f}"
        ])
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Name', 'PGA (g)', 'Max Disp (mm)', 'Max Damage', 'Max Stress (MPa)'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    plt.suptitle('Earthquake Simulation Batch Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved batch comparison to: {output_path}")
    
    plt.close()
    return output_path

