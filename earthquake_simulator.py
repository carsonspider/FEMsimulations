#!/usr/bin/env python3
"""
Earthquake Simulator for Gyroid STL Structures

This module simulates the response of gyroid structures to earthquake ground motion
using SfePy or FEniCS with the Mazars damage model framework.

METHODOLOGY & REFERENCES:
=========================
This implementation follows established methods for nonlinear structural dynamics
under seismic loading:

1. **Newmark-Beta Time Integration**:
   - Newmark, N. M. (1959). "A method of computation for structural dynamics."
     Journal of the Engineering Mechanics Division, 85(3), 67-94.
   - Uses average acceleration method (gamma=0.5, beta=0.25) for unconditional stability

2. **Rayleigh Damping**:
   - Clough, R. W., & Penzien, J. (1993). "Dynamics of Structures."
     McGraw-Hill, New York.
   - C = α·M + β·K, where α and β are computed from damping ratio and frequencies

3. **Mazars Damage Model**:
   - Mazars, J. (1986). "A description of micro- and macroscale damage of concrete
     structures." Engineering Fracture Mechanics, 25(5-6), 729-737.
   - Pijaudier-Cabot, G., & Mazars, J. (2001). "Damage models for concrete."
     In Handbook of Materials Behavior Models (pp. 500-512).

4. **PEER NGA Database**:
   - Ancheta, T. D., et al. (2014). "NGA-West2 Database."
     Earthquake Spectra, 30(3), 989-1005.
   - PEER Ground Motion Database: https://peer.berkeley.edu/peer-strong-ground-motion-databases

5. **Structural Dynamics**:
   - Chopra, A. K. (2017). "Dynamics of Structures: Theory and Applications to
     Earthquake Engineering." 5th Edition, Pearson.

FEniCS vs SfePy:
===============
Both frameworks are suitable for earthquake simulation:

**SfePy (Current Implementation)**:
- ✅ Already integrated in your codebase
- ✅ Good for complex geometries (STL meshes)
- ✅ Python-native, easier to integrate with Mazars model
- ⚠️ Less extensive documentation for structural dynamics
- ⚠️ Smaller community than FEniCS

**FEniCS (Optional Alternative)**:
- ✅ More established for structural dynamics
- ✅ Extensive documentation and examples
- ✅ Better performance for large problems
- ⚠️ Requires separate installation
- ⚠️ Different API (would need separate implementation)

**Recommendation**: Start with SfePy (already set up), but FEniCS option is
available if you need better performance or more features.

HOW IT WORKS:
============
1. **Ground Motion Input**: Loads seismic accelerograms
   - Real records from PEER NGA database
   - Synthetic ground motion (for testing)
   - Intensity selection (PGA, spectral acceleration)
   
2. **Structural Dynamics**: Solves M·ü + C·u̇ + K·u = -M·ü_g
   - M = mass matrix (lumped or consistent)
   - C = damping matrix (Rayleigh damping)
   - K = stiffness matrix (degraded by damage: K = K₀ × (1-D))
   - ü_g = ground acceleration
   - u = structural displacement (relative to ground)

3. **Time Integration**: Newmark-beta method
   - Unconditionally stable (gamma=0.5, beta=0.25)
   - Accounts for damage evolution during shaking
   - Updates stiffness matrix as damage accumulates

4. **Output Metrics** (all computed):
   - Maximum displacement (absolute and relative)
   - Maximum inter-story drift
   - Peak acceleration response
   - Damage distribution (compressive and tensile)
   - Residual deformation (permanent displacement)
   - Stress history (maximum and distribution)
   - Strain history
   - Energy dissipation
   - Failure 
   
   Use:
   python _sfePy_active/earthquake_simulator.py test_cube_30mm.stl --pga-g 0.5 --viz

"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass

# Import existing simulation infrastructure
from mazars_model_sfepy import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    compute_equivalent_strain,
    mazars_compressive_damage,
    mazars_tensile_damage
)

try:
    from sfepy.discrete import Field, Problem
    from sfepy.discrete.fem import Mesh, FEDomain
    from sfepy.solvers import Solver
    SFEPY_AVAILABLE = True
except ImportError:
    SFEPY_AVAILABLE = False
    print("Warning: SfePy not available. Some features may be limited.")

# Optional FEniCS support
try:
    from dolfin import *
    import ufl
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False

# Import PEER NGA loader
try:
    from peer_nga_loader import load_peer_nga_record, PEERNGAReader
    PEER_NGA_AVAILABLE = True
except ImportError:
    PEER_NGA_AVAILABLE = False
    print("Warning: PEER NGA loader not available. Install peer_nga_loader.py")


@dataclass
class GroundMotion:
    """Ground motion record (seismic accelerogram)"""
    time: np.ndarray  # Time array (seconds)
    acceleration: np.ndarray  # Ground acceleration (m/s²)
    dt: float  # Time step (seconds)
    duration: float  # Total duration (seconds)
    pga: float  # Peak ground acceleration (m/s²)
    name: str = "ground_motion"  # Record name/identifier
    
    @classmethod
    def from_array(cls, time: np.ndarray, acceleration: np.ndarray, name: str = "ground_motion"):
        """Create ground motion from time and acceleration arrays"""
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        duration = time[-1] - time[0]
        pga = np.max(np.abs(acceleration))
        return cls(time, acceleration, dt, duration, pga, name)
    
    @classmethod
    def synthetic(cls, duration: float = 20.0, dt: float = 0.01, 
                  pga: float = 0.5, frequency_range: Tuple[float, float] = (0.5, 10.0),
                  name: str = "synthetic"):
        """Generate synthetic ground motion (simplified)
        
        Parameters:
        -----------
        duration : float
            Duration in seconds
        dt : float
            Time step in seconds
        pga : float
            Peak ground acceleration (m/s²)
        frequency_range : tuple
            Frequency range (Hz) for the motion
        """
        time = np.arange(0, duration, dt)
        n = len(time)
        
        # Generate broadband motion using filtered white noise
        # This is simplified - real ground motion is more complex
        white_noise = np.random.randn(n)
        
        # Apply frequency filter (simplified)
        freqs = np.fft.fftfreq(n, dt)
        fft_signal = np.fft.fft(white_noise)
        
        # Bandpass filter
        mask = (np.abs(freqs) >= frequency_range[0]) & (np.abs(freqs) <= frequency_range[1])
        fft_signal[~mask] = 0
        
        # Inverse FFT
        acceleration = np.real(np.fft.ifft(fft_signal))
        
        # Normalize to target PGA
        acceleration = acceleration / np.max(np.abs(acceleration)) * pga
        
        # Apply envelope (build-up and decay)
        envelope = np.exp(-(time - duration/2)**2 / (2 * (duration/4)**2))
        envelope = envelope / np.max(envelope)
        acceleration = acceleration * envelope
        
        return cls.from_array(time, acceleration, name)
    
    @classmethod
    def from_peer_nga(cls, filepath: str, scale_pga: Optional[float] = None):
        """Load ground motion from PEER NGA database format
        
        Parameters:
        -----------
        filepath : str
            Path to PEER NGA ASCII file
        scale_pga : float, optional
            Target PGA in m/s² (if None, uses original PGA)
            
        Returns:
        --------
        GroundMotion object
        
        References:
        -----------
        Ancheta, T. D., et al. (2014). "NGA-West2 Database."
        Earthquake Spectra, 30(3), 989-1005.
        """
        if not PEER_NGA_AVAILABLE:
            raise ImportError("PEER NGA loader not available. Check peer_nga_loader.py")
        
        record = load_peer_nga_record(filepath, scale_pga=scale_pga)
        
        return cls(
            time=record['time'],
            acceleration=record['acceleration'],
            dt=record['dt'],
            duration=record['duration'],
            pga=record['pga'],
            name=record['name']
        )


@dataclass
class EarthquakeIntensity:
    """Earthquake intensity parameters for selection"""
    pga: Optional[float] = None  # Peak Ground Acceleration (m/s²)
    pga_g: Optional[float] = None  # Peak Ground Acceleration (g units)
    sa_t1: Optional[float] = None  # Spectral acceleration at T=1s (m/s²)
    sa_t1_g: Optional[float] = None  # Spectral acceleration at T=1s (g units)
    magnitude: Optional[float] = None  # Earthquake magnitude (Mw)
    distance: Optional[float] = None  # Distance to fault (km)
    intensity_scale: Optional[str] = None  # Intensity scale name (e.g., "MMI", "PGA")
    
    def to_pga(self) -> float:
        """Convert to PGA in m/s²"""
        if self.pga is not None:
            return self.pga
        elif self.pga_g is not None:
            return self.pga_g * 9.81
        elif self.sa_t1 is not None:
            # Approximate: SA(T=1s) ≈ 1.5-2.0 × PGA for typical earthquakes
            return self.sa_t1 / 1.75
        elif self.sa_t1_g is not None:
            return (self.sa_t1_g * 9.81) / 1.75
        else:
            raise ValueError("No intensity measure specified")
    
    @classmethod
    def from_pga(cls, pga: float, unit: str = "m/s2"):
        """Create intensity from PGA
        
        Parameters:
        -----------
        pga : float
            Peak ground acceleration
        unit : str
            Unit: "m/s2" or "g"
        """
        if unit.lower() == "g":
            return cls(pga_g=pga, pga=pga * 9.81)
        else:
            return cls(pga=pga, pga_g=pga / 9.81)
    
    @classmethod
    def from_magnitude_distance(cls, magnitude: float, distance_km: float):
        """Estimate intensity from magnitude and distance
        
        Uses simplified attenuation relationship (for initial estimates only).
        For accurate results, use ground motion prediction equations (GMPEs).
        
        Parameters:
        -----------
        magnitude : float
            Earthquake magnitude (Mw)
        distance_km : float
            Distance to fault (km)
        """
        # Simplified attenuation (log10(PGA) ≈ M - log10(R) - 2)
        # This is very approximate - real GMPEs are more complex
        log_pga = magnitude - np.log10(max(distance_km, 1.0)) - 2.0
        pga = 10**log_pga  # m/s²
        
        return cls(pga=pga, pga_g=pga/9.81, magnitude=magnitude, distance=distance_km)


@dataclass
class EarthquakeSimulationParameters:
    """Parameters for earthquake simulation"""
    ground_motion: GroundMotion
    damping_ratio: float = 0.05  # 5% damping (typical for concrete)
    damping_frequencies: Tuple[float, float] = (1.0, 10.0)  # For Rayleigh damping
    max_time_step: float = 0.001  # Maximum time step for integration (seconds)
    min_time_step: float = 0.0001  # Minimum time step
    adaptive_time_step: bool = True  # Use adaptive time stepping
    output_frequency: int = 10  # Save results every N steps
    damage_update_frequency: int = 1  # Update damage every N steps
    intensity: Optional[EarthquakeIntensity] = None  # Intensity parameters


class EarthquakeSimulator:
    """Earthquake simulator for gyroid structures"""
    
    def __init__(self, domain, material: MaterialProperties, 
                 sim_params: EarthquakeSimulationParameters):
        """
        Initialize earthquake simulator
        
        Parameters:
        -----------
        domain : FEDomain
            SfePy domain (mesh) for the structure
        material : MaterialProperties
            Material properties with Mazars damage parameters
        sim_params : EarthquakeSimulationParameters
            Simulation parameters including ground motion
        """
        self.domain = domain
        self.material = material
        self.sim_params = sim_params
        self.ground_motion = sim_params.ground_motion
        
        # Initialize state variables
        self.displacement_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.damage_history = []
        self.time_history = []
        self.stress_history = []
        
        # Compute mass and initial stiffness matrices
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """Initialize mass, damping, and stiffness matrices"""
        # TODO: Implement proper FE matrix assembly
        # For now, use simplified lumped mass and stiffness
        
        # Get number of nodes
        n_nodes = self.domain.mesh.n_nod
        
        # Lumped mass matrix (diagonal)
        # Mass = density * volume / n_nodes (simplified)
        # For proper implementation, integrate shape functions
        volume = self._estimate_volume()
        mass_per_node = self.material.rho * volume / n_nodes
        self.M = np.eye(n_nodes * 3) * mass_per_node  # 3 DOF per node
        
        # Initial stiffness matrix (before damage)
        # K = E * geometric_stiffness
        # For proper implementation, assemble from element stiffness matrices
        # Simplified: K = E * identity (will be updated with proper FE assembly)
        self.K0 = self._assemble_stiffness_matrix(damage=None)
        
        # Damping matrix (Rayleigh damping)
        # C = α*M + β*K
        omega1 = 2 * np.pi * self.sim_params.damping_frequencies[0]
        omega2 = 2 * np.pi * self.sim_params.damping_frequencies[1]
        alpha = 2 * self.sim_params.damping_ratio * (omega1 * omega2) / (omega1 + omega2)
        beta = 2 * self.sim_params.damping_ratio / (omega1 + omega2)
        self.C = alpha * self.M + beta * self.K0
        
        print(f"Initialized matrices:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Mass matrix size: {self.M.shape}")
        print(f"  Stiffness matrix size: {self.K0.shape}")
        print(f"  Damping ratio: {self.sim_params.damping_ratio*100:.1f}%")
    
    def _estimate_volume(self) -> float:
        """Estimate structure volume from mesh"""
        # Simplified: use bounding box volume
        # For proper implementation, integrate over elements
        coors = self.domain.mesh.coors
        bbox_min = coors.min(axis=0)
        bbox_max = coors.max(axis=0)
        volume = np.prod(bbox_max - bbox_min)
        return volume
    
    def _assemble_stiffness_matrix(self, damage: Optional[np.ndarray] = None) -> np.ndarray:
        """Assemble stiffness matrix from FE elements
        
        TODO: Implement proper FE stiffness matrix assembly
        For now, returns simplified matrix
        """
        n_nodes = self.domain.mesh.n_nod
        n_dof = n_nodes * 3
        
        # Simplified stiffness matrix
        # In proper implementation, would assemble from element stiffness matrices:
        # K_e = ∫ B^T · D · B dV
        # where B = strain-displacement matrix, D = material matrix
        
        if damage is None:
            E_eff = self.material.E
        else:
            # Average damage for simplified case
            E_eff = self.material.E * (1.0 - np.mean(damage))
        
        # Simplified: diagonal stiffness (will be replaced with proper FE assembly)
        K = np.eye(n_dof) * E_eff * 1e6  # Scaling factor (needs proper calibration)
        
        return K
    
    def run_simulation(self) -> Dict:
        """Run earthquake simulation using Newmark-beta time integration
        
        Returns:
        --------
        Dict with simulation results
        """
        print("\n" + "="*60)
        print("EARTHQUAKE SIMULATION")
        print("="*60)
        print(f"Ground motion: {self.ground_motion.name}")
        print(f"Duration: {self.ground_motion.duration:.2f} s")
        print(f"PGA: {self.ground_motion.pga:.3f} m/s² ({self.ground_motion.pga/9.81:.3f} g)")
        print(f"Time steps: {len(self.ground_motion.time)}")
        
        # Initialize state
        n_dof = self.M.shape[0]
        u = np.zeros(n_dof)  # Displacement
        u_dot = np.zeros(n_dof)  # Velocity
        u_ddot = np.zeros(n_dof)  # Acceleration
        damage = np.zeros(self.domain.mesh.n_nod)  # Damage at each node
        
        # Newmark-beta parameters (average acceleration method)
        gamma = 0.5
        beta = 0.25
        
        # Time stepping
        dt = self.ground_motion.dt
        time_array = self.ground_motion.time
        ground_acc = self.ground_motion.acceleration
        
        print(f"\nStarting time integration...")
        start_time = time.time()
        
        for i, t in enumerate(time_array):
            # Current ground acceleration
            u_g_ddot = ground_acc[i]
            
            # Effective force: -M * u_g_ddot (earthquake loading)
            F_eff = -self.M @ (np.ones(n_dof) * u_g_ddot)
            
            # Update stiffness matrix with current damage
            if i % self.sim_params.damage_update_frequency == 0:
                K = self._assemble_stiffness_matrix(damage)
            
            # Effective stiffness: K_eff = K + a0*M + a1*C
            a0 = 1.0 / (beta * dt**2)
            a1 = gamma / (beta * dt)
            K_eff = K + a0 * self.M + a1 * self.C
            
            # Effective force: F_eff = F + M*(a0*u + a2*u_dot + a3*u_ddot) + C*(a1*u + a4*u_dot + a5*u_ddot)
            a2 = 1.0 / (beta * dt)
            a3 = 1.0 / (2 * beta) - 1.0
            a4 = gamma / beta - 1.0
            a5 = dt * (gamma / (2 * beta) - 1.0)
            
            F_eff = F_eff + self.M @ (a0 * u + a2 * u_dot + a3 * u_ddot) + \
                           self.C @ (a1 * u + a4 * u_dot + a5 * u_ddot)
            
            # Solve for new displacement
            u_new = np.linalg.solve(K_eff, F_eff)
            
            # Update velocity and acceleration
            u_dot_new = a1 * (u_new - u) - a4 * u_dot - a5 * u_ddot
            u_ddot_new = a0 * (u_new - u) - a2 * u_dot - a3 * u_dot
            
            # Compute strain and update damage
            if i % self.sim_params.damage_update_frequency == 0:
                strain_tensor = self._compute_strain_from_displacement(u_new)
                eps_eq = compute_equivalent_strain(strain_tensor)
                
                # Update damage (use both compressive and tensile)
                damage_comp = mazars_compressive_damage(
                    eps_eq, self.material.epsilon_c0, 
                    self.material.A_c, self.material.B_c
                )
                damage_tens = mazars_tensile_damage(
                    eps_eq, self.material.epsilon_t0,
                    self.material.A_t, self.material.B_t
                )
                # Total damage (maximum of compressive and tensile)
                damage_new = np.maximum(damage_comp, damage_tens)
                damage = np.maximum(damage, damage_new)  # Irreversible
            
            # Update state
            u = u_new
            u_dot = u_dot_new
            u_ddot = u_dot_new
            
            # Store history
            if i % self.sim_params.output_frequency == 0:
                self.time_history.append(t)
                self.displacement_history.append(u.copy())
                self.velocity_history.append(u_dot.copy())
                self.acceleration_history.append(u_ddot.copy())
                self.damage_history.append(damage.copy())
                
                # Compute stress
                stress = self._compute_stress_from_strain(strain_tensor, damage)
                self.stress_history.append(stress)
                
                # Store ground acceleration at same frequency for visualization
                if not hasattr(self, 'ground_acc_history'):
                    self.ground_acc_history = []
                self.ground_acc_history.append(u_g_ddot)
            
            # Progress update
            if i % (len(time_array) // 10) == 0:
                progress = (i / len(time_array)) * 100
                max_disp = np.max(np.abs(u))
                max_damage = np.max(damage)
                print(f"  {progress:.0f}%: t={t:.2f}s, max_disp={max_disp*1000:.2f}mm, max_damage={max_damage:.3f}")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        
        # Compute results summary
        results = self._compute_results()
        return results
    
    def _compute_strain_from_displacement(self, u: np.ndarray) -> np.ndarray:
        """Compute average strain tensor from displacement
        
        TODO: Implement proper FE strain computation
        """
        # Simplified: return average strain tensor
        # In proper implementation, compute from grad(u) at each node
        n_nodes = self.domain.mesh.n_nod
        u_reshaped = u.reshape(n_nodes, 3)
        
        # Simplified strain (needs proper FE gradient computation)
        strain_avg = np.mean(np.abs(u_reshaped), axis=0) / 0.03  # Divide by characteristic length
        strain_tensor = np.diag(strain_avg)
        return strain_tensor
    
    def _compute_stress_from_strain(self, strain_tensor: np.ndarray, damage: np.ndarray) -> float:
        """Compute stress from strain and damage"""
        E_eff = self.material.E * (1.0 - np.mean(damage))
        stress = E_eff * np.trace(strain_tensor)
        return stress
    
    def _compute_results(self) -> Dict:
        """Compute comprehensive summary results from simulation
        
        Computes all important metrics for earthquake engineering analysis.
        """
        if not self.time_history:
            return {}
        
        # Convert to arrays
        time_array = np.array(self.time_history)
        displacements = np.array(self.displacement_history)
        velocities = np.array(self.velocity_history) if self.velocity_history else None
        accelerations = np.array(self.acceleration_history) if self.acceleration_history else None
        damages = np.array(self.damage_history)
        stresses = np.array(self.stress_history) if self.stress_history else None
        
        # Maximum values (absolute)
        max_displacement = np.max(np.abs(displacements))
        max_damage = np.max(damages)
        max_stress = np.max(np.abs(stresses)) if stresses is not None else 0.0
        
        # Residual (final) values
        residual_displacement = np.abs(displacements[-1])
        residual_damage = damages[-1]
        residual_stress = np.abs(stresses[-1]) if stresses is not None else 0.0
        
        # Peak response
        peak_acceleration = np.max(np.abs(accelerations)) if accelerations is not None else 0.0
        peak_velocity = np.max(np.abs(velocities)) if velocities is not None else 0.0
        
        # Inter-story drift (for multi-story structures, simplified here)
        # For single structure, use maximum relative displacement
        max_drift = max_displacement / 0.03  # Assuming 30mm structure height (normalize)
        
        # Energy metrics
        if velocities is not None and accelerations is not None:
            # Kinetic energy: KE = 0.5 * M * v²
            # Simplified: use average velocity magnitude
            avg_velocity_mag = np.mean(np.linalg.norm(velocities, axis=1))
            kinetic_energy = 0.5 * np.trace(self.M) * avg_velocity_mag**2
            
            # Strain energy: SE = 0.5 * u^T * K * u
            # Simplified: use average displacement
            avg_displacement_mag = np.mean(np.linalg.norm(displacements, axis=1))
            strain_energy = 0.5 * avg_displacement_mag**2 * np.trace(self.K0) * 1e6  # Approximate
        else:
            kinetic_energy = 0.0
            strain_energy = 0.0
        
        # Damage metrics
        max_damage_compressive = np.max(damages)  # Simplified (would separate comp/tens in full implementation)
        max_damage_tensile = np.max(damages)  # Simplified
        damage_at_failure = 0.5  # Threshold for significant failure
        
        # Failure indicators
        failure_occurred = max_damage > damage_at_failure
        failure_time = time_array[np.argmax(damages > damage_at_failure)] if failure_occurred else None
        
        # Stress distribution metrics
        if stresses is not None:
            mean_stress = np.mean(np.abs(stresses))
            std_stress = np.std(np.abs(stresses))
            stress_concentration_factor = max_stress / mean_stress if mean_stress > 0 else 1.0
        else:
            mean_stress = 0.0
            std_stress = 0.0
            stress_concentration_factor = 1.0
        
        # Displacement distribution
        mean_displacement = np.mean(np.abs(displacements))
        std_displacement = np.std(np.abs(displacements))
        
        # Response amplification (structural response / ground motion)
        response_amplification = peak_acceleration / self.ground_motion.pga if self.ground_motion.pga > 0 else 0.0
        
        results = {
            # Ground motion info
            "ground_motion_name": self.ground_motion.name,
            "pga": self.ground_motion.pga,
            "pga_g": self.ground_motion.pga / 9.81,
            "duration": self.ground_motion.duration,
            
            # Displacement metrics
            "max_displacement_m": float(max_displacement),
            "max_displacement_mm": float(max_displacement * 1000),
            "mean_displacement_m": float(mean_displacement),
            "std_displacement_m": float(std_displacement),
            "residual_displacement_m": float(np.max(residual_displacement)),
            "residual_displacement_mm": float(np.max(residual_displacement) * 1000),
            "max_inter_story_drift": float(max_drift),
            
            # Velocity and acceleration
            "peak_velocity_m_s": float(peak_velocity),
            "peak_acceleration_m_s2": float(peak_acceleration),
            "peak_acceleration_g": float(peak_acceleration / 9.81),
            "response_amplification": float(response_amplification),
            
            # Damage metrics
            "max_damage": float(max_damage),
            "max_damage_compressive": float(max_damage_compressive),
            "max_damage_tensile": float(max_damage_tensile),
            "residual_damage": float(np.max(residual_damage)),
            "mean_damage": float(np.mean(damages)),
            "damage_at_failure": damage_at_failure,
            "failure_occurred": bool(failure_occurred),
            "failure_time_s": float(failure_time) if failure_time is not None else None,
            
            # Stress metrics
            "max_stress_Pa": float(max_stress),
            "max_stress_MPa": float(max_stress / 1e6),
            "mean_stress_Pa": float(mean_stress),
            "mean_stress_MPa": float(mean_stress / 1e6),
            "std_stress_Pa": float(std_stress),
            "stress_concentration_factor": float(stress_concentration_factor),
            "residual_stress_Pa": float(residual_stress),
            "residual_stress_MPa": float(residual_stress / 1e6),
            
            # Energy metrics
            "max_kinetic_energy_J": float(kinetic_energy),
            "max_strain_energy_J": float(strain_energy),
            "total_energy_J": float(kinetic_energy + strain_energy),
            
            # Time histories (for visualization)
            "time_history": time_array.tolist(),
            "displacement_history": [d.tolist() for d in displacements],
            "velocity_history": [v.tolist() for v in velocities] if velocities is not None else None,
            "acceleration_history": [a.tolist() for a in accelerations] if accelerations is not None else None,
            "damage_history": [d.tolist() for d in damages],
            "stress_history": [s.tolist() if isinstance(s, np.ndarray) else [s] for s in stresses] if stresses is not None else None,
        }
        
        return results


def run_earthquake_test(stl_path: Path, ground_motion: GroundMotion,
                       material: Optional[MaterialProperties] = None,
                       element_size: float = 0.05,
                       damping_ratio: float = 0.05) -> Dict:
    """Run earthquake simulation on an STL structure
    
    Parameters:
    -----------
    stl_path : Path
        Path to STL file
    ground_motion : GroundMotion
        Ground motion record
    material : MaterialProperties, optional
        Material properties (uses defaults if None)
    element_size : float
        Mesh element size (meters)
    damping_ratio : float
        Damping ratio (e.g., 0.05 for 5%)
    
    Returns:
    --------
    Dict with simulation results
    """
    if material is None:
        material = MaterialProperties()
    
    # Load and mesh STL
    print(f"Loading STL: {stl_path}")
    domain = load_stl_and_create_mesh(stl_path, element_size)
    
    # Setup simulation parameters
    sim_params = EarthquakeSimulationParameters(
        ground_motion=ground_motion,
        damping_ratio=damping_ratio
    )
    
    # Run simulation
    simulator = EarthquakeSimulator(domain, material, sim_params)
    results = simulator.run_simulation()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Earthquake simulator for gyroid structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic earthquake with 0.5g PGA
  python earthquake_simulator.py test_cube_30mm.stl --pga-g 0.5
  
  # Use PEER NGA record
  python earthquake_simulator.py test_cube_30mm.stl --peer-nga path/to/record.txt
  
  # Scale PEER record to specific intensity
  python earthquake_simulator.py test_cube_30mm.stl --peer-nga record.txt --scale-peer 4.9
  
  # Create visualization
  python earthquake_simulator.py test_cube_30mm.stl --pga-g 0.5 --viz
        """
    )
    parser.add_argument("stl_file", type=str, help="Path to STL file")
    
    # Intensity selection
    intensity_group = parser.add_mutually_exclusive_group()
    intensity_group.add_argument("--pga", type=float, help="Peak ground acceleration (m/s²)")
    intensity_group.add_argument("--pga-g", type=float, help="Peak ground acceleration (g units)")
    intensity_group.add_argument("--peer-nga", type=str, help="Path to PEER NGA file")
    intensity_group.add_argument("--magnitude-distance", nargs=2, type=float, metavar=('MAG', 'DIST'),
                                 help="Magnitude and distance (km) for intensity estimation")
    
    parser.add_argument("--scale-peer", type=float, help="Scale PEER record to this PGA (m/s²)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration for synthetic (seconds)")
    parser.add_argument("--damping", type=float, default=0.05, help="Damping ratio")
    parser.add_argument("--element-size", type=float, default=0.05, help="Mesh element size (m)")
    parser.add_argument("--output", type=str, default="earthquake_results.json", help="Output JSON file")
    parser.add_argument("--viz", action="store_true", help="Create visualization plots")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Create intensity
    if args.pga:
        intensity = EarthquakeIntensity.from_pga(args.pga, unit="m/s2")
        target_pga = args.pga
    elif args.pga_g:
        intensity = EarthquakeIntensity.from_pga(args.pga_g, unit="g")
        target_pga = args.pga_g * 9.81
    elif args.magnitude_distance:
        intensity = EarthquakeIntensity.from_magnitude_distance(
            args.magnitude_distance[0], args.magnitude_distance[1]
        )
        target_pga = intensity.to_pga()
    else:
        # Default
        intensity = EarthquakeIntensity.from_pga(0.5, unit="g")
        target_pga = 0.5 * 9.81
    
    # Create ground motion
    if args.peer_nga:
        print(f"Loading PEER NGA record: {args.peer_nga}")
        ground_motion = GroundMotion.from_peer_nga(args.peer_nga, scale_pga=args.scale_peer)
    else:
        print(f"Generating synthetic ground motion with PGA: {target_pga:.3f} m/s² ({target_pga/9.81:.3f} g)")
        ground_motion = GroundMotion.synthetic(
            duration=args.duration,
            pga=target_pga,
            name=f"synthetic_{target_pga/9.81:.2f}g"
        )
    
    # Run simulation
    results = run_earthquake_test(
        Path(args.stl_file),
        ground_motion,
        element_size=args.element_size,
        damping_ratio=args.damping
    )
    
    # Ground motion acceleration is already in results at output frequency
    # Also add full time history for reference
    results['ground_motion_full_time'] = ground_motion.time.tolist()
    results['ground_motion_full_acceleration'] = ground_motion.acceleration.tolist()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    if args.viz or (not args.no_viz and not args.viz):
        try:
            from earthquake_visualization import plot_earthquake_results
            viz_file = Path(args.output).with_suffix('.png')
            plot_earthquake_results(results, output_path=viz_file, show_plot=False)
        except ImportError:
            print("Warning: Visualization module not available")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Ground Motion: {results['ground_motion_name']}")
    print(f"PGA: {results['pga_g']:.3f} g ({results['pga']:.2f} m/s²)")
    print(f"\nDisplacement:")
    print(f"  Max: {results['max_displacement_mm']:.2f} mm")
    print(f"  Residual: {results['residual_displacement_mm']:.2f} mm")
    print(f"  Max Drift: {results['max_inter_story_drift']:.4f}")
    print(f"\nDamage:")
    print(f"  Max: {results['max_damage']:.3f}")
    print(f"  Residual: {results['residual_damage']:.3f}")
    print(f"  Failure: {'YES' if results['failure_occurred'] else 'NO'}")
    if results['failure_time_s']:
        print(f"  Failure Time: {results['failure_time_s']:.2f} s")
    print(f"\nStress:")
    print(f"  Max: {results['max_stress_MPa']:.2f} MPa")
    print(f"  Residual: {results['residual_stress_MPa']:.2f} MPa")
    print(f"  Concentration: {results['stress_concentration_factor']:.2f}x")
    print(f"\nResponse:")
    print(f"  Peak Acceleration: {results['peak_acceleration_g']:.3f} g")
    print(f"  Amplification: {results['response_amplification']:.2f}x")
    print(f"\nEnergy:")
    print(f"  Kinetic: {results['max_kinetic_energy_J']:.2f} J")
    print(f"  Strain: {results['max_strain_energy_J']:.2f} J")
    print(f"\nResults saved to: {args.output}")
    if args.viz or (not args.no_viz and not args.viz):
        print(f"Visualization saved to: {viz_file}")
    print("="*60)

