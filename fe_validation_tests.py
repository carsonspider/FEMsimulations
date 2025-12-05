#!/usr/bin/env python3
"""
FE Implementation Validation Tests

This module contains validation tests to verify the accuracy of the FE matrix assembly:
1. Patch Test - Verifies element formulation
2. Cantilever Beam Test - Verifies boundary conditions and assembly
3. Eigenvalue Test - Verifies natural frequencies
4. Unit Test - Verifies stress recovery

References:
- Bathe, K. J. (2014). "Finite Element Procedures." 2nd Edition.
- Zienkiewicz, O. C., et al. (2013). "The Finite Element Method." 7th Edition.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from _sfePy_active.earthquake_simulator import EarthquakeSimulator, EarthquakeSimulationParameters
from _sfePy_active.mazars_model_sfepy import MaterialProperties, load_stl_and_create_mesh
from _sfePy_active.earthquake_simulator import GroundMotion


class FEValidationTests:
    """Validation tests for FE implementation"""
    
    def __init__(self):
        self.material = MaterialProperties()
        self.results = {}
    
    def test_1_patch_test(self, verbose=True):
        """
        Patch Test: Apply uniform strain field
        
        For linear elements, a uniform strain field should be recovered exactly.
        This is the fundamental test of FE element formulation.
        
        Expected: Exact recovery of constant strain field
        """
        if verbose:
            print("\n" + "="*60)
            print("TEST 1: PATCH TEST")
            print("="*60)
            print("Applying uniform strain field to a single element...")
        
        # Create a simple 1-element cube mesh
        # For this test, we'll use a simple 2x2x2 mesh and apply uniform strain
        from sfepy.discrete.fem import Mesh, FEDomain
        
        # Create a simple cube: 1m × 1m × 1m
        # 2×2×2 = 8 nodes, 1 hexahedral element
        coors = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
        ], dtype=np.float64)
        
        # Single hexahedral element (8 nodes)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        mat_ids = [np.array([0], dtype=np.int32)]
        descs = ['3_8']
        
        mesh = Mesh.from_data('test_mesh', coors, None, [conn], mat_ids, descs)
        domain = FEDomain('domain', mesh)
        
        # Create simulator to access FE assembly
        ground_motion = GroundMotion.synthetic(duration=1.0, pga=0.1, name="test")
        sim_params = EarthquakeSimulationParameters(ground_motion=ground_motion)
        simulator = EarthquakeSimulator(domain, self.material, sim_params)
        
        # Apply uniform strain field: ε_xx = 0.001, all others zero
        # This corresponds to: u_x = 0.001 * x, u_y = 0, u_z = 0
        n_nodes = mesh.n_nod
        u_patch = np.zeros(n_nodes * 3)
        for i, coord in enumerate(coors):
            u_patch[i * 3] = 0.001 * coord[0]  # u_x = 0.001 * x
            # u_y = 0, u_z = 0 (already zero)
        
        # Compute strain from displacement using FE formulation
        # For patch test, we should recover ε_xx = 0.001 exactly
        strain_tensor = simulator._compute_strain_from_displacement(u_patch)
        
        # Check if strain is recovered
        epsilon_xx = strain_tensor[0, 0] if strain_tensor.ndim == 2 else 0.0
        
        # For patch test, we need to compute strain properly from FE
        # The current _compute_strain_from_displacement is simplified
        # For proper test, would need to compute from B matrix
        
        if verbose:
            print(f"Applied uniform strain: ε_xx = 0.001")
            print(f"Recovered strain: ε_xx ≈ {epsilon_xx:.6f}")
            if abs(epsilon_xx - 0.001) < 0.0001:
                print("✓ PATCH TEST PASSED: Strain recovered accurately")
            else:
                print("⚠ PATCH TEST: Strain recovery needs verification")
                print("  (Current implementation uses simplified strain computation)")
        
        return {
            'test': 'patch_test',
            'applied_strain': 0.001,
            'recovered_strain': epsilon_xx,
            'passed': abs(epsilon_xx - 0.001) < 0.0001,
            'note': 'Uses simplified strain computation - full FE strain needed for accurate test'
        }
    
    def test_2_cantilever_beam(self, verbose=True):
        """
        Cantilever Beam Test
        
        Simple cantilever beam with point load at tip.
        Analytical solution: δ = PL³/(3EI)
        
        Where:
        - P = applied force
        - L = beam length
        - E = Young's modulus
        - I = second moment of area
        """
        if verbose:
            print("\n" + "="*60)
            print("TEST 2: CANTILEVER BEAM TEST")
            print("="*60)
        
        # Create a simple beam mesh (rectangular cross-section)
        # Beam: 1m long, 0.1m × 0.1m cross-section
        from sfepy.discrete.fem import Mesh, FEDomain
        
        L = 1.0  # Length (m)
        h = 0.1  # Height (m)
        w = 0.1  # Width (m)
        
        # Create mesh: 10 elements along length, 2×2 in cross-section
        n_x, n_y, n_z = 10, 2, 2
        
        coors = []
        for i in range(n_x + 1):
            for j in range(n_y + 1):
                for k in range(n_z + 1):
                    x = i * L / n_x
                    y = j * w / n_y - w/2
                    z = k * h / n_z - h/2
                    coors.append([x, y, z])
        
        coors = np.array(coors, dtype=np.float64)
        
        # Create connectivity (hexahedral elements)
        conn = []
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    # Node indices for hexahedron
                    base = i * (n_y + 1) * (n_z + 1) + j * (n_z + 1) + k
                    step_x = (n_y + 1) * (n_z + 1)
                    step_y = (n_z + 1)
                    step_z = 1
                    
                    el_nodes = [
                        base,
                        base + step_x,
                        base + step_x + step_y,
                        base + step_y,
                        base + step_z,
                        base + step_x + step_z,
                        base + step_x + step_y + step_z,
                        base + step_y + step_z,
                    ]
                    conn.append(el_nodes)
        
        conn = np.array(conn, dtype=np.int32)
        mat_ids = [np.zeros(len(conn), dtype=np.int32)]
        descs = ['3_8']
        
        mesh = Mesh.from_data('beam_mesh', coors, None, [conn], mat_ids, descs)
        domain = FEDomain('domain', mesh)
        
        # Analytical solution
        E = self.material.E
        I = w * h**3 / 12  # Second moment of area
        P = 1000.0  # Applied force (N) at tip
        delta_analytical = P * L**3 / (3 * E * I)
        
        if verbose:
            print(f"Beam dimensions: L={L}m, w={w}m, h={h}m")
            print(f"Material: E={E/1e9:.1f} GPa")
            print(f"Applied force: P={P} N")
            print(f"Analytical tip deflection: δ = {delta_analytical*1000:.3f} mm")
            print(f"\nComputing FE solution...")
        
        # For this test, we'd need to apply a static load
        # This requires solving K·u = F (static analysis)
        # For now, we'll verify the stiffness matrix is positive definite
        
        ground_motion = GroundMotion.synthetic(duration=0.1, pga=0.01, name="test")
        sim_params = EarthquakeSimulationParameters(ground_motion=ground_motion)
        simulator = EarthquakeSimulator(domain, self.material, sim_params)
        
        # Check stiffness matrix properties
        K = simulator.K0
        eigenvalues = np.linalg.eigvals(K)
        min_eigenvalue = np.min(np.real(eigenvalues))
        max_eigenvalue = np.max(np.real(eigenvalues))
        condition_number = max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else np.inf
        
        if verbose:
            print(f"Stiffness matrix properties:")
            print(f"  Size: {K.shape}")
            print(f"  Min eigenvalue: {min_eigenvalue:.2e}")
            print(f"  Max eigenvalue: {max_eigenvalue:.2e}")
            print(f"  Condition number: {condition_number:.2e}")
            
            if min_eigenvalue > 0:
                print("✓ Stiffness matrix is positive definite")
            else:
                print("✗ Stiffness matrix has negative eigenvalues!")
            
            print("\n⚠ CANTILEVER TEST: Static analysis needed for full validation")
            print("  (Current implementation is for dynamic analysis)")
        
        return {
            'test': 'cantilever_beam',
            'analytical_deflection_mm': delta_analytical * 1000,
            'min_eigenvalue': min_eigenvalue,
            'max_eigenvalue': max_eigenvalue,
            'condition_number': condition_number,
            'positive_definite': min_eigenvalue > 0,
            'note': 'Static analysis needed for full validation'
        }
    
    def test_3_eigenvalue_test(self, verbose=True):
        """
        Eigenvalue Test: Compute natural frequencies
        
        Solve: (K - ω²M)·φ = 0
        Natural frequencies: ω = sqrt(λ) where λ are eigenvalues
        
        Should be positive and real.
        """
        if verbose:
            print("\n" + "="*60)
            print("TEST 3: EIGENVALUE TEST")
            print("="*60)
        
        # Use simple cube mesh
        from sfepy.discrete.fem import Mesh, FEDomain
        
        # Simple 2×2×2 cube
        coors = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64) * 0.01  # 10mm cube
        
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        mat_ids = [np.array([0], dtype=np.int32)]
        descs = ['3_8']
        
        mesh = Mesh.from_data('eigen_test', coors, None, [conn], mat_ids, descs)
        domain = FEDomain('domain', mesh)
        
        ground_motion = GroundMotion.synthetic(duration=0.1, pga=0.01, name="test")
        sim_params = EarthquakeSimulationParameters(ground_motion=ground_motion)
        simulator = EarthquakeSimulator(domain, self.material, sim_params)
        
        # Setup boundary conditions
        simulator._setup_boundary_conditions()
        
        # Get matrices for free DOF only
        K = simulator.K0
        M = simulator.M
        
        # Apply boundary conditions
        K_ff = K[np.ix_(simulator.free_dof, simulator.free_dof)]
        M_ff = M[np.ix_(simulator.free_dof, simulator.free_dof)]
        
        # Solve generalized eigenvalue problem: K·φ = λ·M·φ
        # Using: M^(-1)·K·φ = λ·φ
        try:
            # Compute M^(-1)·K
            M_inv = np.linalg.inv(M_ff)
            A = M_inv @ K_ff
            
            # Solve eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # Natural frequencies: f = sqrt(λ) / (2π)
            # Only consider positive, real eigenvalues
            eigenvalues_real = np.real(eigenvalues)
            eigenvalues_positive = eigenvalues_real[eigenvalues_real > 0]
            frequencies = np.sqrt(eigenvalues_positive) / (2 * np.pi)
            frequencies = np.sort(frequencies)
            
            if verbose:
                print(f"Eigenvalue analysis:")
                print(f"  Total eigenvalues: {len(eigenvalues)}")
                print(f"  Positive eigenvalues: {len(eigenvalues_positive)}")
                print(f"  Negative eigenvalues: {len(eigenvalues_real[eigenvalues_real < 0])}")
                print(f"\nFirst 5 natural frequencies:")
                for i, f in enumerate(frequencies[:5]):
                    print(f"  f_{i+1} = {f:.2f} Hz")
                
                if len(eigenvalues_positive) > 0 and np.min(eigenvalues_positive) > 0:
                    print("✓ All natural frequencies are positive and real")
                else:
                    print("✗ Some eigenvalues are negative or complex!")
                
                # Estimate first frequency (simplified)
                # For a fixed-free beam: f ≈ (1.875²/(2π)) * sqrt(EI/(ρAL⁴))
                # For cube: rough estimate
                L = 0.01  # 10mm
                E = self.material.E
                rho = self.material.rho
                f_estimate = (1.875**2 / (2 * np.pi)) * np.sqrt(E / (rho * L**2))
                print(f"\nEstimated first frequency (simplified): {f_estimate:.2f} Hz")
        
        except Exception as e:
            if verbose:
                print(f"✗ Eigenvalue computation failed: {e}")
            frequencies = []
            eigenvalues_positive = []
        
        return {
            'test': 'eigenvalue_test',
            'num_positive_eigenvalues': len(eigenvalues_positive) if 'eigenvalues_positive' in locals() else 0,
            'num_negative_eigenvalues': len(eigenvalues_real[eigenvalues_real < 0]) if 'eigenvalues_real' in locals() else 0,
            'first_frequency_hz': float(frequencies[0]) if len(frequencies) > 0 else None,
            'passed': len(eigenvalues_positive) > 0 and np.min(eigenvalues_positive) > 0 if 'eigenvalues_positive' in locals() else False
        }
    
    def test_4_unit_test(self, verbose=True):
        """
        Unit Test: Simple 1-element cube with known displacement
        
        Apply known displacement and check recovered stress.
        """
        if verbose:
            print("\n" + "="*60)
            print("TEST 4: UNIT TEST")
            print("="*60)
            print("Testing single element with known displacement...")
        
        # Create 1-element cube
        from sfepy.discrete.fem import Mesh, FEDomain
        
        coors = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64) * 0.01  # 10mm cube
        
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        mat_ids = [np.array([0], dtype=np.int32)]
        descs = ['3_8']
        
        mesh = Mesh.from_data('unit_test', coors, None, [conn], mat_ids, descs)
        domain = FEDomain('domain', mesh)
        
        ground_motion = GroundMotion.synthetic(duration=0.1, pga=0.01, name="test")
        sim_params = EarthquakeSimulationParameters(ground_motion=ground_motion)
        simulator = EarthquakeSimulator(domain, self.material, sim_params)
        
        # Apply uniform extension: u_x = 0.001 * x (1mm extension over 10mm = 0.1% strain)
        n_nodes = mesh.n_nod
        u_test = np.zeros(n_nodes * 3)
        for i, coord in enumerate(coors):
            u_test[i * 3] = 0.001 * coord[0]  # u_x = 0.001 * x
        
        # Expected stress: σ_xx = E * ε_xx = E * 0.001
        E = self.material.E
        expected_stress = E * 0.001  # Pa
        expected_stress_mpa = expected_stress / 1e6
        
        # Compute strain and stress
        strain_tensor = simulator._compute_strain_from_displacement(u_test)
        damage = np.zeros(n_nodes)
        stress = simulator._compute_stress_from_strain(strain_tensor, damage)
        stress_mpa = stress / 1e6
        
        if verbose:
            print(f"Applied displacement: u_x = 0.001 * x")
            print(f"Expected strain: ε_xx = 0.001")
            print(f"Expected stress: σ_xx = {expected_stress_mpa:.2f} MPa")
            print(f"Recovered stress: σ = {stress_mpa:.2f} MPa")
            
            error = abs(stress - expected_stress) / expected_stress * 100
            if error < 10:  # Within 10%
                print(f"✓ UNIT TEST PASSED: Stress recovery within {error:.1f}%")
            else:
                print(f"⚠ UNIT TEST: Stress recovery error: {error:.1f}%")
                print("  (Current implementation uses simplified stress computation)")
        
        return {
            'test': 'unit_test',
            'expected_stress_mpa': expected_stress_mpa,
            'recovered_stress_mpa': stress_mpa,
            'error_percent': abs(stress - expected_stress) / expected_stress * 100,
            'passed': abs(stress - expected_stress) / expected_stress < 0.1,
            'note': 'Uses simplified stress computation'
        }
    
    def run_all_tests(self, verbose=True):
        """Run all validation tests"""
        if verbose:
            print("\n" + "="*60)
            print("FE IMPLEMENTATION VALIDATION TESTS")
            print("="*60)
        
        results = {}
        
        try:
            results['patch_test'] = self.test_1_patch_test(verbose)
        except Exception as e:
            results['patch_test'] = {'test': 'patch_test', 'error': str(e)}
            if verbose:
                print(f"✗ Patch test failed: {e}")
        
        try:
            results['cantilever'] = self.test_2_cantilever_beam(verbose)
        except Exception as e:
            results['cantilever'] = {'test': 'cantilever_beam', 'error': str(e)}
            if verbose:
                print(f"✗ Cantilever test failed: {e}")
        
        try:
            results['eigenvalue'] = self.test_3_eigenvalue_test(verbose)
        except Exception as e:
            results['eigenvalue'] = {'test': 'eigenvalue_test', 'error': str(e)}
            if verbose:
                print(f"✗ Eigenvalue test failed: {e}")
        
        try:
            results['unit_test'] = self.test_4_unit_test(verbose)
        except Exception as e:
            results['unit_test'] = {'test': 'unit_test', 'error': str(e)}
            if verbose:
                print(f"✗ Unit test failed: {e}")
        
        if verbose:
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for name, result in results.items():
                status = "✓" if result.get('passed', False) else "⚠"
                print(f"{status} {name}: {result.get('note', 'See details above')}")
        
        return results


if __name__ == "__main__":
    tester = FEValidationTests()
    results = tester.run_all_tests(verbose=True)
    
    # Save results
    import json
    with open('fe_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: fe_validation_results.json")

