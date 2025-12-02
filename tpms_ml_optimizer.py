#!/usr/bin/env python3
"""
TPMS ML Optimizer using Gaussian Process Regression

This model predicts optimal TPMS geometries for maximum weight-to-strength ratio
and learns complex relationships between parameters that humans might miss.

Key Features:
- Predicts compressive/tensile strength from TPMS parameters
- Identifies optimal geometry for given load constraints
- Learns relational properties between parameters
- Answers inverse design questions: "Given load N, how porous can it be?"
- Replaces brute force FEA with ML predictions

Uses Gaussian Process Regression (GPR) for:
- Uncertainty quantification
- Non-parametric learning of complex relationships
- Multi-output predictions (compressive, tensile, R-value, mass)

Requirements:
    pip install scikit-learn scipy pandas numpy

Usage:
    python tpms_ml_optimizer.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from dataclasses import dataclass

# Scikit-learn imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Optimization
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TPMSFeatures:
    """Feature set for TPMS prediction."""
    tpms_type: str
    unit_cell_size_mm: float
    wall_thickness_mm: float
    porosity_min: float
    porosity_max: float
    func_degree: int
    # Derived features
    porosity_avg: float = None
    porosity_range: float = None
    aspect_ratio: float = None  # wall_thickness / unit_cell_size
    
    def to_array(self, label_encoder: LabelEncoder = None) -> np.ndarray:
        """Convert to feature array for ML model."""
        features = []
        
        # Encode TPMS type
        if label_encoder:
            tpms_encoded = label_encoder.transform([self.tpms_type])[0]
        else:
            # Fallback: one-hot encoding manually
            tpms_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
            tpms_encoded = tpms_types.index(self.tpms_type) if self.tpms_type in tpms_types else 0
        
        features.append(tpms_encoded)
        features.append(self.unit_cell_size_mm)
        features.append(self.wall_thickness_mm)
        features.append(self.porosity_min)
        features.append(self.porosity_max)
        features.append(self.func_degree)
        
        # Derived features
        if self.porosity_avg is None:
            self.porosity_avg = (self.porosity_min + self.porosity_max) / 2.0
        if self.porosity_range is None:
            self.porosity_range = self.porosity_max - self.porosity_min
        if self.aspect_ratio is None:
            self.aspect_ratio = self.wall_thickness_mm / self.unit_cell_size_mm if self.unit_cell_size_mm > 0 else 0
        
        features.append(self.porosity_avg)
        features.append(self.porosity_range)
        features.append(self.aspect_ratio)
        
        return np.array(features, dtype=np.float64)


class TPMSMLOptimizer:
    """
    ML-based TPMS optimizer using Gaussian Process Regression.
    
    Learns relationships between TPMS parameters and mechanical properties,
    enabling inverse design and optimal geometry prediction.
    """
    
    def __init__(self, dataset_path: str = 'dataset_full.csv'):
        """Initialize the optimizer with dataset path."""
        self.dataset_path = Path(dataset_path)
        self.models = {}  # Dictionary of GPR models for different outputs
        self.scalers = {}  # Feature scalers
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'tpms_type_encoded',
            'unit_cell_size_mm',
            'wall_thickness_mm',
            'porosity_min',
            'porosity_max',
            'func_degree',
            'porosity_avg',
            'porosity_range',
            'aspect_ratio'
        ]
        self.target_names = [
            'compressive_strength_MPa',
            'tensile_strength_MPa',  # Will be predicted if not in dataset
            'energy_absorption_J',
            'mass_kg',  # Will be calculated from geometry
            'weight_to_strength_ratio'  # Derived metric
        ]
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        print(f"Loading dataset from: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Filter successful simulations only
        if 'status' in df.columns:
            df = df[df['status'] == 'success'].copy()
            print(f"Filtered to {len(df)} successful simulations")
        
        # Handle missing values
        df = df.dropna(subset=['compressive_strength_MPa'])
        
        # Encode TPMS types
        if 'tpms_type' in df.columns:
            self.label_encoder.fit(df['tpms_type'].unique())
            df['tpms_type_encoded'] = self.label_encoder.transform(df['tpms_type'])
        
        # Create derived features
        if 'porosity_min' in df.columns and 'porosity_max' in df.columns:
            df['porosity_avg'] = (df['porosity_min'] + df['porosity_max']) / 2.0
            df['porosity_range'] = df['porosity_max'] - df['porosity_min']
        
        if 'wall_thickness_mm' in df.columns and 'unit_cell_size_mm' in df.columns:
            df['aspect_ratio'] = df['wall_thickness_mm'] / df['unit_cell_size_mm']
        
        # Calculate or estimate mass (if not present)
        if 'mass_kg' not in df.columns:
            # Estimate mass from geometry (simplified)
            # Mass ≈ volume × density × (1 - porosity)
            # Assuming unit cell volume and material density
            material_density = 2000.0  # kg/m³ (typical for cement/concrete)
            # Approximate volume from unit cell size
            df['estimated_volume_m3'] = (df['unit_cell_size_mm'] / 1000.0) ** 3
            df['mass_kg'] = df['estimated_volume_m3'] * material_density * (1 - df['porosity_avg'])
        
        # Calculate weight-to-strength ratio (higher is better for lightweight structures)
        if 'compressive_strength_MPa' in df.columns and 'mass_kg' in df.columns:
            df['weight_to_strength_ratio'] = df['compressive_strength_MPa'] / df['mass_kg']
            # Replace inf/nan with 0
            df['weight_to_strength_ratio'] = df['weight_to_strength_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Estimate tensile strength if not present (typically 10-15% of compressive for concrete)
        if 'tensile_strength_MPa' not in df.columns:
            df['tensile_strength_MPa'] = df['compressive_strength_MPa'] * 0.12  # 12% estimate
        
        self.data = df
        print(f"Final dataset: {len(df)} samples")
        print(f"Features: {self.feature_names}")
        print(f"Targets: {self.target_names}")
        
        return df
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature and target arrays for training."""
        if self.data is None:
            self.load_data()
        
        # Select feature columns
        feature_cols = [
            'tpms_type_encoded',
            'unit_cell_size_mm',
            'wall_thickness_mm',
            'porosity_min',
            'porosity_max',
            'func_degree',
            'porosity_avg',
            'porosity_range',
            'aspect_ratio'
        ]
        
        # Select target columns (only those that exist)
        target_cols = [col for col in self.target_names if col in self.data.columns]
        
        X = self.data[feature_cols].values.astype(np.float64)
        y = self.data[target_cols].values.astype(np.float64)
        
        # Remove rows with NaN or Inf
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | 
                      np.isnan(y).any(axis=1) | np.isinf(y).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Prepared {len(X)} valid samples")
        print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """Train GPR models for each target variable."""
        X, y = self.prepare_features()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['features'] = scaler
        
        # Get target column names
        target_cols = [col for col in self.target_names if col in self.data.columns]
        
        print(f"\nTraining GPR models for {len(target_cols)} targets...")
        print("="*60)
        
        # Train separate GPR model for each target
        for i, target_name in enumerate(target_cols):
            print(f"\nTraining model for: {target_name}")
            
            y_train_target = self.y_train[:, i]
            y_test_target = self.y_test[:, i]
            
            # Remove any remaining NaN/Inf
            valid_train = ~(np.isnan(y_train_target) | np.isinf(y_train_target))
            valid_test = ~(np.isnan(y_test_target) | np.isinf(y_test_target))
            
            X_train_clean = self.X_train_scaled[valid_train]
            y_train_clean = y_train_target[valid_train]
            X_test_clean = self.X_test_scaled[valid_test]
            y_test_clean = y_test_target[valid_test]
            
            if len(y_train_clean) == 0:
                print(f"  ⚠ Skipping {target_name}: no valid training data")
                continue
            
            # Define kernel for GPR
            # RBF (Radial Basis Function) for smooth functions
            # ConstantKernel for signal variance
            # WhiteKernel for noise
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                     WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
            
            # Create GPR model
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Regularization
                n_restarts_optimizer=10,  # Multiple restarts for better optimization
                normalize_y=True,  # Normalize target
                random_state=random_state
            )
            
            # Train model
            print(f"  Training on {len(y_train_clean)} samples...")
            gpr.fit(X_train_clean, y_train_clean)
            
            # Predictions
            y_pred_train = gpr.predict(X_train_clean)
            y_pred_test = gpr.predict(X_test_clean)
            
            # Evaluate
            train_r2 = r2_score(y_train_clean, y_pred_train)
            test_r2 = r2_score(y_test_clean, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train_clean, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_test))
            
            print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4e}")
            print(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4e}")
            
            # Store model
            self.models[target_name] = gpr
        
        print("\n" + "="*60)
        print(f"✓ Trained {len(self.models)} GPR models")
    
    def predict(self, features: TPMSFeatures) -> Dict[str, Tuple[float, float]]:
        """
        Predict mechanical properties for given TPMS features.
        
        Returns:
            Dictionary mapping property names to (mean, std) predictions
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Convert features to array
        X = features.to_array(self.label_encoder).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Predict for each target
        predictions = {}
        for target_name, model in self.models.items():
            y_pred, y_std = model.predict(X_scaled, return_std=True)
            predictions[target_name] = (float(y_pred[0]), float(y_std[0]))
        
        return predictions
    
    def find_optimal_geometry(self, 
                             target_load_N: float,
                             constraints: Optional[Dict] = None,
                             objective: str = 'weight_to_strength_ratio') -> Dict:
        """
        Find optimal TPMS geometry for given load constraint.
        
        Answers: "Given load N, how porous can it be? And what shape?"
        
        Parameters:
        -----------
        target_load_N : float
            Target load in Newtons
        constraints : dict, optional
            Additional constraints (e.g., {'porosity_min': 0.3, 'porosity_max': 0.8})
        objective : str
            Objective to maximize ('weight_to_strength_ratio', 'compressive_strength_MPa', etc.)
        
        Returns:
        --------
        dict with optimal parameters and predicted properties
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if objective not in self.models:
            raise ValueError(f"Objective '{objective}' not available. Available: {list(self.models.keys())}")
        
        # Parameter bounds (from dataset ranges)
        if self.data is None:
            self.load_data()
        
        # Get bounds from data
        tpms_types = self.data['tpms_type'].unique()
        bounds = [
            (0, len(tpms_types) - 1),  # tpms_type_encoded
            (self.data['unit_cell_size_mm'].min(), self.data['unit_cell_size_mm'].max()),
            (self.data['wall_thickness_mm'].min(), self.data['wall_thickness_mm'].max()),
            (self.data['porosity_min'].min(), self.data['porosity_max'].max()),
            (self.data['porosity_min'].min(), self.data['porosity_max'].max()),
            (self.data['func_degree'].min(), self.data['func_degree'].max()),
        ]
        
        # Apply constraints
        if constraints:
            if 'porosity_min' in constraints:
                bounds[3] = (constraints['porosity_min'], bounds[3][1])
            if 'porosity_max' in constraints:
                bounds[4] = (bounds[4][0], constraints['porosity_max'])
            if 'unit_cell_size_mm' in constraints:
                if isinstance(constraints['unit_cell_size_mm'], (list, tuple)):
                    bounds[1] = tuple(constraints['unit_cell_size_mm'])
            if 'wall_thickness_mm' in constraints:
                if isinstance(constraints['wall_thickness_mm'], (list, tuple)):
                    bounds[2] = tuple(constraints['wall_thickness_mm'])
        
        # Objective function: maximize objective while meeting load requirement
        def objective_func(x):
            # Create features from optimization vector
            features = TPMSFeatures(
                tpms_type=tpms_types[int(x[0])],
                unit_cell_size_mm=x[1],
                wall_thickness_mm=x[2],
                porosity_min=x[3],
                porosity_max=x[4],
                func_degree=int(x[5])
            )
            
            # Ensure porosity_min <= porosity_max
            if features.porosity_min > features.porosity_max:
                return 1e10  # Penalty
            
            # Predict properties
            predictions = self.predict(features)
            
            # Check if compressive strength meets load requirement
            if 'compressive_strength_MPa' in predictions:
                comp_strength = predictions['compressive_strength_MPa'][0]
                # Estimate required strength from load
                # Assuming cross-sectional area from unit cell size
                area_m2 = (features.unit_cell_size_mm / 1000.0) ** 2
                required_strength_MPa = (target_load_N / area_m2) / 1e6
                
                if comp_strength < required_strength_MPa:
                    return 1e10  # Penalty if doesn't meet load requirement
            
            # Maximize objective (return negative for minimization)
            if objective in predictions:
                return -predictions[objective][0]
            else:
                return 1e10
        
        # Optimize using differential evolution (global optimizer)
        print(f"\nOptimizing for load: {target_load_N:.2f} N")
        print(f"Objective: maximize {objective}")
        print("="*60)
        
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            polish=True,
            atol=1e-6,
            tol=1e-6
        )
        
        if not result.success:
            print(f"⚠ Optimization warning: {result.message}")
        
        # Extract optimal parameters
        optimal_features = TPMSFeatures(
            tpms_type=tpms_types[int(result.x[0])],
            unit_cell_size_mm=result.x[1],
            wall_thickness_mm=result.x[2],
            porosity_min=result.x[3],
            porosity_max=result.x[4],
            func_degree=int(result.x[5])
        )
        
        # Get predictions for optimal geometry
        optimal_predictions = self.predict(optimal_features)
        
        return {
            'optimal_geometry': {
                'tpms_type': optimal_features.tpms_type,
                'unit_cell_size_mm': float(optimal_features.unit_cell_size_mm),
                'wall_thickness_mm': float(optimal_features.wall_thickness_mm),
                'porosity_min': float(optimal_features.porosity_min),
                'porosity_max': float(optimal_features.porosity_max),
                'func_degree': int(optimal_features.func_degree),
            },
            'predicted_properties': {
                k: {'mean': v[0], 'std': v[1]} 
                for k, v in optimal_predictions.items()
            },
            'target_load_N': target_load_N,
            'objective_value': -result.fun,
            'optimization_success': result.success
        }
    
    def analyze_relationships(self) -> Dict:
        """
        Analyze relationships between parameters and properties.
        
        HOW RELATIONSHIPS ARE DETERMINED:
        ==================================
        
        1. KERNEL HYPERPARAMETERS (RBF length scales):
           - Each feature has a length_scale in the RBF kernel
           - Small length_scale = feature changes rapidly (high sensitivity)
           - Large length_scale = feature changes slowly (low sensitivity)
           - This reveals which parameters matter most for each property
        
        2. KERNEL STRUCTURE:
           - The learned kernel structure shows how features interact
           - RBF captures smooth, non-linear relationships
           - ConstantKernel variance shows overall signal strength
           - WhiteKernel noise shows data uncertainty
        
        3. MARGINAL LIKELIHOOD:
           - Higher = better fit to data structure
           - Reveals how well the relationships are captured
        
        4. FEATURE INTERACTIONS:
           - GPR implicitly learns all feature interactions
           - The covariance function captures multi-parameter relationships
           - Predictions reveal how parameters combine (not just individually)
        
        Returns:
        --------
        Dictionary with relationship analysis for each target property
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        relationships = {}
        
        print("\n" + "="*60)
        print("ANALYZING PARAMETER RELATIONSHIPS")
        print("="*60)
        print("\nHOW GPR LEARNS RELATIONSHIPS:")
        print("1. RBF length scales reveal feature sensitivity")
        print("2. Kernel structure shows interaction patterns")
        print("3. Covariance function captures multi-parameter relationships")
        print("="*60)
        
        for target_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"TARGET: {target_name}")
            print(f"{'='*60}")
            
            # Get kernel hyperparameters
            kernel = model.kernel_
            print(f"\nKernel Structure: {kernel}")
            
            # Extract RBF length scales (if available)
            # The RBF kernel has length_scale parameter(s) that reveal feature importance
            try:
                # Try to extract length scales from kernel
                if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k2'):
                    # Composite kernel: ConstantKernel * RBF
                    rbf_kernel = kernel.k1.k2
                    if hasattr(rbf_kernel, 'length_scale'):
                        length_scales = rbf_kernel.length_scale
                        
                        # If length_scale is a scalar, it's the same for all features
                        # If it's an array, each feature has its own scale
                        if np.isscalar(length_scales):
                            print(f"\nRBF Length Scale (all features): {length_scales:.4f}")
                            print("  → All features have similar sensitivity")
                        else:
                            print(f"\nRBF Length Scales (per feature):")
                            for i, (feature_name, scale) in enumerate(zip(self.feature_names, length_scales)):
                                # Smaller scale = more sensitive = more important
                                importance = 1.0 / scale if scale > 0 else 0
                                print(f"  {feature_name:25s}: {scale:8.4f} (importance: {importance:.4f})")
                            
                            # Rank features by importance (inverse of length scale)
                            if not np.isscalar(length_scales):
                                importances = 1.0 / length_scales
                                sorted_indices = np.argsort(importances)[::-1]
                                print(f"\nFeature Importance Ranking:")
                                for rank, idx in enumerate(sorted_indices[:5], 1):
                                    print(f"  {rank}. {self.feature_names[idx]:25s} (importance: {importances[idx]:.4f})")
            except Exception as e:
                print(f"  Could not extract length scales: {e}")
            
            # Extract ConstantKernel variance (signal strength)
            try:
                if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k1'):
                    const_kernel = kernel.k1.k1
                    if hasattr(const_kernel, 'constant_value'):
                        signal_variance = const_kernel.constant_value
                        print(f"\nSignal Variance (ConstantKernel): {signal_variance:.4f}")
                        print("  → Higher = stronger signal, more predictable")
            except:
                pass
            
            # Extract WhiteKernel noise level
            try:
                if hasattr(kernel, 'k2'):
                    white_kernel = kernel.k2
                    if hasattr(white_kernel, 'noise_level'):
                        noise_level = white_kernel.noise_level
                        print(f"Noise Level (WhiteKernel): {noise_level:.4f}")
                        print("  → Higher = more uncertainty/noise in data")
            except:
                pass
            
            # Log marginal likelihood (model fit quality)
            log_likelihood = model.log_marginal_likelihood()
            print(f"\nLog Marginal Likelihood: {log_likelihood:.4f}")
            print("  → Higher = better fit to data structure")
            print("  → Reveals how well relationships are captured")
            
            # Store relationship data
            relationships[target_name] = {
                'kernel': str(kernel),
                'log_marginal_likelihood': float(log_likelihood),
                'feature_names': self.feature_names.copy()
            }
            
            # Try to extract length scales for storage
            try:
                if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k2'):
                    rbf_kernel = kernel.k1.k2
                    if hasattr(rbf_kernel, 'length_scale'):
                        length_scales = rbf_kernel.length_scale
                        if np.isscalar(length_scales):
                            relationships[target_name]['length_scale'] = float(length_scales)
                            relationships[target_name]['feature_importances'] = {
                                name: 1.0/float(length_scales) for name in self.feature_names
                            }
                        else:
                            relationships[target_name]['length_scales'] = length_scales.tolist()
                            relationships[target_name]['feature_importances'] = {
                                name: float(1.0/scale) if scale > 0 else 0.0
                                for name, scale in zip(self.feature_names, length_scales)
                            }
            except:
                pass
        
        print(f"\n{'='*60}")
        print("RELATIONSHIP DISCOVERY SUMMARY")
        print(f"{'='*60}")
        print("\nKey Insights:")
        print("1. Features with SMALL length scales are MOST IMPORTANT")
        print("2. Features with LARGE length scales have LESS IMPACT")
        print("3. The kernel structure reveals NON-LINEAR interactions")
        print("4. Multi-parameter relationships are captured in the COVARIANCE FUNCTION")
        print("\nThese relationships are learned AUTOMATICALLY from the data,")
        print("revealing patterns that humans might miss!")
        
        return relationships
    
    def get_feature_interactions(self, target_name: str = 'compressive_strength_MPa') -> Dict:
        """
        Extract feature interaction patterns from the trained GPR model.
        
        WHERE RELATIONSHIPS ARE LEARNED:
        =================================
        
        The relationships are learned in THREE places:
        
        1. DURING TRAINING (line ~288: gpr.fit()):
           - GPR optimizes kernel hyperparameters to maximize log marginal likelihood
           - This optimization finds the best length scales for each feature
           - The kernel structure captures how features interact
        
        2. IN THE KERNEL FUNCTION (RBF covariance):
           - k(x, x') = exp(-0.5 * sum((x_i - x'_i)² / length_scale_i²))
           - This formula shows how similar inputs produce similar outputs
           - Different length scales mean different features have different importance
        
        3. IN PREDICTIONS (line ~328: model.predict()):
           - Predictions use the learned covariance to interpolate/extrapolate
           - The covariance matrix captures ALL feature interactions implicitly
           - Multi-parameter relationships emerge from the kernel structure
        
        Returns:
        --------
        Dictionary with feature importance and interaction patterns
        """
        if target_name not in self.models:
            raise ValueError(f"Model for '{target_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[target_name]
        kernel = model.kernel_
        
        interactions = {
            'target': target_name,
            'feature_names': self.feature_names,
            'feature_importances': {},
            'kernel_structure': str(kernel),
            'log_likelihood': float(model.log_marginal_likelihood())
        }
        
        # Extract length scales (feature importance)
        try:
            if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k2'):
                rbf_kernel = kernel.k1.k2
                if hasattr(rbf_kernel, 'length_scale'):
                    length_scales = rbf_kernel.length_scale
                    
                    if np.isscalar(length_scales):
                        # Same importance for all features
                        importance = 1.0 / length_scales if length_scales > 0 else 0
                        interactions['feature_importances'] = {
                            name: float(importance) for name in self.feature_names
                        }
                        interactions['length_scale'] = float(length_scales)
                    else:
                        # Different importance per feature
                        importances = 1.0 / length_scales
                        interactions['feature_importances'] = {
                            name: float(imp) for name, imp in zip(self.feature_names, importances)
                        }
                        interactions['length_scales'] = length_scales.tolist()
                        
                        # Normalize importances to 0-1 scale
                        max_imp = max(importances)
                        if max_imp > 0:
                            interactions['normalized_importances'] = {
                                name: float(imp / max_imp) 
                                for name, imp in zip(self.feature_names, importances)
                            }
        except Exception as e:
            interactions['error'] = str(e)
        
        return interactions
    
    def explain_relationship(self, feature1: str, feature2: str, 
                            target_name: str = 'compressive_strength_MPa') -> str:
        """
        Explain how two features interact to affect a target property.
        
        This uses the learned kernel structure to understand feature interactions.
        """
        if target_name not in self.models:
            return f"Model for '{target_name}' not found."
        
        interactions = self.get_feature_interactions(target_name)
        
        if feature1 not in interactions['feature_importances']:
            return f"Feature '{feature1}' not found."
        if feature2 not in interactions['feature_importances']:
            return f"Feature '{feature2}' not found."
        
        imp1 = interactions['feature_importances'][feature1]
        imp2 = interactions['feature_importances'][feature2]
        
        explanation = f"\nRelationship Analysis: {feature1} × {feature2} → {target_name}\n"
        explanation += "="*60 + "\n"
        explanation += f"Feature 1 ({feature1}) importance: {imp1:.4f}\n"
        explanation += f"Feature 2 ({feature2}) importance: {imp2:.4f}\n"
        explanation += "\n"
        
        if imp1 > imp2 * 1.5:
            explanation += f"→ {feature1} is MUCH MORE important than {feature2}\n"
        elif imp2 > imp1 * 1.5:
            explanation += f"→ {feature2} is MUCH MORE important than {feature1}\n"
        else:
            explanation += f"→ Both features have SIMILAR importance\n"
        
        explanation += "\n"
        explanation += "HOW GPR LEARNS THIS INTERACTION:\n"
        explanation += "1. The RBF kernel computes similarity: k(x, x') = exp(-distance²/scale²)\n"
        explanation += "2. Distance is computed in ALL feature dimensions simultaneously\n"
        explanation += "3. Different length scales mean features contribute differently to distance\n"
        explanation += "4. The covariance function captures how features COMBINE (not just individually)\n"
        explanation += "5. This reveals COMPLEX INTERACTIONS that linear models would miss!\n"
        
        return explanation
    
    def save_models(self, filepath: str = 'tpms_ml_models.pkl'):
        """Save trained models to file."""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Models saved to: {filepath}")
    
    def load_models(self, filepath: str = 'tpms_ml_models.pkl'):
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.label_encoder = save_data['label_encoder']
        self.feature_names = save_data['feature_names']
        self.target_names = save_data['target_names']
        
        print(f"✓ Models loaded from: {filepath}")


def main():
    """Example usage of TPMS ML Optimizer."""
    print("="*60)
    print("TPMS ML OPTIMIZER - Gaussian Process Regression")
    print("="*60)
    
    # Initialize optimizer
    optimizer = TPMSMLOptimizer(dataset_path='dataset_full.csv')
    
    # Load data
    optimizer.load_data()
    
    # Train models
    optimizer.train_models()
    
    # Save models
    optimizer.save_models()
    
    # Analyze relationships
    relationships = optimizer.analyze_relationships()
    
    # Example: Find optimal geometry for given load
    print("\n" + "="*60)
    print("EXAMPLE: Optimal geometry for 1000 N load")
    print("="*60)
    
    result = optimizer.find_optimal_geometry(
        target_load_N=1000.0,
        constraints={'porosity_min': 0.2, 'porosity_max': 0.9},
        objective='weight_to_strength_ratio'
    )
    
    print("\nOptimal Geometry:")
    for key, value in result['optimal_geometry'].items():
        print(f"  {key}: {value}")
    
    print("\nPredicted Properties:")
    for prop, values in result['predicted_properties'].items():
        print(f"  {prop}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    print("\n" + "="*60)
    print("✓ ML Optimizer ready for use!")
    print("="*60)


if __name__ == "__main__":
    main()

