#!/usr/bin/env python3
"""
TPMS Shape Optimizer using Machine Learning

This module creates a machine learning model that:
1. Analyzes TPMS dataset with different structure types
2. Learns the relationship between TPMS parameters and properties (porosity, strength, etc.)
3. Generates NEW optimal TPMS equations (not just selects from existing ones)
4. Predicts optimal shape coefficients for desired properties

The model uses a parametric TPMS equation that can interpolate and extrapolate
beyond existing TPMS types to create novel optimal structures.

results at optimal_tpms.py
to use:
python tpms_ml_optimizer.py --load-model tpms_model.pkl --predict \
    --unit-cell-size 10.0 --wall-thickness 0.1 \
    --porosity-min 0.2 --porosity-max 0.5 \
    --target-compressive-strength 40.0 \
    --target-tensile-strength 5.8 \
    --export-stl --output-stl outputs/stl/ml_optimal.stl
    
"""

import numpy as np
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class TPMSEquationCoefficients:
    """Coefficients for parametric TPMS equation
    
    The parametric equation allows interpolation between existing TPMS types
    and creation of novel optimal structures.
    """
    # Base frequency coefficients (control periodicity)
    kx_coeff: float = 1.0
    ky_coeff: float = 1.0
    kz_coeff: float = 1.0
    
    # Sin/Cos term coefficients (control shape characteristics)
    # These coefficients weight different trigonometric combinations
    sin_x_cos_y: float = 0.0
    sin_y_cos_z: float = 0.0
    sin_z_cos_x: float = 0.0
    cos_x: float = 0.0
    cos_y: float = 0.0
    cos_z: float = 0.0
    sin_x_sin_y_sin_z: float = 0.0
    sin_x_cos_y_cos_z: float = 0.0
    cos_x_sin_y_cos_z: float = 0.0
    cos_x_cos_y_sin_z: float = 0.0
    
    # Higher frequency terms (for complex structures)
    sin_2x_cos_y_sin_z: float = 0.0
    sin_x_sin_2y_cos_z: float = 0.0
    cos_x_sin_y_sin_2z: float = 0.0
    cos_2x_cos_2y: float = 0.0
    cos_2y_cos_2z: float = 0.0
    cos_2z_cos_2x: float = 0.0
    
    # Cross-product terms (for advanced structures)
    cos_x_cos_y_cos_z: float = 0.0
    
    # Bias/offset
    bias: float = 0.0
    
    # Normalization factor
    normalization: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'kx_coeff': self.kx_coeff,
            'ky_coeff': self.ky_coeff,
            'kz_coeff': self.kz_coeff,
            'sin_x_cos_y': self.sin_x_cos_y,
            'sin_y_cos_z': self.sin_y_cos_z,
            'sin_z_cos_x': self.sin_z_cos_x,
            'cos_x': self.cos_x,
            'cos_y': self.cos_y,
            'cos_z': self.cos_z,
            'sin_x_sin_y_sin_z': self.sin_x_sin_y_sin_z,
            'sin_x_cos_y_cos_z': self.sin_x_cos_y_cos_z,
            'cos_x_sin_y_cos_z': self.cos_x_sin_y_cos_z,
            'cos_x_cos_y_sin_z': self.cos_x_cos_y_sin_z,
            'sin_2x_cos_y_sin_z': self.sin_2x_cos_y_sin_z,
            'sin_x_sin_2y_cos_z': self.sin_x_sin_2y_cos_z,
            'cos_x_sin_y_sin_2z': self.cos_x_sin_y_sin_2z,
            'cos_2x_cos_2y': self.cos_2x_cos_2y,
            'cos_2y_cos_2z': self.cos_2y_cos_2z,
            'cos_2z_cos_2x': self.cos_2z_cos_2x,
            'cos_x_cos_y_cos_z': self.cos_x_cos_y_cos_z,
            'bias': self.bias,
            'normalization': self.normalization,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TPMSEquationCoefficients':
        """Create from dictionary"""
        return cls(**d)


def parametric_tpms_equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                           coeffs: TPMSEquationCoefficients) -> np.ndarray:
    """
    Generate TPMS field using parametric equation with learnable coefficients.
    
    IMPORTANT: Coordinates (x, y, z) are already scaled by span in solver.compute_surface,
    so we use them directly like the existing TPMS functions (gyroid, schwarz, etc.)
    The solver uses mgrid[-1:1:res] * span, so coordinates are in range [-span, span].
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays (already scaled by span, typically in range [-2π, 2π])
    coeffs : TPMSEquationCoefficients
        Learned coefficients for the equation
        
    Returns
    -------
    np.ndarray
        TPMS field values
    """
    # Apply frequency coefficients to coordinates (like existing TPMS functions)
    # kx_coeff, ky_coeff, kz_coeff control the periodicity
    scaled_x = coeffs.kx_coeff * x
    scaled_y = coeffs.ky_coeff * y
    scaled_z = coeffs.kz_coeff * z
    
    # Compute trigonometric terms (using numpy functions like existing TPMS)
    from numpy import sin, cos
    
    sin_x = sin(scaled_x)
    cos_x = cos(scaled_x)
    sin_y = sin(scaled_y)
    cos_y = cos(scaled_y)
    sin_z = sin(scaled_z)
    cos_z = cos(scaled_z)
    
    # Higher frequency terms
    sin_2x = sin(2.0 * scaled_x)
    cos_2x = cos(2.0 * scaled_x)
    sin_2y = sin(2.0 * scaled_y)
    cos_2y = cos(2.0 * scaled_y)
    sin_2z = sin(2.0 * scaled_z)
    cos_2z = cos(2.0 * scaled_z)
    
    # Build parametric equation
    field = (
        # Basic gyroid-like terms (cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x))
        coeffs.sin_x_cos_y * cos_x * sin_y +  # cos(x) * sin(y)
        coeffs.sin_y_cos_z * cos_y * sin_z +  # cos(y) * sin(z)
        coeffs.sin_z_cos_x * cos_z * sin_x +  # cos(z) * sin(x)
        
        # Schwarz-like terms
        coeffs.cos_x * cos_x +
        coeffs.cos_y * cos_y +
        coeffs.cos_z * cos_z +
        
        # Diamond-like terms
        coeffs.sin_x_sin_y_sin_z * sin_x * sin_y * sin_z +
        coeffs.sin_x_cos_y_cos_z * sin_x * cos_y * cos_z +
        coeffs.cos_x_sin_y_cos_z * cos_x * sin_y * cos_z +
        coeffs.cos_x_cos_y_sin_z * cos_x * cos_y * sin_z +
        
        # Lidinoid-like terms
        coeffs.sin_2x_cos_y_sin_z * sin_2x * cos_y * sin_z +
        coeffs.sin_x_sin_2y_cos_z * sin_x * sin_2y * cos_z +
        coeffs.cos_x_sin_y_sin_2z * cos_x * sin_y * sin_2z +
        coeffs.cos_2x_cos_2y * cos_2x * cos_2y +
        coeffs.cos_2y_cos_2z * cos_2y * cos_2z +
        coeffs.cos_2z_cos_2x * cos_2z * cos_2x +
        
        # Cross-product terms
        coeffs.cos_x_cos_y_cos_z * cos_x * cos_y * cos_z +
        
        # Bias
        coeffs.bias
    )
    
    # Normalize
    if coeffs.normalization != 0:
        field = field / coeffs.normalization
    
    return field


def map_tpms_type_to_coefficients(tpms_type: str) -> TPMSEquationCoefficients:
    """
    Map existing TPMS types to their coefficient representation.
    
    This allows the model to learn from existing structures and interpolate.
    """
    tpms_type = tpms_type.lower()
    
    if tpms_type == 'gyroid':
        # Match lib/surfaces.py: cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x)
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            sin_x_cos_y=1.0,  # cos(x) * sin(y) term
            sin_y_cos_z=1.0,  # cos(y) * sin(z) term  
            sin_z_cos_x=1.0,  # cos(z) * sin(x) term
            normalization=1.0
        )
    elif tpms_type == 'schwarz' or tpms_type == 'schwarz_p':
        # Match lib/surfaces.py: -(cos(x) + cos(y) + cos(z))
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            cos_x=-1.0,
            cos_y=-1.0,
            cos_z=-1.0,
            normalization=1.0
        )
    elif tpms_type == 'diamond':
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            sin_x_sin_y_sin_z=1.0/4.0,
            sin_x_cos_y_cos_z=1.0/4.0,
            cos_x_sin_y_cos_z=1.0/4.0,
            cos_x_cos_y_sin_z=1.0/4.0,
            normalization=1.0
        )
    elif tpms_type == 'lidinoid' or tpms_type == 'l_surface':
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            sin_2x_cos_y_sin_z=1.0/6.0,
            sin_x_sin_2y_cos_z=1.0/6.0,
            cos_x_sin_y_sin_2z=1.0/6.0,
            cos_2x_cos_2y=-1.0/6.0,
            cos_2y_cos_2z=-1.0/6.0,
            cos_2z_cos_2x=-1.0/6.0,
            bias=0.3/6.0,
            normalization=1.0
        )
    elif tpms_type == 'split-p':
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            sin_2x_cos_y_sin_z=1.1/5.0,
            sin_x_sin_2y_cos_z=1.1/5.0,
            cos_x_sin_y_sin_2z=1.1/5.0,
            cos_2x_cos_2y=-0.2/5.0,
            cos_2y_cos_2z=-0.2/5.0,
            cos_2z_cos_2x=-0.2/5.0,
            cos_x=-0.4/5.0,
            cos_y=-0.4/5.0,
            cos_z=-0.4/5.0,
            normalization=1.0
        )
    else:
        # Default to gyroid
        return TPMSEquationCoefficients(
            kx_coeff=1.0, ky_coeff=1.0, kz_coeff=1.0,
            sin_x_cos_y=1.0,
            sin_y_cos_z=1.0,
            sin_z_cos_x=1.0,
            normalization=1.0
        )


class TPMSOptimizer:
    """
    Machine Learning model to optimize TPMS shapes.
    
    Learns from dataset to predict optimal TPMS equation coefficients
    for desired properties (porosity, strength, etc.).
    """
    
    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0):
        """
        Initialize TPMS optimizer.
        
        Parameters
        ----------
        model_type : str
            Type of regression model: 'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'
        alpha : float
            Regularization parameter (for ridge/lasso)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.alpha = alpha
        
        # Create model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif model_type == 'random_forest':
            # RandomForestRegressor supports multi-output natively
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            # GradientBoostingRegressor needs MultiOutputRegressor wrapper for multi-output
            base_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model = MultiOutputRegressor(base_gb)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Store feature names and coefficient names
        self.feature_names = []
        self.coefficient_names = []
    
    def load_dataset(self, csv_path: Path, additional_csv_files: Optional[List[Path]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dataset from CSV file(s), combining multiple sources.
        
        Parameters
        ----------
        csv_path : Path
            Primary CSV file path
        additional_csv_files : List[Path], optional
            Additional CSV files to combine (e.g., simulation_results.csv)
            
        Returns
        -------
        X : np.ndarray
            Feature matrix (TPMS parameters + properties)
        y : np.ndarray
            Target matrix (TPMS equation coefficients)
        feature_names : List[str]
            Names of features
        """
        print(f"Loading dataset from {csv_path}...")
        
        # Read primary CSV
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        # Load additional CSV files if provided
        if additional_csv_files:
            for additional_path in additional_csv_files:
                if additional_path.exists():
                    print(f"Loading additional data from {additional_path}...")
                    with open(additional_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Normalize column names to match primary format
                            normalized_row = self._normalize_row(row)
                            if normalized_row:
                                data.append(normalized_row)
        
        if len(data) == 0:
            raise ValueError(f"Dataset is empty: {csv_path}")
        
        print(f"Loaded {len(data)} total samples")
        
        # Extract features (input: TPMS parameters + desired properties)
        # Features: unit_cell_size, wall_thickness, porosity_min, porosity_max, func_degree
        # Plus: actual properties as features (for optimization)
        feature_names = [
            'unit_cell_size_mm',
            'wall_thickness_mm',
            'porosity_min',
            'porosity_max',
            'func_degree',
            # Actual properties from simulations (used as features for optimization)
            'compressive_strength_MPa',
            'tensile_strength_MPa',
            'energy_absorption_J',
            'max_displacement_mm',
            'max_strain',
        ]
        
        # Extract targets (output: TPMS equation coefficients)
        # Convert TPMS types to coefficients
        X_list = []
        y_list = []
        
        for row in data:
            # Normalize row if needed (handles different CSV formats)
            if 'formula_name' in row and 'tpms_type' not in row:
                normalized_row = self._normalize_row(row)
                if normalized_row:
                    row = normalized_row
                else:
                    continue  # Skip if normalization failed
            
            # Features
            features = []
            for feat_name in feature_names:
                try:
                    val = float(row.get(feat_name, 0.0))
                    features.append(val)
                except (ValueError, KeyError):
                    features.append(0.0)
            
            X_list.append(features)
            
            # Targets: TPMS coefficients
            tpms_type = row.get('tpms_type', 'gyroid')
            coeffs = map_tpms_type_to_coefficients(tpms_type)
            
            # Convert coefficients to array
            coeff_dict = coeffs.to_dict()
            coeff_array = [coeff_dict[k] for k in sorted(coeff_dict.keys())]
            y_list.append(coeff_array)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Store coefficient names
        self.coefficient_names = sorted(coeff_dict.keys())
        self.feature_names = feature_names
        
        print(f"Features shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        print(f"Feature names: {feature_names}")
        print(f"Coefficient names: {self.coefficient_names}")
        
        return X, y, feature_names
    
    def _normalize_row(self, row: Dict) -> Optional[Dict]:
        """
        Normalize row from different CSV formats to common format.
        
        Handles:
        - dataset_full.csv format: tpms_type, unit_cell_size_mm, etc.
        - simulation_results.csv format: formula_name, thickness, threshold, etc.
        """
        normalized = {}
        
        # Map TPMS type
        if 'tpms_type' in row:
            normalized['tpms_type'] = row['tpms_type']
        elif 'formula_name' in row:
            # Map formula_name to tpms_type
            formula = row['formula_name'].lower()
            if formula in ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p', 'holes', 'lamella']:
                normalized['tpms_type'] = formula
            else:
                return None  # Skip unknown types
        else:
            return None  # Need TPMS type
        
        # Map unit_cell_size (estimate from size if not available)
        if 'unit_cell_size_mm' in row:
            normalized['unit_cell_size_mm'] = float(row['unit_cell_size_mm'])
        elif 'size' in row:
            # Estimate: size is total size, approximate unit cell
            size = float(row.get('size', 30))
            normalized['unit_cell_size_mm'] = size / 3.0  # Rough estimate
        else:
            normalized['unit_cell_size_mm'] = 10.0  # Default
        
        # Map wall_thickness
        if 'wall_thickness_mm' in row:
            normalized['wall_thickness_mm'] = float(row['wall_thickness_mm'])
        elif 'thickness' in row:
            normalized['wall_thickness_mm'] = float(row['thickness'])
        else:
            normalized['wall_thickness_mm'] = 0.5  # Default
        
        # Map porosity (estimate from threshold if needed)
        if 'porosity_min' in row:
            normalized['porosity_min'] = float(row['porosity_min'])
        else:
            # Estimate from threshold (negative threshold = higher porosity)
            threshold = float(row.get('threshold', 0.0))
            normalized['porosity_min'] = max(0.1, 0.5 - abs(threshold) * 0.2)
        
        if 'porosity_max' in row:
            normalized['porosity_max'] = float(row['porosity_max'])
        else:
            threshold = float(row.get('threshold', 0.0))
            normalized['porosity_max'] = min(0.9, 0.5 + abs(threshold) * 0.2)
        
        # Map func_degree
        if 'func_degree' in row:
            normalized['func_degree'] = int(row['func_degree'])
        else:
            normalized['func_degree'] = 1  # Default
        
        # Map simulation results (use Mazars results, fallback to earthquake)
        if 'compressive_strength_MPa' in row:
            normalized['compressive_strength_MPa'] = float(row['compressive_strength_MPa'])
        elif 'mazars_compressive_strength_MPa' in row:
            normalized['compressive_strength_MPa'] = float(row['mazars_compressive_strength_MPa'])
        else:
            normalized['compressive_strength_MPa'] = 0.0
        
        if 'tensile_strength_MPa' in row:
            normalized['tensile_strength_MPa'] = float(row['tensile_strength_MPa'])
        elif 'mazars_tensile_strength_MPa' in row:
            normalized['tensile_strength_MPa'] = float(row['mazars_tensile_strength_MPa'])
        else:
            normalized['tensile_strength_MPa'] = 0.0
        
        if 'energy_absorption_J' in row:
            normalized['energy_absorption_J'] = float(row['energy_absorption_J'])
        elif 'mazars_total_energy_absorption_J' in row:
            normalized['energy_absorption_J'] = float(row['mazars_total_energy_absorption_J'])
        elif 'earthquake_max_strain_energy_J' in row:
            normalized['energy_absorption_J'] = float(row['earthquake_max_strain_energy_J'])
        else:
            normalized['energy_absorption_J'] = 0.0
        
        if 'max_displacement_mm' in row:
            normalized['max_displacement_mm'] = float(row['max_displacement_mm'])
        elif 'earthquake_max_displacement_mm' in row:
            normalized['max_displacement_mm'] = float(row['earthquake_max_displacement_mm'])
        else:
            normalized['max_displacement_mm'] = 0.0
        
        if 'max_strain' in row:
            normalized['max_strain'] = float(row['max_strain'])
        else:
            normalized['max_strain'] = 0.0
        
        return normalized
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Train the model to predict TPMS coefficients from features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target matrix (coefficients)
        test_size : float
            Fraction of data to use for testing
        """
        print(f"\nTraining {self.model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (predict each coefficient separately or use multi-output)
        # For simplicity, train separate models for each coefficient
        # Or use multi-output regression if available
        
        if hasattr(self.model, 'fit'):
            # Single model approach: predict all coefficients at once
            # This works for linear models
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            print(f"Training MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
            print(f"Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        else:
            raise ValueError("Model does not support fit method")
        
        self.is_fitted = True
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }
    
    def predict_optimal_coefficients(self, 
                                     unit_cell_size: float,
                                     wall_thickness: float,
                                     porosity_min: float,
                                     porosity_max: float,
                                     func_degree: int,
                                     target_compressive_strength: Optional[float] = None,
                                     target_tensile_strength: Optional[float] = None,
                                     target_energy_absorption: Optional[float] = None,
                                     target_max_displacement: Optional[float] = None,
                                     target_max_strain: Optional[float] = None,
                                     encourage_novelty: bool = True,
                                     novelty_strength: float = 0.3) -> TPMSEquationCoefficients:
        """
        Predict optimal TPMS equation coefficients for given parameters and targets.
        
        Parameters
        ----------
        unit_cell_size : float
            Unit cell size in mm
        wall_thickness : float
            Wall thickness in mm
        porosity_min : float
            Minimum porosity
        porosity_max : float
            Maximum porosity
        func_degree : int
            Function degree (0, 1, or 2)
        target_compressive_strength : float, optional
            Target compressive strength in MPa
        target_tensile_strength : float, optional
            Target tensile strength in MPa
        target_energy_absorption : float, optional
            Target energy absorption in J
        target_max_displacement : float, optional
            Target max displacement in mm
        target_max_strain : float, optional
            Target max strain
            
        Returns
        -------
        TPMSEquationCoefficients
            Predicted optimal coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first. Call train() method.")
        
        # Build feature vector
        features = np.array([[
            unit_cell_size,
            wall_thickness,
            porosity_min,
            porosity_max,
            func_degree,
            target_compressive_strength or 0.0,
            target_tensile_strength or 0.0,
            target_energy_absorption or 0.0,
            target_max_displacement or 0.0,
            target_max_strain or 0.0,
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict coefficients
        coeff_array = self.model.predict(features_scaled)[0]
        
        # Convert to TPMSEquationCoefficients
        coeff_dict = dict(zip(self.coefficient_names, coeff_array))
        coeffs = TPMSEquationCoefficients.from_dict(coeff_dict)
        
        # Encourage novelty by moving away from known TPMS types
        if encourage_novelty:
            coeffs = self._apply_novelty_enhancement(coeffs, novelty_strength)
        
        return coeffs
    
    def _apply_novelty_enhancement(self, coeffs: TPMSEquationCoefficients, strength: float = 0.3) -> TPMSEquationCoefficients:
        """
        Enhance novelty by moving coefficients away from known TPMS types.
        
        This encourages the model to create truly novel shapes rather than
        just selecting the best existing type.
        """
        # Get all known TPMS types
        known_types = ['gyroid', 'schwarz', 'diamond', 'lidinoid', 'split-p']
        known_coeffs = [map_tpms_type_to_coefficients(t) for t in known_types]
        
        # Calculate distance to each known type
        coeff_dict = coeffs.to_dict()
        distances = []
        for known in known_coeffs:
            known_dict = known.to_dict()
            # Euclidean distance in coefficient space
            dist = sum((coeff_dict[k] - known_dict[k])**2 for k in coeff_dict.keys())
            distances.append(dist)
        
        # If too close to any known type, push away
        min_distance = min(distances)
        if min_distance < 0.1:  # Too close to a known type
            # Find which known type we're closest to
            closest_idx = distances.index(min_distance)
            closest_known = known_coeffs[closest_idx]
            closest_dict = closest_known.to_dict()
            
            # Push coefficients away from the closest known type
            # by interpolating towards a more novel combination
            new_coeff_dict = {}
            for k in coeff_dict.keys():
                # Move away from closest known type
                # strength=0.3 means move 30% away from known, 70% keep prediction
                new_coeff_dict[k] = coeff_dict[k] + strength * (coeff_dict[k] - closest_dict[k])
            
            # Also add some random exploration in unused coefficient dimensions
            # to encourage novel combinations
            for k in new_coeff_dict.keys():
                if abs(new_coeff_dict[k]) < 0.01:  # Unused coefficient
                    # Add small random value to explore this dimension
                    import random
                    new_coeff_dict[k] += random.uniform(-0.1, 0.1) * strength
            
            coeffs = TPMSEquationCoefficients.from_dict(new_coeff_dict)
        
        return coeffs
    
    def save_model(self, path: Path):
        """Save trained model to file"""
        import pickle
        
        model_data = {
            'model_type': self.model_type,
            'alpha': self.alpha,
            'scaler': self.scaler,
            'model': self.model,
            'feature_names': self.feature_names,
            'coefficient_names': self.coefficient_names,
            'is_fitted': self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load trained model from file"""
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.alpha = model_data['alpha']
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.coefficient_names = model_data['coefficient_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {path}")


def generate_optimal_tpms_function(coeffs: TPMSEquationCoefficients) -> str:
    """
    Generate Python function code for the optimal TPMS equation.
    
    Parameters
    ----------
    coeffs : TPMSEquationCoefficients
        Learned coefficients
        
    Returns
    -------
    str
        Python function code
    """
    code = f"""def optimal_tpms(x, y, z):
    \"\"\"
    ML-optimized TPMS equation.
    
    Generated coefficients:
    {json.dumps(coeffs.to_dict(), indent=4)}
    \"\"\"
    import numpy as np
    
    # Frequency coefficients
    kx = 2.0 * np.pi * {coeffs.kx_coeff}
    ky = 2.0 * np.pi * {coeffs.ky_coeff}
    kz = 2.0 * np.pi * {coeffs.kz_coeff}
    
    # Trigonometric terms
    sin_x = np.sin(kx * x)
    cos_x = np.cos(kx * x)
    sin_y = np.sin(ky * y)
    cos_y = np.cos(ky * y)
    sin_z = np.sin(kz * z)
    cos_z = np.cos(kz * z)
    
    sin_2x = np.sin(2 * kx * x)
    cos_2x = np.cos(2 * kx * x)
    sin_2y = np.sin(2 * ky * y)
    cos_2y = np.cos(2 * ky * y)
    sin_2z = np.sin(2 * kz * z)
    cos_2z = np.cos(2 * kz * z)
    
    # ML-optimized equation
    field = (
        {coeffs.sin_x_cos_y} * sin_x * cos_y +
        {coeffs.sin_y_cos_z} * sin_y * cos_z +
        {coeffs.sin_z_cos_x} * sin_z * cos_x +
        {coeffs.cos_x} * cos_x +
        {coeffs.cos_y} * cos_y +
        {coeffs.cos_z} * cos_z +
        {coeffs.sin_x_sin_y_sin_z} * sin_x * sin_y * sin_z +
        {coeffs.sin_x_cos_y_cos_z} * sin_x * cos_y * cos_z +
        {coeffs.cos_x_sin_y_cos_z} * cos_x * sin_y * cos_z +
        {coeffs.cos_x_cos_y_sin_z} * cos_x * cos_y * sin_z +
        {coeffs.sin_2x_cos_y_sin_z} * sin_2x * cos_y * sin_z +
        {coeffs.sin_x_sin_2y_cos_z} * sin_x * sin_2y * cos_z +
        {coeffs.cos_x_sin_y_sin_2z} * cos_x * sin_y * sin_2z +
        {coeffs.cos_2x_cos_2y} * cos_2x * cos_2y +
        {coeffs.cos_2y_cos_2z} * cos_2y * cos_2z +
        {coeffs.cos_2z_cos_2x} * cos_2z * cos_2x +
        {coeffs.cos_x_cos_y_cos_z} * cos_x * cos_y * cos_z +
        {coeffs.bias}
    ) / {coeffs.normalization}
    
    return field
"""
    return code


def generate_stl_from_coefficients(coeffs: TPMSEquationCoefficients,
                                   output_path: Path,
                                   size: float = 30.0,
                                   subdivisions: int = 150,
                                   threshold: float = 0.0,
                                   thickness: float = 0.5,
                                   granularity: float = 0.2):
    """
    Generate STL file directly from ML-optimized TPMS coefficients.
    
    This integrates with the existing lib/stl.py and lib/solver.py pipeline.
    
    Parameters
    ----------
    coeffs : TPMSEquationCoefficients
        ML-predicted optimal coefficients
    output_path : Path
        Path to save STL file
    size : float
        Size of the structure
    subdivisions : int
        Number of subdivisions for surface computation
    threshold : float
        Threshold for surface extraction
    thickness : float
        Wall thickness
    granularity : float
        Granularity parameter
    """
    try:
        from lib import solver, stl
        from lib.types import PlotParams
        from numpy import pi
        
        # Create a TPMS function from coefficients
        def ml_optimized_formula(x, y, z):
            return parametric_tpms_equation(x, y, z, coeffs)
        
        # Create parameters
        params = PlotParams(
            name='ml_optimized',
            subdivisions=subdivisions,
            span=pi * 2.0,
            formula=ml_optimized_formula,
            threshold=threshold,
            size=size,
            thickness=thickness,
            granularity=granularity,
        )
        
        # Compute surface
        print(f"Computing ML-optimized TPMS surface...")
        vertices, faces = solver.compute_surface(
            params.span, 
            params.subdivisions, 
            params.formula, 
            params.threshold
        )
        
        # Save STL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stl.save_volume_stl(vertices, faces, params, output_path)
        
        print(f"✓ Generated STL: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating STL: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TPMS Shape Optimizer using Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on dataset
  python tpms_ml_optimizer.py --train --dataset simulation_results.csv
  
  # Predict optimal shape and export STL directly
  python tpms_ml_optimizer.py --load-model tpms_model.pkl --predict \\
      --unit-cell-size 10.0 --wall-thickness 0.1 \\
      --porosity-min 0.2 --porosity-max 0.5 \\
      --target-compressive-strength 40.0 \\
      --export-stl --output-stl outputs/stl/ml_optimal.stl
  
  # Generate optimal TPMS function (for use in code)
  python tpms_ml_optimizer.py --load-model tpms_model.pkl --predict \\
      --unit-cell-size 10.0 --wall-thickness 0.1 \\
      --generate-function --output optimal_tpms.py
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--dataset', type=str, default='dataset_full.csv', help='Dataset CSV file')
    parser.add_argument('--model-type', type=str, default='ridge', 
                       choices=['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
                       help='Type of regression model')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--save-model', type=str, help='Save trained model to file')
    parser.add_argument('--load-model', type=str, help='Load trained model from file')
    
    # Prediction arguments
    parser.add_argument('--predict', action='store_true', help='Predict optimal coefficients')
    parser.add_argument('--unit-cell-size', type=float, help='Unit cell size (mm)')
    parser.add_argument('--wall-thickness', type=float, help='Wall thickness (mm)')
    parser.add_argument('--porosity-min', type=float, help='Minimum porosity')
    parser.add_argument('--porosity-max', type=float, help='Maximum porosity')
    parser.add_argument('--func-degree', type=int, default=1, help='Function degree')
    parser.add_argument('--target-compressive-strength', type=float, help='Target compressive strength (MPa)')
    parser.add_argument('--target-tensile-strength', type=float, help='Target tensile strength (MPa)')
    parser.add_argument('--target-energy-absorption', type=float, help='Target energy absorption (J)')
    parser.add_argument('--encourage-novelty', action='store_true', default=True, help='Encourage novel shapes (default: True)')
    parser.add_argument('--no-novelty', dest='encourage_novelty', action='store_false', help='Disable novelty enhancement')
    parser.add_argument('--novelty-strength', type=float, default=0.3, help='Novelty enhancement strength (0.0-1.0, default: 0.3)')
    
    # Generate function
    parser.add_argument('--generate-function', action='store_true', help='Generate Python function code')
    parser.add_argument('--output', type=str, help='Output file for generated function')
    
    # STL export
    parser.add_argument('--export-stl', action='store_true', help='Export STL file directly')
    parser.add_argument('--output-stl', type=str, help='Output STL file path')
    parser.add_argument('--stl-size', type=float, default=30.0, help='STL size parameter')
    parser.add_argument('--stl-subdivisions', type=int, default=150, help='STL subdivisions')
    parser.add_argument('--stl-threshold', type=float, default=0.0, help='STL threshold')
    parser.add_argument('--stl-thickness', type=float, default=0.5, help='STL thickness')
    parser.add_argument('--stl-granularity', type=float, default=0.2, help='STL granularity')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = TPMSOptimizer(model_type=args.model_type, alpha=args.alpha)
    
    # Load or train model
    if args.load_model:
        optimizer.load_model(Path(args.load_model))
    elif args.train:
        # Use simulation_results.csv as the primary dataset
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            # Try simulation_results.csv as fallback
            sim_results_path = Path('simulation_results.csv')
            if sim_results_path.exists():
                dataset_path = sim_results_path
                print(f"Using simulation_results.csv as dataset")
            else:
                print(f"Error: Dataset file not found: {args.dataset}")
                exit(1)
        
        X, y, feature_names = optimizer.load_dataset(dataset_path)
        results = optimizer.train(X, y)
        
        if args.save_model:
            optimizer.save_model(Path(args.save_model))
    else:
        print("Error: Must specify --train or --load-model")
        exit(1)
    
    # Predict optimal coefficients
    if args.predict:
        if not args.unit_cell_size or not args.wall_thickness:
            print("Error: --unit-cell-size and --wall-thickness are required for prediction")
            exit(1)
        
        coeffs = optimizer.predict_optimal_coefficients(
            unit_cell_size=args.unit_cell_size,
            wall_thickness=args.wall_thickness,
            porosity_min=args.porosity_min or 0.2,
            porosity_max=args.porosity_max or 0.5,
            func_degree=args.func_degree,
            target_compressive_strength=args.target_compressive_strength,
            target_tensile_strength=args.target_tensile_strength,
            target_energy_absorption=args.target_energy_absorption,
            encourage_novelty=args.encourage_novelty,
            novelty_strength=args.novelty_strength,
        )
        
        print("\n" + "="*60)
        print("PREDICTED OPTIMAL TPMS COEFFICIENTS")
        print("="*60)
        print(json.dumps(coeffs.to_dict(), indent=2))
        
        # Export STL if requested
        if args.export_stl:
            stl_path = Path(args.output_stl) if args.output_stl else Path('outputs/stl/ml_optimal.stl')
            print(f"\n{'='*60}")
            print("GENERATING STL FILE FROM ML-OPTIMIZED SHAPE")
            print("="*60)
            generate_stl_from_coefficients(
                coeffs=coeffs,
                output_path=stl_path,
                size=args.stl_size,
                subdivisions=args.stl_subdivisions,
                threshold=args.stl_threshold,
                thickness=args.stl_thickness,
                granularity=args.stl_granularity
            )
            print(f"\n✓ ML-optimized STL saved to: {stl_path}")
            print(f"  This is a NOVEL shape optimized for your requirements!")
            print(f"  Use this STL for simulations or 3D printing.")
        
        # Generate function code if requested
        if args.generate_function:
            code = generate_optimal_tpms_function(coeffs)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(code)
                print(f"\nGenerated function saved to {args.output}")
            else:
                print("\n" + "="*60)
                print("GENERATED OPTIMAL TPMS FUNCTION")
                print("="*60)
                print(code)

