# shap_analysis.py - Fixed SHAP Analysis Module
"""
SHAP Analysis Module that properly integrates with your existing models.py

Key Features:
1. Uses your existing prepare_features_for_model() function (22 UI -> 67 model features)
2. Uses your existing get_trained_model() function  
3. Uses your existing prepare_isbsg_sample_data() function
4. Properly handles feature dimension alignment
5. Integrates seamlessly with your current codebase

Architecture:
- UI Input (22 features) → prepare_features_for_model() → Model Features (66-67)
- ISBSG Background Data → Same pipeline → Model-ready features
- Both processed through same pipeline for consistency
"""

import numpy as np
import pandas as pd
import shap
import warnings
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import yaml
import os

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
logging.getLogger('shap').setLevel(logging.WARNING)

# Import from your models.py - all the functions we need are already there!
try:
    from models import (
        prepare_isbsg_sample_data,
        prepare_features_for_model,
        get_trained_model,
        list_available_models,
        get_field_options,
        FIELDS
    )
    MODELS_AVAILABLE = True
    print("✅ Models module loaded - ISBSG data available for SHAP")
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"❌ Models module not available: {e}")
    
    # Fallback function definitions
    def get_field_options(field_name: str) -> List:
        """Fallback function for field options"""
        return ['option1', 'option2', 'option3']

    def prepare_features_for_model(ui_features: Dict) -> Optional[pd.DataFrame]:
        """Fallback feature preparation"""
        return None

# Global cache for explainers to avoid recreating them
_explainer_cache = {}

def clear_explainer_cache():
    """Clear the explainer cache to free memory."""
    global _explainer_cache
    _explainer_cache.clear()
    print("🗑️ SHAP explainer cache cleared")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current explainer cache."""
    return {
        'cached_models': list(_explainer_cache.keys()),
        'cache_size': len(_explainer_cache)
    }

def get_best_sample_data(n_samples: int = 100, model_name: str = None) -> Optional[np.ndarray]:
    """
    Get the best available sample data using your existing ISBSG function.
    
    Args:
        n_samples: Number of sample instances to return
        model_name: Model name for context (optional)
        
    Returns:
        numpy array with sample data ready for SHAP, or None if all methods fail
    """
    try:
        if MODELS_AVAILABLE:
            # Use your existing ISBSG function - it already returns properly formatted data
            print("🔍 Attempting to load ISBSG sample data...")
            isbsg_data = prepare_isbsg_sample_data(n_samples)
            if isbsg_data is not None:
                print(f"✅ Using ISBSG sample data for SHAP baseline: {isbsg_data.shape}")
                return isbsg_data
            else:
                print("❌ ISBSG data preparation failed")
        
        # Fallback: Generate synthetic data via your feature pipeline
        print("⚠️ No real data available, generating synthetic data via your pipeline...")
        return generate_synthetic_data_via_pipeline(n_samples)
        
    except Exception as e:
        print(f"❌ Error getting sample data: {e}")
        return generate_synthetic_data_via_pipeline(n_samples)

def generate_synthetic_data_via_pipeline(n_samples: int) -> Optional[np.ndarray]:
    """
    Generate synthetic data by creating realistic UI inputs and processing them 
    through your existing feature preparation pipeline.
    
    This ensures the synthetic data has the same structure as real predictions.
    """
    try:
        if not MODELS_AVAILABLE:
            # Final fallback to basic synthetic data
            print("⚠️ No field configuration available - generating basic synthetic data")
            return np.random.normal(0, 1, (n_samples, 67)).astype(np.float32)
        
        print(f"🔄 Generating {n_samples} synthetic samples via your feature pipeline...")
        
        synthetic_samples = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_samples):
            # Create realistic UI inputs (22 features) based on your FIELDS config
            sample_inputs = create_realistic_ui_inputs()
            
            # Process through your existing feature pipeline
            try:
                processed_features = prepare_features_for_model(sample_inputs)
                
                if processed_features is not None and not processed_features.empty:
                    # Convert to numpy array
                    feature_vector = processed_features.values.flatten()
                    synthetic_samples.append(feature_vector)
                else:
                    print(f"⚠️ Feature pipeline failed for sample {i}")
                    # Skip this sample
                    continue
                    
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                continue
        
        if synthetic_samples:
            result = np.array(synthetic_samples, dtype=np.float32)
            print(f"✅ Generated synthetic sample data via your pipeline: {result.shape}")
            return result
        else:
            print("❌ No synthetic samples could be generated")
            return None
            
    except Exception as e:
        print(f"❌ Error generating synthetic data via pipeline: {e}")
        # Final fallback
        return np.random.normal(0, 1, (n_samples, 67)).astype(np.float32)

def create_realistic_ui_inputs() -> Dict:
    """
    Create realistic UI inputs that match your 22 UI features from FIELDS config.
    """
    if not MODELS_AVAILABLE or not FIELDS:
        # Basic fallback if no config available
        return {
            'project_prf_year_of_project': np.random.randint(2020, 2025),
            'project_prf_functional_size': int(np.random.lognormal(6, 1)),
            'project_prf_max_team_size': np.random.randint(3, 15),
            'external_eef_industry_sector': np.random.choice(['finance', 'healthcare', 'retail']),
            'tech_tf_primary_programming_language': np.random.choice(['java', 'python', 'csharp']),
        }
    
    # Generate realistic values based on your actual FIELDS configuration
    ui_inputs = {}
    
    for field_name, field_config in FIELDS.items():
        field_type = field_config.get('type', 'numeric')
        
        if field_type == 'numeric':
            min_val = field_config.get('min', 1)
            max_val = field_config.get('max', 100)
            default_val = field_config.get('default', (min_val + max_val) / 2)
            
            # Generate realistic values based on field name
            if 'functional_size' in field_name.lower():
                # Log-normal for project sizes
                ui_inputs[field_name] = int(np.random.lognormal(np.log(max(default_val, 100)), 1))
            elif 'team_size' in field_name.lower():
                # Realistic team sizes
                ui_inputs[field_name] = np.random.randint(max(min_val, 2), min(max_val, 20))
            elif 'year' in field_name.lower():
                # Recent years
                ui_inputs[field_name] = np.random.randint(2020, 2025)
            else:
                # Normal distribution around default
                value = np.random.normal(default_val, (max_val - min_val) * 0.2)
                ui_inputs[field_name] = np.clip(value, min_val, max_val)
                
        elif field_type == 'categorical':
            # Get options from field config or use get_field_options
            options = field_config.get('options', [])
            if not options:
                options = get_field_options(field_name)
            if options:
                ui_inputs[field_name] = np.random.choice(options)
            else:
                ui_inputs[field_name] = 'option1'
                
        elif field_type == 'boolean':
            # Realistic boolean probabilities
            if 'agile' in field_name.lower():
                probability = 0.65
            elif 'cloud' in field_name.lower():
                probability = 0.55
            else:
                probability = 0.50
            ui_inputs[field_name] = np.random.choice([True, False], p=[probability, 1-probability])
            
        else:
            # Default fallback
            ui_inputs[field_name] = 0
    
    return ui_inputs

def get_shap_explainer(
    model_name: str, 
    get_trained_model_func: Callable = None,
    prepare_sample_data_func: Optional[Callable] = None,
    sample_size: int = 100
) -> Optional[shap.Explainer]:
    """
    Get or create a SHAP explainer for the specified model.
    
    Args:
        model_name: Technical name of the model
        get_trained_model_func: Function to retrieve model (uses your get_trained_model if None)
        prepare_sample_data_func: Deprecated - uses your ISBSG data automatically
        sample_size: Number of samples for background data
    
    Returns:
        SHAP explainer object or None if creation fails
    """
    # Check cache first
    cache_key = f"{model_name}_{sample_size}"
    if cache_key in _explainer_cache:
        print(f"📋 Using cached SHAP explainer for {model_name}")
        return _explainer_cache[cache_key]
    
    try:
        # Use your existing function to get the model
        if get_trained_model_func is None:
            if not MODELS_AVAILABLE:
                print("❌ Models module not available")
                return None
            get_trained_model_func = get_trained_model
        
        # Get the trained model using your function
        model = get_trained_model_func(model_name)
        if model is None:
            print(f"❌ Could not retrieve model '{model_name}'")
            return None
        
        # Get background data using your existing functions
        background_data = get_best_sample_data(sample_size, model_name)
        if background_data is None:
            print(f"⚠️ No background data available for SHAP analysis")
            
        # Create appropriate explainer based on model type
        explainer = None
        model_type = type(model).__name__.lower()
        
        print(f"🔍 Creating SHAP explainer for model type: {model_type}")
        
        # Try TreeExplainer first (for tree-based models)
        if any(t in model_type for t in ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient', 'randomforest', 'extratrees']):
            try:
                if background_data is not None:
                    explainer = shap.TreeExplainer(model, background_data)
                    print(f"✅ Created TreeExplainer with ISBSG background for {model_name}")
                else:
                    explainer = shap.TreeExplainer(model)
                    print(f"⚠️ Created TreeExplainer without background for {model_name}")
            except Exception as tree_error:
                print(f"TreeExplainer failed for {model_name}: {tree_error}")
        
        # Try LinearExplainer for linear models
        elif any(t in model_type for t in ['linear', 'lasso', 'ridge', 'elastic', 'bayesianridge']):
            if background_data is not None:
                try:
                    explainer = shap.LinearExplainer(model, background_data)
                    print(f"✅ Created LinearExplainer with ISBSG background for {model_name}")
                except Exception as linear_error:
                    print(f"LinearExplainer failed for {model_name}: {linear_error}")
            else:
                print(f"❌ LinearExplainer requires background data, but none available")
        
        # Fallback to KernelExplainer (model-agnostic but slower)
        if explainer is None and background_data is not None:
            try:
                # Create prediction function wrapper
                def predict_func(X):
                    if hasattr(model, 'predict'):
                        return model.predict(X)
                    elif hasattr(model, '__call__'):
                        return model(X)
                    else:
                        raise ValueError("Model has no predict method or is not callable")
                
                # Use smaller sample for KernelExplainer (it's computationally expensive)
                kernel_sample = background_data[:min(50, len(background_data))]
                explainer = shap.KernelExplainer(predict_func, kernel_sample)
                print(f"✅ Created KernelExplainer with ISBSG background for {model_name}")
            except Exception as kernel_error:
                print(f"KernelExplainer failed for {model_name}: {kernel_error}")
        
        # Cache the explainer if successful
        if explainer is not None:
            _explainer_cache[cache_key] = explainer
            
            # Test if explainer supports interaction values
            has_interactions = hasattr(explainer, 'shap_interaction_values')
            print(f"✅ Explainer {model_name}: PASS")
            print(f"   Type: {type(explainer).__name__}")
            print(f"   Shap values method: {hasattr(explainer, 'shap_values')}")
            print(f"   Interaction values: {has_interactions}")
        else:
            print(f"❌ All SHAP explainer methods failed for {model_name}")
        
        return explainer
        
    except Exception as e:
        print(f"❌ Error creating SHAP explainer for {model_name}: {e}")
        return None

def get_shap_values_for_input(
    explainer: shap.Explainer, 
    user_inputs: Union[Dict, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Calculate SHAP values for a specific input using your existing feature pipeline.
    
    Args:
        explainer: SHAP explainer object
        user_inputs: Dictionary of user input values or numpy array
        feature_names: Optional list of feature names (not used, kept for compatibility)
    
    Returns:
        SHAP values array or None if calculation fails
    """
    try:
        if explainer is None:
            print("❌ No explainer provided for SHAP calculation")
            return None
        
        # Handle dict input (from UI) using your existing pipeline
        if isinstance(user_inputs, dict):
            if not MODELS_AVAILABLE:
                print("❌ Models module not available for input conversion")
                return None
                
            # Use your existing function to convert UI inputs to model features
            try:
                # This is the key fix - use prepare_features_for_model instead
                input_df = prepare_features_for_model(user_inputs)
                if input_df is None:
                    print("❌ Could not convert user inputs using your pipeline")
                    return None
                
                # Convert DataFrame to numpy array
                input_data = input_df.values
                
            except Exception as e:
                print(f"❌ Error preparing input data: {e}")
                return None
        else:
            input_data = user_inputs
        
        # Ensure 2D input for single prediction
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Calculate SHAP values
        print(f"🔄 Calculating SHAP values for input shape: {input_data.shape}")
        shap_values = explainer.shap_values(input_data)
        
        # Handle different return formats
        if isinstance(shap_values, list):
            # Multi-class or multi-output - return first class/output
            result = shap_values[0]
        else:
            result = shap_values
        
        print(f"✅ SHAP values calculated successfully: {result.shape}")
        return result
        
    except Exception as e:
        print(f"❌ Error calculating SHAP values: {e}")
        return None    

def get_feature_interaction_values(
    explainer: shap.Explainer,
    user_inputs: Union[Dict, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Calculate SHAP interaction values for feature pairs (TreeExplainer only).
    
    Args:
        explainer: SHAP explainer object (must support interaction values)
        user_inputs: Dictionary of user input values or numpy array
        feature_names: Optional list of feature names
    
    Returns:
        SHAP interaction values matrix or None if calculation fails
    """
    try:
        if explainer is None or not hasattr(explainer, 'shap_interaction_values'):
            print("ℹ️ Interaction values not available for this explainer type")
            return None
        
        # Handle dict input using your existing pipeline
        if isinstance(user_inputs, dict):
            if not MODELS_AVAILABLE:
                print("❌ Models module not available for input conversion")
                return None
                
            # Use prepare_features_for_model instead of prepare_input_data
            try:
                input_df = prepare_features_for_model(user_inputs)
                if input_df is None:
                    print("❌ Could not convert user inputs for interaction analysis")
                    return None
                input_data = input_df.values
            except Exception as e:
                print(f"❌ Error preparing input data: {e}")
                return None
        else:
            input_data = user_inputs
        
        # Ensure input is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        print(f"🔄 Calculating SHAP interaction values...")
        interaction_values = explainer.shap_interaction_values(input_data)
        
        # Return interaction matrix for single prediction
        if isinstance(interaction_values, list):
            result = interaction_values[0][0]  # First class, first instance
        else:
            result = interaction_values[0]  # First instance
        
        print(f"✅ SHAP interaction values calculated: {result.shape}")
        return result
            
    except Exception as e:
        print(f"❌ Error calculating SHAP interaction values: {e}")
        return None
    
def get_feature_names_from_fields(fields: Dict) -> List[str]:
    """
    Extract feature names from fields configuration.
    
    Args:
        fields: Fields configuration dictionary
    
    Returns:
        List of feature names (excluding UI fields)
    """
    exclude_fields = {
        'selected_model', 'selected_models', 'submit', 
        'clear_results', 'show_history', 'comparison_mode'
    }
    return [name for name in sorted(fields.keys()) if name not in exclude_fields]

def get_feature_names_from_inputs(user_inputs: Dict) -> List[str]:
    """
    Extract feature names from user inputs.
    
    Args:
        user_inputs: User input dictionary
    
    Returns:
        List of feature names (excluding UI fields)
    """
    exclude_fields = {
        'selected_model', 'selected_models', 'submit', 
        'clear_results', 'show_history', 'comparison_mode'
    }
    return [name for name in sorted(user_inputs.keys()) if name not in exclude_fields]

def validate_shap_inputs(user_inputs: Dict, required_fields: List[str] = None) -> bool:
    """
    Validate that user inputs are suitable for SHAP analysis.
    
    Args:
        user_inputs: User input dictionary
        required_fields: Optional list of required fields
    
    Returns:
        True if inputs are valid, False otherwise
    """
    if not user_inputs:
        return False
    
    # Check for required fields if specified
    if required_fields:
        for field in required_fields:
            if field not in user_inputs or user_inputs[field] is None:
                return False
    
    # Check that we have some meaningful inputs
    exclude_fields = {
        'selected_model', 'selected_models', 'submit', 
        'clear_results', 'show_history', 'comparison_mode'
    }
    meaningful_inputs = {k: v for k, v in user_inputs.items() 
                        if k not in exclude_fields and v is not None and v != ""}
    
    return len(meaningful_inputs) > 0

def create_shap_summary_data(
    shap_values: np.ndarray,
    feature_names: List[str],
    user_inputs: Dict,
    top_n: int = 10
) -> List[Dict]:
    """
    Create summary data for SHAP values display.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        user_inputs: User input values
        top_n: Number of top features to include
    
    Returns:
        List of dictionaries containing feature impact data
    """
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else np.array([])
        else:
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        if len(shap_vals) == 0:
            return []
        
        # Create summary data
        summary_data = []
        for i, (name, shap_val) in enumerate(zip(feature_names, shap_vals)):
            if i >= len(shap_vals):
                break
                
            summary_data.append({
                'feature_name': name,
                'shap_value': float(shap_val),
                'input_value': user_inputs.get(name, 'N/A'),
                'abs_impact': abs(float(shap_val)),
                'direction': 'Increases' if shap_val > 0 else 'Decreases'
            })
        
        # Sort by absolute impact and return top N
        summary_data.sort(key=lambda x: x['abs_impact'], reverse=True)
        return summary_data[:top_n]
        
    except Exception as e:
        print(f"❌ Error creating SHAP summary data: {e}")
        return []

def get_sample_data_info() -> Dict[str, Any]:
    """
    Get information about available sample data sources using your existing functions.
    
    Returns:
        Dictionary with sample data availability and statistics
    """
    info = {
        'isbsg_available': False,
        'training_csv_available': False,
        'synthetic_fallback': True,
        'recommended_source': 'synthetic'
    }
    
    try:
        if MODELS_AVAILABLE:
            # Check ISBSG availability using your function
            from models import get_isbsg_dataset_info
            isbsg_info = get_isbsg_dataset_info()
            if isbsg_info.get('available', False):
                info['isbsg_available'] = True
                info['recommended_source'] = 'isbsg'
                info['isbsg_rows'] = isbsg_info.get('total_rows', 0)
                info['isbsg_features'] = isbsg_info.get('feature_columns', 0)
            
            # Test your prepare_isbsg_sample_data function
            try:
                test_data = prepare_isbsg_sample_data(10)  # Small test
                if test_data is not None:
                    info['isbsg_available'] = True
                    if not info.get('isbsg_available'):
                        info['recommended_source'] = 'isbsg'
            except Exception:
                pass
    
    except Exception as e:
        info['error'] = str(e)
    
    return info

def validate_shap_compatibility(model_name: str) -> Dict[str, Any]:
    """
    Validate if a model is compatible with SHAP analysis.
    """
    try:
        result = {
            'compatible': False,
            'explainer_type': None,
            'issues': [],
            'recommendations': []
        }
        
        # Load the model
        model = get_trained_model(model_name)
        if model is None:
            result['issues'].append("Could not load model")
            return result
        
        model_type = type(model).__name__
        print(f"Checking SHAP compatibility for model type: {model_type}")
        
        # Check for tree-based models (best SHAP support)
        tree_models = [
            'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor',
            'LGBMRegressor', 'CatBoostRegressor', 'ExtraTreesRegressor',
            'DecisionTreeRegressor'
        ]
        
        if any(tree_model in model_type for tree_model in tree_models):
            result['compatible'] = True
            result['explainer_type'] = 'TreeExplainer'
            result['recommendations'].append("Excellent SHAP support with TreeExplainer")
        
        # Check for linear models
        elif any(linear_model in model_type for linear_model in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge']):
            result['compatible'] = True
            result['explainer_type'] = 'LinearExplainer'
            result['recommendations'].append("Good SHAP support with LinearExplainer")
        
        # Check if model has predict method (required for KernelExplainer)
        elif hasattr(model, 'predict'):
            result['compatible'] = True
            result['explainer_type'] = 'KernelExplainer'
            result['recommendations'].append("Basic SHAP support with KernelExplainer (slower)")
            result['issues'].append("KernelExplainer may be slow for complex models")
        
        else:
            result['issues'].append(f"Model type {model_type} may not be fully compatible with SHAP")
            result['recommendations'].append("Consider using a tree-based model for better SHAP support")
        
        return result
        
    except Exception as e:
        return {
            'compatible': False,
            'explainer_type': None,
            'issues': [f"Error checking compatibility: {e}"],
            'recommendations': ["Ensure model can be loaded properly"]
        }

def prepare_sample_data(n_samples, fields, get_field_options_func):
    """
    Main sample data preparation function - now uses your ISBSG data.
    Maintains backward compatibility with original function signature.
    
    Args:
        n_samples: Number of samples to generate
        fields: Field configuration dictionary (unused - uses your FIELDS)
        get_field_options_func: Function to get field options (unused - uses your function)
    
    Returns:
        numpy array with sample data
    """
    try:
        # Use your existing sample data functions
        return get_best_sample_data(n_samples)
        
    except Exception as e:
        print(f"❌ Sample data preparation failed: {e}")
        return None

def prepare_input_data(user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Wrapper function for UI compatibility.
    Converts user inputs to numpy array for SHAP analysis.
    """
    try:
        if not MODELS_AVAILABLE:
            print("❌ Models module not available")
            return None
            
        # Use your existing feature preparation
        features_df = prepare_features_for_model(user_inputs)
        if features_df is None or features_df.empty:
            print("❌ Feature preparation failed")
            return None
        
        # Convert to numpy array
        return features_df.values
        
    except Exception as e:
        print(f"❌ Error in prepare_input_data wrapper: {e}")
        return None

def get_shap_feature_names(model_name: str, user_inputs: Dict[str, Any]) -> List[str]:
    """
    Get feature names for SHAP analysis.
    """
    try:
        # Try to get from prepared features
        if MODELS_AVAILABLE:
            features_df = prepare_features_for_model(user_inputs)
            if features_df is not None:
                return list(features_df.columns)
        
        # Fallback to input names
        return get_feature_names_from_inputs(user_inputs)
        
    except Exception as e:
        print(f"❌ Error getting SHAP feature names: {e}")
        return get_feature_names_from_inputs(user_inputs)

# Export all functions for use in UI - matching original interface
__all__ = [
    'get_shap_explainer',
    'prepare_sample_data', 
    'get_shap_values_for_input',
    'get_feature_interaction_values',
    'get_feature_names_from_fields',
    'get_feature_names_from_inputs',
    'validate_shap_inputs',
    'clear_explainer_cache',
    'get_cache_info',
    'get_sample_data_info',
    'create_shap_summary_data',
    'get_best_sample_data'
]