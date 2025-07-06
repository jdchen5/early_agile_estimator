# shap_analysis.py - Fixed SHAP Analysis Module for PyCaret Models
"""
SHAP Analysis Module that properly handles PyCaret-wrapped models.

Key Features:
1. Properly extracts the underlying estimator from PyCaret models
2. Handles the feature transformation pipeline correctly
3. Returns actual SHAP explainer objects, not dictionaries
4. Properly aligns features between UI (22) and model (54-67)
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
from constants import PipelineConstants, ModelConstants, UIConstants

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
logging.getLogger('shap').setLevel(logging.WARNING)

# Import from your models.py
try:
    from models import (
        prepare_isbsg_sample_data,
        prepare_features_for_model,
        get_trained_model,
        list_available_models,
        get_model_expected_features,
        align_df_to_model,
        get_isbsg_dataset_info,
        load_model,  # Add this import
        load_preprocessing_pipeline,
        FIELDS  
    )
    MODELS_AVAILABLE = True
    print("Models module loaded - ISBSG data available for SHAP")
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Models module not available: {e}")
    
    # Fallback function definitions
    def get_field_options(field_name: str) -> List:
        """Fallback function for field options"""
        return ['option1', 'option2', 'option3']

    def prepare_features_for_model(ui_features: Dict) -> Optional[pd.DataFrame]:
        """Fallback feature preparation"""
        return None

# Global cache for explainers
_explainer_cache = {}

def clear_explainer_cache():
    """Clear the explainer cache to free memory."""
    global _explainer_cache
    _explainer_cache.clear()
    print("SHAP explainer cache cleared")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current explainer cache."""
    return {
        'cached_models': list(_explainer_cache.keys()),
        'cache_size': len(_explainer_cache)
    }

def extract_pycaret_estimator(model):
    """
    Extract the underlying estimator from a PyCaret model.
    PyCaret wraps models in containers, we need the actual estimator.
    """
    try:
        # Check if it's a PyCaret model
        if hasattr(model, '_final_estimator'):
            actual_model = model._final_estimator
            print(f"Extracted final estimator from PyCaret: {type(actual_model).__name__}")
            return actual_model
            
        # Check if it's a pipeline
        elif hasattr(model, 'named_steps'):
            # Look for the estimator in the pipeline
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'predict') and not step_name.startswith(('scaler', 'encoder', 'imputer')):
                    print(f"Extracted estimator from pipeline step '{step_name}': {type(step).__name__}")
                    return step
            
            # If no suitable step found, try the last step
            step_names = list(model.named_steps.keys())
            if step_names:
                final_step = model.named_steps[step_names[-1]]
                print(f"Extracted final pipeline step: {type(final_step).__name__}")
                return final_step
                
        # Check for sklearn pipeline format
        elif hasattr(model, 'steps'):
            if len(model.steps) > 0:
                final_step = model.steps[-1][1]
                print(f"Extracted from sklearn pipeline: {type(final_step).__name__}")
                return final_step
        
        # If it's already a raw estimator, return it
        print(f"Model appears to be raw estimator: {type(model).__name__}")
        return model
        
    except Exception as e:
        print(f"Error extracting estimator: {e}")
        return model

def get_best_sample_data(n_samples: int = 100, model_name: str = None) -> Optional[np.ndarray]:
    """Hybrid: Try ISBSG mapping first, fallback to synthetic"""
    try:
        # Try ISBSG approach first
        return get_isbsg_mapped_data(n_samples)
    except Exception as e:
        print(f"ISBSG mapping failed: {e}")
        print("Falling back to synthetic generation...")
        # Fallback to synthetic approach
        return get_synthetic_data_from_stats(n_samples)

def get_isbsg_mapped_data(n_samples: int) -> Optional[np.ndarray]:
    """Simple ISBSG→UI mapping approach"""
    # Get raw ISBSG data
    raw_isbsg = prepare_isbsg_sample_data(n_samples)
    if raw_isbsg is None:
        raise Exception("ISBSG data not available")
    
    processed_samples = []
    
    for i in range(min(n_samples, len(raw_isbsg))):
        # Simple mapping: extract basic features from ISBSG
        ui_input = {
            'project_prf_functional_size': raw_isbsg[i][3] if len(raw_isbsg[i]) > 3 else 100,
            'project_prf_max_team_size': raw_isbsg[i][13] if len(raw_isbsg[i]) > 13 else 5,
            # Add defaults for other required UI features
            'external_eef_industry_sector': 'financial',
            'tech_tf_primary_programming_language': 'Java',
        }
        
        # Process through pipeline
        processed = prepare_features_for_model(ui_input)
        if processed is not None:
            processed_samples.append(processed.values.flatten())
    
    return np.array(processed_samples, dtype=np.float32) if processed_samples else None

def get_synthetic_data_from_stats(n_samples: int) -> Optional[np.ndarray]:
    """Fallback: Generate synthetic UI inputs"""
    processed_samples = []
    
    for i in range(n_samples):
        # Generate realistic UI inputs
        ui_input = create_realistic_ui_inputs()
        
        # Process through pipeline  
        processed = prepare_features_for_model(ui_input)
        if processed is not None:
            processed_samples.append(processed.values.flatten())
    
    return np.array(processed_samples, dtype=np.float32) if processed_samples else None

def generate_synthetic_data_via_pipeline(n_samples: int) -> Optional[np.ndarray]:
    """
    Generate synthetic data by creating realistic UI inputs and processing them 
    through your existing feature preparation pipeline.
    """
    try:
        if not MODELS_AVAILABLE:
            print("No models available - generating random synthetic data")
            # Match your model's expected feature count
            return np.random.normal(0, 1, (n_samples, 54)).astype(np.float32)
        
        print(f"Generating {n_samples} synthetic samples via feature pipeline...")
        
        synthetic_samples = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_samples):
            # Create realistic UI inputs
            sample_inputs = create_realistic_ui_inputs()
            
            # Process through your existing feature pipeline
            try:
                processed_features = prepare_features_for_model(sample_inputs)
                
                if processed_features is not None and not processed_features.empty:
                    feature_vector = processed_features.values.flatten()
                    synthetic_samples.append(feature_vector)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if synthetic_samples:
            result = np.array(synthetic_samples, dtype=np.float32)
            print(f"Generated synthetic data: {result.shape}")
            return result
        else:
            print("No synthetic samples could be generated")
            # Return random data as last resort
            return np.random.normal(0, 1, (n_samples, 54)).astype(np.float32)
            
    except Exception as e:
        print(f"Error in synthetic data generation: {e}")
        return np.random.normal(0, 1, (n_samples, 54)).astype(np.float32)

def create_realistic_ui_inputs() -> Dict:
    """
    Create realistic UI inputs that match your 22 UI features.
    """
    return {
        'project_prf_year_of_project': np.random.randint(2020, 2025),
        'external_eef_industry_sector': np.random.choice(['Financial', 'Banking',]),
        'tech_tf_primary_programming_language': np.random.choice(['Java', 'Python', 'C#', 'Agile platform']),
        'tech_tf_tools_used': np.random.randint(0, 5),
        'project_prf_relative_size': np.random.choice(['XXS', 'XS', 'S', 'M', 'L']),
        'project_prf_functional_size': int(np.random.lognormal(5, 1.5)),
        'project_prf_development_type': '',
        'tech_tf_language_type': '',
        'project_prf_application_type': None,
        'external_eef_organisation_type': np.random.choice(['Amusement/Game Center', 'Banking', 
                                                            'Banking, Insurance, Stock', 'Credit Card Processor', 
                                                            'Financial', 'Financial, Property & Business Services', 
                                                            'Insurance', 'Revenue', 'Revenue Collection']),
        'tech_tf_architecture': '',
        'tech_tf_development_platform': '',
        'project_prf_team_size_group': '',
        'project_prf_max_team_size': np.random.randint(3, 15),
        'tech_tf_server_roles': None,
        'tech_tf_client_roles': None,
        'tech_tf_web_development': np.random.choice([True, False]),
        'tech_tf_dbms_used': np.random.choice([True, False]),
        'process_pmf_prototyping_used': np.random.choice([True, False]),
        'project_prf_case_tool_used': np.random.choice([True, False]),
        'process_pmf_docs': np.random.randint(0, 10),
        'people_prf_project_user_involvement': np.random.randint(0, 5)
    }

def get_shap_explainer(
    model_name: str, 
    get_trained_model_func: Callable = None,
    prepare_sample_data_func: Optional[Callable] = None,
    sample_size: int = PipelineConstants.DEFAULT_SAMPLE_SIZE
) -> Optional[shap.Explainer]:
    """
    Get or create a SHAP explainer for the specified model.
    This version supports both pipeline and non-pipeline workflows.
    """
    # Check cache first
    cache_key = f"{model_name}_{sample_size}"
    if cache_key in _explainer_cache:
        print(f"📋 Using cached SHAP explainer for {model_name}")
        return _explainer_cache[cache_key]
    
    try:
        # === NEW PIPELINE APPROACH ===
        # Try pipeline approach first
        from models import load_preprocessing_pipeline, get_pipeline_background_data
        
        pipeline = load_preprocessing_pipeline()
        if pipeline is not None:
            print("🔧 Using pipeline-based SHAP approach")
            
            # Get background data through pipeline
            background_data_raw = get_best_sample_data(sample_size, model_name)
            pipeline = load_preprocessing_pipeline()
            background_data = pipeline.transform(background_data_raw)

            
            # Load model
            if get_trained_model_func:
                model = get_trained_model_func(model_name)
            else:
                model = get_trained_model(model_name)
                
            if model is None:
                print(f"Could not load model '{model_name}'")
                return None
            
            # Extract actual estimator
            actual_model = model
            
            # Create explainer with pipeline-processed background
            model_type = type(actual_model).__name__.lower()
            explainer = None
            
            # Try appropriate explainer type
            if any(keyword in model_type for keyword in ['forest', 'tree', 'xgb', 'lgb', 'catboost']):
                try:
                    explainer = shap.TreeExplainer(actual_model, background_data)
                    print(f"Created TreeExplainer with pipeline background")
                except Exception as e:
                    print(f"TreeExplainer failed: {e}")
            
            elif any(keyword in model_type for keyword in ['linear', 'lasso', 'ridge', 'elastic']):
                try:
                    explainer = shap.LinearExplainer(actual_model, background_data)
                    print(f"Created LinearExplainer with pipeline background")
                except Exception as e:
                    print(f"LinearExplainer failed: {e}")
            
            # Fallback to general Explainer
            if explainer is None:
                try:
                    explainer = shap.Explainer(actual_model, background_data[:PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE])  # Smaller sample
                    print(f"Created general Explainer with pipeline background")
                except Exception as e:
                    print(f"General Explainer failed: {e}")
            
            if explainer is not None:
                _explainer_cache[cache_key] = explainer
                return explainer
        
        # === EXISTING LOGIC (FALLBACK) ===
        # This is your current implementation from the original get_shap_explainer
        print("Using traditional SHAP approach (no pipeline)")
        
        # Load the full PyCaret model
        if not MODELS_AVAILABLE:
            print("Models module not available")
            return None
            
        # Load the complete model (with PyCaret wrapper)
        full_model = load_model(model_name)
        if full_model is None:
            print(f"Could not load model '{model_name}'")
            return None
        
        # Extract the actual estimator from PyCaret wrapper
        actual_model = extract_pycaret_estimator(full_model)
        if actual_model is None:
            print(f"Could not extract estimator from PyCaret model")
            return None
        
        # Get background data using the traditional method
        background_data = get_best_sample_data(sample_size, model_name)
        
        # Prepare background data through the same pipeline
        if background_data is not None and MODELS_AVAILABLE:
            print("Processing background data through feature pipeline...")
            # Create a function that applies the full prediction pipeline
            def model_predict_func(X):
                try:
                    # If X is raw ISBSG data, ensure it goes through the model's pipeline
                    if isinstance(X, np.ndarray):
                        X_df = pd.DataFrame(X)
                    else:
                        X_df = X
                    
                    # Use the full PyCaret model for prediction (includes preprocessing)
                    predictions = full_model.predict(X_df)
                    return predictions
                except Exception as e:
                    print(f"Prediction error in SHAP: {e}")
                    # Fallback to direct prediction
                    return actual_model.predict(X)
        else:
            model_predict_func = lambda X: actual_model.predict(X)
        
        # Create appropriate explainer based on model type
        explainer = None
        model_type = type(actual_model).__name__.lower()
        
        print(f"🔍 Creating SHAP explainer for {model_type}")
        
        # Try TreeExplainer first (for tree-based models)
        tree_keywords = ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient', 
                        'randomforest', 'extratrees', 'decisiontree']
        
        if any(keyword in model_type for keyword in tree_keywords):
            try:
                if background_data is not None:
                    explainer = shap.TreeExplainer(actual_model, background_data)
                else:
                    explainer = shap.TreeExplainer(actual_model)
                print(f"Created TreeExplainer for {model_name}")
            except Exception as e:
                print(f"TreeExplainer failed: {e}")
        
        # Try LinearExplainer for linear models
        elif any(keyword in model_type for keyword in ['linear', 'lasso', 'ridge', 'elastic', 'bayesianridge']):
            if background_data is not None:
                try:
                    explainer = shap.LinearExplainer(actual_model, background_data)
                    print(f"Created LinearExplainer for {model_name}")
                except Exception as e:
                    print(f"LinearExplainer failed: {e}")
        
        # Fallback to KernelExplainer
        if explainer is None and background_data is not None:
            try:
                # Use smaller sample for KernelExplainer
                kernel_sample = background_data[:min(50, len(background_data))]
                
                # Use the full model's predict function for KernelExplainer
                explainer = shap.KernelExplainer(model_predict_func, kernel_sample)
                print(f"Created KernelExplainer for {model_name}")
            except Exception as e:
                print(f"KernelExplainer failed: {e}")
        
        # Cache successful explainer
        if explainer is not None:
            _explainer_cache[cache_key] = explainer
            print(f"SHAP explainer created and cached for {model_name}")
        else:
            print(f"Failed to create any SHAP explainer for {model_name}")
        
        return explainer
        
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_shap_values_for_input(
    explainer: Union[shap.Explainer, Dict], 
    user_inputs: Union[Dict, np.ndarray],
    model=None,
    feature_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """Enhanced to support pipeline workflow"""
    
    try:
        # If dict input, transform through pipeline
        if isinstance(user_inputs, dict):
            from models import transform_with_pipeline
            
            # Try pipeline transformation first
            input_data = transform_with_pipeline(user_inputs)

            # --- PLACE DEBUG PRINTS HERE ---
            print("User input keys:", user_inputs.keys())
            print("Transformed input shape:", input_data.shape)  # should be (1, 67)
            # --------------------------------
            
            if input_data is None:
                # Fallback to existing method
                input_data = prepare_features_for_model(user_inputs)
            
            # Ensure it's numpy for SHAP
            if isinstance(input_data, pd.DataFrame):
                input_data = input_data.values
        else:
            input_data = user_inputs
        
        # Rest of the existing logic...
        return explainer.shap_values(input_data)
        
    except Exception as e:
        logging.error(f"Error in pipeline SHAP values: {e}")
        return None

def get_feature_interaction_values(
    explainer: shap.Explainer,
    user_inputs: Union[Dict, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Calculate SHAP interaction values (TreeExplainer only).
    """
    try:
        if explainer is None or not hasattr(explainer, 'shap_interaction_values'):
            print("Interaction values not available for this explainer")
            return None
        
        # Prepare input data
        if isinstance(user_inputs, dict):
            if not MODELS_AVAILABLE:
                return None
                
            input_df = prepare_features_for_model(user_inputs)
            if input_df is None:
                return None
            input_data = input_df.values
        else:
            input_data = user_inputs
        
        # Ensure 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        print("Calculating SHAP interaction values...")
        interaction_values = explainer.shap_interaction_values(input_data)
        
        # Extract first instance
        if isinstance(interaction_values, list):
            result = interaction_values[0][0]
        else:
            result = interaction_values[0]
        
        print(f"Interaction values calculated: {result.shape}")
        return result
            
    except Exception as e:
        print(f"Error calculating interaction values: {e}")
        return None

def get_feature_names_from_fields(fields: Dict) -> List[str]:
    """Extract feature names from fields configuration."""
    exclude_fields = {
        'selected_model', 'selected_models', 'submit', 
        'clear_results', 'show_history', 'comparison_mode'
    }
    return [name for name in sorted(fields.keys()) if name not in exclude_fields]

def get_feature_names_from_inputs(user_inputs: Dict) -> List[str]:
    """Extract feature names from user inputs."""
    exclude_fields = {
        'selected_model', 'selected_models', 'submit', 
        'clear_results', 'show_history', 'comparison_mode'
    }
    return [name for name in sorted(user_inputs.keys()) if name not in exclude_fields]

def validate_shap_inputs(user_inputs: Dict, required_fields: List[str] = None) -> bool:
    """Validate that user inputs are suitable for SHAP analysis."""
    if not user_inputs:
        return False
    
    if required_fields:
        for field in required_fields:
            if field not in user_inputs or user_inputs[field] is None:
                return False
    
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
    top_n: int = PipelineConstants.TOP_N_FEATURES
) -> List[Dict]:
    """Create summary data for SHAP values display."""
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else np.array([])
        else:
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        if len(shap_vals) == 0:
            return []
        
        # Create summary
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
        
        # Sort by impact
        summary_data.sort(key=lambda x: x['abs_impact'], reverse=True)
        return summary_data[:top_n]
        
    except Exception as e:
        print(f"Error creating summary: {e}")
        return []

def get_sample_data_info() -> Dict[str, Any]:
    """Get information about available sample data sources."""
    info = {
        'isbsg_available': False,
        'training_csv_available': False,
        'synthetic_fallback': True,
        'recommended_source': 'synthetic'
    }
    
    try:
        if MODELS_AVAILABLE:
            # Check ISBSG availability
            isbsg_info = get_isbsg_dataset_info()
            if isbsg_info.get('available', False):
                info['isbsg_available'] = True
                info['recommended_source'] = 'isbsg'
                info['isbsg_rows'] = isbsg_info.get('total_rows', 0)
                info['isbsg_features'] = isbsg_info.get('feature_columns', 0)
    except Exception as e:
        info['error'] = str(e)
    
    return info

def prepare_sample_data(n_samples, fields, get_field_options_func):
    """Main sample data preparation function."""
    try:
        return get_best_sample_data(n_samples)
    except Exception as e:
        print(f"Sample data preparation failed: {e}")
        return None

def prepare_input_data(user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Wrapper function for UI compatibility."""
    try:
        if not MODELS_AVAILABLE:
            print("Models module not available")
            return None
            
        features_df = prepare_features_for_model(user_inputs)
        if features_df is None or features_df.empty:
            print("Feature preparation failed")
            return None
        
        return features_df.values
        
    except Exception as e:
        print(f"Error in prepare_input_data: {e}")
        return None

def get_parameter_index(param_name, feature_names):
    """Get the index of a parameter in the feature names list."""
    try:
        return feature_names.index(param_name)
    except (ValueError, AttributeError):
        return None

# Export all functions
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
    'get_best_sample_data',
    'get_parameter_index',
    'prepare_input_data'
]