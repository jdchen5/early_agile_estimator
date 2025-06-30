# shap_analysis.py
"""
SHAP Analysis Utilities for ML Effort Estimator
This module contains all SHAP explainer, data prep, and SHAP value calculation logic.
UI/Streamlit display should be handled in ui.py.
"""

import shap
import numpy as np

# You may need to import or pass in:
# - get_trained_model
# - FIELDS
# - get_field_options

def get_shap_explainer(model_name, get_trained_model_func, prepare_sample_data_func):
    """
    Create a SHAP explainer for the given model.
    Args:
        model_name: str, name of model to load
        get_trained_model_func: function(model_name) -> model object
        prepare_sample_data_func: function(n_samples) -> np.ndarray sample data
    Returns:
        SHAP explainer object or None
    """
    try:
        model = get_trained_model_func(model_name)
        if model is None:
            return None

        model_type = type(model).__name__.lower()
        # Tree-based models
        if any(t in model_type for t in ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient']):
            explainer = shap.TreeExplainer(model)
        # Linear models
        elif any(t in model_type for t in ['linear', 'lasso', 'ridge', 'elastic', 'bayesianridge']):
            sample_data = prepare_sample_data_func(100)
            explainer = shap.LinearExplainer(model, sample_data)
        # Neural networks or others
        elif any(t in model_type for t in ['neural', 'mlp', 'perceptron']):
            sample_data = prepare_sample_data_func(20)
            explainer = shap.KernelExplainer(model.predict, sample_data)
        # Fallback
        else:
            sample_data = prepare_sample_data_func(20)
            explainer = shap.KernelExplainer(model.predict, sample_data)
        return explainer
    except Exception as e:
        print(f"Error creating SHAP explainer for {model_name}: {e}")
        return None

def prepare_sample_data(fields, n_samples=100, get_field_options_func=None):
    """
    Prepare sample data for SHAP background (using your FIELDS config).
    Args:
        fields: dict of field config (like FIELDS from your YAML)
        n_samples: int
        get_field_options_func: optional function(field_name) -> options list
    Returns:
        np.ndarray of shape (n_samples, num_features)
    """
    sample_data = []
    feature_names = list(fields.keys())
    for _ in range(n_samples):
        row = []
        for fname in feature_names:
            fconfig = fields[fname]
            ftype = fconfig.get('type', 'numeric')
            if ftype == 'numeric':
                minv = fconfig.get('min', 1)
                maxv = fconfig.get('max', 100)
                default = fconfig.get('default', (minv + maxv) / 2)
                value = np.random.normal(default, (maxv - minv) * 0.2)
                value = np.clip(value, minv, maxv)
                row.append(value)
            elif ftype == 'categorical':
                if get_field_options_func:
                    opts = get_field_options_func(fname)
                else:
                    opts = fconfig.get('options', [])
                if opts:
                    row.append(np.random.randint(0, len(opts)))
                else:
                    row.append(0)
            elif ftype == 'boolean':
                row.append(np.random.choice([0, 1]))
            else:
                row.append(0)
        sample_data.append(row)
    return np.array(sample_data)

def get_shap_values_for_input(explainer, input_data):
    """
    Compute SHAP values for a single prediction input.
    Args:
        explainer: SHAP explainer object
        input_data: np.ndarray (1D or 2D, single row)
    Returns:
        SHAP values for the instance (np.ndarray)
    """
    try:
        # Ensure 2D input for a single prediction
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        shap_values = explainer.shap_values(input_data)
        # Return regression (or first class for multi-class)
        if isinstance(shap_values, list):
            return shap_values[0]
        return shap_values
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None

def get_feature_interaction_values(explainer, input_data):
    """
    Compute SHAP feature interaction values (for TreeExplainer).
    Args:
        explainer: SHAP explainer
        input_data: np.ndarray
    Returns:
        interaction matrix (np.ndarray) or None
    """
    try:
        if hasattr(explainer, 'shap_interaction_values'):
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            return explainer.shap_interaction_values(input_data)[0]
        else:
            return None
    except Exception as e:
        print(f"Error calculating SHAP interaction values: {e}")
        return None

def get_feature_names_from_fields(fields):
    """
    Utility to get feature names in training order (matches model input).
    """
    return list(fields.keys())


# Helper functions for SHAP analysis
def get_feature_names_from_inputs(user_inputs):
    """Extract feature names from user inputs"""
    exclude_keys = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
    return [k for k in user_inputs.keys() if k not in exclude_keys]

def get_parameter_index(parameter_name, feature_names):
    """Get the index of a parameter in the feature array"""
    # This should return the index of the parameter in your model's feature array
    # You'll need to implement this based on your feature ordering
    try:
        return feature_names.index(parameter_name)
    except ValueError:
        return None