# models.py - PyCaret compatible module for managing trained models and predictions
"""
- Train Your Models in Jupyter Notebook using PyCaret and Save the trained models 
   to the models folder with its own names ad pkl as file extension. 
   Hence the model file names can be different.
- Make Predictions: Input your project parameters, Select your 
  preferred model and Click "Predict Man-Hours"

    SEQUENTIAL APPROACH:
    1. UI Input → Pipeline Transformation (pipeline.py)
    2. Pipeline Output → Feature Engineering (fill missing features)  
    3. Complete Features → Model Prediction (columns dynamically aligned to trained model)
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, List, Optional, Union, Any

# Import  existing pipeline functions
try:
    from pipeline import convert_feature_dict_to_dataframe, create_preprocessing_pipeline
    PIPELINE_AVAILABLE = True
    logging.info("Pipeline module loaded successfully")
except ImportError as e:
    PIPELINE_AVAILABLE = False
    logging.warning(f"Pipeline module not available: {e}")

try:
    from feature_engineering import (
        estimate_target_value,
        calculate_derived_features,
        validate_features,
        get_feature_summary
    )
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

# --- Load unified YAML config ---
CONFIG_FOLDER = 'config'
MODELS_FOLDER = 'models'
UI_FIELDS_FILE = os.path.join(CONFIG_FOLDER, 'ui_fields.yaml')

def load_yaml_config(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {path}: {e}")
        return {}

APP_CONFIG = load_yaml_config(UI_FIELDS_FILE)
FIELDS = APP_CONFIG.get('fields', {})
TAB_ORG = APP_CONFIG.get('tab_organization', {})

def get_numeric_features():
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'numeric']

def get_categorical_features():
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'categorical']

def get_boolean_features():
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'boolean']

def get_expected_feature_names_from_config() -> List[str]:
    # Preserves order: numeric > categorical > boolean > rest
    feature_names = []
    feature_names += get_numeric_features()
    feature_names += get_categorical_features()
    feature_names += get_boolean_features()
    feature_names += [f for f in FIELDS if f not in feature_names]
    # Remove duplicates
    seen = set()
    unique = []
    for f in feature_names:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique

def validate_feature_dict_against_config(feature_dict: Dict) -> Dict[str, Any]:
    """
    Validate a feature dictionary against the configuration and return missing/extra features.
    """
    expected_features = set(get_expected_feature_names_from_config())
    provided_features = set(feature_dict.keys()) - {'selected_model', 'submit'}
    missing_features = expected_features - provided_features
    extra_features = provided_features - expected_features
    return {
        'missing_features': list(missing_features),
        'extra_features': list(extra_features),
        'valid': len(missing_features) == 0,
        'expected_count': len(expected_features),
        'provided_count': len(provided_features)
    }

def create_feature_vector_from_dict(feature_dict: Dict, expected_features: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a properly ordered feature vector from a feature dictionary.
    """
    if expected_features is None:
        expected_features = get_expected_feature_names_from_config()
    feature_vector = []
    for feature_name in expected_features:
        value = feature_dict.get(feature_name, 0)  # Default to 0 for missing features
        # Ensure numeric type
        try:
            numeric_value = float(value) if value is not None else 0.0
            feature_vector.append(numeric_value)
        except (ValueError, TypeError):
            feature_vector.append(0.0)
    return np.array(feature_vector)

# === Dynamic Feature Alignment Utilities ===

def get_model_expected_features(model) -> List[str]:
    """Get expected feature names from model, robustly."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    if hasattr(model, 'X') and hasattr(model.X, 'columns'):
        return list(model.X.columns)
    return []

def align_features_to_model(feature_dict: Dict[str, Any], model_features: List[str]) -> Dict[str, Any]:
    """Align dict to model columns. Missing columns = 0, extras dropped."""
    return {f: feature_dict.get(f, 0) for f in model_features}

def align_df_to_model(df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]

def ensure_models_folder():
    """Ensure the models folder directory exists."""
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

def list_available_models() -> list:
    """
    Lists all available model files in the models directory (without extension).
    Returns a list of dicts with 'technical_name' and 'display_name'.
    """
    ensure_models_folder()
    model_files = []
    for f in os.listdir(MODELS_FOLDER):
        if f.endswith('.pkl') and not ('scaler' in f.lower()) and not ('pipeline' in f.lower()):
            technical_name = os.path.splitext(f)[0]
            display_name = get_model_display_name(technical_name)
            model_files.append({
                'technical_name': technical_name,
                'display_name': display_name
            })
    model_files.sort(key=lambda x: x['display_name'])
    return model_files

def get_model_display_name(model_filename: str) -> str:
    return " ".join(word.capitalize() for word in model_filename.split("_"))

def check_required_models() -> dict:
    """
    Checks for the presence of model files and a scaler in the models folder.
    Returns a dictionary summarizing their availability and listing the models.
    """
    ensure_models_folder()
    existing_files = os.listdir(MODELS_FOLDER)
    existing_models = [f for f in existing_files if f.endswith('.pkl')]
    has_models = any(f for f in existing_models if not f.startswith('scaler') and 'pipeline' not in f)
    found_models = []
    for f in existing_models:
        if not ('scaler' in f.lower()) and not ('pipeline' in f.lower()):
            technical_name = os.path.splitext(f)[0]
            display_name = get_model_display_name(technical_name)
            found_models.append({
                'technical_name': technical_name,
                'display_name': display_name
            })
    return {
        "models_available": has_models,
        "found_models": found_models,
        "technical_names": [model['technical_name'] for model in found_models]
    }

def load_model(model_name: str) -> Optional[Any]:
    # Try PyCaret, then joblib, then pickle
    model_path = os.path.join(MODELS_FOLDER, model_name)
    try:
        from pycaret.regression import load_model as pc_load_model
        return pc_load_model(model_path)
    except Exception:
        pass
    try:
        import joblib
        if os.path.exists(model_path + ".pkl"):
            return joblib.load(model_path + ".pkl")
        elif os.path.exists(model_path + ".joblib"):
            return joblib.load(model_path + ".joblib")
    except Exception:
        pass
    try:
        with open(model_path + ".pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    return None

def get_model_expected_features(model) -> List[str]:
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    if hasattr(model, 'X') and hasattr(model.X, 'columns'):
        return list(model.X.columns)
    return []

def align_features_to_model(feature_dict: Dict[str, Any], model_features: List[str]) -> Dict[str, Any]:
    return {f: feature_dict.get(f, 0) for f in model_features}

def prepare_features_manually_from_config(features):
    """
    Manual feature preparation using configuration as fallback
    """
    try:
        if isinstance(features, dict):
            # Use the existing logic from create_feature_vector_from_dict
            expected_features = get_expected_feature_names_from_config()
            feature_vector = create_feature_vector_from_dict(features, expected_features)
            features_df = pd.DataFrame([feature_vector], columns=expected_features)
            logging.info(f"Manual feature preparation successful: {features_df.shape}")
            return features_df
        else:
            # Handle array input
            expected_features = get_expected_feature_names_from_config()
            features_array = np.array(features).reshape(1, -1)
            
            # Ensure we have the right number of features
            if features_array.shape[1] != len(expected_features):
                logging.warning(f"Feature count mismatch: got {features_array.shape[1]}, expected {len(expected_features)}")
                # Pad or truncate as needed
                if features_array.shape[1] < len(expected_features):
                    padding = np.zeros((1, len(expected_features) - features_array.shape[1]))
                    features_array = np.hstack([features_array, padding])
                else:
                    features_array = features_array[:, :len(expected_features)]
            
            features_df = pd.DataFrame(features_array, columns=expected_features)
            logging.info(f"Manual array preparation successful: {features_df.shape}")
            return features_df
            
    except Exception as e:
        logging.error(f"Manual feature preparation failed: {e}")
        return None

def prepare_features_for_model(ui_features: Dict[str, Any]) -> pd.DataFrame:
    """Pipeline + feature engineering for user input."""
    # Step 1: Pipeline transformation (if available)
    if PIPELINE_AVAILABLE:
        try:
            feature_df = convert_feature_dict_to_dataframe(ui_features, FIELDS)
            pipeline = create_preprocessing_pipeline(
                target_col=None,
                high_missing_threshold=0.9,
                max_categorical_cardinality=20
            )
            features_transformed = pipeline.fit_transform(feature_df)
            # Optionally drop any label/target columns
            to_drop = [col for col in features_transformed.columns if "target" in col or "effort" in col]
            features_transformed = features_transformed.drop(columns=to_drop, errors="ignore")
        except Exception as e:
            logging.warning(f"Pipeline transformation failed: {e}")
            features_transformed = pd.DataFrame([ui_features])
    else:
        features_transformed = pd.DataFrame([ui_features])
    # Step 2: Feature engineering (if available)
    if FEATURE_ENGINEERING_AVAILABLE:
        try:
            # (Insert any custom logic for derived/calculated fields)
            # Here, you could do: features_transformed = calculate_derived_features(features_transformed)
            pass
        except Exception as e:
            logging.warning(f"Feature engineering failed: {e}")
    return features_transformed

def predict_man_hours(
    features: Union[np.ndarray, Dict, List], 
    model_name: str, 
    use_scaler: bool = False,
    use_preprocessing_pipeline: bool = True
) -> Optional[float]:
    """
    Updated main prediction function that uses the sequential approach for dict input.
    Maintains backward compatibility for array/list inputs.
    """
    
    # Use sequential approach for dictionary input (from UI)
    if isinstance(features, dict):
        # Transform and engineer features
        features_df = prepare_features_for_model(features)
        model = load_model(model_name)
        if not model:
            return None
        model_expected_features = get_model_expected_features(model)
        if not model_expected_features:
            features_aligned = features_df
        else:
            for col in model_expected_features:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_aligned = features_df[model_expected_features]
        try:
            from pycaret.regression import predict_model
            preds = predict_model(model, data=features_aligned)
            for col in ['prediction_label', 'Label', 'pred', 'prediction']:
                if col in preds.columns:
                    return float(preds[col].iloc[0])
            return float(preds.iloc[0, -1])
        except Exception:
            if hasattr(model, 'predict'):
                pred = model.predict(features_aligned)
                return float(pred[0]) if hasattr(pred, '__len__') else float(pred)
        return None
    # else, fallback to previous pipeline logic (not likely needed)
    return None

def get_feature_importance(model_name: str) -> Optional[np.ndarray]:
    """
    Get feature importance for a given model if available.
    """
    model = load_model(model_name)
    if model is None:
        return None
    try:
        if hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_importances_'):
                    return step.feature_importances_
        elif hasattr(model, '_final_estimator'):
            est = model._final_estimator
            if hasattr(est, 'feature_importances_'):
                return est.feature_importances_
            elif hasattr(est, 'coef_'):
                return np.abs(est.coef_).flatten()
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        return None
    except Exception:
        return None

def analyze_what_if(
    base_features: Union[np.ndarray, Dict],
    model_name: str,
    param_name: str,
    param_values: List[float]
) -> Dict[str, List]:
    results = {"param_values": [], "predictions": []}
    base_dict = dict(base_features) if isinstance(base_features, dict) else {}
    for value in param_values:
        test_features = base_dict.copy()
        test_features[param_name] = value
        pred = predict_man_hours(test_features, model_name)
        if pred is not None:
            results["param_values"].append(value)
            results["predictions"].append(pred)
    return results

# Exports for Streamlit UI
__all__ = [
    'list_available_models',
    'get_model_display_name',
    'check_required_models',
    'predict_man_hours',
    'get_feature_importance',
    'analyze_what_if'
]
