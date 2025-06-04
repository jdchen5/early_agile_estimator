# models.py - PyCaret compatible module for managing trained models and predictions
"""
- Train Your Models in Jupyter Notebook using PyCaret and Save the trained models 
   to the models folder with its own names ad pkl as file extension. 
   Hence the model file names can be different.
- Make Predictions: Input your project parameters, Select your 
  preferred model and Click "Predict Man-Hours"
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, List, Optional, Union, Any, Tuple

# Paths
DATA_FOLDER = 'data'
CONFIG_FOLDER = 'config'
MODELS_FOLDER = 'models'
FEATURE_COLS_FILE = os.path.join(DATA_FOLDER, 'expected_features.txt')
MODEL_DISPLAY_NAMES_FILE = os.path.join(CONFIG_FOLDER, 'model_display_names.json')
FEATURE_MAPPING_FILE = os.path.join(CONFIG_FOLDER, 'feature_mapping.yaml')
UI_CONFIG_FILE = os.path.join(CONFIG_FOLDER, 'ui_config.yaml')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_yaml_config(path: str) -> Dict:
    """Load YAML configuration file with error handling"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {path}: {e}")
        return {}

# Load configurations
FEATURE_CONFIG = load_yaml_config(FEATURE_MAPPING_FILE)
UI_CONFIG = load_yaml_config(UI_CONFIG_FILE)


def get_expected_feature_names_from_config() -> List[str]:
    """
    Get expected feature names from the feature mapping configuration.
    This creates the complete feature vector based on all possible features.
    """
    feature_names = []
    
    # Add numeric features
    numeric_features = FEATURE_CONFIG.get("numeric_features", [])
    feature_names.extend(numeric_features)
    
    # Add categorical features (they remain as-is in the feature vector)
    categorical_features = FEATURE_CONFIG.get("categorical_features", {})
    feature_names.extend(categorical_features.keys())
    
    # Add one-hot encoded features
    one_hot_features = FEATURE_CONFIG.get("one_hot_features", {})
    for group_name, group_config in one_hot_features.items():
        mapping = group_config.get("mapping", {})
        feature_names.extend(mapping.values())
    
    # Add special case features
    special_cases = FEATURE_CONFIG.get("special_cases", {})
    for group_name, group_config in special_cases.items():
        if "output_keys" in group_config:
            feature_names.extend(group_config["output_keys"].values())
    
    # Add binary features
    binary_features = FEATURE_CONFIG.get("binary_features", {})
    for group_name, group_config in binary_features.items():
        mapping = group_config.get("mapping", {})
        feature_names.extend(mapping.values())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for name in feature_names:
        if name not in seen:
            seen.add(name)
            unique_features.append(name)
    
    logging.info(f"Generated {len(unique_features)} expected feature names from configuration")
    return unique_features

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
    
    # Validate the feature dictionary
    validation = validate_feature_dict_against_config(feature_dict)
    if not validation['valid']:
        logging.warning(f"Feature validation failed. Missing: {validation['missing_features']}")
    
    # Create feature vector
    feature_vector = []
    for feature_name in expected_features:
        value = feature_dict.get(feature_name, 0)  # Default to 0 for missing features
        # Ensure numeric type
        try:
            numeric_value = float(value) if value is not None else 0.0
            feature_vector.append(numeric_value)
        except (ValueError, TypeError):
            logging.warning(f"Non-numeric value for feature '{feature_name}': {value}, using 0.0")
            feature_vector.append(0.0)
    
    return np.array(feature_vector)

def list_available_models() -> list:
    """
    Lists all available model files in the models directory (without extension).
    Returns a list of dicts with 'technical_name' and 'display_name'.
    """
    ensure_models_folder()
    model_files = []
    for f in os.listdir(MODELS_FOLDER):
        if f.endswith('.pkl') and not ('scaler' in f.lower()):
            technical_name = os.path.splitext(f)[0]
            display_name = get_model_display_name(technical_name)
            model_files.append({
                'technical_name': technical_name,
                'display_name': display_name
            })
    # Sort by display name for user-friendly dropdown
    model_files.sort(key=lambda x: x['display_name'])
    return model_files

def check_required_models() -> dict:
    """
    Checks for the presence of model files and a scaler in the models folder.
    Returns a dictionary summarizing their availability and listing the models.
    """
    ensure_models_folder()
    existing_files = os.listdir(MODELS_FOLDER)
    existing_models = [f for f in existing_files if f.endswith('.pkl')]
    has_models = any(f for f in existing_models if not f.startswith('scaler'))
    has_scaler = any(f for f in existing_models if f.startswith('scaler'))
    found_models = []
    for f in existing_models:
        if not ('scaler' in f.lower()):
            technical_name = os.path.splitext(f)[0]
            display_name = get_model_display_name(technical_name)
            found_models.append({
                'technical_name': technical_name,
                'display_name': display_name
            })
    return {
        "models_available": has_models,
        "scaler_available": has_scaler,
        "found_models": found_models,
        "technical_names": [model['technical_name'] for model in found_models]
    }

def load_model_display_names() -> Dict[str, str]:
    """Load model display names from config file"""
    if os.path.exists(MODEL_DISPLAY_NAMES_FILE):
        try:
            with open(MODEL_DISPLAY_NAMES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error loading model display names: {e}")
    else:
        logging.warning("Model display names file not found. Using empty mapping.")
    return {}

MODEL_DISPLAY_NAMES = load_model_display_names()

def get_model_display_name(model_filename: str) -> str:
    """Get human-readable display name for a model"""
    if model_filename in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_filename]
    
    # Try to extract algorithm name from PyCaret naming convention
    parts = model_filename.split("_")
    if len(parts) >= 4 and model_filename.startswith("top_model_"):
        algorithm_name = "_".join(parts[3:])
        readable_name = ""
        for i, char in enumerate(algorithm_name):
            if char.isupper() and i > 0:
                readable_name += " "
            readable_name += char
        if not readable_name.endswith("Regressor") and not readable_name.endswith("Regression"):
            if "Regression" not in readable_name:
                readable_name += " Regressor"
        return readable_name
    return " ".join(word.capitalize() for word in model_filename.split("_"))

def get_expected_feature_names() -> List[str]:
    """
    Get expected feature names from file or configuration.
    Prioritizes the configuration-based approach.
    """
    # First try to get from configuration
    config_features = get_expected_feature_names_from_config()
    if config_features:
        return config_features
    
    # Fallback to file-based approach
    if os.path.exists(FEATURE_COLS_FILE):
        with open(FEATURE_COLS_FILE, "r") as f:
            file_features = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(file_features)} feature names from file")
        return file_features
    else:
        logging.warning(f"Expected features file not found at {FEATURE_COLS_FILE}")
        return []

def ensure_models_folder():
    """Ensure the models folder directory exists."""
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        logging.info(f"Created models folder at '{MODELS_FOLDER}'")

def check_pycaret_safely():
    """Safely check if PyCaret is available and functional"""
    try:
        from pycaret.regression import predict_model
        return True, predict_model
    except ImportError:
        logging.warning("PyCaret not available")
        return False, None
    except Exception as e:
        logging.warning(f"PyCaret import error: {str(e)}")
        return False, None

# Use this at the module level
PYCARET_AVAILABLE, predict_model = check_pycaret_safely()

def pycaret_load_model(model_path):
    """
    Safely load a PyCaret model using the correct PyCaret function if available.
    Accepts either a path with or without extension.
    """
    if not PYCARET_AVAILABLE:
        raise ImportError("PyCaret is not installed or not available.")
    from pycaret.regression import load_model as pc_load_model
    # Remove .pkl extension if present for PyCaret compatibility
    model_name = model_path
    if model_name.endswith('.pkl'):
        model_name = model_name[:-4]
    return pc_load_model(model_name)

def prepare_features_for_pycaret(features, model=None):
    """
    Converts input features into a pandas DataFrame with correct columns/order.
    Now uses preprocessing pipeline for feature transformation.

    Args:
        features: numpy array, list-like, or dict
        model: optional, can provide column names if present
    Returns:
        pd.DataFrame with proper columns/order
    """
    if isinstance(features, pd.DataFrame):
        return features
    
    # Handle dict input (from UI) - use preprocessing pipeline
    if isinstance(features, dict):
        try:
            # Import here to avoid circular imports
            from pipeline import preprocess_for_prediction
            
            # Use preprocessing pipeline to transform features
            features_df = preprocess_for_prediction(features)
            logging.info(f"Used preprocessing pipeline to transform features: {features_df.shape}")
            return features_df
        except Exception as e:
            logging.warning(f"Preprocessing pipeline failed, using fallback: {e}")
            # Fallback to manual conversion
            expected_columns = get_expected_feature_names_from_model(model)
            feature_vector = create_feature_vector_from_dict(features, expected_columns)
            features_df = pd.DataFrame([feature_vector], columns=expected_columns)
            return features_df
    
    # Handle array/list input - convert to DataFrame
    expected_columns = get_expected_feature_names_from_model(model)
    features = np.array(features).reshape(1, -1)  # Ensure 2D
    
    # Ensure we have the right number of features
    if features.shape[1] != len(expected_columns):
        logging.warning(f"Feature count mismatch: got {features.shape[1]}, expected {len(expected_columns)}")
        # Pad with zeros or truncate as needed
        if features.shape[1] < len(expected_columns):
            padding = np.zeros((1, len(expected_columns) - features.shape[1]))
            features = np.hstack([features, padding])
        else:
            features = features[:, :len(expected_columns)]
    
    features_df = pd.DataFrame(features, columns=expected_columns)
    return features_df

def load_model(model_name: str) -> Optional[Any]:
    """
    Load a model from the models directory, supporting both PyCaret and pickle formats.
    
    Args:
        model_name (str): Name of the model to load (without extension)
        
    Returns:
        Optional[Any]: Loaded model object or None if not found/error
    """
    model_path = os.path.join(MODELS_FOLDER, model_name)
    model_path_with_ext = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    
    # First, try PyCaret's load_model (without extension)
    if PYCARET_AVAILABLE:
        try:
            model = pycaret_load_model(model_path)
            logging.info(f"Successfully loaded PyCaret model: {model_name}")
            logging.info(f"Model type: {type(model)}")
            return model
        except Exception as e:
            logging.warning(f"PyCaret load failed for '{model_name}': {str(e)}")
    
    # Fall back to pickle loading (with extension)
    if os.path.exists(model_path_with_ext):
        try:
            with open(model_path_with_ext, 'rb') as f:
                model = pickle.load(f)
            
            logging.info(f"Successfully loaded pickle model: {model_name}")
            logging.info(f"Loaded object type: {type(model)}")
            
            # For PyCaret models saved with pickle, they might be wrapped
            if hasattr(model, 'predict') or hasattr(model, '_predict'):
                return model
            else:
                logging.error(f"Loaded object '{model_name}' does not have prediction capability")
                return None
                
        except Exception as e:
            logging.error(f"Error loading model '{model_name}' with pickle: {str(e)}")
    
    logging.error(f"Model file '{model_name}' not found in either format")
    return None

def load_scaler() -> Optional[Any]:
    """
    Load the scaler if available in the models directory.
    
    Returns:
        Optional[Any]: Loaded scaler object or None if not found/error
    """
    ensure_models_folder()
    scaler_files = [f for f in os.listdir(MODELS_FOLDER) 
                   if 'scaler' in f.lower() and f.endswith('.pkl')]
    
    if not scaler_files:
        logging.info("No scaler found. Proceeding without scaling.")
        return None
    
    # Use the first scaler found
    scaler_path = os.path.join(MODELS_FOLDER, scaler_files[0])
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        logging.info(f"Successfully loaded scaler: {scaler_files[0]}")
        logging.info(f"Scaler type: {type(scaler)}")
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler '{scaler_files[0]}': {str(e)}")
        return None

def get_expected_feature_names_from_model(model=None) -> List[str]:
    """Get expected feature names from model or configuration"""
    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'X') and hasattr(model.X, 'columns'):
            return list(model.X.columns)
        elif hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
    
    # Fallback to configuration-based feature names
    return get_expected_feature_names()

def align_features_to_model(features_df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Ensures the features_df has all columns (in correct order) that the model expects.
    Missing columns are filled with 0.
    """
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    return features_df[expected_columns]

def predict_man_hours(
    features: Union[np.ndarray, Dict, List], 
    model_name: str, 
    use_scaler: bool = True,
    use_preprocessing_pipeline: bool = True
) -> Optional[float]:
    """
    Make a prediction using the specified model and features, with preprocessing pipeline support.
    
    Args:
        features: Array/dict/list of input features
        model_name (str): Name of the model to use (technical name)
        use_scaler (bool): Whether to apply scaling if available
        use_preprocessing_pipeline (bool): Whether to use preprocessing pipeline
        
    Returns:
        Optional[float]: Predicted man-hours value or None if error
    """
    try:
        # Debug logging
        logging.info(f"Starting prediction with model '{model_name}'")
        logging.info(f"Input features type: {type(features)}")
        
        # Load the model
        model = load_model(model_name)
        if model is None:
            logging.error(f"Failed to load model '{model_name}' for prediction")
            return None
        
        logging.info(f"Loaded model type: {type(model)}")

        # Preprocessing with pipeline integration
        if use_preprocessing_pipeline and isinstance(features, dict):
            try:
                from pipeline import preprocess_for_prediction, load_preprocessing_pipeline, validate_pipeline_compatibility

                # Use preprocessing pipeline for feature transformation
                features_df = preprocess_for_prediction(features)
                logging.info(f"Used preprocessing pipeline: DataFrame shape {features_df.shape}")
                
                # Validate pipeline compatibility
                pipeline = load_preprocessing_pipeline()
                if pipeline:
                    validation = validate_pipeline_compatibility(pipeline, features)
                    if not validation.get('compatible', False):
                        logging.warning(f"Pipeline compatibility issues: {validation}")
                
            except Exception as e:
                logging.warning(f"Preprocessing pipeline failed: {e}")
                # Fallback to manual preparation
                features_df = prepare_features_for_pycaret(features, model=model)
        else:
            # Standard feature preparation
            features_df = prepare_features_for_pycaret(features, model=model)
            
        logging.info(f"Features DataFrame columns: {list(features_df.columns)}")
        logging.info(f"Features DataFrame shape: {features_df.shape}")
        
        # Validate features against configuration
        if isinstance(features, dict):
            validation = validate_feature_dict_against_config(features)
            if not validation['valid']:
                logging.warning(f"Feature validation issues: {validation}")
        
        # Load scaler if requested and available (only if not using preprocessing pipeline)
        if use_scaler and not use_preprocessing_pipeline:
            scaler = load_scaler()
            if scaler is not None and hasattr(scaler, 'transform'):
                try:
                    features_scaled = scaler.transform(features_df)
                    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
                    logging.info("Applied scaler to features.")
                except Exception as e:
                    logging.warning(f"Scaler could not be applied: {str(e)}")
        else:
            if use_preprocessing_pipeline:
                logging.info("Skipping separate scaling - preprocessing pipeline handles normalization")
            else:
                logging.info("No scaling applied or scaler not available.")
        
        # Try PyCaret prediction first
        if PYCARET_AVAILABLE:
            try:
                logging.info("Attempting PyCaret prediction...")
                predictions = predict_model(model, data=features_df)
                possible_columns = ['prediction_label', 'Label', 'pred', 'prediction', 'target']
                prediction_value = None
                for col in possible_columns:
                    if col in predictions.columns:
                        prediction_value = float(predictions[col].iloc[0])
                        logging.info(f"Found prediction in column '{col}': {prediction_value}")
                        break
                if prediction_value is None:
                    prediction_value = float(predictions.iloc[0, -1])
                    logging.info(f"Used last column for prediction: {prediction_value}")
                if np.isnan(prediction_value) or np.isinf(prediction_value):
                    logging.error(f"Invalid prediction value: {prediction_value}")
                    return None
                return max(0.1, prediction_value)
            except Exception as e:
                logging.warning(f"PyCaret prediction failed: {str(e)}")
                import traceback
                logging.warning(traceback.format_exc())

        # Fallback: standard sklearn prediction
        if hasattr(model, 'predict'):
            try:
                prediction = model.predict(features_df)
                if isinstance(prediction, np.ndarray):
                    prediction_value = float(prediction.flat[0])
                elif isinstance(prediction, (list, tuple)):
                    prediction_value = float(prediction[0])
                else:
                    prediction_value = float(prediction)
                if np.isnan(prediction_value) or np.isinf(prediction_value):
                    logging.error(f"Invalid prediction value: {prediction_value}")
                    return None
                return max(0.1, prediction_value)
            except Exception as e:
                logging.error(f"Standard prediction failed: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())

        logging.error(f"No valid prediction method found for model '{model_name}'")
        return None

    except Exception as e:
        logging.error(f"Error making prediction with model '{model_name}': {str(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return None

def get_feature_importance(model_name: str) -> Optional[np.ndarray]:
    """
    Get feature importance for a given model if available.
    
    Args:
        model_name (str): Name of the model to analyze (technical name)
        
    Returns:
        Optional[np.ndarray]: Feature importance values or None if not available
    """
    model = load_model(model_name)
    
    if model is None:
        return None
    
    try:
        # For PyCaret models, try to access the underlying estimator
        if hasattr(model, 'named_steps'):
            # Pipeline case
            estimator = None
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    estimator = step
                    break
        elif hasattr(model, '_final_estimator'):
            estimator = model._final_estimator
        else:
            estimator = model
        
        # Check for feature importances
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            logging.info(f"Retrieved feature importance from '{model_name}' (feature_importances_)")
            return importances
        
        # For linear models, use coefficients
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_)
            if importances.ndim > 1:
                importances = importances.flatten()
            logging.info(f"Retrieved feature importance from '{model_name}' (coefficients)")
            return importances
        
        logging.warning(f"Model '{model_name}' does not support feature importance")
        return None
        
    except Exception as e:
        logging.error(f"Error getting feature importance for '{model_name}': {str(e)}")
        return None

def analyze_what_if(
    base_features: Union[np.ndarray, Dict],
    model_name: str,
    param_name: str,
    param_values: List[float],
    use_scaler: bool = True,
    use_preprocessing_pipeline: bool = True
) -> Dict[str, List]:
    """
    Perform what-if analysis by varying one parameter and observing predictions.
    
    Args:
        base_features: Base feature values (array or dict)
        model_name (str): Name of the model to use (technical name)
        param_name (str): Name of the parameter to vary
        param_values (List[float]): Values to use for the parameter
        use_scaler (bool): Whether to apply scaling if available
        use_preprocessing_pipeline (bool): Whether to use preprocessing pipeline
        
    Returns:
        Dict[str, List]: Dictionary with parameter values and predictions
    """
    results = {
        "param_values": [],
        "predictions": []
    }
    
    # Convert to dict if needed for easier manipulation
    if isinstance(base_features, np.ndarray):
        expected_features = get_expected_feature_names()
        base_dict = {name: float(value) for name, value in zip(expected_features, base_features)}
    else:
        base_dict = base_features.copy()
    
    for value in param_values:
        try:
            # Create a copy of features and modify the specified parameter
            modified_features = base_dict.copy()
            modified_features[param_name] = value
            
            # Make prediction using the main prediction function
            prediction = predict_man_hours(
                modified_features, 
                model_name, 
                use_scaler, 
                use_preprocessing_pipeline
            )
            
            if prediction is not None:
                results["param_values"].append(value)
                results["predictions"].append(prediction)
            
        except Exception as e:
            logging.error(f"Error in what-if analysis for value {value}: {str(e)}")
    
    logging.info(f"What-if analysis completed for parameter {param_name} with {len(results['predictions'])} data points")
    return results

def get_feature_statistics() -> Dict[str, Any]:
    """
    Get statistics about the feature configuration for debugging/info purposes.
    """
    stats = {
        "total_features": len(get_expected_feature_names()),
        "numeric_features": len(FEATURE_CONFIG.get("numeric_features", [])),
        "categorical_features": len(FEATURE_CONFIG.get("categorical_features", {})),
        "one_hot_groups": len(FEATURE_CONFIG.get("one_hot_features", {})),
        "special_cases": len(FEATURE_CONFIG.get("special_cases", {})),
        "binary_features": len(FEATURE_CONFIG.get("binary_features", {}))
    }
    
    # Count actual one-hot features
    one_hot_count = 0
    for group_config in FEATURE_CONFIG.get("one_hot_features", {}).values():
        one_hot_count += len(group_config.get("mapping", {}))
    stats["one_hot_features_total"] = one_hot_count
    
    # Count special case features
    special_count = 0
    for group_config in FEATURE_CONFIG.get("special_cases", {}).values():
        special_count += len(group_config.get("output_keys", {}))
    stats["special_case_features_total"] = special_count
    
    # Count binary features
    binary_count = 0
    for group_config in FEATURE_CONFIG.get("binary_features", {}).values():
        binary_count += len(group_config.get("mapping", {}))
    stats["binary_features_total"] = binary_count
    
    # Add preprocessing pipeline info
    try:
        from pipeline import load_preprocessing_pipeline
        pipeline = load_preprocessing_pipeline()
        if pipeline:
            stats["preprocessing_pipeline_available"] = True
            stats["pipeline_steps"] = len(pipeline.steps)
        else:
            stats["preprocessing_pipeline_available"] = False
            stats["pipeline_steps"] = 0
    except ImportError:
        stats["preprocessing_pipeline_available"] = False
        stats["pipeline_steps"] = 0
    
    return stats

# === New utility functions for pipeline integration ===
def check_preprocessing_pipeline_compatibility() -> Dict[str, Any]:
    """Check if preprocessing pipeline is compatible with current configuration"""
    pipeline = load_preprocessing_pipeline()
    if not pipeline:
        return {
            "compatible": False,
            "error": "No preprocessing pipeline found",
            "recommendations": ["Train a new model with preprocessing pipeline", "Save preprocessing pipeline after training"]
        }
    
    try:
        from pipeline import load_preprocessing_pipeline, validate_pipeline_compatibility
        
        pipeline = load_preprocessing_pipeline()
        if not pipeline:
            return {
                "compatible": False,
                "error": "No preprocessing pipeline found",
                "recommendations": ["Train a new model with preprocessing pipeline", "Save preprocessing pipeline after training"]
            }
        
        # Create dummy features based on current config
        dummy_features = {}
        
        # Add numeric features
        for feature in FEATURE_CONFIG.get("numeric_features", []):
            dummy_features[feature] = 1.0
        
        # Add categorical features  
        for feature, config in FEATURE_CONFIG.get("categorical_features", {}).items():
            options = config.get("options", [])
            dummy_features[feature] = options[0] if options else "default"
        
        # Add one-hot features
        for group_name, group_config in FEATURE_CONFIG.get("one_hot_features", {}).items():
            input_key = group_config.get("input_key")
            mapping = group_config.get("mapping", {})
            if mapping:
                # Set first option as active
                first_key = list(mapping.keys())[0]
                dummy_features[input_key] = first_key
                # Set one-hot encoded features
                for label, feature_key in mapping.items():
                    dummy_features[feature_key] = 1 if label == first_key else 0
        
        # Test preprocessing
        validation = validate_pipeline_compatibility(pipeline, dummy_features)
        
        return {
            "compatible": validation.get("compatible", False),
            "validation_details": validation,
            "recommendations": [
                "Pipeline appears compatible" if validation.get("compatible") else "Pipeline needs retraining",
                f"Expected {validation.get('expected_features', 0)} features, got {validation.get('actual_features', 0)}"
            ]
        }
    except ImportError:
        return {
            "compatible": False,
            "error": "Pipeline module not available",
            "recommendations": ["Install pipeline dependencies", "Check pipeline.py file"]
        }
        
    except Exception as e:
        return {
            "compatible": False,
            "error": str(e),
            "recommendations": ["Check preprocessing pipeline configuration", "Retrain model with current feature configuration"]
        }

def get_preprocessing_pipeline_info() -> Dict[str, Any]:
    """Get information about the current preprocessing pipeline"""
    pipeline = load_preprocessing_pipeline()
    if not pipeline:
        return {
            "available": False,
            "message": "No preprocessing pipeline found"
        }
    
    try:
        from pipeline import load_preprocessing_pipeline
        
        pipeline = load_preprocessing_pipeline()
        if not pipeline:
            return {
                "available": False,
                "message": "No preprocessing pipeline found"
            }
        
        pipeline_info = {
            "available": True,
            "steps": [step[0] for step in pipeline.steps],
            "step_count": len(pipeline.steps),
            "pipeline_type": str(type(pipeline)),
        }
        
        # Try to get pipeline parameters for each step
        step_details = {}
        for step_name, step_obj in pipeline.named_steps.items():
            step_details[step_name] = {
                "class": step_obj.__class__.__name__,
                "parameters": step_obj.get_params() if hasattr(step_obj, 'get_params') else {}
            }
        
        pipeline_info["step_details"] = step_details
        
        return pipeline_info

    except ImportError:
        return {
            "available": False,
            "error": "Pipeline module not available"
        }
        
    except Exception as e:
        return {
            "available": True,
            "error": str(e),
            "message": "Pipeline found but error getting details"
        }

def test_feature_pipeline_integration(feature_dict: Dict) -> Dict[str, Any]:
    """Test the complete pipeline integration with given features"""
    results = {
        "success": False,
        "steps_completed": [],
        "errors": [],
        "warnings": [],
        "final_shape": None,
        "processing_time": None
    }
    
    import time
    start_time = time.time()
    
    try:
        # Step 1: Validate input
        results["steps_completed"].append("input_validation")
        if not isinstance(feature_dict, dict):
            results["errors"].append("Input must be a dictionary")
            return results
        
        # Step 2: Check configuration
        results["steps_completed"].append("configuration_check")
        validation = validate_feature_dict_against_config(feature_dict)
        if not validation["valid"]:
            results["warnings"].append(f"Missing features: {validation['missing_features']}")
        
        # Step 3: Convert to DataFrame
        results["steps_completed"].append("dataframe_conversion")
        try:
            from pipeline import convert_feature_dict_to_dataframe
            df = convert_feature_dict_to_dataframe(feature_dict, FEATURE_CONFIG)
        except ImportError:
            results["warnings"].append("Pipeline module not available for DataFrame conversion")
        
        # Step 4: Preprocessing pipeline
        results["steps_completed"].append("preprocessing")
        try:
            from pipeline import preprocess_for_prediction
            processed_df = preprocess_for_prediction(feature_dict)
            results["final_shape"] = processed_df.shape
        except ImportError:
            results["warnings"].append("Pipeline module not available for preprocessing")
        
        # Step 5: Pipeline compatibility check
        results["steps_completed"].append("compatibility_check")
        try:
            from pipeline import load_preprocessing_pipeline, validate_pipeline_compatibility
            pipeline = load_preprocessing_pipeline()
            if pipeline:
                compat = validate_pipeline_compatibility(pipeline, feature_dict)
                if not compat.get("compatible", False):
                    results["warnings"].append("Pipeline compatibility issues detected")
        except ImportError:
            results["warnings"].append("Pipeline module not available for compatibility check")
        
        results["success"] = True
        results["processing_time"] = time.time() - start_time
        
    except Exception as e:
        results["errors"].append(str(e))
        results["processing_time"] = time.time() - start_time
    
    return results

# === Model training integration utilities ===
def prepare_training_data_with_pipeline(
    df: pd.DataFrame,
    target_col: str = 'project_prf_normalised_work_effort',
    save_pipeline: bool = True,
    **pipeline_kwargs
) -> Tuple[pd.DataFrame, Any, Dict]:
    """
    Prepare training data using preprocessing pipeline and optionally save it.
    
    Args:
        df: Raw training DataFrame
        target_col: Name of target column
        save_pipeline: Whether to save the fitted pipeline
        **pipeline_kwargs: Additional arguments for pipeline creation
        
    Returns:
        Tuple of (processed_df, fitted_pipeline, metadata)
    """
    try:
        # Import here to avoid circular imports
        from pipeline import preprocess_dataframe, save_preprocessing_pipeline, create_preprocessing_pipeline
        
        # Preprocess the data
        processed_df, metadata = preprocess_dataframe(
            df, target_col=target_col, **pipeline_kwargs
        )
        
        # Get the fitted pipeline from preprocessing

        pipeline = create_preprocessing_pipeline(target_col=target_col, **pipeline_kwargs)
        fitted_pipeline = pipeline.fit(df)
        
        if save_pipeline:
            save_preprocessing_pipeline(fitted_pipeline)
            logging.info("Preprocessing pipeline saved successfully")
        
        return processed_df, fitted_pipeline, metadata
        
    except Exception as e:
        logging.error(f"Error in training data preparation: {e}")
        raise

def save_model_with_metadata(
    model: Any,
    model_name: str,
    feature_names: List[str],
    model_metadata: Optional[Dict] = None
) -> bool:
    """
    Save model along with feature names and metadata.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        feature_names: List of feature names expected by model
        model_metadata: Additional metadata about the model
        
    Returns:
        bool: Success status
    """
    try:
        ensure_models_folder()
        
        # Save the model
        model_path = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names
        feature_names_path = os.path.join(DATA_FOLDER, 'expected_features.txt')
        os.makedirs(DATA_FOLDER, exist_ok=True)
        with open(feature_names_path, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        # Save model metadata
        if model_metadata:
            metadata_path = os.path.join(MODELS_FOLDER, f'{model_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
        
        logging.info(f"Model '{model_name}' saved successfully with {len(feature_names)} features")
        return True
        
    except Exception as e:
        logging.error(f"Error saving model '{model_name}': {e}")
        return False

def load_model_metadata(model_name: str) -> Optional[Dict]:
    """Load metadata for a specific model"""
    try:
        metadata_path = os.path.join(MODELS_FOLDER, f'{model_name}_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logging.error(f"Error loading metadata for model '{model_name}': {e}")
        return None

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get comprehensive information about a specific model"""
    info = {
        "model_name": model_name,
        "display_name": get_model_display_name(model_name),
        "model_available": False,
        "scaler_available": False,
        "metadata_available": False,
        "feature_importance_available": False,
        "expected_features": 0,
        "model_type": None,
        "metadata": None
    }
    
    try:
        # Check if model exists and load it
        model = load_model(model_name)
        if model:
            info["model_available"] = True
            info["model_type"] = str(type(model))
            
            # Check feature importance
            feature_importance = get_feature_importance(model_name)
            info["feature_importance_available"] = feature_importance is not None
            
            # Get expected features
            expected_features = get_expected_feature_names_from_model(model)
            info["expected_features"] = len(expected_features)
        
        # Check for scaler
        scaler = load_scaler()
        info["scaler_available"] = scaler is not None
        
        # Load metadata
        metadata = load_model_metadata(model_name)
        if metadata:
            info["metadata_available"] = True
            info["metadata"] = metadata
            
    except Exception as e:
        info["error"] = str(e)
        logging.error(f"Error getting model info for '{model_name}': {e}")
    
    return info

def validate_model_setup() -> Dict[str, Any]:
    """Validate the complete model setup including pipeline compatibility"""
    validation = {
        "overall_status": "unknown",
        "models_found": 0,
        "pipeline_available": False,
        "configuration_valid": False,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Check available models
        models = list_available_models()
        validation["models_found"] = len(models)
        
        if len(models) == 0:
            validation["issues"].append("No trained models found")
            validation["recommendations"].append("Train and save at least one model")
        
        # Check preprocessing pipeline
        pipeline_info = get_preprocessing_pipeline_info()
        validation["pipeline_available"] = pipeline_info.get("available", False)
        
        if not validation["pipeline_available"]:
            validation["issues"].append("No preprocessing pipeline found")
            validation["recommendations"].append("Save preprocessing pipeline during model training")
        
        # Check configuration
        try:
            feature_stats = get_feature_statistics()
            validation["configuration_valid"] = feature_stats["total_features"] > 0
            validation["feature_stats"] = feature_stats
        except Exception as e:
            validation["issues"].append(f"Configuration validation failed: {e}")
            validation["recommendations"].append("Check feature mapping configuration file")
        
        # Check pipeline compatibility if available
        if validation["pipeline_available"]:
            compat_info = check_preprocessing_pipeline_compatibility()
            if not compat_info.get("compatible", False):
                validation["issues"].append("Preprocessing pipeline compatibility issues")
                validation["recommendations"].extend(compat_info.get("recommendations", []))
        
        # Determine overall status
        if len(validation["issues"]) == 0:
            validation["overall_status"] = "healthy"
        elif validation["models_found"] > 0:
            validation["overall_status"] = "functional_with_issues"
        else:
            validation["overall_status"] = "needs_setup"
            
    except Exception as e:
        validation["overall_status"] = "error"
        validation["issues"].append(f"Validation failed: {e}")
    
    return validation

# === Backwards compatibility functions ===
def prepare_feature_vector(user_inputs, model=None):
    """
    Backwards compatibility function - redirects to new preprocessing approach
    """
    logging.warning("prepare_feature_vector is deprecated. Use prepare_features_for_pycaret instead.")
    return prepare_features_for_pycaret(user_inputs, model)

# === Export list for easier imports ===
__all__ = [
    # Model management
    'list_available_models',
    'check_required_models', 
    'load_model',
    'get_model_display_name',
    
    # Prediction functions
    'predict_man_hours',
    'analyze_what_if',
    
    # Feature handling
    'get_expected_feature_names',
    'get_expected_feature_names_from_model',
    'prepare_features_for_pycaret',
    'create_feature_vector_from_dict',
    'validate_feature_dict_against_config',
    
    # Feature importance
    'get_feature_importance',
    
    # Pipeline integration
    'check_preprocessing_pipeline_compatibility',
    'get_preprocessing_pipeline_info',
    'test_feature_pipeline_integration',
    
    # Statistics and validation
    'get_feature_statistics',
    'validate_model_setup',
    'get_model_info',
    
    # Training utilities
    'prepare_training_data_with_pipeline',
    'save_model_with_metadata',
    'load_model_metadata'
]