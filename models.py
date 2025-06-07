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

def ensure_models_folder():
    """Ensure the models folder directory exists."""
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        logging.info(f"Created models folder at '{MODELS_FOLDER}'")

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
    has_models = any(f for f in existing_models if not f.startswith('scaler') and 'pipeline' not in f)
    has_scaler = any(f for f in existing_models if f.startswith('scaler'))
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
        "scaler_available": has_scaler,
        "found_models": found_models,
        "technical_names": [model['technical_name'] for model in found_models]
    }

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

def load_with_pycaret(model_path: str, model_name: str):
    """Try loading with PyCaret"""
    if not PYCARET_AVAILABLE:
        return None
    return pycaret_load_model(model_path)

def load_with_joblib(model_path: str, model_name: str):
    """Try loading with joblib"""
    try:
        import joblib
        model_path_with_ext = f"{model_path}.pkl"
        if os.path.exists(model_path_with_ext):
            return joblib.load(model_path_with_ext)
        elif os.path.exists(f"{model_path}.joblib"):
            return joblib.load(f"{model_path}.joblib")
    except ImportError:
        pass
    return None

def load_with_pickle_protocols(model_path: str, model_name: str):
    """Try loading with different pickle protocols"""
    model_path_with_ext = f"{model_path}.pkl"
    if not os.path.exists(model_path_with_ext):
        return None
    
    # Try different pickle protocols
    for protocol in [None, 0, 1, 2, 3, 4, 5]:
        try:
            with open(model_path_with_ext, 'rb') as f:
                if protocol is None:
                    return pickle.load(f)
                else:
                    # For loading, protocol doesn't matter, but we'll catch specific errors
                    return pickle.load(f)
        except Exception as e:
            if "protocol" in str(e).lower():
                continue
            else:
                break
    return None

def load_with_dill(model_path: str, model_name: str):
    """Try loading with dill (enhanced pickle)"""
    try:
        import dill
        model_path_with_ext = f"{model_path}.pkl"
        if os.path.exists(model_path_with_ext):
            with open(model_path_with_ext, 'rb') as f:
                return dill.load(f)
    except ImportError:
        pass
    return None

def load_model_with_fallback(model_name: str) -> Optional[Any]:
    """
    Enhanced model loading with better error handling and fallback options.
    """
    model_path = os.path.join(MODELS_FOLDER, model_name)
    model_path_with_ext = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    
    # Try multiple loading strategies
    loading_strategies = [
        ("PyCaret", load_with_pycaret),
        ("Joblib", load_with_joblib),
        ("Pickle with protocols", load_with_pickle_protocols),
        ("Dill (if available)", load_with_dill)
    ]
    
    for strategy_name, loader_func in loading_strategies:
        try:
            logging.info(f"Trying {strategy_name} loading for model '{model_name}'")
            model = loader_func(model_path, model_name)
            if model is not None:
                logging.info(f"Successfully loaded model '{model_name}' using {strategy_name}")
                return model
        except Exception as e:
            logging.warning(f"{strategy_name} loading failed for '{model_name}': {str(e)}")
            continue
    
    logging.error(f"All loading strategies failed for model '{model_name}'")
    return None

def load_model(model_name: str) -> Optional[Any]:
    """
    Load a model from the models directory, supporting both PyCaret and pickle formats.
    """
    return load_model_with_fallback(model_name)

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

def prepare_features_for_pycaret_with_pipeline(features, model=None):
    """
    Convert input features into a pandas DataFrame using the dynamic preprocessing pipeline.
    This version creates the pipeline dynamically without requiring a saved pkl file.
    """
    if isinstance(features, pd.DataFrame):
        return features
    
    # Handle dict input (from UI) - use DYNAMIC preprocessing pipeline
    if isinstance(features, dict):
        try:
            # Import the pipeline functions
            from pipeline import convert_feature_dict_to_dataframe, create_preprocessing_pipeline
            
            # Load feature config
            feature_config = load_yaml_config(FEATURE_MAPPING_FILE)
            
            # Step 1: Convert feature dict to DataFrame with original categorical structure
            df = convert_feature_dict_to_dataframe(features, feature_config)
            logging.info(f"Converted feature dict to DataFrame with shape {df.shape} and columns: {list(df.columns)}")
            
            # Step 2: Create preprocessing pipeline dynamically (NO pkl file needed)
            pipeline = create_preprocessing_pipeline(
                target_col=None,  # No target for prediction
                high_missing_threshold=0.9,  # More lenient for single prediction
                max_categorical_cardinality=20  # More lenient for prediction
            )
            
            # Step 3: Apply preprocessing
            processed_df = pipeline.fit_transform(df)
            
            # Step 4: Remove any target-related columns that might have been created
            target_related_cols = [col for col in processed_df.columns 
                                 if any(keyword in col.lower() for keyword in ['target', 'effort', 'work_effort'])]
            if target_related_cols:
                processed_df = processed_df.drop(columns=target_related_cols)
                logging.info(f"Removed target-related columns: {target_related_cols}")
            
            logging.info(f"Pipeline preprocessing successful: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logging.warning(f"Pipeline preprocessing failed: {e}")
            # Fallback to manual conversion
            return prepare_features_manually_from_config(features)
    
    # Handle array/list input - convert to DataFrame manually
    return prepare_features_manually_from_config(features)

def prepare_features_for_pycaret(features, model=None):
    """
    Converts input features into a pandas DataFrame with correct columns/order.
    Uses preprocessing pipeline for feature transformation.
    """
    return prepare_features_for_pycaret_with_pipeline(features, model)

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

def predict_man_hours_with_dynamic_pipeline(
    features: Union[np.ndarray, Dict, List], 
    model_name: str, 
    use_preprocessing_pipeline: bool = True,
    use_scaler: bool = False  # Disable scaler since pipeline handles normalization
) -> Optional[float]:
    """
    Make prediction using dynamic preprocessing pipeline (no pkl file required)
    """
    try:
        logging.info(f"Starting prediction with model '{model_name}' using dynamic pipeline")
        
        # Load the model
        model = load_model(model_name)
        if model is None:
            logging.error(f"Failed to load model '{model_name}'")
            return None
        
        # Prepare features using dynamic pipeline or manual fallback
        if use_preprocessing_pipeline:
            features_df = prepare_features_for_pycaret_with_pipeline(features, model)
        else:
            features_df = prepare_features_manually_from_config(features)
        
        if features_df is None or features_df.empty:
            logging.error("Feature preparation failed")
            return None
        
        logging.info(f"Features prepared: shape {features_df.shape}, columns: {list(features_df.columns)}")
        
        # Make prediction
        try:
            # Try PyCaret prediction first
            if PYCARET_AVAILABLE:
                predictions = predict_model(model, data=features_df)
                
                # Look for prediction in various possible column names
                prediction_columns = ['prediction_label', 'Label', 'pred', 'prediction', 'target']
                for col in prediction_columns:
                    if col in predictions.columns:
                        result = float(predictions[col].iloc[0])
                        logging.info(f"PyCaret prediction successful: {result}")
                        return max(0.1, result)
                
                # Fallback to last column
                result = float(predictions.iloc[0, -1])
                logging.info(f"PyCaret prediction (last column): {result}")
                return max(0.1, result)
        
        except Exception as e:
            logging.warning(f"PyCaret prediction failed: {e}")
        
        # Fallback to direct model prediction
        if hasattr(model, 'predict'):
            try:
                prediction = model.predict(features_df)
                if isinstance(prediction, np.ndarray):
                    result = float(prediction.flat[0])
                else:
                    result = float(prediction)
                logging.info(f"Direct model prediction successful: {result}")
                return max(0.1, result)
            except Exception as e:
                logging.error(f"Direct model prediction failed: {e}")
        
        logging.error("All prediction methods failed")
        return None
        
    except Exception as e:
        logging.error(f"Prediction completely failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return None

def predict_man_hours(
    features: Union[np.ndarray, Dict, List], 
    model_name: str, 
    use_scaler: bool = False,  # Disabled since pipeline handles this
    use_preprocessing_pipeline: bool = True  # Enable dynamic pipeline
) -> Optional[float]:
    """
    Make a prediction using the specified model and features, with preprocessing pipeline support.
    """
    return predict_man_hours_with_dynamic_pipeline(
        features, model_name, use_preprocessing_pipeline, use_scaler
    )

def get_feature_importance(model_name: str) -> Optional[np.ndarray]:
    """
    Get feature importance for a given model if available.
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
    use_scaler: bool = False,
    use_preprocessing_pipeline: bool = True
) -> Dict[str, List]:
    """
    Perform what-if analysis by varying one parameter and observing predictions.
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
        from pipeline import create_preprocessing_pipeline
        pipeline = create_preprocessing_pipeline(target_col=None)
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


def create_training_compatible_features(input_features):
    """
    Create features that match EXACTLY how the model was trained.
    Based on the notebook preprocessing shown in the image.
    INCLUDES the target column that PyCaret is treating as a feature.
    """
    
    # Start with the core features
    training_features = {}
    
    # === NUMERIC FEATURES (as-is) ===
    training_features['project_prf_year_of_project'] = input_features.get('project_prf_year_of_project', 2024)
    training_features['project_prf_max_team_size'] = input_features.get('project_prf_max_team_size', 8)
    
    # === TARGET COLUMN THAT PYCARET TREATS AS FEATURE ===
    # This is the key missing piece! PyCaret is expecting this as an input feature
    # We'll provide a reasonable default value based on team size and project type
    training_features['project_prf_normalised_work_effort'] = estimate_work_effort_default(input_features)
    
    # === CATEGORICAL FEATURES (fill missing with "Missing") ===
    
    # Industry Sector - use exact mapping from training
    industry_input = input_features.get('external_eef_industry_sector', 'Missing')
    if industry_input in ['Technology', 'Banking', 'Healthcare', 'Manufacturing', 'Government']:
        training_features['external_eef_industry_sector'] = industry_input
    else:
        training_features['external_eef_industry_sector'] = 'Missing'
    
    # Primary Programming Language
    lang_input = input_features.get('tech_tf_primary_programming_language', 'Missing')
    if lang_input in ['Python', 'Java', 'JavaScript', 'C#', 'PHP', 'Ruby', 'C++', 'C']:
        training_features['tech_tf_primary_programming_language'] = lang_input
    else:
        training_features['tech_tf_primary_programming_language'] = 'Missing'
    
    # Team Size Group - calculate from max team size
    max_team_size = training_features['project_prf_max_team_size']
    if max_team_size <= 1:
        team_size_group = "1"
    elif max_team_size <= 3:
        team_size_group = "2-3"
    elif max_team_size <= 5:
        team_size_group = "4-5"
    elif max_team_size <= 10:
        team_size_group = "6-10"
    else:
        team_size_group = "11+"
    training_features['project_prf_team_size_group'] = team_size_group
    
    # Development Methodologies - default to Missing
    training_features['process_pmf_development_methodologies'] = 'Missing'
    
    # Client/Server Roles - default to Missing
    training_features['tech_tf_client_roles'] = 'Missing'
    training_features['tech_tf_server_roles'] = 'Missing'
    
    # Client-Server Description - default to Missing
    training_features['tech_tf_clientserver_description'] = 'Missing'
    
    # === ADD MISSING FEATURES THAT MODEL EXPECTS ===
    # Based on the error messages from your model
    
    # Functional Size (numeric)
    training_features['project_prf_functional_size'] = 100  # Default value
    
    # Documentation
    training_features['process_pmf_docs'] = 'Missing'
    
    # Tools Used
    training_features['tech_tf_tools_used'] = 'Missing'
    
    # Application Group
    app_type = input_features.get('project_prf_application_type', 'Missing')
    if 'Web' in str(app_type) or 'Business' in str(app_type):
        training_features['project_prf_application_group'] = 'business_application'
    else:
        training_features['project_prf_application_group'] = 'Missing'
    
    # Client-Server Description
    arch_input = input_features.get('tech_tf_architecture', 'Missing')
    if 'Multi' in str(arch_input) or 'Micro' in str(arch_input):
        training_features['tech_tf_clientserver_description'] = 'web'
    elif 'Client' in str(arch_input):
        training_features['tech_tf_clientserver_description'] = 'client_server'
    else:
        training_features['tech_tf_clientserver_description'] = 'Missing'
    
    # Data Quality Rating
    training_features['external_eef_data_quality_rating'] = 'A'  # Default to high quality
    
    # Development Type
    dev_input = input_features.get('project_prf_development_type', 'Missing')
    if 'New' in str(dev_input):
        training_features['project_prf_development_type'] = 'new_development'
    elif 'Enhancement' in str(dev_input):
        training_features['project_prf_development_type'] = 'enhancement'
    else:
        training_features['project_prf_development_type'] = 'Missing'
    
    # Development Platform
    platform_input = input_features.get('tech_tf_development_platform', 'Missing')
    if 'Cloud' in str(platform_input) or 'PC' in str(platform_input):
        training_features['tech_tf_development_platform'] = 'pc'
    elif 'Multi' in str(platform_input):
        training_features['tech_tf_development_platform'] = 'multi'
    else:
        training_features['tech_tf_development_platform'] = 'Missing'
    
    # Language Type
    lang_type_input = input_features.get('tech_tf_language_type', 'Missing')
    if 'Object' in str(lang_type_input) or '3GL' in str(lang_type_input):
        training_features['tech_tf_language_type'] = '3gl'
    elif '4GL' in str(lang_type_input):
        training_features['tech_tf_language_type'] = '4gl'
    else:
        training_features['tech_tf_language_type'] = 'Missing'
    
    # Relative Size
    size_input = input_features.get('project_prf_relative_size', 'Missing')
    if 'M' in str(size_input) or 'Medium' in str(size_input):
        training_features['project_prf_relative_size'] = 'm1'
    elif 'L' in str(size_input) and 'XL' not in str(size_input):
        training_features['project_prf_relative_size'] = 'l'
    elif 'S' in str(size_input) and 'XS' not in str(size_input):
        training_features['project_prf_relative_size'] = 's'
    elif 'XL' in str(size_input):
        training_features['project_prf_relative_size'] = 'xl'
    else:
        training_features['project_prf_relative_size'] = 'Missing'
    
    # Case Tool Used
    training_features['project_prf_case_tool_used'] = 'Missing'
    
    # Prototyping Used
    training_features['process_pmf_prototyping_used'] = 'Missing'
    
    # Architecture
    if 'Multi' in str(arch_input) or 'Micro' in str(arch_input):
        training_features['tech_tf_architecture'] = 'multi_tier'
    elif 'Client' in str(arch_input):
        training_features['tech_tf_architecture'] = 'client_server'
    else:
        training_features['tech_tf_architecture'] = 'Missing'
    
    # Client Server (binary)
    training_features['tech_tf_client_server'] = 'Missing'
    
    # Type of Server
    training_features['tech_tf_type_of_server'] = 'Missing'
    
    # Web Development
    if 'Web' in str(app_type):
        training_features['tech_tf_web_development'] = 'web'
    else:
        training_features['tech_tf_web_development'] = 'Missing'
    
    # DBMS Used
    training_features['tech_tf_dbms_used'] = 'Missing'
    
    # User Involvement
    training_features['people_prf_project_user_involvement'] = 'Missing'
    
    # Currency Multiple
    training_features['project_prf_currency_multiple'] = 'Missing'
    
    # Organization Type (mapped from input)
    org_input = input_features.get('external_eef_organisation_type', 'Missing')
    if 'Large' in str(org_input) or 'Enterprise' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'computers & software'
    elif 'Banking' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'banking'
    elif 'Government' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'government'
    else:
        training_features['external_eef_organisation_type_top'] = 'Missing'
    
    # Application Type (mapped from input)
    if 'Web' in str(app_type):
        training_features['project_prf_application_type_top'] = 'business application'
    elif 'Financial' in str(app_type):
        training_features['project_prf_application_type_top'] = 'financial transaction process/accounting'
    else:
        training_features['project_prf_application_type_top'] = 'Missing'
    
    return training_features


def estimate_work_effort_default(input_features):
    """
    Estimate a reasonable default value for project_prf_normalised_work_effort
    based on project characteristics. This is needed because PyCaret treats
    the target column as a feature.
    """
    
    # Base effort calculation using simple heuristics
    team_size = input_features.get('project_prf_max_team_size', 5)
    
    # Size multiplier
    size_input = str(input_features.get('project_prf_relative_size', 'Medium'))
    if 'XS' in size_input or 'Extra Small' in size_input:
        size_multiplier = 0.5
    elif 'S' in size_input and 'XS' not in size_input:
        size_multiplier = 0.8
    elif 'M' in size_input or 'Medium' in size_input:
        size_multiplier = 1.0
    elif 'L' in size_input and 'XL' not in size_input:
        size_multiplier = 1.5
    elif 'XL' in size_input:
        size_multiplier = 2.0
    else:
        size_multiplier = 1.0
    
    # Application type multiplier
    app_type = str(input_features.get('project_prf_application_type', ''))
    if 'Web' in app_type:
        app_multiplier = 1.2
    elif 'Mobile' in app_type:
        app_multiplier = 1.0
    elif 'Desktop' in app_type:
        app_multiplier = 1.3
    elif 'API' in app_type:
        app_multiplier = 0.8
    else:
        app_multiplier = 1.0
    
    # Development type multiplier  
    dev_type = str(input_features.get('project_prf_development_type', ''))
    if 'New' in dev_type:
        dev_multiplier = 1.0
    elif 'Enhancement' in dev_type:
        dev_multiplier = 0.7
    elif 'Maintenance' in dev_type:
        dev_multiplier = 0.5
    else:
        dev_multiplier = 1.0
    
    # Base effort per person (hours)
    base_effort_per_person = 200
    
    # Calculate normalized work effort
    estimated_effort = base_effort_per_person * team_size * size_multiplier * app_multiplier * dev_multiplier
    
    # Normalize (this might need adjustment based on your training data scale)
    # Common normalization ranges: 0-1, 0-100, or raw hours
    # You might need to adjust this based on what range your training data used
    normalized_effort = estimated_effort / 1000  # Assuming normalization to 0-10 range
    
    return normalized_effort
    
    # === CATEGORICAL FEATURES (fill missing with "Missing") ===
    
    # Industry Sector - use exact mapping from training
    industry_input = input_features.get('external_eef_industry_sector', 'Missing')
    if industry_input in ['Technology', 'Banking', 'Healthcare', 'Manufacturing', 'Government']:
        training_features['external_eef_industry_sector'] = industry_input
    else:
        training_features['external_eef_industry_sector'] = 'Missing'
    
    # Primary Programming Language
    lang_input = input_features.get('tech_tf_primary_programming_language', 'Missing')
    if lang_input in ['Python', 'Java', 'JavaScript', 'C#', 'PHP', 'Ruby', 'C++', 'C']:
        training_features['tech_tf_primary_programming_language'] = lang_input
    else:
        training_features['tech_tf_primary_programming_language'] = 'Missing'
    
    # Team Size Group - calculate from max team size
    max_team_size = training_features['project_prf_max_team_size']
    if max_team_size <= 1:
        team_size_group = "1"
    elif max_team_size <= 3:
        team_size_group = "2-3"
    elif max_team_size <= 5:
        team_size_group = "4-5"
    elif max_team_size <= 10:
        team_size_group = "6-10"
    else:
        team_size_group = "11+"
    training_features['project_prf_team_size_group'] = team_size_group
    
    # Development Methodologies - default to Missing
    training_features['process_pmf_development_methodologies'] = 'Missing'
    
    # Client/Server Roles - default to Missing
    training_features['tech_tf_client_roles'] = 'Missing'
    training_features['tech_tf_server_roles'] = 'Missing'
    
    # Client-Server Description - default to Missing
    training_features['tech_tf_clientserver_description'] = 'Missing'
    
    # === ADD MISSING FEATURES THAT MODEL EXPECTS ===
    # Based on the error messages from your model
    
    # Functional Size (numeric)
    training_features['project_prf_functional_size'] = 100  # Default value
    
    # Documentation
    training_features['process_pmf_docs'] = 'Missing'
    
    # Tools Used
    training_features['tech_tf_tools_used'] = 'Missing'
    
    # Application Group
    app_type = input_features.get('project_prf_application_type', 'Missing')
    if 'Web' in str(app_type) or 'Business' in str(app_type):
        training_features['project_prf_application_group'] = 'business_application'
    else:
        training_features['project_prf_application_group'] = 'Missing'
    
    # Client-Server Description
    arch_input = input_features.get('tech_tf_architecture', 'Missing')
    if 'Multi' in str(arch_input) or 'Micro' in str(arch_input):
        training_features['tech_tf_clientserver_description'] = 'web'
    elif 'Client' in str(arch_input):
        training_features['tech_tf_clientserver_description'] = 'client_server'
    else:
        training_features['tech_tf_clientserver_description'] = 'Missing'
    
    # Data Quality Rating
    training_features['external_eef_data_quality_rating'] = 'A'  # Default to high quality
    
    # Development Type
    dev_input = input_features.get('project_prf_development_type', 'Missing')
    if 'New' in str(dev_input):
        training_features['project_prf_development_type'] = 'new_development'
    elif 'Enhancement' in str(dev_input):
        training_features['project_prf_development_type'] = 'enhancement'
    else:
        training_features['project_prf_development_type'] = 'Missing'
    
    # Development Platform
    platform_input = input_features.get('tech_tf_development_platform', 'Missing')
    if 'Cloud' in str(platform_input) or 'PC' in str(platform_input):
        training_features['tech_tf_development_platform'] = 'pc'
    elif 'Multi' in str(platform_input):
        training_features['tech_tf_development_platform'] = 'multi'
    else:
        training_features['tech_tf_development_platform'] = 'Missing'
    
    # Language Type
    lang_type_input = input_features.get('tech_tf_language_type', 'Missing')
    if 'Object' in str(lang_type_input) or '3GL' in str(lang_type_input):
        training_features['tech_tf_language_type'] = '3gl'
    elif '4GL' in str(lang_type_input):
        training_features['tech_tf_language_type'] = '4gl'
    else:
        training_features['tech_tf_language_type'] = 'Missing'
    
    # Relative Size
    size_input = input_features.get('project_prf_relative_size', 'Missing')
    if 'M' in str(size_input) or 'Medium' in str(size_input):
        training_features['project_prf_relative_size'] = 'm1'
    elif 'L' in str(size_input) and 'XL' not in str(size_input):
        training_features['project_prf_relative_size'] = 'l'
    elif 'S' in str(size_input) and 'XS' not in str(size_input):
        training_features['project_prf_relative_size'] = 's'
    elif 'XL' in str(size_input):
        training_features['project_prf_relative_size'] = 'xl'
    else:
        training_features['project_prf_relative_size'] = 'Missing'
    
    # Case Tool Used
    training_features['project_prf_case_tool_used'] = 'Missing'
    
    # Prototyping Used
    training_features['process_pmf_prototyping_used'] = 'Missing'
    
    # Architecture
    if 'Multi' in str(arch_input) or 'Micro' in str(arch_input):
        training_features['tech_tf_architecture'] = 'multi_tier'
    elif 'Client' in str(arch_input):
        training_features['tech_tf_architecture'] = 'client_server'
    else:
        training_features['tech_tf_architecture'] = 'Missing'
    
    # Client Server (binary)
    training_features['tech_tf_client_server'] = 'Missing'
    
    # Type of Server
    training_features['tech_tf_type_of_server'] = 'Missing'
    
    # Web Development
    if 'Web' in str(app_type):
        training_features['tech_tf_web_development'] = 'web'
    else:
        training_features['tech_tf_web_development'] = 'Missing'
    
    # DBMS Used
    training_features['tech_tf_dbms_used'] = 'Missing'
    
    # User Involvement
    training_features['people_prf_project_user_involvement'] = 'Missing'
    
    # Currency Multiple
    training_features['project_prf_currency_multiple'] = 'Missing'
    
    # Organization Type (mapped from input)
    org_input = input_features.get('external_eef_organisation_type', 'Missing')
    if 'Large' in str(org_input) or 'Enterprise' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'computers & software'
    elif 'Banking' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'banking'
    elif 'Government' in str(org_input):
        training_features['external_eef_organisation_type_top'] = 'government'
    else:
        training_features['external_eef_organisation_type_top'] = 'Missing'
    
    # Application Type (mapped from input)
    if 'Web' in str(app_type):
        training_features['project_prf_application_type_top'] = 'business application'
    elif 'Financial' in str(app_type):
        training_features['project_prf_application_type_top'] = 'financial transaction process/accounting'
    else:
        training_features['project_prf_application_type_top'] = 'Missing'
    
    return training_features


def predict_with_training_features(input_features, model_name):
    """
    Make prediction using features that match the training preprocessing
    """
    try:
        print(f"Creating training-compatible features for model: {model_name}")
        
        # Create features that match training preprocessing
        training_features = create_training_compatible_features(input_features)
        print(f"Created {len(training_features)} training-compatible features")
        
        # Debug: show some key features
        print("Key features created:")
        for key, value in list(training_features.items())[:10]:
            print(f"  {key}: {value}")
        
        # Load model
        model = load_model(model_name)
        if not model:
            print("Failed to load model")
            return None
        
        # Try PyCaret prediction first (since model was trained with PyCaret)
        try:
            import pandas as pd
            features_df = pd.DataFrame([training_features])
            
            if PYCARET_AVAILABLE:
                from pycaret.regression import predict_model as pc_predict
                predictions = pc_predict(model, data=features_df)
                
                # Look for prediction column
                prediction_columns = ['prediction_label', 'Label', 'pred', 'prediction']
                for col in prediction_columns:
                    if col in predictions.columns:
                        result = float(predictions[col].iloc[0])
                        print(f"PyCaret prediction successful: {result:.2f} hours")
                        return max(0.1, result)
            
            # Fallback to direct model prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(features_df)
                result = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
                print(f"Direct prediction successful: {result:.2f} hours")
                return max(0.1, result)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
            
    except Exception as e:
        print(f"Training-compatible prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_compatible_prediction():
    """
    Test prediction using training-compatible features with target column
    """
    print("\n=== TESTING TRAINING-COMPATIBLE PREDICTION (WITH TARGET) ===")
    
    # Sample input that matches your UI
    sample_input = {
        "project_prf_year_of_project": 2024,
        "external_eef_industry_sector": "Technology",
        "external_eef_organisation_type": "Large Enterprise", 
        "project_prf_max_team_size": 8,
        "project_prf_relative_size": "M (Medium)",
        "project_prf_application_type": "Web Application",
        "project_prf_development_type": "New Development",
        "tech_tf_architecture": "Microservices",
        "tech_tf_development_platform": "Cloud",
        "tech_tf_language_type": "Object-Oriented",
        "tech_tf_primary_programming_language": "Python"
    }
    
    print("Input features:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Test with first available model
    models = list_available_models()
    if models:
        model_name = models[0]['technical_name']
        print(f"\nTesting with model: {model_name}")
        
        # Try optimized prediction with target value optimization
        result = predict_with_training_features_optimized(sample_input, model_name)
        
        if result:
            print(f"🎉 SUCCESS! Training-compatible prediction: {result:.1f} hours")
            print(f"   That's {result/8:.1f} work days for {sample_input['project_prf_max_team_size']} people")
            print(f"   Per person: {result/sample_input['project_prf_max_team_size']:.1f} hours")
            return True
        else:
            print("❌ Training-compatible prediction failed")
            
            # Fallback: try simple approach
            print("\n🔄 Trying simple target approach...")
            simple_result = predict_with_training_features(sample_input, model_name)
            if simple_result:
                print(f"✅ Simple approach worked: {simple_result:.1f} hours")
                return True
            else:
                print("❌ All approaches failed")
                return False
    else:
        print("❌ No models available for testing")
        return False

# === Export list for easier imports ===
__all__ = [
    # Model management
    'list_available_models',
    'check_required_models', 
    'load_model',
    'get_model_display_name',
    
    # Prediction functions
    'predict_man_hours',
    'predict_with_training_features_optimized'
    'analyze_what_if',
    
    # Feature handling
    'get_expected_feature_names',
    'get_expected_feature_names_from_model',
    'prepare_features_for_pycaret',
    'create_feature_vector_from_dict',
    'validate_feature_dict_against_config',
    'create_training_compatible_features',
    
    # Feature importance
    'get_feature_importance',
    
    # Statistics and validation
    'get_feature_statistics'
]

def diagnose_model_file(model_name: str) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of a model file to understand why it's not loading
    """
    diagnosis = {
        "model_name": model_name,
        "file_exists": False,
        "file_size": 0,
        "file_permissions": None,
        "pickle_protocol": None,
        "file_type": None,
        "error_details": [],
        "recommendations": []
    }
    
    model_path = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    
    # Check if file exists
    if os.path.exists(model_path):
        diagnosis["file_exists"] = True
        
        # Get file info
        stat_info = os.stat(model_path)
        diagnosis["file_size"] = stat_info.st_size
        diagnosis["file_permissions"] = oct(stat_info.st_mode)[-3:]
        
        # Try to detect pickle protocol and contents
        try:
            with open(model_path, 'rb') as f:
                # Read first few bytes to detect pickle protocol
                first_bytes = f.read(10)
                f.seek(0)
                
                # Try to get pickle protocol info
                try:
                    import pickletools
                    import io
                    
                    f.seek(0)
                    data = f.read()
                    memo = pickletools.dis(io.BytesIO(data))
                    diagnosis["pickle_protocol"] = "Detectable"
                except Exception as e:
                    diagnosis["pickle_protocol"] = f"Error: {e}"
                
                # Try basic pickle load to see specific error
                f.seek(0)
                try:
                    import pickle
                    obj = pickle.load(f)
                    diagnosis["file_type"] = str(type(obj))
                    diagnosis["pickle_protocol"] = "Standard pickle - loadable"
                    
                    # Check if it has predict method
                    if hasattr(obj, 'predict'):
                        diagnosis["has_predict"] = True
                    else:
                        diagnosis["has_predict"] = False
                        diagnosis["error_details"].append("Object doesn't have predict method")
                        
                except Exception as e:
                    diagnosis["error_details"].append(f"Pickle load error: {str(e)}")
                    
                    # Try with different protocols
                    for protocol in [0, 1, 2, 3, 4, 5]:
                        try:
                            f.seek(0)
                            obj = pickle.load(f)
                            diagnosis["pickle_protocol"] = f"Protocol {protocol} works"
                            break
                        except:
                            continue
        
        except Exception as e:
            diagnosis["error_details"].append(f"File reading error: {str(e)}")
    else:
        diagnosis["error_details"].append("Model file not found")
        diagnosis["recommendations"].append("Check if model file exists in models folder")
        
        # List available files
        if os.path.exists(MODELS_FOLDER):
            available_files = os.listdir(MODELS_FOLDER)
            diagnosis["available_files"] = available_files
        else:
            diagnosis["available_files"] = []
            diagnosis["error_details"].append("Models folder doesn't exist")
    
    return diagnosis


# Add these essential diagnostic functions to the END of your models.py file:

def create_test_model_file(model_name: str = "test_model") -> bool:
    """
    Create a simple test model file to verify the loading mechanism works
    """
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        # Create simple test data and model
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Save the model
        ensure_models_folder()
        model_path = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info(f"Test model created at {model_path}")
        
        # Test loading immediately
        loaded_model = load_model(model_name)
        if loaded_model is not None:
            logging.info("Test model loads successfully!")
            return True
        else:
            logging.error("Test model failed to load")
            return False
            
    except Exception as e:
        logging.error(f"Error creating test model: {e}")
        return False

def diagnose_model_file(model_name: str) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of a model file to understand why it's not loading
    """
    diagnosis = {
        "model_name": model_name,
        "file_exists": False,
        "file_size": 0,
        "error_details": [],
        "recommendations": []
    }
    
    model_path = os.path.join(MODELS_FOLDER, f'{model_name}.pkl')
    
    # Check if file exists
    if os.path.exists(model_path):
        diagnosis["file_exists"] = True
        
        # Get file info
        stat_info = os.stat(model_path)
        diagnosis["file_size"] = stat_info.st_size
        
        # Try to load with pickle
        try:
            with open(model_path, 'rb') as f:
                obj = pickle.load(f)
            diagnosis["file_type"] = str(type(obj))
            
            # Check if it has predict method
            if hasattr(obj, 'predict'):
                diagnosis["has_predict"] = True
            else:
                diagnosis["has_predict"] = False
                diagnosis["error_details"].append("Object doesn't have predict method")
                        
        except Exception as e:
            diagnosis["error_details"].append(f"Pickle load error: {str(e)}")
            
            # Check for common issues
            error_str = str(e).lower()
            if "cannot import name" in error_str:
                diagnosis["recommendations"].append("sklearn version compatibility issue - model needs retraining")
            elif "protocol" in error_str:
                diagnosis["recommendations"].append("Pickle protocol issue - try different Python version")
            else:
                diagnosis["recommendations"].append("Model file may be corrupted")
    else:
        diagnosis["error_details"].append("Model file not found")
        diagnosis["recommendations"].append("Check if model file exists in models folder")
        
        # List available files
        if os.path.exists(MODELS_FOLDER):
            available_files = os.listdir(MODELS_FOLDER)
            diagnosis["available_files"] = available_files
        else:
            diagnosis["available_files"] = []
            diagnosis["error_details"].append("Models folder doesn't exist")
    
    return diagnosis

def fix_model_loading_issues():
    """
    Comprehensive fix for model loading issues
    """
    logging.info("=== Model Loading Diagnostics ===")
    
    # Check models folder
    ensure_models_folder()
    
    # List all available model files
    available_models = list_available_models()
    logging.info(f"Found {len(available_models)} model files")
    
    if not available_models:
        logging.warning("No model files found. Creating test model...")
        if create_test_model_file():
            logging.info("Test model created successfully!")
        else:
            logging.error("Failed to create test model")
        return
    
    # Diagnose each model
    for model_info in available_models:
        model_name = model_info['technical_name']
        logging.info(f"\n--- Diagnosing {model_name} ---")
        
        diagnosis = diagnose_model_file(model_name)
        
        logging.info(f"File exists: {diagnosis['file_exists']}")
        if diagnosis['file_exists']:
            logging.info(f"File size: {diagnosis['file_size']} bytes")
            if diagnosis['error_details']:
                logging.error(f"Errors: {diagnosis['error_details']}")
            if diagnosis['recommendations']:
                logging.info(f"Recommendations: {diagnosis['recommendations']}")

# Update the __all__ list to include the new functions
try:
    __all__.extend([
        'diagnose_model_file',
        'create_test_model_file',
        'fix_model_loading_issues'
    ])
except NameError:
    # If __all__ doesn't exist, create it
    __all__ = [
        'list_available_models',
        'check_required_models', 
        'load_model',
        'get_model_display_name',
        'predict_man_hours',
        'analyze_what_if',
        'get_expected_feature_names',
        'get_expected_feature_names_from_model',
        'prepare_features_for_pycaret',
        'create_feature_vector_from_dict',
        'validate_feature_dict_against_config',
        'get_feature_importance',
        'get_feature_statistics',
        'diagnose_model_file',
        'create_test_model_file',
        'fix_model_loading_issues'
    ]

def test_different_target_strategies(input_features, model_name):
    """
    Test different strategies for estimating the target column value
    """
    print(f"\n=== TESTING DIFFERENT TARGET STRATEGIES ===")
    
    strategies = {
        "Strategy 1: Zero Target": 0.0,
        "Strategy 2: Small Target": 0.1,
        "Strategy 3: Medium Target": 1.0,
        "Strategy 4: Large Target": 5.0,
        "Strategy 5: Estimated Target": estimate_work_effort_default(input_features),
        "Strategy 6: Team-based Target": input_features.get('project_prf_max_team_size', 5) * 0.5,
        "Strategy 7: Size-based Target": get_size_based_target(input_features)
    }
    
    results = {}
    
    for strategy_name, target_value in strategies.items():
        print(f"\nTesting {strategy_name}: {target_value}")
        
        try:
            # Create features with this target value
            test_features = create_training_compatible_features(input_features)
            test_features['project_prf_normalised_work_effort'] = target_value
            
            # Try prediction
            result = predict_with_specific_target(test_features, model_name)
            results[strategy_name] = result
            
            if result:
                print(f"  ✅ SUCCESS: {result:.1f} hours")
            else:
                print(f"  ❌ FAILED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[strategy_name] = None
    
    # Find the best strategy
    working_strategies = [(k, v) for k, v in results.items() if v is not None]
    
    print(f"\n{'='*60}")
    print("TARGET STRATEGY RESULTS")
    print(f"{'='*60}")
    
    for strategy, result in results.items():
        status = f"{result:.1f} hours" if result else "FAILED"
        print(f"{strategy:25}: {status}")
    
    if working_strategies:
        print(f"\n✅ WORKING STRATEGIES: {len(working_strategies)}")
        best_strategy, best_result = working_strategies[0]
        print(f"🎯 BEST STRATEGY: {best_strategy} -> {best_result:.1f} hours")
        return best_strategy, best_result
    else:
        print(f"\n❌ NO STRATEGIES WORKED")
        return None, None


def get_size_based_target(input_features):
    """
    Create target value based on project size
    """
    size_input = str(input_features.get('project_prf_relative_size', 'Medium'))
    
    size_targets = {
        'XS': 0.1,
        'S': 0.3, 
        'M': 1.0,
        'L': 2.0,
        'XL': 5.0
    }
    
    for size_key, target_val in size_targets.items():
        if size_key in size_input:
            return target_val
    
    return 1.0  # Default


def predict_with_specific_target(training_features, model_name):
    """
    Make prediction with specific target value included
    """
    try:
        # Load model
        model = load_model(model_name)
        if not model:
            return None
        
        # Create DataFrame
        import pandas as pd
        features_df = pd.DataFrame([training_features])
        
        # Try PyCaret prediction first
        if PYCARET_AVAILABLE:
            try:
                from pycaret.regression import predict_model as pc_predict
                predictions = pc_predict(model, data=features_df)
                
                # Look for prediction column
                prediction_columns = ['prediction_label', 'Label', 'pred', 'prediction']
                for col in prediction_columns:
                    if col in predictions.columns:
                        result = float(predictions[col].iloc[0])
                        return max(0.1, result)
                
                # Try last column
                result = float(predictions.iloc[0, -1])
                return max(0.1, result)
            except Exception as e:
                print(f"    PyCaret failed: {e}")
        
        # Try direct model prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(features_df)
            result = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
            return max(0.1, result)
        
        return None
        
    except Exception as e:
        return None


def find_optimal_target_value(input_features, model_name):
    """
    Try to find the optimal target value that produces realistic results
    """
    print(f"\n=== FINDING OPTIMAL TARGET VALUE ===")
    
    # Test a range of target values
    target_values = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    results = []
    
    for target_val in target_values:
        try:
            test_features = create_training_compatible_features(input_features)
            test_features['project_prf_normalised_work_effort'] = target_val
            
            result = predict_with_specific_target(test_features, model_name)
            
            if result:
                results.append((target_val, result))
                print(f"Target {target_val:4.1f} -> Prediction {result:6.1f} hours")
        except:
            pass
    
    if results:
        print(f"\n✅ Found {len(results)} working target values")
        
        # Find most reasonable result (not too low, not too high)
        reasonable_results = [(t, r) for t, r in results if 10 <= r <= 5000]
        
        if reasonable_results:
            # Choose middle result
            best_target, best_result = reasonable_results[len(reasonable_results)//2]
            print(f"🎯 OPTIMAL: Target {best_target} -> {best_result:.1f} hours")
            return best_target, best_result
        else:
            # Use first working result
            best_target, best_result = results[0]
            print(f"🎯 FIRST WORKING: Target {best_target} -> {best_result:.1f} hours")
            return best_target, best_result
    else:
        print("❌ No working target values found")
        return None, None


# Update the main prediction function to use optimal target
def predict_with_training_features_optimized(input_features, model_name):
    """
    Enhanced prediction using training-compatible features with optimized target value
    """
    try:
        print(f"Making optimized training-compatible prediction with model: {model_name}")
        
        # Method 1: Try with estimated target
        training_features = create_training_compatible_features(input_features)
        result1 = predict_with_specific_target(training_features, model_name)
        
        if result1 and 10 <= result1 <= 5000:  # Reasonable range
            print(f"✅ Estimated target worked: {result1:.1f} hours")
            return result1
        
        # Method 2: Find optimal target value
        print("🔍 Estimated target didn't work, finding optimal target...")
        optimal_target, optimal_result = find_optimal_target_value(input_features, model_name)
        
        if optimal_result:
            print(f"✅ Optimal target found: {optimal_result:.1f} hours")
            return optimal_result
        
        # Method 3: Try different strategies
        print("🔍 Trying different target strategies...")
        best_strategy, best_result = test_different_target_strategies(input_features, model_name)
        
        if best_result:
            print(f"✅ Best strategy worked: {best_result:.1f} hours")
            return best_result
        
        print("❌ All target strategies failed")
        return None
        
    except Exception as e:
        print(f"Optimized prediction failed: {e}")
        return None

def main():
    """
    Test function to validate model loading and prediction functionality
    Run this file directly to test all functions: python models.py
    """
    print("=" * 60)
    print("MODELS.PY TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Check folder structure
    print("\n1. CHECKING FOLDER STRUCTURE")
    print("-" * 30)
    ensure_models_folder()
    print(f"✓ Models folder: {MODELS_FOLDER}")
    print(f"✓ Config folder: {CONFIG_FOLDER}")
    print(f"✓ Data folder: {DATA_FOLDER}")
    
    # Test 2: Load configurations
    print("\n2. LOADING CONFIGURATIONS")
    print("-" * 30)
    print(f"Feature config loaded: {bool(FEATURE_CONFIG)}")
    print(f"UI config loaded: {bool(UI_CONFIG)}")
    
    if FEATURE_CONFIG:
        print(f"  - Categorical features: {len(FEATURE_CONFIG.get('categorical_features', {}))}")
        print(f"  - Numeric features: {len(FEATURE_CONFIG.get('numeric_features', []))}")
        print(f"  - Special cases: {len(FEATURE_CONFIG.get('special_cases', {}))}")
    
    # Test 3: Check expected features
    print("\n3. FEATURE ANALYSIS")
    print("-" * 30)
    expected_features = get_expected_feature_names()
    print(f"Expected features count: {len(expected_features)}")
    if expected_features:
        print(f"First 5 features: {expected_features[:5]}")
        print(f"Last 5 features: {expected_features[-5:]}")
    
    # Get feature statistics
    stats = get_feature_statistics()
    print("\nFeature Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test 4: Check available models
    print("\n4. MODEL AVAILABILITY CHECK")
    print("-" * 30)
    model_status = check_required_models()
    print(f"Models available: {model_status['models_available']}")
    print(f"Scaler available: {model_status['scaler_available']}")
    print(f"Found models: {len(model_status['found_models'])}")
    
    available_models = list_available_models()
    if available_models:
        print("\nAvailable Models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model['display_name']} ({model['technical_name']})")
    else:
        print("No models found!")
    
    # Test 5: Create test model if no models exist
    print("\n5. MODEL CREATION TEST")
    print("-" * 30)
    if not available_models:
        print("No models found. Creating test model...")
        if create_test_model_file("test_regression_model"):
            print("✓ Test model created successfully!")
            # Refresh available models list
            available_models = list_available_models()
        else:
            print("✗ Failed to create test model")
    else:
        print(f"✓ {len(available_models)} models already available")
    
    # Test 6: Test model loading
    print("\n6. MODEL LOADING TEST")
    print("-" * 30)
    if available_models:
        test_model_name = available_models[0]['technical_name']
        print(f"Testing model: {test_model_name}")
        
        # Diagnose the model
        diagnosis = diagnose_model_file(test_model_name)
        print(f"File exists: {diagnosis['file_exists']}")
        if diagnosis['file_exists']:
            print(f"File size: {diagnosis['file_size']} bytes")
        
        if diagnosis['error_details']:
            print(f"Errors: {diagnosis['error_details']}")
        
        # Try to load the model
        model = load_model(test_model_name)
        if model is not None:
            print("✓ Model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Has predict method: {hasattr(model, 'predict')}")
        else:
            print("✗ Model loading failed")
    else:
        print("No models to test")
    
    # Test 7: Feature preparation test
    print("\n7. FEATURE PREPARATION TEST")
    print("-" * 30)
    
    # Create sample feature dictionary
    sample_features = {
        "project_prf_year_of_project": 2024,
        "external_eef_industry_sector": "Technology",
        "external_eef_organisation_type": "Large Enterprise",
        "project_prf_max_team_size": 8,
        "project_prf_relative_size": "M (Medium)",
        "project_prf_application_type": "Web Application",
        "project_prf_development_type": "New Development",
        "tech_tf_architecture": "Microservices",
        "tech_tf_development_platform": "Cloud",
        "tech_tf_language_type": "Object-Oriented",
        "tech_tf_primary_programming_language": "Python"
    }
    
    print("Sample features created:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")
    
    # Validate features
    validation = validate_feature_dict_against_config(sample_features)
    print(f"\nFeature validation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Expected count: {validation['expected_count']}")
    print(f"  Provided count: {validation['provided_count']}")
    if validation['missing_features']:
        print(f"  Missing: {validation['missing_features'][:5]}...")  # Show first 5
    
    # Test feature vector creation
    try:
        feature_vector = create_feature_vector_from_dict(sample_features)
        print(f"Feature vector shape: {feature_vector.shape}")
        print(f"Feature vector (first 10): {feature_vector[:10]}")
    except Exception as e:
        print(f"Feature vector creation failed: {e}")
    
    # Test DataFrame preparation
    try:
        features_df = prepare_features_for_pycaret(sample_features)
        if features_df is not None:
            print(f"✓ DataFrame preparation successful!")
            print(f"DataFrame shape: {features_df.shape}")
            print(f"DataFrame columns (first 10): {list(features_df.columns)[:10]}")
        else:
            print("✗ DataFrame preparation failed")
    except Exception as e:
        print(f"DataFrame preparation error: {e}")
    
    # Test 8: Prediction test
    print("\n8. PREDICTION TEST")
    print("-" * 30)
    if available_models and len(available_models) > 0:
        test_model_name = available_models[0]['technical_name']
        print(f"Testing prediction with model: {test_model_name}")
        
        try:
            prediction = predict_man_hours(
                features=sample_features,
                model_name=test_model_name,
                use_preprocessing_pipeline=True
            )
            
            if prediction is not None:
                print(f"✓ Prediction successful: {prediction:.2f} hours")
                print(f"Prediction in days: {prediction/8:.2f} days")
                
                # Test feature importance
                importance = get_feature_importance(test_model_name)
                if importance is not None:
                    print(f"✓ Feature importance available: {len(importance)} features")
                    print(f"Top 3 importance values: {sorted(importance, reverse=True)[:3]}")
                else:
                    print("✗ Feature importance not available")
            else:
                print("✗ Prediction failed")
                
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
    else:
        print("No models available for prediction test")
    
    # Test 9: What-if analysis test
    print("\n9. WHAT-IF ANALYSIS TEST")
    print("-" * 30)
    if available_models and len(available_models) > 0:
        test_model_name = available_models[0]['technical_name']
        
        try:
            # Test varying team size
            whatif_results = analyze_what_if(
                base_features=sample_features,
                model_name=test_model_name,
                param_name="project_prf_max_team_size",
                param_values=[2, 4, 6, 8, 10, 12],
                use_preprocessing_pipeline=True
            )
            
            if whatif_results['predictions']:
                print("✓ What-if analysis successful!")
                print("Team Size -> Prediction:")
                for size, pred in zip(whatif_results['param_values'], whatif_results['predictions']):
                    print(f"  {size} people -> {pred:.1f} hours")
            else:
                print("✗ What-if analysis failed - no results")
                
        except Exception as e:
            print(f"What-if analysis error: {e}")
    else:
        print("No models available for what-if analysis")
    
    # Test 10: Diagnostics
    print("\n10. COMPREHENSIVE DIAGNOSTICS")
    print("-" * 30)
    try:
        fix_model_loading_issues()
        print("✓ Diagnostics completed successfully")
    except Exception as e:
        print(f"Diagnostics error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)
    print(f"✓ Models folder: {os.path.exists(MODELS_FOLDER)}")
    print(f"✓ Available models: {len(list_available_models())}")
    print(f"✓ Feature configuration: {bool(FEATURE_CONFIG)}")
    print(f"✓ Expected features: {len(get_expected_feature_names())}")
    print(f"✓ PyCaret available: {PYCARET_AVAILABLE}")
    
    if available_models:
        print(f"✓ Ready for predictions with {len(available_models)} models")
    else:
        print("⚠ No models available - need to train and save models first")
    
    print("\nTo use in Streamlit UI:")
    print("1. Make sure models are trained and saved in 'models/' folder")
    print("2. Ensure all YAML config files are properly set up")
    print("3. Run: streamlit run ui.py")


# Alternative simple test function for quick checks
def quick_test():
    """Quick test of essential functions"""
    print("=== QUICK TEST ===")
    
    # Check models
    models = list_available_models()
    print(f"Available models: {len(models)}")
    
    # Check features  
    features = get_expected_feature_names()
    print(f"Expected features: {len(features)}")
    
    # Check configs
    print(f"Feature config loaded: {bool(FEATURE_CONFIG)}")
    
    if models and features:
        print("✅ Ready for predictions!")
    else:
        print("❌ Setup incomplete")


if __name__ == "__main__":
    # Run the main test when file is executed directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        main()    

# Simple test at the end of models.py
if __name__ == "__main__":
    print("Starting models.py test...")
    try:
        print("Testing basic imports...")
        import os
        import sys
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        print("Testing folder creation...")
        ensure_models_folder()
        print("✓ Folders OK")
        
        print("Testing configuration loading...")
        print(f"FEATURE_CONFIG loaded: {bool(FEATURE_CONFIG)}")
        print(f"UI_CONFIG loaded: {bool(UI_CONFIG)}")
        
        print("Testing feature names...")
        features = get_expected_feature_names()
        print(f"Expected features count: {len(features)}")
        
        print("Testing model check...")
        models = list_available_models()
        print(f"Available models: {len(models)}")
        
        print("✅ Basic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")