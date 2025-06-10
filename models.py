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

# Import existing pipeline functions
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
UI_INFO_FILE = os.path.join(CONFIG_FOLDER, 'ui_info.yaml')  # Updated to match merged config

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

# Load merged configuration
APP_CONFIG = load_yaml_config(UI_INFO_FILE)
FIELDS = APP_CONFIG.get('fields', {})
TAB_ORG = APP_CONFIG.get('tab_organization', {})

def get_numeric_features():
    """Get list of numeric feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'numeric']

def get_categorical_features():
    """Get list of categorical feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'categorical']

def get_boolean_features():
    """Get list of boolean feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'boolean']

def get_expected_feature_names_from_config() -> List[str]:
    """
    Get expected feature names in a consistent order from configuration.
    Preserves order: numeric > categorical > boolean > rest
    """
    feature_names = []
    feature_names += get_numeric_features()
    feature_names += get_categorical_features()
    feature_names += get_boolean_features()
    feature_names += [f for f in FIELDS if f not in feature_names]
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for f in feature_names:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique

def get_expected_feature_names_from_model(model) -> List[str]:
    """
    Get expected feature names from a trained model.
    This is an alias for consistency with UI imports.
    """
    return get_model_expected_features(model)

def validate_feature_dict_against_config(feature_dict: Dict) -> Dict[str, Any]:
    """
    Validate a feature dictionary against the configuration and return missing/extra features.
    """
    expected_features = set(get_expected_feature_names_from_config())
    provided_features = set(feature_dict.keys()) - {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
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
            # For categorical values that can't be converted, use 0
            feature_vector.append(0.0)
    
    return np.array(feature_vector)

# === Dynamic Feature Alignment Utilities ===

def get_model_expected_features(model) -> List[str]:
    """Get expected feature names from model, robustly."""
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        if hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
        if hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'feature_names_in_'):
            return list(model._final_estimator.feature_names_in_)
        if hasattr(model, 'X') and hasattr(model.X, 'columns'):
            return list(model.X.columns)
        
        # Fallback: try to get from model training data if available
        if hasattr(model, 'feature_names_'):
            return list(model.feature_names_)
            
    except Exception as e:
        logging.warning(f"Could not extract feature names from model: {e}")
    
    return []

def align_features_to_model(feature_dict: Dict[str, Any], model_features: List[str]) -> Dict[str, Any]:
    """Align feature dictionary to model columns. Missing columns = 0, extras dropped."""
    return {f: feature_dict.get(f, 0) for f in model_features}

def align_df_to_model(df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    """Align DataFrame to model expected features"""
    # Add missing columns with default value 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    
    # Return only the columns expected by the model in the right order
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
    
    # Load display names configuration once
    display_names_map = load_model_display_names()
    
    for f in os.listdir(MODELS_FOLDER):
        if f.endswith('.pkl') and not ('scaler' in f.lower()) and not ('pipeline' in f.lower()):
            technical_name = os.path.splitext(f)[0]
            # Use configured display name if available, otherwise generate dynamically
            display_name = get_model_display_name_from_config(technical_name, display_names_map)
            model_files.append({
                'technical_name': technical_name,
                'display_name': display_name
            })
    
    model_files.sort(key=lambda x: x['display_name'])
    return model_files

def get_model_display_name(model_filename: str) -> str:
    """
    Convert model filename to human-readable display name.
    This is the basic dynamic generation function.
    """
    return " ".join(word.capitalize() for word in model_filename.split("_"))

def load_model_display_names() -> Dict[str, str]:
    """Load model display names from JSON configuration file."""
    try:
        model_config_path = os.path.join(CONFIG_FOLDER, 'model_display_names.json')
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                display_names = json.load(f)
            logging.info(f"Loaded {len(display_names)} model display names from JSON")
            return display_names
        else:
            logging.warning(f"model_display_names.json not found at {model_config_path}")
    except Exception as e:
        logging.error(f"Failed to load model display names: {e}")
    
    return {}

def get_model_display_name_from_config(model_filename: str, display_names_map: Optional[Dict[str, str]] = None) -> str:
    """
    Get model display name, first checking configuration, then falling back to dynamic generation.
    """
    if display_names_map is None:
        display_names_map = load_model_display_names()
    
    # Check if we have a configured display name (exact match first)
    if model_filename in display_names_map:
        return display_names_map[model_filename]
    
    # Try case-insensitive lookup
    model_filename_lower = model_filename.lower()
    for key, value in display_names_map.items():
        if key.lower() == model_filename_lower:
            return value
    
    # Fallback to dynamic generation
    return get_model_display_name(model_filename)

def get_all_model_display_names() -> Dict[str, str]:
    """
    Get display names for all available models.
    Returns a mapping of technical_name -> display_name.
    """
    display_names_map = load_model_display_names()
    all_display_names = {}
    
    try:
        # Get all available models
        available_models = list_available_models()
        
        for model_info in available_models:
            technical_name = model_info['technical_name']
            # Use configured name if available, otherwise generate dynamically
            display_name = get_model_display_name_from_config(technical_name, display_names_map)
            all_display_names[technical_name] = display_name
        
        logging.info(f"Generated display names for {len(all_display_names)} models")
        
    except Exception as e:
        logging.error(f"Failed to get all model display names: {e}")
    
    return all_display_names

def save_model_display_names(display_names: Dict[str, str]) -> bool:
    """
    Save model display names to configuration file.
    """
    try:
        ensure_models_folder()  # Ensure config folder exists too
        if not os.path.exists(CONFIG_FOLDER):
            os.makedirs(CONFIG_FOLDER)
        
        config_path = os.path.join(CONFIG_FOLDER, 'model_display_names.yaml')
        
        config_data = {
            'model_display_names': display_names,
            'last_updated': pd.Timestamp.now().isoformat(),
            'description': 'Custom display names for ML models'
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
        
        logging.info(f"Saved {len(display_names)} model display names to {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save model display names: {e}")
        return False
    """
    Load model display names from configuration or file.
    Returns a mapping of technical_name -> display_name.
    Falls back to dynamic generation if no configuration is found.
    """
    display_names = {}
    
    try:
        # Try to load from configuration file first
        model_config_path = os.path.join(CONFIG_FOLDER, 'model_display_names.yaml')
        if os.path.exists(model_config_path):
            model_config = load_yaml_config(model_config_path)
            display_names = model_config.get('model_display_names', {})
            logging.info(f"Loaded {len(display_names)} model display names from configuration")
        
        # If no config file, try to load from main UI config
        elif 'model_display_names' in APP_CONFIG:
            display_names = APP_CONFIG.get('model_display_names', {})
            logging.info(f"Loaded {len(display_names)} model display names from main config")
        
        else:
            logging.info("No model display names configuration found, will use dynamic generation")
    
    except Exception as e:
        logging.warning(f"Failed to load model display names configuration: {e}")
    
    return display_names

def get_model_display_name_from_config(model_filename: str, display_names_map: Optional[Dict[str, str]] = None) -> str:
    """
    Get model display name, first checking configuration, then falling back to dynamic generation.
    """
    if display_names_map is None:
        display_names_map = load_model_display_names()
    
    # Check if we have a configured display name
    if model_filename in display_names_map:
        return display_names_map[model_filename]
    
    # Fallback to dynamic generation
    return get_model_display_name(model_filename)

def get_all_model_display_names() -> Dict[str, str]:
    """
    Get display names for all available models.
    Returns a mapping of technical_name -> display_name.
    """
    display_names_map = load_model_display_names()
    all_display_names = {}
    
    try:
        # Get all available models
        available_models = list_available_models()
        
        for model_info in available_models:
            technical_name = model_info['technical_name']
            # Use configured name if available, otherwise generate dynamically
            display_name = get_model_display_name_from_config(technical_name, display_names_map)
            all_display_names[technical_name] = display_name
        
        logging.info(f"Generated display names for {len(all_display_names)} models")
        
    except Exception as e:
        logging.error(f"Failed to get all model display names: {e}")
    
    return all_display_names

def debug_model_files():
    """Debug function to check actual model filenames"""
    print("=== ACTUAL MODEL FILES ===")
    
    ensure_models_folder()
    files = os.listdir(MODELS_FOLDER)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    
    for f in pkl_files:
        technical_name = os.path.splitext(f)[0]
        print(f"File: {f}")
        print(f"Technical name: {technical_name}")
        
        # Check if it's in JSON
        display_names = load_model_display_names()
        if technical_name in display_names:
            print(f"✅ Found in JSON: {display_names[technical_name]}")
        else:
            print(f"❌ NOT found in JSON")
            # Show close matches
            close_matches = [k for k in display_names.keys() if technical_name.lower() in k.lower() or k.lower() in technical_name.lower()]
            if close_matches:
                print(f"   Similar keys: {close_matches}")
        print("-" * 50)

def save_model_display_names(display_names: Dict[str, str]) -> bool:
    """
    Save model display names to configuration file.
    """
    try:
        ensure_models_folder()  # Ensure config folder exists too
        if not os.path.exists(CONFIG_FOLDER):
            os.makedirs(CONFIG_FOLDER)
        
        config_path = os.path.join(CONFIG_FOLDER, 'model_display_names.yaml')
        
        config_data = {
            'model_display_names': display_names,
            'last_updated': pd.Timestamp.now().isoformat(),
            'description': 'Custom display names for ML models'
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
        
        logging.info(f"Saved {len(display_names)} model display names to {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save model display names: {e}")
        return False

def check_required_models() -> dict:
    """
    Checks for the presence of model files in the models folder.
    Returns a dictionary summarizing their availability and listing the models.
    """
    ensure_models_folder()
    existing_files = os.listdir(MODELS_FOLDER)
    existing_models = [f for f in existing_files if f.endswith('.pkl')]
    
    # Filter out scaler and pipeline files
    model_files = [f for f in existing_models if not ('scaler' in f.lower()) and not ('pipeline' in f.lower())]
    has_models = len(model_files) > 0
    
    # Load display names configuration once
    display_names_map = load_model_display_names()
    
    found_models = []
    for f in model_files:
        technical_name = os.path.splitext(f)[0]
        # Use configured display name if available, otherwise generate dynamically
        display_name = get_model_display_name_from_config(technical_name, display_names_map)
        found_models.append({
            'technical_name': technical_name,
            'display_name': display_name,
            'file_path': os.path.join(MODELS_FOLDER, f)
        })
    
    return {
        "models_available": has_models,
        "found_models": found_models,
        "technical_names": [model['technical_name'] for model in found_models],
        "model_count": len(found_models)
    }

def load_model(model_name: str) -> Optional[Any]:
    """
    Load a model using multiple fallback methods: PyCaret, joblib, pickle
    """
    model_path = os.path.join(MODELS_FOLDER, model_name)
    
    # Try PyCaret first
    try:
        from pycaret.regression import load_model as pc_load_model
        model = pc_load_model(model_path)
        logging.info(f"Model {model_name} loaded successfully using PyCaret")
        return model
    except Exception as e:
        logging.debug(f"PyCaret loading failed for {model_name}: {e}")
    
    # Try joblib
    try:
        import joblib
        if os.path.exists(model_path + ".pkl"):
            model = joblib.load(model_path + ".pkl")
            logging.info(f"Model {model_name} loaded successfully using joblib")
            return model
        elif os.path.exists(model_path + ".joblib"):
            model = joblib.load(model_path + ".joblib")
            logging.info(f"Model {model_name} loaded successfully using joblib")
            return model
    except Exception as e:
        logging.debug(f"Joblib loading failed for {model_name}: {e}")
    
    # Try pickle
    try:
        pkl_path = model_path + ".pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
                logging.info(f"Model {model_name} loaded successfully using pickle")
                return model
    except Exception as e:
        logging.debug(f"Pickle loading failed for {model_name}: {e}")
    
    logging.error(f"Failed to load model {model_name} with all methods")
    return None

def prepare_features_manually_from_config(features):
    """
    Manual feature preparation using configuration as fallback when pipeline is not available
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

def apply_pipeline_transformation(ui_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply preprocessing pipeline transformation to user input features.
    This is the first step in the feature preparation process.
    """
    try:
        if not PIPELINE_AVAILABLE:
            logging.info("Pipeline module not available, using manual preparation")
            return prepare_features_manually_from_config(ui_features)
        
        logging.info("Applying pipeline transformation")
        
        # Convert UI features to DataFrame
        try:
            feature_df = convert_feature_dict_to_dataframe(ui_features, FIELDS)
            logging.info(f"Converted UI features to DataFrame: {feature_df.shape}")
        except Exception as e:
            logging.error(f"Failed to convert features to DataFrame: {e}")
            return prepare_features_manually_from_config(ui_features)
        
        # Create and apply preprocessing pipeline
        try:
            pipeline = create_preprocessing_pipeline(
                target_col=None,
                high_missing_threshold=0.9,
                max_categorical_cardinality=20
            )
            
            features_transformed = pipeline.fit_transform(feature_df)
            logging.info(f"Pipeline transformation successful: {features_transformed.shape}")
            
            # Clean up: drop any target/label columns that shouldn't be used for prediction
            to_drop = [col for col in features_transformed.columns 
                      if any(keyword in col.lower() for keyword in ["target", "effort", "label", "prediction"])]
            if to_drop:
                features_transformed = features_transformed.drop(columns=to_drop, errors="ignore")
                logging.info(f"Dropped target-related columns: {to_drop}")
            
            return features_transformed
            
        except Exception as e:
            logging.error(f"Pipeline transformation failed: {e}")
            return prepare_features_manually_from_config(ui_features)
            
    except Exception as e:
        logging.error(f"Pipeline transformation error: {e}")
        return prepare_features_manually_from_config(ui_features)

def apply_pipeline_transformation_with_custom_params(
    ui_features: Dict[str, Any],
    target_col: Optional[str] = None,
    high_missing_threshold: float = 0.9,
    max_categorical_cardinality: int = 20
) -> pd.DataFrame:
    """
    Apply preprocessing pipeline transformation with custom parameters.
    Provides more control over pipeline configuration.
    """
    try:
        if not PIPELINE_AVAILABLE:
            logging.info("Pipeline module not available, using manual preparation")
            return prepare_features_manually_from_config(ui_features)
        
        logging.info(f"Applying pipeline transformation with custom params: "
                    f"target_col={target_col}, missing_threshold={high_missing_threshold}, "
                    f"max_cardinality={max_categorical_cardinality}")
        
        # Convert UI features to DataFrame
        feature_df = convert_feature_dict_to_dataframe(ui_features, FIELDS)
        
        # Create pipeline with custom parameters
        pipeline = create_preprocessing_pipeline(
            target_col=target_col,
            high_missing_threshold=high_missing_threshold,
            max_categorical_cardinality=max_categorical_cardinality
        )
        
        # Apply transformation
        features_transformed = pipeline.fit_transform(feature_df)
        
        # Clean up target columns if specified
        if target_col:
            features_transformed = features_transformed.drop(columns=[target_col], errors="ignore")
        
        logging.info(f"Custom pipeline transformation successful: {features_transformed.shape}")
        return features_transformed
        
    except Exception as e:
        logging.error(f"Custom pipeline transformation failed: {e}")
        return prepare_features_manually_from_config(ui_features)
    """
    Apply feature engineering transformations to the prepared features.
    This function handles derived features, calculations, and feature validation.
    """
    try:
        if not FEATURE_ENGINEERING_AVAILABLE:
            logging.info("Feature engineering module not available, returning features as-is")
            return features_df
        
        logging.info("Applying feature engineering transformations")
        engineered_features = features_df.copy()
        
        # Apply feature engineering functions if available
        try:
            # Calculate derived features
            engineered_features = calculate_derived_features(engineered_features)
            logging.info("Derived features calculated successfully")
        except Exception as e:
            logging.warning(f"Derived features calculation failed: {e}")
        
        try:
            # Validate features
            validation_result = validate_features(engineered_features)
            if not validation_result.get('valid', True):
                logging.warning(f"Feature validation warnings: {validation_result.get('warnings', [])}")
        except Exception as e:
            logging.warning(f"Feature validation failed: {e}")
        
        try:
            # Get feature summary for logging
            summary = get_feature_summary(engineered_features)
            logging.info(f"Feature engineering complete. Summary: {summary}")
        except Exception as e:
            logging.debug(f"Could not generate feature summary: {e}")
        
        return engineered_features
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        return features_df  # Return original features if engineering fails

def estimate_missing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate or fill missing features using domain knowledge or statistical methods.
    """
    try:
        if not FEATURE_ENGINEERING_AVAILABLE:
            # Fallback: simple imputation
            return features_df.fillna(0)
        
        # Use feature engineering module if available
        try:
            # This would use your estimate_target_value function
            estimated_features = estimate_target_value(features_df)
            logging.info("Missing features estimated using feature engineering module")
            return estimated_features
        except Exception as e:
            logging.warning(f"Feature estimation failed, using simple imputation: {e}")
            return features_df.fillna(0)
            
    except Exception as e:
        logging.error(f"Missing feature estimation failed: {e}")
        return features_df.fillna(0)
def apply_feature_engineering(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to the prepared features.
    This function handles derived features, calculations, and feature validation.
    
    Args:
        features_df: DataFrame with features after pipeline transformation
        
    Returns:
        pd.DataFrame: Features with engineering transformations applied
    """
    try:
        if not FEATURE_ENGINEERING_AVAILABLE:
            logging.info("Feature engineering module not available, returning features as-is")
            return features_df
        
        logging.info("Applying feature engineering transformations")
        engineered_features = features_df.copy()
        
        # Apply feature engineering functions if available
        try:
            # Calculate derived features
            engineered_features = calculate_derived_features(engineered_features)
            logging.info("Derived features calculated successfully")
        except Exception as e:
            logging.warning(f"Derived features calculation failed: {e}")
        
        try:
            # Validate features
            validation_result = validate_features(engineered_features)
            if not validation_result.get('valid', True):
                logging.warning(f"Feature validation warnings: {validation_result.get('warnings', [])}")
        except Exception as e:
            logging.warning(f"Feature validation failed: {e}")
        
        try:
            # Get feature summary for logging
            summary = get_feature_summary(engineered_features)
            logging.info(f"Feature engineering complete. Summary: {summary}")
        except Exception as e:
            logging.debug(f"Could not generate feature summary: {e}")
        
        return engineered_features
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        return features_df  # Return original features if engineering fails

def prepare_features_for_model(ui_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Complete feature preparation pipeline for user input.
    This is the main feature preparation function used by predict_man_hours.
    
    Args:
        ui_features: Dictionary of user input features from the UI
        
    Returns:
        pd.DataFrame: Prepared features ready for model prediction
        
    Raises:
        Exception: If all feature preparation methods fail
    """
    
    try:
        logging.info(f"Starting feature preparation for {len(ui_features)} input features")
        
        # Validate input
        if not ui_features:
            raise ValueError("No input features provided")
        
        # Remove UI-specific keys that shouldn't be used for prediction
        ui_keys_to_remove = {
            'selected_model', 'selected_models', 'submit', 'clear_results', 
            'show_history', 'save_config', 'config_name', 'comparison_mode'
        }
        clean_features = {k: v for k, v in ui_features.items() if k not in ui_keys_to_remove}
        
        logging.info(f"Cleaned features: {len(clean_features)} features after removing UI keys")
        
        # Step 1: Apply pipeline transformation
        try:
            features_transformed = apply_pipeline_transformation(clean_features)
            logging.info(f"Pipeline transformation successful: {features_transformed.shape}")
        except Exception as e:
            logging.warning(f"Pipeline transformation failed: {e}")
            # Fallback to manual preparation
            features_transformed = prepare_features_manually_from_config(clean_features)
            if features_transformed is None:
                raise Exception("Both pipeline and manual feature preparation failed")
        
        # Step 2: Apply feature engineering
        try:
            features_engineered = apply_feature_engineering(features_transformed)
            logging.info(f"Feature engineering successful: {features_engineered.shape}")
        except Exception as e:
            logging.warning(f"Feature engineering failed: {e}")
            # Continue with transformed features if engineering fails
            features_engineered = features_transformed
        
        # Step 3: Handle missing features and validate
        try:
            features_final = estimate_missing_features(features_engineered)
            logging.info(f"Missing feature handling successful: {features_final.shape}")
        except Exception as e:
            logging.warning(f"Missing feature estimation failed: {e}")
            # Simple fallback: fill with zeros
            features_final = features_engineered.fillna(0)
        
        # Step 4: Final validation and cleanup
        try:
            # Ensure all values are numeric
            features_final = features_final.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Remove any remaining target/label columns
            target_keywords = ['target', 'effort', 'label', 'prediction', 'actual', 'ground_truth']
            cols_to_drop = [col for col in features_final.columns 
                           if any(keyword in col.lower() for keyword in target_keywords)]
            if cols_to_drop:
                features_final = features_final.drop(columns=cols_to_drop, errors='ignore')
                logging.info(f"Removed potential target columns: {cols_to_drop}")
            
            # Ensure we have at least some features
            if features_final.empty or features_final.shape[1] == 0:
                raise ValueError("No features remaining after preparation")
            
            # Check for infinite or extremely large values
            features_final = features_final.replace([np.inf, -np.inf], 0)
            
            # Log final statistics
            logging.info(f"Feature preparation complete:")
            logging.info(f"  - Final shape: {features_final.shape}")
            logging.info(f"  - Feature columns: {list(features_final.columns)}")
            logging.info(f"  - Missing values: {features_final.isnull().sum().sum()}")
            logging.info(f"  - Data types: {features_final.dtypes.value_counts().to_dict()}")
            
            return features_final
            
        except Exception as e:
            logging.error(f"Final validation failed: {e}")
            raise Exception(f"Feature preparation failed at final validation: {e}")
    
    except Exception as e:
        logging.error(f"Complete feature preparation failed: {e}")
        
        # Last resort: try basic DataFrame creation
        try:
            logging.warning("Attempting emergency feature preparation")
            expected_features = get_expected_feature_names_from_config()
            feature_vector = create_feature_vector_from_dict(clean_features, expected_features)
            emergency_df = pd.DataFrame([feature_vector], columns=expected_features)
            logging.warning(f"Emergency preparation successful: {emergency_df.shape}")
            return emergency_df
        except Exception as emergency_e:
            logging.error(f"Emergency feature preparation also failed: {emergency_e}")
            raise Exception(f"All feature preparation methods failed. Original error: {e}, Emergency error: {emergency_e}")

def validate_prepared_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate prepared features for common issues.
    
    Args:
        features_df: Prepared features DataFrame
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    try:
        # Basic shape validation
        if features_df.empty:
            validation_result['errors'].append("DataFrame is empty")
            validation_result['valid'] = False
            return validation_result
        
        # Check for missing values
        missing_count = features_df.isnull().sum().sum()
        if missing_count > 0:
            validation_result['warnings'].append(f"Found {missing_count} missing values")
        
        # Check for infinite values
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            validation_result['errors'].append(f"Found {inf_count} infinite values")
            validation_result['valid'] = False
        
        # Check data types
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            validation_result['warnings'].append(f"Non-numeric columns found: {non_numeric_cols}")
        
        # Statistics
        validation_result['stats'] = {
            'shape': features_df.shape,
            'missing_values': missing_count,
            'infinite_values': inf_count,
            'numeric_columns': len(features_df.select_dtypes(include=[np.number]).columns),
            'non_numeric_columns': len(non_numeric_cols),
            'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
    except Exception as e:
        validation_result['errors'].append(f"Validation failed: {e}")
        validation_result['valid'] = False
    
    return validation_result
    """
    Pipeline + feature engineering for user input.
    This is the main feature preparation function used by predict_man_hours.
    """
    
    # Step 1: Pipeline transformation (if available)
    if PIPELINE_AVAILABLE:
        try:
            logging.info("Using pipeline transformation for feature preparation")
            feature_df = convert_feature_dict_to_dataframe(ui_features, FIELDS)
            pipeline = create_preprocessing_pipeline(
                target_col=None,
                high_missing_threshold=0.9,
                max_categorical_cardinality=20
            )
            features_transformed = pipeline.fit_transform(feature_df)
            
            # Optionally drop any label/target columns that shouldn't be used for prediction
            to_drop = [col for col in features_transformed.columns 
                      if any(keyword in col.lower() for keyword in ["target", "effort", "label", "prediction"])]
            if to_drop:
                features_transformed = features_transformed.drop(columns=to_drop, errors="ignore")
                logging.info(f"Dropped target-related columns: {to_drop}")
                
        except Exception as e:
            logging.warning(f"Pipeline transformation failed, falling back to manual preparation: {e}")
            features_transformed = prepare_features_manually_from_config(ui_features)
    else:
        logging.info("Pipeline not available, using manual feature preparation")
        features_transformed = prepare_features_manually_from_config(ui_features)
    
    # Step 2: Feature engineering (if available)
    if FEATURE_ENGINEERING_AVAILABLE:
        try:
            logging.info("Applying feature engineering")
            # Add any derived features or calculations here
            # features_transformed = calculate_derived_features(features_transformed)
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
    
    try:
        # Use sequential approach for dictionary input (from UI)
        if isinstance(features, dict):
            logging.info(f"Starting prediction with model: {model_name}")
            
            # Step 1: Transform and engineer features
            features_df = prepare_features_for_model(features)
            if features_df is None:
                logging.error("Feature preparation failed")
                return None
            
            # Step 2: Load model
            model = load_model(model_name)
            if not model:
                logging.error(f"Failed to load model: {model_name}")
                return None
            
            # Step 3: Align features to model expectations
            model_expected_features = get_model_expected_features(model)
            if model_expected_features:
                logging.info(f"Aligning {len(features_df.columns)} features to {len(model_expected_features)} model features")
                features_aligned = align_df_to_model(features_df, model_expected_features)
            else:
                logging.warning("Could not determine model expected features, using all prepared features")
                features_aligned = features_df
            
            # Step 4: Make prediction
            try:
                # Try PyCaret prediction first
                from pycaret.regression import predict_model
                preds = predict_model(model, data=features_aligned)
                
                # Look for prediction column with common names
                for col in ['prediction_label', 'Label', 'pred', 'prediction']:
                    if col in preds.columns:
                        result = float(preds[col].iloc[0])
                        logging.info(f"Prediction successful: {result}")
                        return result
                
                # Fallback to last column
                result = float(preds.iloc[0, -1])
                logging.info(f"Prediction successful (last column): {result}")
                return result
                
            except Exception as e:
                logging.warning(f"PyCaret prediction failed, trying direct model prediction: {e}")
                
                # Fallback to direct model prediction
                if hasattr(model, 'predict'):
                    pred = model.predict(features_aligned)
                    result = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                    logging.info(f"Direct prediction successful: {result}")
                    return result
                
        else:
            # Handle array/list input (backward compatibility)
            logging.warning("Array/list input not fully supported in sequential approach")
            return None
            
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        
    return None

def get_feature_importance(model_name: str) -> Optional[np.ndarray]:
    """
    Get feature importance for a given model if available.
    """
    model = load_model(model_name)
    if model is None:
        return None
    
    try:
        # Try different ways to get feature importance
        if hasattr(model, 'named_steps'):
            # For pipeline models
            for step in model.named_steps.values():
                if hasattr(step, 'feature_importances_'):
                    return step.feature_importances_
                elif hasattr(step, 'coef_'):
                    return np.abs(step.coef_).flatten()
                    
        elif hasattr(model, '_final_estimator'):
            # For PyCaret models
            est = model._final_estimator
            if hasattr(est, 'feature_importances_'):
                return est.feature_importances_
            elif hasattr(est, 'coef_'):
                return np.abs(est.coef_).flatten()
                
        elif hasattr(model, 'feature_importances_'):
            # Direct feature importance
            return model.feature_importances_
            
        elif hasattr(model, 'coef_'):
            # For linear models
            return np.abs(model.coef_).flatten()
            
        return None
        
    except Exception as e:
        logging.error(f"Failed to get feature importance for {model_name}: {e}")
        return None

def analyze_what_if(
    base_features: Union[np.ndarray, Dict],
    model_name: str,
    param_name: str,
    param_values: List[float]
) -> Dict[str, List]:
    """
    Analyze how changing a parameter affects predictions
    """
    results = {"param_values": [], "predictions": []}
    
    if not isinstance(base_features, dict):
        logging.warning("What-if analysis requires dictionary input")
        return results
    
    base_dict = dict(base_features)
    
    for value in param_values:
        test_features = base_dict.copy()
        test_features[param_name] = value
        
        pred = predict_man_hours(test_features, model_name)
        if pred is not None:
            results["param_values"].append(value)
            results["predictions"].append(pred)
    
    return results

# Pipeline integration functions (if needed by UI)
def check_preprocessing_pipeline_compatibility():
    """Check if preprocessing pipeline is compatible with current configuration"""
    try:
        if not PIPELINE_AVAILABLE:
            return {"compatible": False, "error": "Pipeline module not available"}
        
        # Add any specific compatibility checks here
        return {
            "compatible": True,
            "recommendations": []
        }
    except Exception as e:
        return {"compatible": False, "error": str(e)}

def get_preprocessing_pipeline_info():
    """Get information about the preprocessing pipeline"""
    try:
        if not PIPELINE_AVAILABLE:
            return {"available": False, "error": "Pipeline module not available"}
        
        return {
            "available": True,
            "step_count": 3,  # Update based on your pipeline
            "steps": [
                "Feature type conversion",
                "Missing value imputation", 
                "Categorical encoding"
            ]
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def test_feature_pipeline_integration(feature_dict):
    """Test if feature pipeline integration works with given features"""
    try:
        if not PIPELINE_AVAILABLE:
            return {"success": False, "errors": ["Pipeline module not available"]}
        
        # Test the pipeline with sample data
        test_df = prepare_features_for_model(feature_dict)
        return {
            "success": test_df is not None,
            "errors": [] if test_df is not None else ["Pipeline processing failed"]
        }
    except Exception as e:
        return {"success": False, "errors": [str(e)]}

def get_feature_statistics():
    """Get statistics about features from configuration"""
    try:
        stats = {
            "total_features": len(FIELDS),
            "numeric_features": len(get_numeric_features()),
            "categorical_features": len(get_categorical_features()),
            "boolean_features": len(get_boolean_features())
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

# Exports for Streamlit UI
__all__ = [
    'list_available_models',
    'get_model_display_name', 
    'check_required_models',
    'predict_man_hours',
    'get_feature_importance',
    'analyze_what_if',
    'get_expected_feature_names_from_model',
    'check_preprocessing_pipeline_compatibility',
    'get_preprocessing_pipeline_info',
    'test_feature_pipeline_integration',
    'get_feature_statistics',
    'apply_pipeline_transformation',
    'apply_pipeline_transformation_with_custom_params',
    'apply_feature_engineering',
    'estimate_missing_features',
    'load_model_display_names',
    'get_model_display_name_from_config',
    'get_all_model_display_names',
    'save_model_display_names',
    'prepare_features_for_model',
    'validate_prepared_features'
]