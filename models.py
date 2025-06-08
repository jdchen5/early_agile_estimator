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
from typing import Dict, List, Optional, Union, Any, Tuple

# Import  existing pipeline functions
try:
    from pipeline import convert_feature_dict_to_dataframe, create_preprocessing_pipeline
    PIPELINE_AVAILABLE = True
    logging.info("Pipeline module loaded successfully")
except ImportError as e:
    PIPELINE_AVAILABLE = False
    logging.warning(f"Pipeline module not available: {e}")

# Import feature engineering functions
from feature_engineering import (
    estimate_target_value,
    calculate_derived_features,
    validate_features,
    get_feature_summary
)


# Ensure required packages are installed
try:        
    # Paths
    DATA_FOLDER = 'data'
    CONFIG_FOLDER = 'config'
    MODELS_FOLDER = 'models'
    FEATURE_COLS_FILE = os.path.join(DATA_FOLDER, 'expected_features.txt')
    MODEL_DISPLAY_NAMES_FILE = os.path.join(CONFIG_FOLDER, 'model_display_names.json')
    FEATURE_MAPPING_FILE = os.path.join(CONFIG_FOLDER, 'feature_mapping.yaml')
    UI_CONFIG_FILE = os.path.join(CONFIG_FOLDER, 'ui_config.yaml')
    # Ensure directories exist
    os.makedirs(DATA_FOLDER, exist_ok=True) 
    os.makedirs(CONFIG_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
except Exception as e:
    logging.error(f"Error setting up directories: {e}")


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

# === Sequential Prediction Core Functions ===

def apply_pipeline_transformation(ui_features: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if not PIPELINE_AVAILABLE:
        logging.warning("Pipeline not available, skipping pipeline transformation")
        return None
    try:
        feature_config = load_yaml_config(FEATURE_MAPPING_FILE)
        if not feature_config:
            logging.warning("Feature config not available for pipeline")
            return None
        logging.info("Converting UI features to DataFrame...")
        initial_df = convert_feature_dict_to_dataframe(ui_features, feature_config)
        if initial_df is None or initial_df.empty:
            logging.error("Failed to convert UI features to DataFrame")
            return None
        logging.info(f"Initial DataFrame shape: {initial_df.shape}")
        logging.info("Creating preprocessing pipeline...")
        pipeline = create_preprocessing_pipeline(
            target_col=None,
            high_missing_threshold=0.9,
            max_categorical_cardinality=20
        )
        if pipeline is None:
            logging.error("Failed to create preprocessing pipeline")
            return None
        logging.info("Applying pipeline transformation...")
        transformed_df = pipeline.fit_transform(initial_df)
        target_related_cols = [col for col in transformed_df.columns 
                               if any(keyword in col.lower() for keyword in ['target', 'effort', 'work_effort'])]
        if target_related_cols:
            transformed_df = transformed_df.drop(columns=target_related_cols)
            logging.info(f"Removed target-related columns: {target_related_cols}")
        logging.info(f"Pipeline transformation successful: {transformed_df.shape}")
        return transformed_df
    except Exception as e:
        logging.error(f"Pipeline transformation failed: {e}")
        return None

def apply_feature_engineering(
    ui_features: Dict[str, Any], 
    pipeline_features: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    try:
        complete_features = {}
        if pipeline_features is not None and not pipeline_features.empty:
            pipeline_dict = pipeline_features.iloc[0].to_dict()
            complete_features.update(pipeline_dict)
            logging.info(f"Added {len(pipeline_dict)} features from pipeline")
        else:
            logging.info("No pipeline features available, starting with empty feature set")        
        logging.info(f"Feature engineering complete: {len(complete_features)} total features")
        return complete_features
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


def make_model_prediction(complete_features: Dict[str, Any], model_name: str) -> Optional[float]:
    """
    STEP 3: Make prediction using the complete feature set (dynamically aligned).
    
    Args:
        complete_features: Complete feature dictionary 
        model_name: Name of the trained model
        
    Returns:
        Predicted man-hours or None if prediction fails
    """
    
    try:
        # Load model
        model = load_model(model_name)
        if not model:
            logging.error(f"Failed to load model: {model_name}")
            return None
        
        # --- DYNAMIC COLUMN ALIGNMENT ---
        model_expected_features = get_model_expected_features(model)
        if not model_expected_features:
            logging.warning("Could not extract expected features from model. Using all features as-is.")
            features_df = pd.DataFrame([complete_features])
        else:
            aligned_features = align_features_to_model(complete_features, model_expected_features)
            features_df = pd.DataFrame([aligned_features], columns=model_expected_features)
        logging.info(f"Created prediction DataFrame: {features_df.shape}")
        key_features = [
            'project_prf_normalised_work_effort', 'project_prf_max_team_size',
            'external_eef_industry_sector', 'tech_tf_primary_programming_language'
        ]
        for key in key_features:
            if key in complete_features:
                logging.info(f"Key feature {key}: {complete_features[key]}")
        if PYCARET_AVAILABLE:
            try:
                from pycaret.regression import predict_model
                predictions = predict_model(model, data=features_df)
                for col in ['prediction_label', 'Label', 'pred', 'prediction']:
                    if col in predictions.columns:
                        result = float(predictions[col].iloc[0])
                        logging.info(f"PyCaret prediction successful: {result:.2f}")
                        return max(0.1, result)  # Ensure positive
                
                # Fallback to last column
                result = float(predictions.iloc[0, -1])
                logging.info(f"PyCaret prediction (last col): {result:.2f}")
                return max(0.1, result)
                
            except Exception as e:
                logging.warning(f"PyCaret prediction failed: {e}")
        
        # Fallback to direct model prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(features_df)
            result = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
            logging.info(f"Direct prediction: {result:.2f}")
            return max(0.1, result)
        
        logging.error("No prediction method available")
        return None
        
    except Exception as e:
        logging.error(f"Model prediction failed: {e}")
        return None

def predict_with_sequential_approach(
    ui_features: Dict[str, Any], 
    model_name: str
) -> Optional[float]:
    try:
        logging.info("=== STARTING SEQUENTIAL PREDICTION ===")
        logging.info("Step 1: Applying pipeline transformation...")
        pipeline_features = apply_pipeline_transformation(ui_features)
        if pipeline_features is None:
            logging.error("Pipeline transformation failed")
            return None
        logging.info(f"Pipeline produced {pipeline_features.shape[1]} features")
        logging.info("Step 2: Applying feature engineering to fill gaps...")
        complete_features = apply_feature_engineering(ui_features, pipeline_features)
        if complete_features is None:
            logging.error("Feature engineering failed")
            return None
        logging.info(f"Feature engineering produced {len(complete_features)} complete features")
        logging.info("Step 3: Making model prediction...")
        prediction = make_model_prediction(complete_features, model_name)
        if prediction is not None:
            logging.info(f"✅ Sequential prediction successful: {prediction:.2f} hours")
        else:
            logging.error("❌ Model prediction failed")
        return prediction
    except Exception as e:
        logging.error(f"Sequential prediction failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return None


def predict_from_ui_features_sequential(
    ui_features: Dict[str, Any], 
    model_name: str
) -> Dict[str, Any]:
    """
    Main UI integration function using the sequential approach.
    
    This is the function your Streamlit UI should call.
    
    Args:
        ui_features: Features from the Streamlit UI
        model_name: Selected model name
        
    Returns:
        Dictionary with prediction result and detailed metadata
    """
    
    try:
        logging.info("=== STARTING UI PREDICTION WITH SEQUENTIAL APPROACH ===")
        
        # Make prediction using sequential approach
        prediction = predict_with_sequential_approach(ui_features, model_name)
        
        if prediction is not None:
            # Create complete features for validation and summary
            pipeline_features = apply_pipeline_transformation(ui_features)
            complete_features = apply_feature_engineering(ui_features, pipeline_features)
            
            if complete_features:
                validation = validate_features(complete_features)
                summary = get_feature_summary(complete_features)
                
                return {
                    'success': True,
                    'prediction': prediction,
                    'prediction_days': prediction / 8,
                    'method_used': 'sequential_pipeline_engineering',
                    'pipeline_features': pipeline_features.shape[1] if pipeline_features is not None else 0,
                    'total_features': len(complete_features),
                    'target_value_used': complete_features.get('project_prf_normalised_work_effort'),
                    'validation': validation,
                    'summary': summary,
                    'error': None
                }
            else:
                return {
                    'success': True,
                    'prediction': prediction,
                    'prediction_days': prediction / 8,
                    'method_used': 'sequential_pipeline_engineering',
                    'error': 'Feature details unavailable but prediction succeeded'
                }
        else:
            return {
                'success': False,
                'error': "Sequential prediction failed at one of the steps",
                'prediction': None,
                'method_used': 'sequential_pipeline_engineering'
            }
    
    except Exception as e:
        logging.error(f"UI prediction with sequential approach failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'prediction': None,
            'method_used': 'sequential_pipeline_engineering'
        }


# === BACKWARD COMPATIBILITY ===

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
        return predict_with_sequential_approach(features, model_name)
    
    # Fallback to existing pipeline approach for array/list input
    else:
        return predict_with_sequential_approach(
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


# === TESTING FUNCTIONS ===

def test_sequential_approach():
    """
    Test the complete sequential approach: Pipeline → Feature Engineering → Prediction
    """
    
    print("=== TESTING SEQUENTIAL APPROACH ===")
    print("Pipeline → Feature Engineering → Prediction")
    
    # Sample UI features
    ui_features = {
        "project_prf_year_of_project": 2024,
        "external_eef_industry_sector": "Technology",
        "external_eef_organisation_type": "Large Enterprise", 
        "project_prf_max_team_size": 8,
        "project_prf_relative_size": "Medium",
        "project_prf_application_type": "Web Application",
        "project_prf_development_type": "New Development",
        "tech_tf_architecture": "Microservices",
        "tech_tf_development_platform": "Cloud",
        "tech_tf_language_type": "Object-Oriented",
        "tech_tf_primary_programming_language": "Python"
    }
    
    print("Input UI features:")
    for key, value in ui_features.items():
        print(f"  {key}: {value}")
    
    # Test each step individually
    print(f"\n{'='*60}")
    print("STEP-BY-STEP TESTING")
    print(f"{'='*60}")
    
    # Step 1: Pipeline transformation
    print("\n1. PIPELINE TRANSFORMATION")
    print("-" * 30)
    pipeline_features = apply_pipeline_transformation(ui_features)
    
    if pipeline_features is not None:
        print(f"✅ Pipeline successful: {pipeline_features.shape}")
        print(f"   Columns: {list(pipeline_features.columns)[:5]}... ({len(pipeline_features.columns)} total)")
    else:
        print("❌ Pipeline failed or not available")
    
    # Step 2: Feature engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 30)
    complete_features = apply_feature_engineering(ui_features, pipeline_features)
    
    if complete_features:
        print(f"✅ Feature engineering successful: {len(complete_features)} features")
        
        # Show key features
        key_features = [
            'project_prf_normalised_work_effort',
            'project_prf_max_team_size', 
            'external_eef_industry_sector',
            'tech_tf_primary_programming_language'
        ]
        
        print("   Key features created:")
        for key in key_features:
            if key in complete_features:
                print(f"     {key}: {complete_features[key]}")
    else:
        print("❌ Feature engineering failed")
    
    # Step 3: Model prediction
    print("\n3. MODEL PREDICTION")
    print("-" * 30)
    
    available_models = list_available_models()
    if available_models:
        model_name = available_models[0]['technical_name']
        print(f"Testing with model: {model_name}")
        
        if complete_features:
            prediction = make_model_prediction(complete_features, model_name)
            
            if prediction:
                print(f"✅ Prediction successful: {prediction:.1f} hours")
                print(f"   That's {prediction/8:.1f} work days")
                print(f"   Per person: {prediction/ui_features['project_prf_max_team_size']:.1f} hours")
            else:
                print("❌ Model prediction failed")
        else:
            print("❌ Cannot test prediction - no complete features")
    else:
        print("❌ No models available for testing")
    
    # Full sequential test
    print(f"\n{'='*60}")
    print("FULL SEQUENTIAL TEST")
    print(f"{'='*60}")
    
    if available_models:
        result = predict_from_ui_features_sequential(ui_features, model_name)
        
        if result['success']:
            print(f"🎉 SEQUENTIAL APPROACH SUCCESS!")
            print(f"   Prediction: {result['prediction']:.1f} hours")
            print(f"   Work days: {result['prediction_days']:.1f}")
            print(f"   Pipeline features: {result.get('pipeline_features', 'Unknown')}")
            print(f"   Total features: {result.get('total_features', 'Unknown')}")
            print(f"   Target value: {result.get('target_value_used', 'Unknown')}")
        else:
            print(f"❌ SEQUENTIAL APPROACH FAILED: {result['error']}")
    
    print(f"\n{'='*60}")
    print("SEQUENTIAL APPROACH TEST COMPLETE")
    print(f"{'='*60}")


# Update the main function
if __name__ == "__main__":
    print("=== SEQUENTIAL PIPELINE + FEATURE ENGINEERING TESTING ===")
    test_sequential_approach()

# === Export list for easier imports ===
__all__ = [
    # Model management
    'diagnose_model_file',
    'create_test_model_file',
    'fix_model_loading_issues',

    # Sequential pipeline + feature engineering
    'predict_with_sequential_approach',
    'predict_from_ui_features_sequential',
    'apply_pipeline_transformation',
    'apply_feature_engineering',
    'make_model_prediction',
    'test_sequential_approach'
]
