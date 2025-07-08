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
import joblib
import numpy as np
import pandas as pd
import logging
import re
import yaml
from typing import Dict, List, Optional, Union, Any
from constants import FileConstants, ModelConstants, DataConstants, PipelineConstants
from config_loader import ConfigLoader
from model_display_names import ModelDisplayNameManager

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



# Load merged configuration
app_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.UI_INFO_FILE)
APP_CONFIG = ConfigLoader.load_yaml_config(app_config_path)
if APP_CONFIG is None:
    APP_CONFIG = {}

FIELDS = APP_CONFIG.get('fields', {})
TAB_ORG = APP_CONFIG.get('tab_organization', {})

# Create global instance (or pass around as needed)
_display_name_manager = ModelDisplayNameManager()

# Add this debug line temporarily:
print(f"DEBUG: ModelDisplayNameManager initialized with {len(_display_name_manager.display_names)} display names")

# --- Display Name and Feature Name Helpers (GROUPED TOGETHER) ---

# ---- Feature Name Helpers ----
def get_numeric_features():
    """Get list of numeric feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'numeric']

def get_categorical_features():
    """Get list of categorical feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'categorical']

def get_boolean_features():
    """Get list of boolean feature names from configuration"""
    return [f for f, meta in FIELDS.items() if meta.get('type') == 'boolean']

def get_model_display_name(model_filename: str) -> str:
    """Wrapper for backward compatibility with ui.py"""
    return _display_name_manager.get_display_name(model_filename)

def get_model_display_name_from_config(model_filename: str, display_names_map: Optional[Dict[str, str]] = None) -> str:
    """Wrapper for backward compatibility"""
    return _display_name_manager.get_display_name(model_filename)

def get_all_model_display_names() -> Dict[str, str]:
    """Wrapper for backward compatibility"""
    # Get all available models first
    available_models = list_available_models()
    technical_names = [model['technical_name'] for model in available_models]
    return _display_name_manager.get_all_display_names(technical_names)

def load_model_display_names() -> Dict[str, str]:
    """
    Wrapper function for backward compatibility with ui.py
    Uses the new ModelDisplayNameManager internally
    """
    return _display_name_manager.display_names

def save_model_display_names(display_names: Dict[str, str]) -> bool:
    """Wrapper for backward compatibility"""
    return _display_name_manager.save_display_names(display_names)

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
        if hasattr(model, 'feature_names_'):
            return list(model.feature_names_)
    except Exception as e:
        logging.warning(f"Could not extract feature names from model: {e}")
    return []

def get_expected_feature_names_from_model(model) -> List[str]:
    """
    Get expected feature names from a trained model.
    This is an alias for consistency with UI imports.
    """
    return get_model_expected_features(model)

def get_expected_feature_names_from_config() -> List[str]:
    """
    Get expected feature names in a consistent order from configuration.
    Preserves order: numeric > categorical > boolean > rest
    """
    # These would be defined by your config loading
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



def load_preprocessing_pipeline(pipeline_name: str = FileConstants.PIPELINE_MODEL_FILE) -> Optional[Any]:
    """Load preprocessing pipeline WITHOUT global caching to avoid import-time recursion"""
    try:
        # FIX: Don't add extra .pkl - the filename already includes it
        pipeline_path = os.path.join(FileConstants.CONFIG_FOLDER, pipeline_name)
        
        logging.info(f"Attempting to load pipeline from: {pipeline_path}")
        
        if not os.path.exists(pipeline_path):
            logging.warning(f"Pipeline file not found: {pipeline_path}")
            return None
        
        # Load the PyCaret pipeline
        pipeline = joblib.load(pipeline_path)
        logging.info(f"Pipeline loaded successfully from {pipeline_path}")
        logging.info(f"Pipeline type: {type(pipeline).__name__}")
        
        # Basic validation for PyCaret pipeline
        if hasattr(pipeline, 'steps'):
            logging.info(f"Pipeline has {len(pipeline.steps)} steps")
        
        if hasattr(pipeline, 'feature_names_in_'):
            logging.info(f"Pipeline expects {len(pipeline.feature_names_in_)} input features")
        
        return pipeline
        
    except Exception as e:
        logging.error(f"Error loading pipeline: {e}")
        return None    

def transform_with_pipeline(ui_inputs: Dict[str, Any], pipeline=None) -> Optional[pd.DataFrame]:
    """
    Transform UI inputs using the sequential pipeline approach
    """
    try:
        # Load pipeline if not provided
        if pipeline is None:
            pipeline = load_preprocessing_pipeline()
            if pipeline is None:
                logging.warning("Pipeline not available, returning None")
                return None
        
        # Use the sequential approach
        result = prepare_features_for_model(ui_inputs)
        
        if result is not None:
            logging.info(f"Sequential pipeline transformation successful: {result.shape}")
            return result
        else:
            logging.warning("Sequential pipeline transformation returned None")
            return None
        
    except Exception as e:
        logging.error(f"Pipeline transformation failed: {e}")
        return None

def get_pipeline_background_data(n_samples: int = 100) -> np.ndarray:
    """Get background data transformed through the pipeline"""
    try:
        pipeline = load_preprocessing_pipeline()
        if pipeline is None:
            return prepare_isbsg_sample_data(n_samples)
        
        # Load raw ISBSG data
        isbsg_df = pd.read_csv(FileConstants.ISBSG_PREPROCESSED_FILE)
        
        # Sample if needed
        if len(isbsg_df) > n_samples:
            isbsg_df = isbsg_df.sample(n=n_samples, random_state=42)
        
        # Transform through pipeline
        background = pipeline.transform(isbsg_df)
        
        logging.info(f"Pipeline background data shape: {background.shape}")
        return background
        
    except Exception as e:
        logging.error(f"Error getting pipeline background: {e}")
        return prepare_isbsg_sample_data(n_samples)

# ---- Display Name Helpers ----






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
    if not os.path.exists(FileConstants.MODELS_FOLDER):
        os.makedirs(FileConstants.MODELS_FOLDER)

def list_available_models() -> list:
    """
    Lists all available model files in the models directory (without extension).
    Returns a list of dicts with 'technical_name' and 'display_name'.
    """
    ensure_models_folder()
    model_files = []
    
    # Add debug
    print(f"DEBUG: Looking for models in: {FileConstants.MODELS_FOLDER}")
    print(f"DEBUG: Folder exists: {os.path.exists(FileConstants.MODELS_FOLDER)}")
    
    if os.path.exists(FileConstants.MODELS_FOLDER):
        all_files = os.listdir(FileConstants.MODELS_FOLDER)
        print(f"DEBUG: All files in models folder: {all_files}")
        
        for f in all_files:
            if f.endswith('.pkl') and not ('scaler' in f.lower()) and not ('pipeline' in f.lower()):
                technical_name = os.path.splitext(f)[0]
                print(f"DEBUG: Processing model file: {f} -> {technical_name}")
                
                # Use the correct method that exists
                display_name = _display_name_manager.get_display_name(technical_name)
                
                model_files.append({
                    'technical_name': technical_name,
                    'display_name': display_name
                })
                print(f"DEBUG: Added model: {technical_name} -> {display_name}")
    
    # Use the correct method for sorting
    model_files.sort(key=lambda x: _display_name_manager._extract_model_number(x['technical_name']))
    
    print(f"DEBUG: Final model list: {len(model_files)} models found")
    return model_files




def check_required_models() -> dict:
    """
    Checks for the presence of model files in the models folder.
    Returns a dictionary summarizing their availability and listing the models.
    """
    ensure_models_folder()
    existing_files = os.listdir(FileConstants.MODELS_FOLDER)
    existing_models = [f for f in existing_files if f.endswith('.pkl')]
    
    # Filter out scaler and pipeline files
    model_files = [f for f in existing_models if not ('scaler' in f.lower()) and not ('pipeline' in f.lower())]
    has_models = len(model_files) > 0
    
    found_models = []
    for f in model_files:
        technical_name = os.path.splitext(f)[0]
        # Use the correct method
        display_name = _display_name_manager.get_display_name(technical_name)
        found_models.append({
            'technical_name': technical_name,
            'display_name': display_name,
            'file_path': os.path.join(FileConstants.MODELS_FOLDER, f)
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
    model_path = os.path.join(FileConstants.MODELS_FOLDER, model_name)
    
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
    Enhanced feature preparation using sequential pipeline approach:
    1. UI features (22) → Custom pipeline.py → Processed features
    2. Processed features → PyCaret pipeline → Model-ready features (54-67)
    
    Fallback: If PyCaret pipeline fails, use only custom pipeline output
    Args:
        ui_features: Dictionary of user input features from the UI
        
    Returns:
        pd.DataFrame: Prepared features ready for model prediction
        
    Raises:
        Exception: If all feature preparation methods fail
    """
    
    try:
        logging.info(f"Starting sequential feature preparation for {len(ui_features)} input features")
        
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
        
        # === STEP 1: CUSTOM PIPELINE TRANSFORMATION ===
        try:
            from pipeline import process_features_for_prediction
            
            logging.info("STEP 1: Applying custom pipeline transformation...")
            custom_processed_features = process_features_for_prediction(clean_features)
            
            if custom_processed_features is not None and not custom_processed_features.empty:
                logging.info(f"Custom pipeline successful: {custom_processed_features.shape}")
                logging.info(f"Custom pipeline features: {list(custom_processed_features.columns)[:10]}...")  # Show first 10
            else:
                raise Exception("Custom pipeline returned None or empty DataFrame")
                
        except Exception as e:
            logging.error(f"Custom pipeline failed: {e}")
            logging.info("Falling back to traditional feature preparation...")
            
            # Fallback to traditional method
            return prepare_features_manually_from_config(clean_features)
        
        # === STEP 2: PYCARET PIPELINE TRANSFORMATION ===
        try:
            logging.info("STEP 2: Attempting PyCaret pipeline transformation...")
            
            # Load PyCaret pipeline
            pycaret_pipeline = load_preprocessing_pipeline()
            
            if pycaret_pipeline is not None:
                logging.info(f"PyCaret pipeline loaded successfully")
                logging.info(f"Pipeline expects {len(pycaret_pipeline.feature_names_in_)} input features")
                
                # Apply PyCaret pipeline transformation
                pycaret_processed_features = pycaret_pipeline.transform(custom_processed_features)
                
                # Convert to DataFrame if it's numpy array
                if isinstance(pycaret_processed_features, np.ndarray):
                    # Get feature names from pipeline if available
                    feature_names = None
                    if hasattr(pycaret_pipeline, 'get_feature_names_out'):
                        try:
                            feature_names = pycaret_pipeline.get_feature_names_out()
                        except:
                            pass
                    elif hasattr(pycaret_pipeline, 'feature_names_'):
                        feature_names = pycaret_pipeline.feature_names_
                    
                    if feature_names is None:
                        # Use generic names
                        feature_names = [f"feature_{i}" for i in range(pycaret_processed_features.shape[1])]
                    
                    pycaret_processed_features = pd.DataFrame(
                        pycaret_processed_features, 
                        columns=feature_names
                    )
                
                logging.info(f"PyCaret pipeline successful: {pycaret_processed_features.shape}")
                
                # Final cleanup
                final_features = pycaret_processed_features.copy()
                
                # Ensure all values are numeric
                final_features = final_features.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Remove any target/label columns
                target_keywords = ['target', 'effort', 'label', 'prediction', 'actual', 'ground_truth']
                cols_to_drop = [col for col in final_features.columns 
                               if any(keyword in col.lower() for keyword in target_keywords)]
                if cols_to_drop:
                    final_features = final_features.drop(columns=cols_to_drop, errors='ignore')
                    logging.info(f"Removed target columns: {cols_to_drop}")
                
                # Final validation
                if final_features.empty or final_features.shape[1] == 0:
                    raise ValueError("No features remaining after PyCaret pipeline processing")
                
                # Check for infinite or extremely large values
                final_features = final_features.replace([np.inf, -np.inf], 0)
                
                logging.info(f"Sequential pipeline complete:")
                logging.info(f"   - UI features: {len(clean_features)}")
                logging.info(f"   - Custom pipeline: {custom_processed_features.shape}")
                logging.info(f"   - PyCaret pipeline: {final_features.shape}")
                logging.info(f"   - Final features: {list(final_features.columns)[:5]}...")  # Show first 5
                
                return final_features
                
            else:
                logging.warning("PyCaret pipeline not available")
                raise Exception("PyCaret pipeline could not be loaded")
                
        except Exception as e:
            logging.warning(f"PyCaret pipeline failed: {e}")
            logging.info("Using custom pipeline output as fallback...")
            
            # === FALLBACK: USE CUSTOM PIPELINE OUTPUT ===
            fallback_features = custom_processed_features.copy()
            
            # Basic cleanup for fallback
            fallback_features = fallback_features.apply(pd.to_numeric, errors='coerce').fillna(0)
            fallback_features = fallback_features.replace([np.inf, -np.inf], 0)
            
            # Remove target columns if any
            target_keywords = ['target', 'effort', 'label', 'prediction']
            cols_to_drop = [col for col in fallback_features.columns 
                           if any(keyword in col.lower() for keyword in target_keywords)]
            if cols_to_drop:
                fallback_features = fallback_features.drop(columns=cols_to_drop, errors='ignore')
                logging.info(f"Removed target columns from fallback: {cols_to_drop}")
            
            logging.info(f"Fallback pipeline complete:")
            logging.info(f"   - Final shape: {fallback_features.shape}")
            logging.info(f"   - Features: {list(fallback_features.columns)[:5]}...")
            
            return fallback_features
    
    except Exception as e:
        logging.error(f"Complete sequential pipeline failed: {e}")
        
        # Last resort: manual feature preparation
        try:
            logging.warning("Attempting emergency feature preparation...")
            ui_keys_to_remove = {
                'selected_model', 'selected_models', 'submit', 'clear_results', 
                'show_history', 'save_config', 'config_name', 'comparison_mode'
            }
            clean_features = {k: v for k, v in ui_features.items() if k not in ui_keys_to_remove}

            expected_features = get_expected_feature_names_from_config()
            feature_vector = create_feature_vector_from_dict(clean_features, expected_features)
            emergency_df = pd.DataFrame([feature_vector], columns=expected_features)
            logging.warning(f"Emergency preparation successful: {emergency_df.shape}")
            return emergency_df
        except Exception as emergency_e:
            logging.error(f"Emergency feature preparation also failed: {emergency_e}")
            raise Exception(f"All feature preparation methods failed. Sequential error: {e}, Emergency error: {emergency_e}")




# Additional helper function to check which method is being used
def get_feature_preparation_method() -> str:
    """
    Check which feature preparation method is available.
    Useful for debugging and UI display.
    
    Returns:
        str: 'pipeline' if pipeline.pkl is available, 'traditional' otherwise
    """
    try:
        pipeline = load_preprocessing_pipeline()
        if pipeline is not None:
            return 'pipeline'
    except:
        pass
    return 'traditional'


# Optional: Add configuration to control pipeline usage
def prepare_features_for_model_with_config(
    ui_features: Dict[str, Any],
    use_pipeline: bool = True,
    force_traditional: bool = False
) -> pd.DataFrame:
    """
    Feature preparation with explicit control over which method to use.
    
    Args:
        ui_features: Dictionary of user input features
        use_pipeline: Whether to try pipeline.pkl approach (default: True)
        force_traditional: Force use of traditional method even if pipeline exists
        
    Returns:
        pd.DataFrame: Prepared features
    """
    if force_traditional:
        # Skip pipeline and use traditional method
        logging.info("Forced to use traditional feature preparation")
        # Create a copy of prepare_features_for_model without pipeline section
        # ... (traditional logic only)
    elif use_pipeline:
        return prepare_features_for_model(ui_features)
    else:
        # Similar to force_traditional
        pass

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
            
            print(f"DEBUG: Prepared features shape: {features_df.shape}")
            print(f"DEBUG: Feature sample values: {features_df.iloc[0].head(5).to_dict()}")
            
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
            
            print(f"DEBUG: Aligned features shape: {features_aligned.shape}")
            print(f"DEBUG: Aligned feature sample: {features_aligned.iloc[0].head(5).to_dict()}")
            
            # Step 4: Make prediction
            try:
                # Try PyCaret prediction first
                from pycaret.regression import predict_model
                preds = predict_model(model, data=features_aligned)
                
                print(f"DEBUG: Raw PyCaret preds shape: {preds.shape}")
                print(f"DEBUG: Raw PyCaret preds columns: {list(preds.columns)}")
                
                # Look for prediction column with common names
                for col in ['prediction_label', 'Label', 'pred', 'prediction']:
                    if col in preds.columns:
                        raw_result = preds[col].iloc[0]
                        result = float(raw_result)
                        print(f"DEBUG: Found prediction in column '{col}': raw={raw_result}, float={result}")
                        logging.info(f"Prediction successful: {result}")
                        return result
                
                # Fallback to last column
                raw_result = preds.iloc[0, -1]
                result = float(raw_result)
                print(f"DEBUG: Using last column: raw={raw_result}, float={result}")
                logging.info(f"Prediction successful (last column): {result}")
                return result
                
            except Exception as e:
                logging.warning(f"PyCaret prediction failed, trying direct model prediction: {e}")
                
                # Fallback to direct model prediction
                if hasattr(model, 'predict'):
                    pred = model.predict(features_aligned)
                    print(f"DEBUG: Direct model predict output: {pred}")
                    result = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                    print(f"DEBUG: Direct prediction result: {result}")
                    logging.info(f"Direct prediction successful: {result}")
                    return result
                
        else:
            # Handle array/list input (backward compatibility)
            logging.warning("Array/list input not fully supported in sequential approach")
            return None
            
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        print(f"DEBUG: Exception in prediction: {e}")

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

def get_trained_model(model_name: str) -> Optional[Any]:
    """
    Get the actual trained model object for SHAP analysis.
    This function returns the underlying model, not wrapped in PyCaret.
    
    Args:
        model_name: Technical name of the model
        
    Returns:
        The actual model object or None if loading fails
    """
    try:
        # First load the model using existing function
        model = load_model(model_name)
        if model is None:
            logging.error(f"Could not load model: {model_name}")
            return None
        
        # Extract the actual estimator from PyCaret model
        if hasattr(model, '_final_estimator'):
            # PyCaret model - get the final estimator
            actual_model = model._final_estimator
            logging.info(f"Extracted final estimator from PyCaret model: {type(actual_model)}")
            return actual_model
            
        elif hasattr(model, 'named_steps'):
            # Pipeline model - get the final step
            step_names = list(model.named_steps.keys())
            if step_names:
                final_step = model.named_steps[step_names[-1]]
                logging.info(f"Extracted final step from pipeline: {type(final_step)}")
                return final_step
            
        elif hasattr(model, 'steps'):
            # Another type of pipeline
            if len(model.steps) > 0:
                final_step = model.steps[-1][1]  # Get the estimator from (name, estimator) tuple
                logging.info(f"Extracted final step from steps pipeline: {type(final_step)}")
                return final_step
        
        # If none of the above, assume it's already the actual model
        logging.info(f"Using model directly: {type(model)}")
        return model
        
    except Exception as e:
        logging.error(f"Error getting trained model {model_name}: {e}")
        return None

def prepare_input_data(user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Prepare user input data for SHAP analysis.
    Converts user inputs to the format expected by the model.
    
    Args:
        user_inputs: Dictionary of user inputs from UI
        
    Returns:
        numpy array ready for model prediction, or None if preparation fails
    """
    try:
        if not user_inputs:
            logging.error("No user inputs provided")
            return None
        
        logging.info(f"Preparing input data from {len(user_inputs)} user inputs")
        
        # Use the existing feature preparation pipeline
        features_df = prepare_features_for_model(user_inputs)
        if features_df is None or features_df.empty:
            logging.error("Feature preparation failed")
            return None
        
        # Convert to numpy array
        input_array = features_df.values
        
        # Ensure it's a 2D array with single row
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        
        logging.info(f"Input data prepared successfully: shape {input_array.shape}")
        return input_array
        
    except Exception as e:
        logging.error(f"Error preparing input data: {e}")
        return None

def prepare_sample_data(n_samples: int = 100, use_training_data: bool = True) -> Optional[np.ndarray]:
    """
    Prepare sample data for SHAP baseline/background.
    Uses actual ISBSG training dataset for more realistic baseline.
    
    Args:
        n_samples: Number of sample instances to return
        use_training_data: Whether to use training data (True) or generate synthetic (False)
        
    Returns:
        numpy array with sample data, or None if loading fails
    """
    
    if use_training_data:
        # Try ISBSG dataset first
        isbsg_data = prepare_isbsg_sample_data(n_samples)
        if isbsg_data is not None:
            return isbsg_data
        
        # Fallback to general training data loading
        return prepare_sample_data_from_training_csv(n_samples)
    else:
        return prepare_sample_data_synthetic(n_samples)

def prepare_sample_data_from_training_csv(n_samples: int = 100) -> Optional[np.ndarray]:
    """
    Load sample data from the original training CSV file.
    This provides the most realistic baseline for SHAP analysis.
    """
    try:
        # Define possible training data file locations (including your specific file)
        possible_paths = [
            'data/synthetic_isbsg2016r1_1_finance_sdv_generated.csv',  # training data file
        ]
        
        training_data_path = None
        
        # Find the training data file
        for path in possible_paths:
            if os.path.exists(path):
                training_data_path = path
                logging.info(f"Found training data at: {path}")
                break
        
        if training_data_path is None:
            logging.warning("Training data CSV file not found. Falling back to synthetic data.")
            return prepare_sample_data_synthetic(n_samples)
        
        # Load the training data
        logging.info(f"Loading training data from: {training_data_path}")
        df = pd.read_csv(training_data_path)
        
        # Remove target/label columns specific to your dataset
        target_columns = [
            # Common effort/prediction columns
            'effort', 'hours', 'man_hours', 'total_effort', 'prediction', 
            'target', 'label', 'actual_effort', 'estimated_effort',
            # Your specific target columns (effort-related)
            'project_prf_normalised_work_effort_level_1',
            'project_prf_normalised_work_effort',
            'project_prf_normalised_level_1_pdr_ufp',
            'project_prf_normalised_pdr_ufp',
            'project_prf_speed_of_delivery',
            'project_prf_project_elapsed_time',
            # ISBSG ID (not a feature)
            'isbsg_project_id'
        ]
        
        # Case-insensitive removal of target columns
        columns_to_drop = []
        for col in df.columns:
            if any(target_col.lower() in col.lower() for target_col in target_columns):
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            logging.info(f"Removed target columns: {columns_to_drop}")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Sample the requested number of rows
        if len(df) > n_samples:
            # Random sampling for diversity
            sample_df = df.sample(n=n_samples, random_state=42)
            logging.info(f"Sampled {n_samples} rows from {len(df)} total rows")
        else:
            sample_df = df.copy()
            logging.info(f"Using all {len(df)} rows (less than requested {n_samples})")
        
        # Convert to the expected feature format
        expected_features = get_expected_feature_names_from_config()
        
        # Process each row through the feature preparation pipeline
        processed_samples = []
        
        for idx, row in sample_df.iterrows():
            try:
                # Convert row to dictionary format expected by prepare_features_for_model
                row_dict = row.to_dict()
                
                # Process through the same pipeline as user inputs
                processed_features = prepare_features_for_model(row_dict)
                
                if processed_features is not None and not processed_features.empty:
                    processed_samples.append(processed_features.values.flatten())
                else:
                    # Fallback: create feature vector directly
                    feature_vector = create_feature_vector_from_dict(row_dict, expected_features)
                    processed_samples.append(feature_vector)
                    
            except Exception as e:
                logging.debug(f"Failed to process training sample {idx}: {e}")
                # Skip this sample
                continue
        
        if not processed_samples:
            logging.error("No training samples could be processed")
            return prepare_sample_data_synthetic(n_samples)
        
        # Convert to numpy array
        sample_array = np.array(processed_samples)
        
        # Ensure proper shape
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)
        
        logging.info(f"Training data sample prepared successfully: shape {sample_array.shape}")
        return sample_array
        
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        logging.info("Falling back to synthetic sample data")
        return prepare_sample_data_synthetic(n_samples)

def prepare_sample_data_synthetic(n_samples: int = 100) -> Optional[np.ndarray]:
    """
    Fallback function to generate synthetic sample data.
    Used when training data is not available.
    """
    try:
        logging.info(f"Generating {n_samples} synthetic sample data points")
        
        expected_features = get_expected_feature_names_from_config()
        n_features = len(expected_features)
        
        if n_features == 0:
            logging.error("No features found in configuration")
            return None
        
        # Generate random data scaled to reasonable ranges
        sample_data = np.random.rand(n_samples, n_features)
        
        # Scale features based on their types from FIELDS
        for i, feature_name in enumerate(expected_features):
            feature_config = FIELDS.get(feature_name, {})
            feature_type = feature_config.get('type', 'numeric')
            
            if feature_type == 'numeric':
                min_val = feature_config.get('min', 1)
                max_val = feature_config.get('max', 100)
                sample_data[:, i] = sample_data[:, i] * (max_val - min_val) + min_val
            else:
                # For categorical/boolean, convert to small integers
                sample_data[:, i] = (sample_data[:, i] * 5).astype(int)
        
        logging.info(f"Synthetic sample data generated: shape {sample_data.shape}")
        return sample_data
        
    except Exception as e:
        logging.error(f"Error generating synthetic sample data: {e}")
        return None

def get_training_data_info() -> Dict[str, Any]:
    """
    Get information about available training data for SHAP analysis.
    """
    try:
        possible_paths = [
            'data/synthetic_isbsg2016r1_1_finance_sdv_generated.csv',  # Training dataset specific file
        ]
        
        info = {
            'training_data_available': False,
            'file_path': None,
            'num_rows': 0,
            'num_features': 0,
            'file_size_mb': 0
        }
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, nrows=1)  # Just read header
                    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                    
                    # Get full row count
                    full_df = pd.read_csv(path)
                    
                    info.update({
                        'training_data_available': True,
                        'file_path': path,
                        'num_rows': len(full_df),
                        'num_features': len(full_df.columns),
                        'file_size_mb': round(file_size, 2)
                    })
                    break
                    
                except Exception as e:
                    logging.debug(f"Could not read {path}: {e}")
                    continue
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting training data info: {e}")
        return {'error': str(e)}

def prepare_isbsg_sample_data(n_samples: int = 100) -> Optional[np.ndarray]:
    """Fixed ISBSG sample data preparation"""
    try:
        # FIXED: Use the correct file path from discovery
        file_path = 'data/synthetic_isbsg2016r1_1_finance_sdv_generated.csv'
        
        if not os.path.exists(file_path):
            logging.error(f"ISBSG dataset not found at: {file_path}")
            return None
        
        logging.info(f"Loading ISBSG data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Remove target columns as identified in discovery
        target_cols = [
            'project_prf_normalised_work_effort_level_1', 
            'project_prf_normalised_work_effort', 
            'project_prf_speed_of_delivery'
        ]
        df = df.drop(columns=target_cols, errors='ignore')
        
        # Sample and convert to numpy
        if len(df) > n_samples:
            sample_df = df.sample(n=n_samples, random_state=42)
        else:
            sample_df = df.copy()
        
        # Handle missing values and convert to numeric
        sample_array = sample_df.fillna(0).select_dtypes(include=[np.number]).values.astype(np.float32)
        
        logging.info(f"ISBSG sample prepared: {sample_array.shape}")
        return sample_array
        
    except Exception as e:
        logging.error(f"Error preparing ISBSG sample data: {e}")
        return None

def apply_pycaret_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing that was used in your PyCaret training.
    This ensures SHAP background data matches your model's training data exactly.
    """
    try:
        # Make column names lowercase (matching your PyCaret setup)
        df.columns = [col.lower() for col in df.columns]
        
        # Apply mixed-type column fixes (same as your PyCaret setup)
        df_processed = fix_mixed_type_columns_simple(df)
        
        # Remove target and ignore columns (same as your PyCaret setup)
        ignore_cols = [
            'isbsg_project_id', 'external_eef_data_quality_rating', 'external_eef_data_quality_rating_b', 
            'project_prf_normalised_work_effort_level_1', 'project_prf_normalised_level_1_pdr_ufp', 
            'project_prf_normalised_pdr_ufp', 'project_prf_project_elapsed_time', 
            'people_prf_ba_team_experience_less_than_1_yr', 'people_prf_ba_team_experience_1_to_3_yr', 
            'people_prf_ba_team_experience_great_than_3_yr', 'people_prf_it_experience_less_than_1_yr', 
            'people_prf_it_experience_1_to_3_yr', 'people_prf_it_experience_great_than_3_yr', 
            'people_prf_it_experience_less_than_3_yr', 'people_prf_it_experience_3_to_9_yr', 
            'people_prf_it_experience_great_than_9_yr', 'people_prf_project_manage_experience', 
            'project_prf_total_project_cost', 'project_prf_cost_currency', 'project_prf_currency_multiple', 
            'project_prf_speed_of_delivery', 'people_prf_project_manage_changes', 
            'project_prf_defect_density', 'project_prf_manpower_delivery_rate'
        ]
        
        # Convert to lowercase and remove target + ignore columns
        ignore_cols = [col.lower() for col in ignore_cols]
        target_col = 'project_prf_normalised_work_effort'
        
        # Remove target and ignore columns
        cols_to_drop = [target_col] + [col for col in ignore_cols if col in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
        
        logging.info(f"Removed {len(cols_to_drop)} columns (target + ignore)")
        
        # Handle missing values (same as your PyCaret setup)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns with mean
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                if len(mode_val) > 0:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                else:
                    df_processed[col] = df_processed[col].fillna('missing')
        
        # Encode categorical variables (same as your PyCaret setup)
        if len(categorical_cols) > 0:
            df_processed = pd.get_dummies(df_processed, drop_first=True)
            logging.info(f"Applied one-hot encoding to {len(categorical_cols)} categorical columns")
        
        # Apply normalization using saved scaler (if available)
        try:
            scaler_path = os.path.join(FileConstants.MODELS_FOLDER, 'standard_scaler.pkl')
            if os.path.exists(scaler_path):
                import joblib
                scaler = joblib.load(scaler_path)
                df_processed = pd.DataFrame(
                    scaler.transform(df_processed),
                    columns=df_processed.columns,
                    index=df_processed.index
                )
                logging.info("Applied saved scaler normalization")
            else:
                logging.warning("No saved scaler found - data may not be normalized")
        except Exception as e:
            logging.warning(f"Could not apply saved scaler: {e}")
        
        logging.info(f"Final processed shape: {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return df


def fix_mixed_type_columns_simple(df):
    """
    Fix mixed type columns (same as your PyCaret setup)
    """
    df_copy = df.copy()
    
    mixed_type_cols = [
        'external_eef_industry_sector',
        'tech_tf_client_roles', 
        'tech_tf_clientserver_description',
        'tech_tf_development_platform_hand_held'
    ]
    
    for col in mixed_type_cols:
        if col in df_copy.columns:
            if col == 'tech_tf_development_platform_hand_held':
                df_copy[col] = df_copy[col].fillna(False).astype(bool)
            else:
                df_copy[col] = df_copy[col].astype(str)
                df_copy[col] = df_copy[col].replace('nan', np.nan)
    
    return df_copy


def validate_shap_sample_data(sample_data: np.ndarray) -> Dict[str, Any]:
    """
    Validate that the sample data is suitable for SHAP analysis
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    try:
        # Check basic properties
        if sample_data is None:
            validation['valid'] = False
            validation['errors'].append("Sample data is None")
            return validation
        
        if not isinstance(sample_data, np.ndarray):
            validation['valid'] = False
            validation['errors'].append("Sample data is not a numpy array")
            return validation
        
        if sample_data.size == 0:
            validation['valid'] = False
            validation['errors'].append("Sample data is empty")
            return validation
        
        # Check for problematic values
        if np.isnan(sample_data).any():
            validation['warnings'].append("Sample data contains NaN values")
        
        if np.isinf(sample_data).any():
            validation['warnings'].append("Sample data contains infinite values")
        
        # Check dimensions
        if len(sample_data.shape) != 2:
            validation['warnings'].append(f"Unexpected data shape: {sample_data.shape}")
        
        # Check feature count
        if len(sample_data.shape) == 2 and sample_data.shape[1] != 67:  # Adjust expected count
            validation['warnings'].append(f"Feature count {sample_data.shape[1]} may not match model expectations")
        
        # Data quality checks
        if sample_data.std() == 0:
            validation['warnings'].append("Sample data has zero variance")
        
        validation['shape'] = sample_data.shape
        validation['dtype'] = str(sample_data.dtype)
        validation['value_range'] = [float(sample_data.min()), float(sample_data.max())]
        
    except Exception as e:
        validation['valid'] = False
        validation['errors'].append(f"Validation error: {e}")
    
    return validation

def get_isbsg_dataset_info() -> Dict[str, Any]:
    """
    Get detailed information about your ISBSG dataset.
    """
    try:
        file_path = 'data/synthetic_isbsg2016r1_1_finance_sdv_generated.csv'
        
        if not os.path.exists(file_path):
            return {'available': False, 'error': 'ISBSG dataset not found'}
        
        # Read just the header first
        df_sample = pd.read_csv(file_path, nrows=100)
        
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Analyze column types
        feature_columns = [col for col in df_sample.columns 
                          if col not in ['isbsg_project_id', 
                                       'project_prf_normalised_work_effort_level_1',
                                       'project_prf_normalised_work_effort',
                                       'project_prf_normalised_level_1_pdr_ufp',
                                       'project_prf_normalised_pdr_ufp',
                                       'project_prf_speed_of_delivery',
                                       'project_prf_project_elapsed_time']]
        
        numeric_cols = df_sample[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        boolean_cols = [col for col in feature_columns if col not in numeric_cols]
        
        # Get full dataset info
        full_df = pd.read_csv(file_path)
        
        info = {
            'available': True,
            'file_path': file_path,
            'total_rows': len(full_df),
            'total_columns': len(full_df.columns),
            'feature_columns': len(feature_columns),
            'numeric_features': len(numeric_cols),
            'boolean_features': len(boolean_cols),
            'file_size_mb': round(file_size_mb, 2),
            'target_columns': [
                'project_prf_normalised_work_effort_level_1',
                'project_prf_normalised_work_effort',
                'project_prf_speed_of_delivery'
            ],
            'sample_statistics': {
                'functional_size_mean': df_sample['project_prf_functional_size'].mean() if 'project_prf_functional_size' in df_sample.columns else None,
                'team_size_mean': df_sample['project_prf_max_team_size'].mean() if 'project_prf_max_team_size' in df_sample.columns else None,
                'missing_values': df_sample.isnull().sum().sum()
            }
        }
        
        return info
        
    except Exception as e:
        return {'available': False, 'error': str(e)}


def set_training_data_path(file_path: str) -> bool:
    """
    Set a custom path for training data.
    You can call this function to specify where your training data is located.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Training data file not found: {file_path}")
            return False
        
        # Test if we can read the file
        test_df = pd.read_csv(file_path, nrows=5)
        logging.info(f"Training data path set successfully: {file_path}")
        logging.info(f"Training data shape: {test_df.shape}")
        
        # You could store this path in a configuration file or global variable
        # For now, we'll just validate it works
        return True
        
    except Exception as e:
        logging.error(f"Error setting training data path: {e}")
        return False
    
def validate_shap_compatibility(model_name: str) -> Dict[str, Any]:
    """
    Validate if a model is compatible with SHAP analysis.
    
    Args:
        model_name: Technical name of the model
        
    Returns:
        Dictionary with compatibility information
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
        logging.info(f"Checking SHAP compatibility for model type: {model_type}")
        
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
        elif any(linear_model in model_type for linear_model in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']):
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

# Additional utility function for SHAP feature name mapping
def get_shap_feature_names(model_name: str, user_inputs: Dict[str, Any]) -> List[str]:
    """
    Get feature names for SHAP analysis, ensuring they match the model's expected features.
    
    Args:
        model_name: Technical name of the model
        user_inputs: User input dictionary to prepare features from
        
    Returns:
        List of feature names that match the model's expectations
    """
    try:
        # Get model expected features
        model = get_trained_model(model_name)
        if model is None:
            logging.error("Could not load model for feature name extraction")
            return get_expected_feature_names_from_config()
        
        model_features = get_model_expected_features(model)
        if model_features:
            logging.info(f"Using model-specific feature names: {len(model_features)} features")
            return model_features
        
        # Fallback to configuration-based features
        config_features = get_expected_feature_names_from_config()
        logging.info(f"Using configuration-based feature names: {len(config_features)} features")
        return config_features
        
    except Exception as e:
        logging.error(f"Error getting SHAP feature names: {e}")
        return get_expected_feature_names_from_config()

# NEW FUNCTION for testing: Test the sequential pipeline
def test_sequential_pipeline(ui_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test function to validate the sequential pipeline approach
    """
    test_result = {
        'success': False,
        'custom_pipeline': {'success': False, 'shape': None, 'error': None},
        'pycaret_pipeline': {'success': False, 'shape': None, 'error': None},
        'final_result': {'shape': None, 'features': None}
    }
    
    try:
        # Test custom pipeline
        try:
            from pipeline import process_features_for_prediction
            custom_result = process_features_for_prediction(ui_features)
            test_result['custom_pipeline']['success'] = True
            test_result['custom_pipeline']['shape'] = custom_result.shape
        except Exception as e:
            test_result['custom_pipeline']['error'] = str(e)
        
        # Test PyCaret pipeline
        try:
            if test_result['custom_pipeline']['success']:
                pycaret_pipeline = load_preprocessing_pipeline()
                if pycaret_pipeline:
                    pycaret_result = pycaret_pipeline.transform(custom_result)
                    test_result['pycaret_pipeline']['success'] = True
                    test_result['pycaret_pipeline']['shape'] = pycaret_result.shape
        except Exception as e:
            test_result['pycaret_pipeline']['error'] = str(e)
        
        # Test full sequential pipeline
        final_result = prepare_features_for_model(ui_features)
        if final_result is not None:
            test_result['success'] = True
            test_result['final_result']['shape'] = final_result.shape
            test_result['final_result']['features'] = list(final_result.columns)[:10]  # First 10 features
    
    except Exception as e:
        test_result['error'] = str(e)
    
    return test_result

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
    'validate_prepared_features',
    'get_trained_model',
    'prepare_input_data', 
    'prepare_sample_data',
    'prepare_sample_data_from_history',
    'validate_shap_compatibility',
    'get_shap_feature_names'
]