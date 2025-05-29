#models.py - PyCaret compatible module for managing trained models and predictions
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

from typing import Dict, List, Optional, Union, Any

# Paths
DATA_FOLDER = 'data'
CONFIG_FOLDER = 'config'
MODELS_FOLDER = 'models'
FEATURE_COLS_FILE = os.path.join(DATA_FOLDER, 'expected_features.txt')
MODEL_DISPLAY_NAMES_FILE = os.path.join(CONFIG_FOLDER, 'model_display_names.json')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load model display names from config file
def load_model_display_names() -> Dict[str, str]:
    if os.path.exists(MODEL_DISPLAY_NAMES_FILE):
        with open(MODEL_DISPLAY_NAMES_FILE, 'r') as f:
            return json.load(f)
    else:
        logging.warning("Model display names file not found. Using empty mapping.")
        return {}

MODEL_DISPLAY_NAMES = load_model_display_names()

def get_model_display_name(model_filename: str) -> str:
    if model_filename in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_filename]
    # ... (rest of your logic as before)
    # Try to extract algorithm name from PyCaret naming convention...
    # (No change here)
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

# --- Expected features loading ---
def get_expected_feature_names() -> List[str]:
    if os.path.exists(FEATURE_COLS_FILE):
        with open(FEATURE_COLS_FILE, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        logging.warning(f"Expected features file not found at {FEATURE_COLS_FILE}")
        return []  # Or your hardcoded fallback

# --- Helper: Ensure Models Folder Exists ---
def ensure_MODELS_FOLDER():
    """Ensure the MODELS_FOLDER directory exists."""
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        logging.info(f"Created models folder at '{MODELS_FOLDER}'")

# --- PyCaret: Safe Load Model Wrapper ---
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

# --- Prepare Features for PyCaret Model ---
def prepare_features_for_pycaret(features, model=None):
    """
    Converts input features into a pandas DataFrame with correct columns/order.
    Args:
        features: numpy array or list-like
        model: optional, can provide column names if present
    Returns:
        pd.DataFrame with proper columns/order
    """
    if isinstance(features, pd.DataFrame):
        return features
    expected_columns = get_expected_feature_names_from_model(model)
    features = np.array(features).reshape(1, -1)  # Ensure 2D
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
    ensure_MODELS_FOLDER()
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
    # ... (keep the logic as you already have, but fallback to get_expected_feature_names)
    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'X') and hasattr(model.X, 'columns'):
            return list(model.X.columns)
        elif hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
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
    features: np.ndarray, 
    model_name: str, 
    use_scaler: bool = True
) -> Optional[float]:
    """
    Make a prediction using the specified model and features, supporting PyCaret models.
    
    Args:
        features (np.ndarray): Array of input features
        model_name (str): Name of the model to use (technical name)
        use_scaler (bool): Whether to apply scaling if available
        
    Returns:
        Optional[float]: Predicted man-Hours value or None if error
    """
    try:
        # Debug logging
        logging.info(f"Starting prediction with model '{model_name}'")
        logging.info(f"Input features shape: {features.shape if hasattr(features, 'shape') else 'No shape'}")
        logging.info(f"Input features: {features}")
        
        # Load the model
        model = load_model(model_name)
        if model is None:
            logging.error(f"Failed to load model '{model_name}' for prediction")
            return None
        
        logging.info(f"Loaded model type: {type(model)}")

        # Prepare features DataFrame using CSV-defined columns
        features_df = prepare_features_for_pycaret(features, model=model)
        logging.info(f"Features DataFrame columns: {list(features_df.columns)}")
        logging.info(f"Features DataFrame shape: {features_df.shape}")
        
        # Load scaler if requested and available
        scaler = load_scaler() if use_scaler else None
        if scaler is not None and hasattr(scaler, 'transform'):
            try:
                features_scaled = scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
                logging.info("Applied scaler to features.")
            except Exception as e:
                logging.warning(f"Scaler could not be applied: {str(e)}")
                # fallback: use unscaled
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

# Additional helper function to safely check PyCaret availability
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

# Use this at the top of your models.py file:
PYCARET_AVAILABLE, predict_model = check_pycaret_safely()

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
    base_features: np.ndarray,
    model_name: str,
    param_index: int,
    param_values: List[float],
    use_scaler: bool = True
) -> Dict[str, List]:
    """
    Perform what-if analysis by varying one parameter and observing predictions.
    
    Args:
        base_features (np.ndarray): Base feature values
        model_name (str): Name of the model to use (technical name)
        param_index (int): Index of the parameter to vary
        param_values (List[float]): Values to use for the parameter
        use_scaler (bool): Whether to apply scaling if available
        
    Returns:
        Dict[str, List]: Dictionary with parameter values and predictions
    """
    results = {
        "param_values": [],
        "predictions": []
    }
    
    for value in param_values:
        try:
            # Create a copy of features and modify the specified parameter
            modified_features = base_features.copy()
            modified_features[param_index] = value
            
            # Make prediction using the main prediction function
            prediction = predict_man_hours(modified_features, model_name, use_scaler)
            
            if prediction is not None:
                results["param_values"].append(value)
                results["predictions"].append(prediction)
            
        except Exception as e:
            logging.error(f"Error in what-if analysis for value {value}: {str(e)}")
    
    logging.info(f"What-if analysis completed for parameter {param_index} with {len(results['predictions'])} data points")
    return results