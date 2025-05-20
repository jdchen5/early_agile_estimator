#models.py
"""
- Train Your Models in Jupyter Notebook and Save the trained models 
   to the models folder with its own names ad pkl as file extension. 
   Hence the model file names can be different.
- Make Predictions: Input your project parameters, Select your 
  preferred model and Click "Predict Man-Months"
"""

import os
import pickle
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
MODELS_DIR = 'models'

def ensure_models_dir() -> str:
    """
    Ensure the models directory exists.
    
    Returns:
        str: Path to the models directory
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logging.info(f"Created models directory at '{MODELS_DIR}'")
    return MODELS_DIR

def list_available_models() -> List[str]:
    """
    List all available models in the models directory.
    
    Returns:
        List[str]: Names of available trained models (without extension)
    """
    ensure_models_dir()
    
    # Get all .pkl files except scaler.pkl
    model_files = [
        os.path.splitext(f)[0] for f in os.listdir(MODELS_DIR)
        if f.endswith('.pkl') and not f.startswith('scaler')
    ]
    
    logging.info(f"Found {len(model_files)} available models: {model_files}")
    return model_files

def check_required_models() -> Dict[str, bool]:
    """
    Check for existing models in the models directory.
    
    Returns:
        Dict[str, bool]: Dictionary with model status (True if found)
    """
    ensure_models_dir()
    
    # Get all model files
    existing_files = os.listdir(MODELS_DIR)
    existing_models = [f for f in existing_files if f.endswith('.pkl')]
    
    # Check for at least one model and optionally a scaler
    has_models = any(f for f in existing_models if not f.startswith('scaler'))
    has_scaler = any(f for f in existing_models if f.startswith('scaler'))
    
    model_status = {
        "models_available": has_models,
        "scaler_available": has_scaler,
        "found_models": [os.path.splitext(f)[0] for f in existing_models]
    }
    
    if has_models:
        logging.info(f"Found models in '{MODELS_DIR}': {existing_models}")
    else:
        logging.warning(f"No trained models found in '{MODELS_DIR}'")
        logging.info("Please train and save models via your Jupyter notebook before using this app.")
    
    return model_status

def load_model(model_name: str) -> Optional[Any]:
    """
    Load a model from the models directory.
    
    Args:
        model_name (str): Name of the model to load (without extension)
        
    Returns:
        Optional[Any]: Loaded model object or None if not found/error
    """
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_name}.pkl' not found.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {str(e)}")
        return None

def load_scaler() -> Optional[Any]:
    """
    Load the scaler if available in the models directory.
    
    Returns:
        Optional[Any]: Loaded scaler object or None if not found/error
    """
    # Look for any scaler in the models directory
    ensure_models_dir()
    scaler_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('scaler') and f.endswith('.pkl')]
    
    if not scaler_files:
        logging.info("No scaler found. Proceeding without scaling.")
        return None
    
    # Use the first scaler found
    scaler_path = os.path.join(MODELS_DIR, scaler_files[0])
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Successfully loaded scaler: {scaler_files[0]}")
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler '{scaler_files[0]}': {str(e)}")
        return None

def predict_man_months(
    features: np.ndarray, 
    model_name: str, 
    use_scaler: bool = True
) -> Optional[float]:
    """
    Make a prediction using the specified model and features.
    
    Args:
        features (np.ndarray): Array of input features
        model_name (str): Name of the model to use
        use_scaler (bool): Whether to apply scaling if available
        
    Returns:
        Optional[float]: Predicted man-months value or None if error
    """
    try:
        # Load the model
        model = load_model(model_name)
        if model is None:
            return None
            
        # Load scaler if requested and available
        scaler = load_scaler() if use_scaler else None
        
        # Apply scaling if provided
        if scaler:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Log prediction details
        logging.info(f"Prediction made with model '{model_name}', features: {features.tolist()} -> {prediction:.2f} man-months")
        
        # Ensure prediction is non-negative
        final_prediction = max(0.1, prediction)
        
        return final_prediction
    
    except Exception as e:
        logging.error(f"Error making prediction with model '{model_name}': {str(e)}")
        return None

def get_feature_importance(model_name: str) -> Optional[np.ndarray]:
    """
    Get feature importance for a given model if available.
    
    Args:
        model_name (str): Name of the model to analyze
        
    Returns:
        Optional[np.ndarray]: Feature importance values or None if not available
    """
    model = load_model(model_name)
    
    if model is None:
        return None
    
    try:
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logging.info(f"Retrieved feature importance from '{model_name}' (feature_importances_)")
            return importances
        
        # For linear models, use coefficients
        elif hasattr(model, 'coef_'):
            importances = model.coef_
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
        model_name (str): Name of the model to use
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
    
    model = load_model(model_name)
    if model is None:
        return results
        
    scaler = load_scaler() if use_scaler else None
    
    for value in param_values:
        try:
            # Create a copy of features and modify the specified parameter
            modified_features = base_features.copy()
            modified_features[param_index] = value
            
            # Make prediction
            if scaler:
                features_scaled = scaler.transform(modified_features.reshape(1, -1))
            else:
                features_scaled = modified_features.reshape(1, -1)
                
            prediction = model.predict(features_scaled)[0]
            prediction = max(0.1, prediction)  # Ensure non-negative
            
            # Add to results
            results["param_values"].append(value)
            results["predictions"].append(prediction)
            
        except Exception as e:
            logging.error(f"Error in what-if analysis for value {value}: {str(e)}")
    
    logging.info(f"What-if analysis completed for parameter {param_index} with {len(results['predictions'])} data points")
    return results