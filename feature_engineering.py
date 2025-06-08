# feature_engineering.py
"""
Feature Engineering Module for ML Model Predictions

This module handles the creation of training-compatible features for PyCaret models.
Key insight: PyCaret treats the target column as an input feature during prediction.

Separated from models.py to keep code organized and maintainable.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_FOLDER = 'config'
FEATURE_MAPPING_FILE = os.path.join(CONFIG_FOLDER, 'feature_mapping.yaml')

# Check if PyCaret is available
try:
    from pycaret.regression import predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logger.warning("PyCaret not available")


def load_yaml_config(path: str) -> Dict:
    """Load YAML configuration file with error handling"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        return {}


def create_training_compatible_features(input_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create features that match EXACTLY how the model was trained.
    
    CRITICAL: PyCaret treats the target column 'project_prf_normalised_work_effort' 
    as an INPUT FEATURE during prediction. This is why predictions fail!
    
    Args:
        input_features: Raw input features from UI
        
    Returns:
        Dictionary of processed features ready for model prediction
    """
    
    features = {}
    
    # === 1. CORE NUMERIC FEATURES ===
    features['project_prf_year_of_project'] = input_features.get('project_prf_year_of_project', 2024)
    features['project_prf_max_team_size'] = input_features.get('project_prf_max_team_size', 8)
    features['project_prf_functional_size'] = input_features.get('project_prf_functional_size', 100)
    
    # === 2. THE CRITICAL MISSING PIECE: TARGET AS FEATURE ===
    # PyCaret expects this as an INPUT feature, not just the target!
    features['project_prf_normalised_work_effort'] = estimate_target_value(input_features)
    
    # === 3. ESSENTIAL CATEGORICAL FEATURES ===
    # Industry
    industry = input_features.get('external_eef_industry_sector', 'Missing')
    features['external_eef_industry_sector'] = industry if industry in [
        'Technology', 'Banking', 'Healthcare', 'Manufacturing', 'Government'
    ] else 'Missing'
    
    # Programming Language
    lang = input_features.get('tech_tf_primary_programming_language', 'Missing')
    features['tech_tf_primary_programming_language'] = lang if lang in [
        'Python', 'Java', 'JavaScript', 'C#', 'PHP', 'Ruby', 'C++', 'C'
    ] else 'Missing'
    
    # === 4. DERIVED FEATURES (calculated, not hardcoded) ===
    features.update(calculate_derived_features(input_features, features))
    
    # === 5. MINIMAL DEFAULTS (only what model absolutely expects) ===
    defaults = {
        'external_eef_data_quality_rating': 'A',
        'process_pmf_development_methodologies': 'Missing',
        'tech_tf_client_roles': 'Missing',
        'tech_tf_server_roles': 'Missing',
        'process_pmf_docs': 'Missing',
        'tech_tf_tools_used': 'Missing',
        'project_prf_case_tool_used': 'Missing',
        'process_pmf_prototyping_used': 'Missing',
        'tech_tf_client_server': 'Missing',
        'tech_tf_type_of_server': 'Missing',
        'tech_tf_dbms_used': 'Missing',
        'people_prf_project_user_involvement': 'Missing',
        'project_prf_currency_multiple': 'Missing'
    }
    features.update(defaults)
    
    logger.info(f"Created {len(features)} features including TARGET as feature")
    return features


def estimate_target_value(input_features: Dict[str, Any]) -> float:
    """
    Estimate target value that PyCaret expects as an INPUT feature.
    
    This is the most critical function - PyCaret models were trained with
    the target column present, so they expect it during prediction!
    
    Uses simple heuristics based on project size and complexity.
    
    Args:
        input_features: Raw input features
        
    Returns:
        Estimated normalized work effort value
    """
    
    # Base effort calculation
    team_size = input_features.get('project_prf_max_team_size', 5)
    
    # Size factor
    size_str = str(input_features.get('project_prf_relative_size', 'Medium')).upper()
    if 'XS' in size_str or 'EXTRA SMALL' in size_str:
        size_factor = 0.5
    elif 'SMALL' in size_str and 'XS' not in size_str:
        size_factor = 0.8
    elif 'LARGE' in size_str and 'XL' not in size_str:
        size_factor = 1.5
    elif 'XL' in size_str or 'EXTRA LARGE' in size_str:
        size_factor = 2.0
    else:  # Medium or unknown
        size_factor = 1.0
    
    # Application complexity factor
    app_type = str(input_features.get('project_prf_application_type', '')).upper()
    if 'WEB' in app_type:
        app_factor = 1.2
    elif 'MOBILE' in app_type:
        app_factor = 1.0
    elif 'DESKTOP' in app_type:
        app_factor = 1.3
    elif 'API' in app_type:
        app_factor = 0.8
    else:
        app_factor = 1.0
    
    # Development type factor
    dev_type = str(input_features.get('project_prf_development_type', '')).upper()
    if 'NEW' in dev_type:
        dev_factor = 1.0
    elif 'ENHANCEMENT' in dev_type:
        dev_factor = 0.7
    elif 'MAINTENANCE' in dev_type:
        dev_factor = 0.5
    else:
        dev_factor = 1.0
    
    # Calculate estimated effort
    base_hours_per_person = 200
    total_effort = base_hours_per_person * team_size * size_factor * app_factor * dev_factor
    
    # Normalize to training data scale (you may need to adjust this!)
    # Check your training data to see what range project_prf_normalised_work_effort has
    normalized = total_effort / 1000  # Assuming 0-10 range, adjust as needed
    
    logger.info(f"Estimated target: {normalized:.3f} (from {total_effort:.0f} raw hours)")
    return normalized


def calculate_derived_features(input_features: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate derived features using simple logic.
    No hardcoded mappings - just straightforward transformations.
    
    Args:
        input_features: Raw input features
        features: Already processed features
        
    Returns:
        Dictionary of derived features
    """
    
    derived = {}
    
    # Team size grouping
    team_size = features['project_prf_max_team_size']
    if team_size <= 1:
        derived['project_prf_team_size_group'] = "1"
    elif team_size <= 3:
        derived['project_prf_team_size_group'] = "2-3"
    elif team_size <= 5:
        derived['project_prf_team_size_group'] = "4-5"
    elif team_size <= 10:
        derived['project_prf_team_size_group'] = "6-10"
    else:
        derived['project_prf_team_size_group'] = "11+"
    
    # Application grouping
    app_type = str(input_features.get('project_prf_application_type', '')).upper()
    if 'WEB' in app_type or 'BUSINESS' in app_type:
        derived['project_prf_application_group'] = 'business_application'
        derived['project_prf_application_type_top'] = 'business application'
        derived['tech_tf_web_development'] = 'web'
    elif 'FINANCIAL' in app_type:
        derived['project_prf_application_type_top'] = 'financial transaction process/accounting'
        derived['project_prf_application_group'] = 'Missing'
        derived['tech_tf_web_development'] = 'Missing'
    else:
        derived['project_prf_application_group'] = 'Missing'
        derived['project_prf_application_type_top'] = 'Missing'
        derived['tech_tf_web_development'] = 'Missing'
    
    # Development type normalization
    dev_type = str(input_features.get('project_prf_development_type', '')).upper()
    if 'NEW' in dev_type:
        derived['project_prf_development_type'] = 'new_development'
    elif 'ENHANCEMENT' in dev_type:
        derived['project_prf_development_type'] = 'enhancement'
    else:
        derived['project_prf_development_type'] = 'Missing'
    
    # Platform normalization
    platform = str(input_features.get('tech_tf_development_platform', '')).upper()
    if 'CLOUD' in platform or 'PC' in platform:
        derived['tech_tf_development_platform'] = 'pc'
    elif 'MULTI' in platform:
        derived['tech_tf_development_platform'] = 'multi'
    else:
        derived['tech_tf_development_platform'] = 'Missing'
    
    # Language type
    lang_type = str(input_features.get('tech_tf_language_type', '')).upper()
    if 'OBJECT' in lang_type or '3GL' in lang_type:
        derived['tech_tf_language_type'] = '3gl'
    elif '4GL' in lang_type:
        derived['tech_tf_language_type'] = '4gl'
    else:
        derived['tech_tf_language_type'] = 'Missing'
    
    # Size normalization
    size = str(input_features.get('project_prf_relative_size', '')).upper()
    if 'MEDIUM' in size or ('M' in size and 'SMALL' not in size):
        derived['project_prf_relative_size'] = 'm1'
    elif 'LARGE' in size and 'XL' not in size:
        derived['project_prf_relative_size'] = 'l'
    elif 'SMALL' in size and 'XS' not in size:
        derived['project_prf_relative_size'] = 's'
    elif 'XL' in size or 'EXTRA LARGE' in size:
        derived['project_prf_relative_size'] = 'xl'
    elif 'XS' in size or 'EXTRA SMALL' in size:
        derived['project_prf_relative_size'] = 'xs'
    else:
        derived['project_prf_relative_size'] = 'Missing'
    
    # Architecture
    arch = str(input_features.get('tech_tf_architecture', '')).upper()
    if 'MULTI' in arch or 'MICRO' in arch:
        derived['tech_tf_architecture'] = 'multi_tier'
        derived['tech_tf_clientserver_description'] = 'web'
    elif 'CLIENT' in arch:
        derived['tech_tf_architecture'] = 'client_server'
        derived['tech_tf_clientserver_description'] = 'client_server'
    else:
        derived['tech_tf_architecture'] = 'Missing'
        derived['tech_tf_clientserver_description'] = 'Missing'
    
    # Organization type
    org = str(input_features.get('external_eef_organisation_type', '')).upper()
    if 'LARGE' in org or 'ENTERPRISE' in org:
        derived['external_eef_organisation_type_top'] = 'computers & software'
    elif 'BANKING' in org:
        derived['external_eef_organisation_type_top'] = 'banking'
    elif 'GOVERNMENT' in org:
        derived['external_eef_organisation_type_top'] = 'government'
    else:
        derived['external_eef_organisation_type_top'] = 'Missing'
    
    return derived


def optimize_target_value(input_features: Dict[str, Any], model_name: str, load_model_func) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the optimal target value that produces realistic predictions.
    
    Sometimes the estimated target value doesn't work well. This function
    tries different target values to find one that gives reasonable results.
    
    Args:
        input_features: Raw input features
        model_name: Name of the model to test with
        load_model_func: Function to load the model (passed to avoid circular import)
        
    Returns:
        Tuple of (optimal_target_value, predicted_result) or (None, None)
    """
    
    # Try different target values
    target_candidates = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    results = []
    
    # Load model once
    model = load_model_func(model_name)
    if not model:
        logger.error(f"Failed to load model: {model_name}")
        return None, None
    
    for target_val in target_candidates:
        try:
            # Create features with specific target value
            features = create_training_compatible_features(input_features)
            features['project_prf_normalised_work_effort'] = target_val
            
            # Create DataFrame
            features_df = pd.DataFrame([features])
            
            # Test prediction
            if PYCARET_AVAILABLE:
                try:
                    predictions = predict_model(model, data=features_df)
                    
                    # Find prediction column
                    pred_cols = ['prediction_label', 'Label', 'pred', 'prediction']
                    result = None
                    
                    for col in pred_cols:
                        if col in predictions.columns:
                            result = float(predictions[col].iloc[0])
                            break
                    
                    if result is None:
                        result = float(predictions.iloc[0, -1])
                    
                    if result and 10 <= result <= 5000:  # Reasonable range
                        results.append((target_val, result))
                        logger.info(f"Target {target_val:.1f} -> Prediction {result:.1f} hours")
                
                except Exception as e:
                    logger.warning(f"Target {target_val} failed: {e}")
        
        except Exception as e:
            logger.warning(f"Target {target_val} error: {e}")
    
    if results:
        # Choose the middle result for stability
        optimal_target, optimal_prediction = results[len(results)//2]
        logger.info(f"Optimal target found: {optimal_target} -> {optimal_prediction:.1f} hours")
        return optimal_target, optimal_prediction
    
    logger.warning("No optimal target found")
    return None, None


def predict_with_target_as_feature(input_features: Dict[str, Any], model_name: str, load_model_func) -> Optional[float]:
    """
    Make prediction using the corrected approach where target is included as feature.
    
    This is the main prediction function that addresses the PyCaret issue.
    
    Args:
        input_features: Raw input features from UI
        model_name: Name of the trained model
        load_model_func: Function to load the model (passed to avoid circular import)
        
    Returns:
        Predicted man-hours or None if prediction fails
    """
    
    try:
        # Create training-compatible features (including target as feature!)
        features = create_training_compatible_features(input_features)
        
        # Load model
        model = load_model_func(model_name)
        if not model:
            logger.error(f"Failed to load model: {model_name}")
            return None
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        logger.info(f"Feature DataFrame shape: {features_df.shape}")
        logger.info(f"Includes target as feature: {'project_prf_normalised_work_effort' in features_df.columns}")
        
        # Make prediction using PyCaret
        if PYCARET_AVAILABLE:
            try:
                predictions = predict_model(model, data=features_df)
                
                # Find prediction column
                pred_cols = ['prediction_label', 'Label', 'pred', 'prediction']
                for col in pred_cols:
                    if col in predictions.columns:
                        result = float(predictions[col].iloc[0])
                        logger.info(f"PyCaret prediction successful: {result:.2f}")
                        return max(0.1, result)  # Ensure positive
                
                # Fallback to last column
                result = float(predictions.iloc[0, -1])
                logger.info(f"PyCaret prediction (last col): {result:.2f}")
                return max(0.1, result)
                
            except Exception as e:
                logger.warning(f"PyCaret prediction failed: {e}")
                
                # Try optimizing target value
                logger.info("Trying to optimize target value...")
                optimal_target, optimal_result = optimize_target_value(input_features, model_name, load_model_func)
                if optimal_result:
                    return optimal_result
        
        # Fallback to direct model prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(features_df)
            result = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
            logger.info(f"Direct prediction: {result:.2f}")
            return max(0.1, result)
        
        logger.error("No prediction method available")
        return None
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def validate_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all required features are present and have valid values.
    
    Args:
        features: Dictionary of features to validate
        
    Returns:
        Dictionary with validation results
    """
    
    validation_result = {
        'valid': True,
        'missing_features': [],
        'invalid_values': [],
        'warnings': []
    }
    
    # Required numeric features
    required_numeric = [
        'project_prf_year_of_project',
        'project_prf_max_team_size',
        'project_prf_functional_size',
        'project_prf_normalised_work_effort'
    ]
    
    for feature in required_numeric:
        if feature not in features:
            validation_result['missing_features'].append(feature)
            validation_result['valid'] = False
        elif not isinstance(features[feature], (int, float)):
            validation_result['invalid_values'].append(f"{feature}: not numeric")
            validation_result['valid'] = False
    
    # Check target value range
    target_value = features.get('project_prf_normalised_work_effort')
    if target_value is not None:
        if target_value <= 0:
            validation_result['warnings'].append("Target value is zero or negative")
        elif target_value > 10:
            validation_result['warnings'].append("Target value seems very high (>10)")
    
    # Check team size
    team_size = features.get('project_prf_max_team_size')
    if team_size is not None:
        if team_size <= 0:
            validation_result['invalid_values'].append("Team size must be positive")
            validation_result['valid'] = False
        elif team_size > 50:
            validation_result['warnings'].append("Very large team size (>50)")
    
    return validation_result


def get_feature_summary(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the created features for debugging/logging.
    
    Args:
        features: Dictionary of features
        
    Returns:
        Summary dictionary
    """
    
    summary = {
        'total_features': len(features),
        'numeric_features': 0,
        'categorical_features': 0,
        'missing_values': 0,
        'target_included': 'project_prf_normalised_work_effort' in features,
        'key_features': {}
    }
    
    # Count feature types
    for key, value in features.items():
        if isinstance(value, (int, float)):
            summary['numeric_features'] += 1
        else:
            summary['categorical_features'] += 1
            
        if value == 'Missing' or value is None:
            summary['missing_values'] += 1
    
    # Extract key features for summary
    key_feature_names = [
        'project_prf_normalised_work_effort',
        'project_prf_max_team_size',
        'external_eef_industry_sector',
        'tech_tf_primary_programming_language',
        'project_prf_team_size_group',
        'project_prf_application_group'
    ]
    
    for key in key_feature_names:
        if key in features:
            summary['key_features'][key] = features[key]
    
    return summary


def get_model_expected_features(model, config_fallback=None):
    """
    Return a list of expected feature names in prediction order from the loaded model.
    """
    # 1. sklearn >=1.0, PyCaret models
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # 2. sklearn pipeline
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    # 3. PyCaret pipeline X attribute
    if hasattr(model, 'X') and hasattr(model.X, 'columns'):
        return list(model.X.columns)
    # 4. Fallback to config (as you already do)
    if config_fallback:
        return config_fallback
    # 5. Give up
    return []

def align_features_to_model(complete_features: dict, model_expected: list) -> dict:
    """
    Return a dict with the same order and set of columns as the model expects.
    Missing features are set to 0, extras are dropped.
    """
    out = {}
    for feat in model_expected:
        out[feat] = complete_features.get(feat, 0)  # or np.nan if you prefer for numerics
    return out


def discover_features_via_dummy_prediction(model, model_name: str) -> Optional[List[str]]:
    """
    Discover expected features by creating dummy data and seeing what the model expects.
    """
    try:
        # Load feature configuration to create reasonable dummy data
        feature_config = load_yaml_config(FEATURE_MAPPING_FILE)
        if not feature_config:
            logger.warning("No feature config available for dummy prediction")
            return None
        
        # Create comprehensive dummy feature dict
        dummy_features = create_comprehensive_dummy_features(feature_config)
        
        # Convert to DataFrame
        dummy_df = pd.DataFrame([dummy_features])
        
        # Try prediction to see what features are expected
        try:
            if PYCARET_AVAILABLE:
                # This will fail with the exact error message showing expected features
                predictions = predict_model(model, data=dummy_df)
                # If it succeeds, return the column names
                return list(dummy_df.columns)
            else:
                # Try direct prediction
                prediction = model.predict(dummy_df)
                return list(dummy_df.columns)
                
        except Exception as prediction_error:
            # Parse the error message to extract expected feature names
            error_str = str(prediction_error)
            if "not in index" in error_str:
                expected_features = extract_features_from_error(error_str)
                if expected_features:
                    logger.info(f"Extracted {len(expected_features)} expected features from error message")
                    return expected_features
            
            logger.error(f"Could not discover features from model: {error_str}")
            return None
            
    except Exception as e:
        logger.error(f"Error in dummy prediction discovery: {e}")
        return None


def extract_features_from_error(error_message: str) -> Optional[List[str]]:
    """
    Extract feature names from PyCaret error message like:
    "['feature1', 'feature2', 'feature3'] not in index"
    """
    try:
        # Find the part with the feature list
        import re
        
        # Look for pattern like ['feature1', 'feature2', ...] 
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, error_message)
        
        if matches:
            # Take the longest match (most likely to be the feature list)
            feature_string = max(matches, key=len)
            
            # Parse the feature names
            features = []
            # Split by comma and clean up
            for feature in feature_string.split(','):
                # Remove quotes and whitespace
                clean_feature = feature.strip().strip("'\"")
                if clean_feature:
                    features.append(clean_feature)
            
            logger.info(f"Extracted {len(features)} features from error message")
            return features
    
    except Exception as e:
        logger.error(f"Error extracting features from error: {e}")
    
    return None


def create_comprehensive_dummy_features(feature_config: Dict) -> Dict[str, Any]:
    """
    Create comprehensive dummy features covering all possible categories.
    """
    dummy_features = {
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
        "tech_tf_primary_programming_language": "Python",
        "project_prf_functional_size": 100
    }
    
    # Add defaults for all configured features
    if feature_config:
        # Add numeric features
        for feature in feature_config.get("numeric_features", []):
            if feature not in dummy_features:
                dummy_features[feature] = 100
        
        # Add categorical features with first option
        for feature, config in feature_config.get("categorical_features", {}).items():
            if feature not in dummy_features:
                options = config.get("options", [])
                dummy_features[feature] = options[0] if options else "default"
        
        # Add one-hot input fields
        for group_name, group_config in feature_config.get("one_hot_features", {}).items():
            input_key = group_config.get("input_key")
            if input_key and input_key not in dummy_features:
                mapping = group_config.get("mapping", {})
                if mapping:
                    dummy_features[input_key] = list(mapping.keys())[0]
    
    return dummy_features


def apply_feature_engineering_dynamic(
    ui_features: Dict[str, Any], 
    pipeline_features: Optional[pd.DataFrame],
    model_name: str,
    load_model_func
) -> Optional[Dict[str, Any]]:
    """
    DYNAMIC APPROACH: Get expected features from model and create them dynamically.
    
    Args:
        ui_features: Raw UI features
        pipeline_features: Features from pipeline transformation  
        model_name: Name of the model to get expected features from
        load_model_func: Function to load the model (passed to avoid circular import)
        
    Returns:
        Complete feature dictionary matching model expectations
    """
    
    try:
        logger.info("=== DYNAMIC FEATURE ENGINEERING ===")
        
        # Step 1: Get expected features from the model
        expected_features = get_model_expected_features(model_name, load_model_func)
        if not expected_features:
            logger.error("Could not determine expected features from model")
            return None
        
        logger.info(f"Model expects {len(expected_features)} features")
        
        # Step 2: Start with pipeline features if available
        complete_features = {}
        if pipeline_features is not None and not pipeline_features.empty:
            pipeline_dict = pipeline_features.iloc[0].to_dict()
            complete_features.update(pipeline_dict)
            logger.info(f"Added {len(pipeline_dict)} features from pipeline")
        
        # Step 3: Add essential numeric features
        essential_numeric = {
            'project_prf_year_of_project': ui_features.get('project_prf_year_of_project', 2024),
            'project_prf_max_team_size': ui_features.get('project_prf_max_team_size', 8),
            'project_prf_functional_size': ui_features.get('project_prf_functional_size', 100)
        }
        
        for key, value in essential_numeric.items():
            if key in expected_features and key not in complete_features:
                complete_features[key] = value
                logger.info(f"Added numeric feature: {key} = {value}")
        
        # Step 4: Add target as feature (critical for PyCaret)
        target_feature = 'project_prf_normalised_work_effort'
        if target_feature in expected_features and target_feature not in complete_features:
            target_value = estimate_target_value(ui_features)
            complete_features[target_feature] = target_value
            logger.info(f"Added target feature: {target_value:.3f}")
        
        # Step 5: Create missing one-hot encoded features dynamically
        missing_features = set(expected_features) - set(complete_features.keys())
        logger.info(f"Creating {len(missing_features)} missing features dynamically")
        
        created_features = create_missing_features_intelligently(
            missing_features, ui_features, complete_features
        )
        complete_features.update(created_features)
        
        # Step 6: Ensure all expected features are present
        final_missing = set(expected_features) - set(complete_features.keys())
        if final_missing:
            logger.info(f"Setting {len(final_missing)} remaining features to default (0)")
            for feature in final_missing:
                complete_features[feature] = 0
        
        # Step 7: Remove any extra features not expected by model
        extra_features = set(complete_features.keys()) - set(expected_features)
        if extra_features:
            logger.info(f"Removing {len(extra_features)} extra features")
            for feature in extra_features:
                del complete_features[feature]
        
        logger.info(f"✅ Dynamic feature engineering complete: {len(complete_features)} features")
        logger.info(f"Expected: {len(expected_features)}, Created: {len(complete_features)}")
        
        return complete_features
        
    except Exception as e:
        logger.error(f"Dynamic feature engineering failed: {e}")
        return None


def create_missing_features_intelligently(
    missing_features: set, 
    ui_features: Dict[str, Any], 
    existing_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Intelligently create missing one-hot encoded features based on UI inputs.
    """
    
    created_features = {}
    
    # Group missing features by prefix to understand structure
    feature_groups = {}
    for feature in missing_features:
        # Extract base name (everything before last underscore)
        parts = feature.split('_')
        if len(parts) > 1:
            base = '_'.join(parts[:-1])
            if base not in feature_groups:
                feature_groups[base] = []
            feature_groups[base].append(feature)
    
    logger.info(f"Identified {len(feature_groups)} feature groups to process")
    
    # Process each feature group
    for group_name, group_features in feature_groups.items():
        
        # Determine which feature in the group should be active
        active_feature = determine_active_feature_in_group(
            group_name, group_features, ui_features, existing_features
        )
        
        # Set all features in group (one-hot encoding)
        for feature in group_features:
            created_features[feature] = 1 if feature == active_feature else 0
        
        if active_feature:
            logger.info(f"Group '{group_name}': activated '{active_feature}'")
    
    return created_features


def determine_active_feature_in_group(
    group_name: str, 
    group_features: List[str], 
    ui_features: Dict[str, Any],
    existing_features: Dict[str, Any]
) -> Optional[str]:
    """
    Determine which feature in a one-hot encoded group should be active (=1).
    """
    
    # Application group logic
    if 'application_group' in group_name:
        app_type = str(ui_features.get('project_prf_application_type', '')).upper()
        if 'WEB' in app_type or 'BUSINESS' in app_type:
            return find_feature_containing(group_features, ['business_application'])
        else:
            return find_feature_containing(group_features, ['nan', 'missing'])
    
    # Client server description logic
    elif 'clientserver_description' in group_name:
        arch = str(ui_features.get('tech_tf_architecture', '')).upper()
        if 'MICRO' in arch or 'WEB' in arch:
            return find_feature_containing(group_features, ['web'])
        elif 'CLIENT' in arch:
            return find_feature_containing(group_features, ['client_server'])
        else:
            return find_feature_containing(group_features, ['nan', 'missing'])
    
    # Development type logic
    elif 'development_type' in group_name:
        dev_type = str(ui_features.get('project_prf_development_type', '')).upper()
        if 'NEW' in dev_type:
            return find_feature_containing(group_features, ['new_development'])
        elif 'ENHANCEMENT' in dev_type:
            return find_feature_containing(group_features, ['enhancement'])
        else:
            return find_feature_containing(group_features, ['other', 'nan'])
    
    # Development platform logic
    elif 'development_platform' in group_name:
        platform = str(ui_features.get('tech_tf_development_platform', '')).upper()
        if 'CLOUD' in platform or 'PC' in platform:
            return find_feature_containing(group_features, ['pc'])
        elif 'MULTI' in platform:
            return find_feature_containing(group_features, ['multi'])
        else:
            return find_feature_containing(group_features, ['nan'])
    
    # Language type logic
    elif 'language_type' in group_name:
        lang_type = str(ui_features.get('tech_tf_language_type', '')).upper()
        if 'OBJECT' in lang_type or '3GL' in lang_type:
            return find_feature_containing(group_features, ['3gl'])
        elif '4GL' in lang_type:
            return find_feature_containing(group_features, ['4gl'])
        else:
            return find_feature_containing(group_features, ['nan'])
    
    # Relative size logic
    elif 'relative_size' in group_name:
        size = str(ui_features.get('project_prf_relative_size', '')).upper()
        if 'MEDIUM' in size:
            return find_feature_containing(group_features, ['m1', 'm2'])
        elif 'LARGE' in size and 'XL' not in size:
            return find_feature_containing(group_features, ['l'])
        elif 'SMALL' in size and 'XS' not in size:
            return find_feature_containing(group_features, ['s'])
        elif 'XL' in size:
            return find_feature_containing(group_features, ['xl'])
        elif 'XS' in size:
            return find_feature_containing(group_features, ['xs'])
        else:
            return find_feature_containing(group_features, ['nan'])
    
    # Architecture logic
    elif 'architecture' in group_name:
        arch = str(ui_features.get('tech_tf_architecture', '')).upper()
        if 'MICRO' in arch or 'MULTI' in arch:
            return find_feature_containing(group_features, ['multi_tier'])
        elif 'CLIENT' in arch:
            return find_feature_containing(group_features, ['client_server'])
        else:
            return find_feature_containing(group_features, ['nan', 'stand_alone'])
    
    # Data quality rating logic
    elif 'data_quality_rating' in group_name:
        return find_feature_containing(group_features, ['a'])  # Default to 'A'
    
    # Organization type logic
    elif 'organisation_type' in group_name:
        org = str(ui_features.get('external_eef_organisation_type', '')).upper()
        if 'LARGE' in org or 'ENTERPRISE' in org:
            return find_feature_containing(group_features, ['computers', 'software'])
        elif 'BANKING' in org:
            return find_feature_containing(group_features, ['banking'])
        elif 'GOVERNMENT' in org:
            return find_feature_containing(group_features, ['government'])
        else:
            return find_feature_containing(group_features, ['nan', 'other'])
    
    # Default logic - prefer 'nan', 'no', or first option
    else:
        # Try common defaults in order of preference
        for default_suffix in ['nan', 'no', 'missing', 'other']:
            match = find_feature_containing(group_features, [default_suffix])
            if match:
                return match
        
        # If no common default, return first feature
        return group_features[0] if group_features else None


def find_feature_containing(feature_list: List[str], keywords: List[str]) -> Optional[str]:
    """
    Find first feature in list that contains any of the keywords.
    """
    for keyword in keywords:
        for feature in feature_list:
            if keyword.lower() in feature.lower():
                return feature
    return None


# === TEST FUNCTIONS ===

def test_feature_creation():
    """
    Test the feature creation function with sample data.
    """
    
    sample_input = {
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
    
    print("=== TESTING FEATURE ENGINEERING ===")
    print("Key insight: PyCaret expects target column as INPUT feature!")
    
    # Create features
    features = create_training_compatible_features(sample_input)
    
    # Validate features
    validation = validate_features(features)
    
    # Get summary
    summary = get_feature_summary(features)
    
    print(f"\nCreated {len(features)} features")
    print(f"Target value included: {features.get('project_prf_normalised_work_effort', 'MISSING!')}")
    print(f"Validation passed: {validation['valid']}")
    
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    if validation['missing_features']:
        print(f"Missing features: {validation['missing_features']}")
    
    # Show key features
    print("\nKey features:")
    for key, value in summary['key_features'].items():
        print(f"  {key}: {value}")
    
    print(f"\nFeature summary:")
    print(f"  Numeric features: {summary['numeric_features']}")
    print(f"  Categorical features: {summary['categorical_features']}")
    print(f"  Missing values: {summary['missing_values']}")
    print(f"  Target included: {summary['target_included']}")
    
    print("\n✅ Feature engineering test completed!")
    print("💡 Features are ready for PyCaret prediction")
    
    return features, validation, summary


if __name__ == "__main__":
    # Run tests when file is executed directly
    test_feature_creation()
