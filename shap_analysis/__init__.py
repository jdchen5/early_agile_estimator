# shap_analysis/__init__.py
"""
SHAP Analysis Module - Safe integration of modular components

This module provides a unified interface for SHAP (SHapley Additive exPlanations) 
analysis functionality. It maintains backward compatibility with existing code
while introducing a new modular system for improved maintainability and extensibility.

Key features:
- Backward compatibility with existing SHAP analysis functions
- New modular system with separate components for data preparation, 
  explainer creation, value calculation, and analysis coordination
- Automatic fallback to backup system if modular system unavailable
- Clean public API for both simple and advanced use cases

Architecture:
- Data Preparer: Handles background and input data preparation
- Explainer Factory: Creates and manages SHAP explainers with caching
- Value Calculator: Computes SHAP values and generates summaries
- Analysis Coordinator: Orchestrates complete analysis workflows
- UI Integration: Provides Streamlit-compatible display functions
"""

import logging

# Configure logging for the module
logger = logging.getLogger(__name__)


# Module-level constants
DEFAULT_SAMPLE_SIZE = 100
FALLBACK_CACHE_INFO = {'cached_models': [], 'cache_size': 0}


# === MODULAR SYSTEM INITIALIZATION ===
try:
    from .data_preparer import SHAPDataPreparer
    from .explainer_factory import SHAPExplainerFactory  
    from .value_calculator import SHAPValueCalculator
    from .analysis_coordinator import SHAPAnalysisCoordinator
    from .ui_integration import SHAPUIIntegration
    
    # Create global component instances
    _data_preparer = SHAPDataPreparer()
    _explainer_factory = SHAPExplainerFactory()
    _value_calculator = SHAPValueCalculator()
    _coordinator = SHAPAnalysisCoordinator()
    _ui_integration = SHAPUIIntegration()
    
    MODULAR_SYSTEM_AVAILABLE = True
    logger.info("New modular SHAP system loaded successfully")
    
except ImportError as e:
    logger.warning(f"New modular system not available: {e}")
    MODULAR_SYSTEM_AVAILABLE = False
    _data_preparer = None
    _explainer_factory = None
    _value_calculator = None
    _coordinator = None
    _ui_integration = None


# === BACKUP SYSTEM INITIALIZATION ===
try:
    from shap_analysis_backup import *
    BACKUP_SYSTEM_AVAILABLE = True
    logger.info("Backup SHAP system loaded as fallback")
except ImportError as e:
    logger.error(f"Backup system also not available: {e}")
    BACKUP_SYSTEM_AVAILABLE = False


# === BACKWARD COMPATIBILITY FUNCTIONS ===

def get_shap_explainer(
    model_name: str,
    get_trained_model_func,
    prepare_sample_data_func=None,
    sample_size: int = DEFAULT_SAMPLE_SIZE
):
    """
    Get or create a SHAP explainer for the specified model.
    
    This function maintains backward compatibility while using the new modular
    system when available. It automatically falls back to the backup system
    if the modular system fails.
    
    Args:
        model_name: Technical name of the model
        get_trained_model_func: Function to retrieve the trained model
        prepare_sample_data_func: Optional function for sample data preparation
        sample_size: Number of background samples for SHAP baseline
        
    Returns:
        SHAP explainer object or None if creation fails
    """
    if MODULAR_SYSTEM_AVAILABLE and _data_preparer and _explainer_factory:
        try:
            # Use new modular system
            background_data = _data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                raise Exception("Background data preparation failed")
                
            explainer = _explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            
            if explainer is not None:
                logger.debug("Using new modular explainer system")
                return explainer
            else:
                raise Exception("Explainer creation failed")
                
        except Exception as e:
            logger.warning(f"New system failed, falling back to backup: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_shap_explainer as backup_get_explainer
            logger.debug("Using backup explainer system")
            return backup_get_explainer(model_name, get_trained_model_func, prepare_sample_data_func, sample_size)
        except Exception as e:
            logger.error(f"Backup system also failed: {e}")
    
    return None


def get_shap_values_for_input(explainer, user_inputs, model=None, feature_names=None):
    """
    Calculate SHAP values for given input data.
    
    Args:
        explainer: SHAP explainer object
        user_inputs: Input data (dict or array)
        model: Optional model object (for compatibility)
        feature_names: Optional list of feature names
        
    Returns:
        SHAP values array or None if calculation fails
    """
    if MODULAR_SYSTEM_AVAILABLE and _data_preparer and _value_calculator and explainer is not None:
        try:
            # Use new modular system
            if isinstance(user_inputs, dict):
                input_data = _data_preparer.prepare_input_data(user_inputs)
            else:
                input_data = user_inputs
            
            if input_data is not None:
                shap_values = _value_calculator.calculate_shap_values(explainer, input_data, feature_names)
                if shap_values is not None:
                    logger.debug("Using new modular value calculation")
                    return shap_values
                
        except Exception as e:
            logger.warning(f"New value calculation failed, falling back: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_shap_values_for_input as backup_get_values
            logger.debug("Using backup value calculation")
            return backup_get_values(explainer, user_inputs, model, feature_names)
        except Exception as e:
            logger.error(f"Backup value calculation failed: {e}")
    
    return None


def prepare_sample_data(n_samples, fields=None, get_field_options_func=None):
    """
    Prepare sample data for SHAP background baseline.
    
    Args:
        n_samples: Number of samples to prepare
        fields: Field configuration (for compatibility)
        get_field_options_func: Field options function (for compatibility)
        
    Returns:
        Background data array or None if preparation fails
    """
    if MODULAR_SYSTEM_AVAILABLE and _data_preparer:
        try:
            background_data = _data_preparer.prepare_background_data(n_samples)
            if background_data is not None:
                logger.debug("Using new modular data preparation")
                return background_data
        except Exception as e:
            logger.warning(f"New data preparation failed, falling back: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import prepare_sample_data as backup_prepare_data
            logger.debug("Using backup data preparation")
            return backup_prepare_data(n_samples, fields, get_field_options_func)
        except Exception as e:
            logger.error(f"Backup data preparation failed: {e}")
    
    return None


def clear_explainer_cache():
    """Clear SHAP explainer cache to free memory."""
    if MODULAR_SYSTEM_AVAILABLE and _explainer_factory:
        try:
            _explainer_factory.clear_cache()
            logger.debug("Cleared new modular cache")
            return
        except Exception as e:
            logger.warning(f"Failed to clear modular cache: {e}")
    
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import clear_explainer_cache as backup_clear_cache
            backup_clear_cache()
            logger.debug("Cleared backup cache")
        except Exception as e:
            logger.warning(f"Failed to clear backup cache: {e}")


def get_cache_info():
    """
    Get information about the explainer cache.
    
    Returns:
        Dictionary containing cache information
    """
    if MODULAR_SYSTEM_AVAILABLE and _explainer_factory:
        try:
            return _explainer_factory.get_cache_info()
        except Exception as e:
            logger.warning(f"Failed to get modular cache info: {e}")
    
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_cache_info as backup_get_cache_info
            return backup_get_cache_info()
        except Exception as e:
            logger.warning(f"Failed to get backup cache info: {e}")
    
    return FALLBACK_CACHE_INFO


def display_instance_specific_shap(user_inputs, model_name, get_trained_model_func):
    """
    Display instance-specific SHAP analysis in Streamlit.
    
    Args:
        user_inputs: Dictionary of user input features
        model_name: Name of the model to analyze
        get_trained_model_func: Function to retrieve trained model
        
    Returns:
        None (displays results in Streamlit)
    """
    if MODULAR_SYSTEM_AVAILABLE and _ui_integration:
        try:
            return _ui_integration.display_instance_specific_shap(
                user_inputs, model_name, get_trained_model_func
            )
        except Exception as e:
            logger.error(f"Modular UI display failed: {e}")
    
    # Fallback behavior
    logger.warning("SHAP UI integration not available")


def display_what_if_shap_analysis(user_inputs, model_name, get_trained_model_func):
    """
    Display what-if SHAP analysis in Streamlit.
    
    Args:
        user_inputs: Dictionary of user input features
        model_name: Name of the model to analyze
        get_trained_model_func: Function to retrieve trained model
        
    Returns:
        None (displays results in Streamlit)
    """
    if MODULAR_SYSTEM_AVAILABLE and _ui_integration:
        try:
            return _ui_integration.display_what_if_shap_analysis(
                user_inputs, model_name, get_trained_model_func
            )
        except Exception as e:
            logger.error(f"Modular UI display failed: {e}")
    
    # Fallback behavior
    logger.warning("SHAP UI integration not available")


def get_feature_names_from_fields(fields):
    """
    Extract feature names from fields configuration.
    
    Args:
        fields: Dictionary of field configurations
        
    Returns:
        List of feature names
    """
    if MODULAR_SYSTEM_AVAILABLE and _data_preparer:
        return _data_preparer.get_feature_names_from_fields(fields)
    
    # Fallback implementation
    exclude_fields = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
    return [name for name in sorted(fields.keys()) if name not in exclude_fields] if fields else []


def get_parameter_index(param_name, feature_names):
    """
    Get the index of a parameter in the feature names list.
    
    Args:
        param_name: Name of the parameter to find
        feature_names: List of feature names
        
    Returns:
        Index of the parameter or None if not found
    """
    if MODULAR_SYSTEM_AVAILABLE and _data_preparer:
        return _data_preparer.get_parameter_index(param_name, feature_names)
    
    # Fallback implementation
    try:
        return feature_names.index(param_name) if feature_names else None
    except (ValueError, AttributeError):
        return None


# === NEW MODULAR INTERFACE ===

def get_coordinator():
    """
    Get the SHAPAnalysisCoordinator instance for advanced analysis workflows.
    
    Returns:
        SHAPAnalysisCoordinator instance or None if not available
    """
    if MODULAR_SYSTEM_AVAILABLE:
        return _coordinator
    return None


def run_complete_analysis(user_inputs, model_name, get_trained_model_func, sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Run complete SHAP analysis using the new modular system.
    
    Args:
        user_inputs: Dictionary of user input features
        model_name: Name of the model to analyze
        get_trained_model_func: Function to retrieve trained model
        sample_size: Number of background samples
        
    Returns:
        Analysis result dictionary or None if system not available
    """
    if MODULAR_SYSTEM_AVAILABLE and _coordinator is not None:
        try:
            return _coordinator.run_instance_analysis(
                user_inputs, model_name, get_trained_model_func, sample_size
            )
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
    
    return None


def get_system_status():
    """
    Get comprehensive status of SHAP analysis systems.
    
    Returns:
        Dictionary containing system availability and status information
    """
    return {
        'modular_system_available': MODULAR_SYSTEM_AVAILABLE,
        'backup_system_available': BACKUP_SYSTEM_AVAILABLE,
        'recommended_system': 'modular' if MODULAR_SYSTEM_AVAILABLE else 'backup',
        'coordinator_available': _coordinator is not None,
        'ui_integration_available': _ui_integration is not None,
        'components': {
            'data_preparer': _data_preparer is not None,
            'explainer_factory': _explainer_factory is not None,
            'value_calculator': _value_calculator is not None,
            'analysis_coordinator': _coordinator is not None,
            'ui_integration': _ui_integration is not None
        }
    }


# === COMPONENT ACCESS FUNCTIONS ===

def get_data_preparer():
    """Get the SHAPDataPreparer instance."""
    return _data_preparer if MODULAR_SYSTEM_AVAILABLE else None


def get_explainer_factory():
    """Get the SHAPExplainerFactory instance."""
    return _explainer_factory if MODULAR_SYSTEM_AVAILABLE else None


def get_value_calculator():
    """Get the SHAPValueCalculator instance."""
    return _value_calculator if MODULAR_SYSTEM_AVAILABLE else None


def get_ui_integration():
    """Get the SHAPUIIntegration instance."""
    return _ui_integration if MODULAR_SYSTEM_AVAILABLE else None


# === MODULE EXPORTS ===
__all__ = [
    # Backward compatibility interface
    'get_shap_explainer',
    'get_shap_values_for_input', 
    'prepare_sample_data',
    'clear_explainer_cache',
    'get_cache_info',
    'display_instance_specific_shap',
    'display_what_if_shap_analysis', 
    'get_feature_names_from_fields',
    'get_parameter_index',
    
    # New modular interface - Classes
    'SHAPAnalysisCoordinator',
    'SHAPDataPreparer',
    'SHAPExplainerFactory', 
    'SHAPValueCalculator',
    'SHAPUIIntegration',
    
    # New modular interface - Functions
    'get_coordinator',
    'run_complete_analysis',
    'get_system_status',
    
    # Component access functions
    'get_data_preparer',
    'get_explainer_factory',
    'get_value_calculator',
    'get_ui_integration',
    
    # System status
    'MODULAR_SYSTEM_AVAILABLE',
    'BACKUP_SYSTEM_AVAILABLE'
]


# === MODULE INITIALIZATION LOG ===
def _log_initialization_status():
    """Log the initialization status of the SHAP analysis module."""
    status = get_system_status()
    logger.info(f"SHAP Analysis Module initialized:")
    logger.info(f"  - Modular system: {'✓' if status['modular_system_available'] else '✗'}")
    logger.info(f"  - Backup system: {'✓' if status['backup_system_available'] else '✗'}")
    logger.info(f"  - Recommended: {status['recommended_system']}")
    
    if status['modular_system_available']:
        components = status['components']
        available_components = sum(1 for available in components.values() if available)
        logger.info(f"  - Components available: {available_components}/{len(components)}")


# Run initialization logging
_log_initialization_status()