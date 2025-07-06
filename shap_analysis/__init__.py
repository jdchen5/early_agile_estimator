# shap_analysis/__init__.py
"""
SHAP Analysis Module - Modular SHAP analysis for ML models
Backward compatibility maintained during transition
"""

from .analysis_coordinator import SHAPAnalysisCoordinator
from .explainer_factory import SHAPExplainerFactory
from .data_preparer import SHAPDataPreparer
from .value_calculator import SHAPValueCalculator

# For now, import everything from old file for backward compatibility
try:
    from ..shap_analysis_backup import *
    print("DEBUG: Using backup shap_analysis for compatibility")
except ImportError:
    # If backup doesn't exist, import from current
    from ..shap_analysis import *
    print("DEBUG: Using current shap_analysis")

# Global coordinator instance for backward compatibility
_coordinator = None

def _get_coordinator():
    """Get or create coordinator instance"""
    global _coordinator
    if _coordinator is None:
        _coordinator = SHAPAnalysisCoordinator()
    return _coordinator

# Backward compatibility wrappers for existing ui.py imports
def get_shap_explainer(
    model_name: str,
    get_trained_model_func,
    prepare_sample_data_func=None,
    sample_size: int = 100
):
    """
    Backward compatibility wrapper for get_shap_explainer
    Returns actual explainer object as before
    """
    try:
        coordinator = _get_coordinator()
        
        # Prepare background data
        background_data = coordinator.data_preparer.prepare_background_data(sample_size, model_name)
        if background_data is None:
            return None
        
        # Create and return explainer
        explainer = coordinator.explainer_factory.create_explainer(
            model_name, get_trained_model_func, background_data, sample_size
        )
        return explainer
        
    except Exception as e:
        print(f"Error in backward compatibility get_shap_explainer: {e}")
        return None

def get_shap_values_for_input(explainer, user_inputs, model=None, feature_names=None):
    """
    Backward compatibility wrapper for get_shap_values_for_input
    """
    try:
        coordinator = _get_coordinator()
        
        # Prepare input data
        if isinstance(user_inputs, dict):
            input_data = coordinator.data_preparer.prepare_input_data(user_inputs)
        else:
            input_data = user_inputs
        
        if input_data is None:
            return None
        
        # Calculate SHAP values
        return coordinator.calculator.calculate_shap_values(explainer, input_data, feature_names)
        
    except Exception as e:
        print(f"Error in backward compatibility get_shap_values_for_input: {e}")
        return None

def prepare_sample_data(n_samples, fields=None, get_field_options_func=None):
    """Backward compatibility wrapper for prepare_sample_data"""
    try:
        coordinator = _get_coordinator()
        return coordinator.data_preparer.prepare_background_data(n_samples)
    except Exception as e:
        print(f"Error in backward compatibility prepare_sample_data: {e}")
        return None

def clear_explainer_cache():
    """Backward compatibility wrapper for clear_explainer_cache"""
    coordinator = _get_coordinator()
    coordinator.explainer_factory.clear_cache()

def get_cache_info():
    """Backward compatibility wrapper for get_cache_info"""
    coordinator = _get_coordinator()
    return coordinator.explainer_factory.get_cache_info()

# Export both old and new interfaces
__all__ = [
    # New modular interface
    'SHAPAnalysisCoordinator',
    'SHAPExplainerFactory',
    'SHAPDataPreparer', 
    'SHAPValueCalculator',
    
    # Backward compatibility
    'get_shap_explainer',
    'get_shap_values_for_input',
    'prepare_sample_data',
    'clear_explainer_cache',
    'get_cache_info'
]