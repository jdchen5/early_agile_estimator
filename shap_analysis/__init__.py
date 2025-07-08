# shap_analysis/__init__.py - FIXED VERSION
"""Enhanced SHAP Analysis Module with reduced feature optimization"""

import logging
from typing import Optional, Callable, Dict, Any, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Create a dummy shap class for type hints when shap is not available
    class shap:
        class Explainer:
            pass

# Module constants
DEFAULT_TOP_N_FEATURES = 15
CORRELATION_THRESHOLD = 0.85

# Initialize components with proper error handling
try:
    from .data_preparer import SHAPDataPreparer
    from .explainer_factory import SHAPExplainerFactory  
    from .value_calculator import SHAPValueCalculator
    from .analysis_coordinator import SHAPAnalysisCoordinator
    from .ui_integration import SHAPUIIntegration
    
    _coordinator = SHAPAnalysisCoordinator()
    _ui_integration = SHAPUIIntegration()
    
    MODULAR_SYSTEM_AVAILABLE = True
    logging.info("Enhanced SHAP system with reduced feature support loaded")
    
except ImportError as e:
    logging.error(f"Enhanced SHAP system not available: {e}")
    MODULAR_SYSTEM_AVAILABLE = False
    _coordinator = None
    _ui_integration = None
    
    # Create stub classes to prevent import errors
    class SHAPAnalysisCoordinator:
        def run_reduced_instance_analysis(self, *args, **kwargs):
            return {"success": False, "error": "SHAP system not available"}
    
    class SHAPUIIntegration:
        def display_reduced_shap_analysis(self, *args, **kwargs):
            import streamlit as st
            st.error("SHAP system not available")

# Backward compatibility functions
def get_shap_explainer_optimized(
    model_name: str,
    get_trained_model_func: Callable,
    top_n_features: int = DEFAULT_TOP_N_FEATURES,
    sample_size: int = 100
) -> Optional[shap.Explainer]:
    """Get optimized SHAP explainer with reduced features"""
    if MODULAR_SYSTEM_AVAILABLE and _coordinator:
        try:
            # FIXED: Use empty dict for explainer creation (no user inputs needed for this step)
            result = _coordinator.run_reduced_instance_analysis(
                {}, model_name, get_trained_model_func, top_n_features, sample_size
            )
            return result.get('explainer') if result.get('success') else None
        except Exception as e:
            logging.error(f"Optimized explainer creation failed: {e}")
    return None

def display_optimized_shap_analysis(
    user_inputs: Dict[str, Any], 
    model_name: str, 
    get_trained_model_func: Callable
) -> None:
    """Display optimized SHAP analysis with transparency"""
    if MODULAR_SYSTEM_AVAILABLE and _ui_integration:
        return _ui_integration.display_reduced_shap_analysis(
            user_inputs, model_name, get_trained_model_func
        )
    else:
        import streamlit as st
        st.error("Optimized SHAP analysis not available")

def get_shap_explainer(
    model_name: str, 
    get_trained_model_func: Callable,
    prepare_sample_data_func: Optional[Callable] = None,
    sample_size: int = 100
) -> Optional[Any]:
    """Enhanced backward compatible function with performance optimization"""
    # Try optimized version first
    optimized_explainer = get_shap_explainer_optimized(
        model_name, get_trained_model_func, sample_size=sample_size
    )
    if optimized_explainer:
        return optimized_explainer
    
    # Fallback to basic implementation
    logging.warning(f"Optimized SHAP not available for {model_name}, using fallback")
    return None

# Stub functions for backward compatibility
def get_shap_values_for_input(*args, **kwargs):
    """Stub function for backward compatibility"""
    logging.warning("get_shap_values_for_input called - functionality moved to coordinator")
    return None

def prepare_sample_data(*args, **kwargs):
    """Stub function for backward compatibility"""
    logging.warning("prepare_sample_data called - functionality moved to data_preparer")
    return None

def clear_explainer_cache():
    """Clear the explainer cache"""
    if MODULAR_SYSTEM_AVAILABLE and _coordinator:
        _coordinator.explainer_factory.clear_cache()

def get_cache_info() -> Dict[str, Any]:
    """Get cache information"""
    if MODULAR_SYSTEM_AVAILABLE and _coordinator:
        return _coordinator.explainer_factory.get_cache_info()
    return {"error": "SHAP system not available"}

# Export functions for UI compatibility
__all__ = [
    # Original exports for backward compatibility
    'get_shap_explainer', 
    'get_shap_values_for_input', 
    'prepare_sample_data',
    'clear_explainer_cache', 
    'get_cache_info',
    
    # Enhanced exports
    'get_shap_explainer_optimized', 
    'display_optimized_shap_analysis',
    'SHAPAnalysisCoordinator', 
    'SHAPUIIntegration',
    'DEFAULT_TOP_N_FEATURES', 
    'CORRELATION_THRESHOLD',
    
    # Module availability flag
    'MODULAR_SYSTEM_AVAILABLE'
]