# shap_analysis/__init__.py
"""Enhanced SHAP Analysis Module with reduced feature optimization"""

import logging

# Module constants
DEFAULT_TOP_N_FEATURES = 15
CORRELATION_THRESHOLD = 0.85

# Load components
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

# New optimized functions
def get_shap_explainer_optimized(
    model_name: str,
    get_trained_model_func,
    top_n_features: int = DEFAULT_TOP_N_FEATURES,
    sample_size: int = 100
):
    """Get optimized SHAP explainer with reduced features"""
    if MODULAR_SYSTEM_AVAILABLE and _coordinator:
        try:
            result = _coordinator.run_reduced_instance_analysis(
                {}, model_name, get_trained_model_func, top_n_features, sample_size
            )
            return result.get('explainer') if result.get('success') else None
        except Exception as e:
            logging.error(f"Optimized explainer creation failed: {e}")
    return None

def display_optimized_shap_analysis(user_inputs, model_name, get_trained_model_func):
    """Display optimized SHAP analysis with transparency"""
    if MODULAR_SYSTEM_AVAILABLE and _ui_integration:
        return _ui_integration.display_reduced_shap_analysis(
            user_inputs, model_name, get_trained_model_func
        )
    else:
        import streamlit as st
        st.error("Optimized SHAP analysis not available")

# Backward compatibility with performance boost
def get_shap_explainer(model_name, get_trained_model_func, prepare_sample_data_func=None, sample_size=100):
    """Enhanced backward compatible function with performance optimization"""
    # Try optimized version first
    optimized_explainer = get_shap_explainer_optimized(model_name, get_trained_model_func, sample_size=sample_size)
    if optimized_explainer:
        return optimized_explainer
    
    # Fallback to original approach
    # ... existing fallback code ...

# Export optimized functions
__all__ = [
    # Original exports
    'get_shap_explainer', 'get_shap_values_for_input', 'prepare_sample_data',
    'clear_explainer_cache', 'get_cache_info',
    
    # Enhanced exports
    'get_shap_explainer_optimized', 'display_optimized_shap_analysis',
    'SHAPAnalysisCoordinator', 'SHAPUIIntegration',
    'DEFAULT_TOP_N_FEATURES', 'CORRELATION_THRESHOLD'
]