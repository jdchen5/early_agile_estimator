# shap_analysis/__init__.py
"""
SHAP Analysis Module - Modular SHAP analysis for ML models
Provides clean interfaces for different types of SHAP analysis
"""

from .analysis_coordinator import SHAPAnalysisCoordinator
from .explainer_factory import SHAPExplainerFactory  
from .data_preparer import SHAPDataPreparer
from .value_calculator import SHAPValueCalculator

# Backward compatibility - main interface
def get_shap_explainer(model_name, get_trained_model_func, prepare_sample_data_func=None, sample_size=100):
    """Backward compatibility wrapper"""
    coordinator = SHAPAnalysisCoordinator()
    # Implementation using new classes
    
# Export main interface
__all__ = [
    'SHAPAnalysisCoordinator',
    'SHAPExplainerFactory', 
    'SHAPDataPreparer',
    'SHAPValueCalculator',
    'get_shap_explainer'  # Backward compatibility
]