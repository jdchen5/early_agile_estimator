# shap_analysis/explainer_factory.py
"""
SHAP Explainer Factory - Creates and manages SHAP explainers
Single responsibility: Explainer creation and caching
"""

import shap
import logging
from typing import Dict, Optional, Callable
from constants import PipelineConstants

class SHAPExplainerFactory:
    """Creates and manages SHAP explainers with caching"""
    
    def __init__(self):
        self._explainer_cache = {}
    
    def create_explainer(
        self, 
        model_name: str, 
        get_trained_model_func: Callable,
        background_data,
        sample_size: int = PipelineConstants.DEFAULT_SAMPLE_SIZE
    ) -> Optional[shap.Explainer]:
        """Create appropriate SHAP explainer for the model"""
        
    def get_cached_explainer(self, model_name: str, sample_size: int) -> Optional[shap.Explainer]:
        """Get explainer from cache if available"""
        
    def clear_cache(self):
        """Clear the explainer cache"""
        
    def get_cache_info(self) -> Dict:
        """Get cache information"""
        
    def _extract_pycaret_estimator(self, model):
        """Extract underlying estimator from PyCaret model"""
        
    def _determine_explainer_type(self, model) -> str:
        """Determine best explainer type for model"""
        
    def _create_tree_explainer(self, model, background_data):
        """Create TreeExplainer for tree-based models"""
        
    def _create_linear_explainer(self, model, background_data):
        """Create LinearExplainer for linear models"""
        
    def _create_kernel_explainer(self, model, background_data):
        """Create KernelExplainer as fallback"""