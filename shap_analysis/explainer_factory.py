# shap_analysis/explainer_factory.py
"""
SHAP Explainer Factory - Creates and manages SHAP explainers with caching
Migrated from shap_analysis_backup.py with complete implementation
"""

import shap
import logging
import warnings
from typing import Dict, Optional, Callable, Any
from constants import PipelineConstants

# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

class SHAPExplainerFactory:
    """Creates and manages SHAP explainers with caching"""
    
    def __init__(self):
        self._explainer_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def create_explainer(
        self, 
        model_name: str, 
        get_trained_model_func: Callable,
        background_data: Any,
        sample_size: int = PipelineConstants.DEFAULT_SAMPLE_SIZE
    ) -> Optional[shap.Explainer]:
        """Create appropriate SHAP explainer for the model"""
        
        # Check cache first
        cache_key = f"{model_name}_{sample_size}"
        cached_explainer = self._get_cached_explainer(cache_key)
        if cached_explainer is not None:
            self.logger.info(f"Using cached SHAP explainer for {model_name}")
            return cached_explainer
        
        try:
            # Load the model
            model = get_trained_model_func(model_name)
            if model is None:
                self.logger.error(f"Could not load model '{model_name}'")
                return None
            
            # Extract actual estimator from PyCaret wrapper
            actual_model = self._extract_pycaret_estimator(model)
            if actual_model is None:
                self.logger.error(f"Could not extract estimator from model")
                return None
            
            # Determine explainer type
            explainer_type = self._determine_explainer_type(actual_model)
            self.logger.info(f"Creating {explainer_type} for {model_name}")
            
            # Create appropriate explainer
            explainer = None
            if explainer_type == "tree":
                explainer = self._create_tree_explainer(actual_model, background_data)
            elif explainer_type == "linear":
                explainer = self._create_linear_explainer(actual_model, background_data)
            else:
                explainer = self._create_kernel_explainer(model, background_data)  # Use full model for kernel
            
            # Cache successful explainer
            if explainer is not None:
                self._explainer_cache[cache_key] = explainer
                self.logger.info(f"SHAP explainer created and cached for {model_name}")
            
            return explainer
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer for {model_name}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the explainer cache"""
        self._explainer_cache.clear()
        self.logger.info("SHAP explainer cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get cache information"""
        return {
            'cached_models': list(self._explainer_cache.keys()),
            'cache_size': len(self._explainer_cache)
        }
    
    def _get_cached_explainer(self, cache_key: str) -> Optional[shap.Explainer]:
        """Get explainer from cache if available"""
        return self._explainer_cache.get(cache_key)
    
    def _extract_pycaret_estimator(self, model):
        """Extract underlying estimator from PyCaret model"""
        try:
            # Check if it's a PyCaret model
            if hasattr(model, '_final_estimator'):
                actual_model = model._final_estimator
                self.logger.debug(f"Extracted final estimator: {type(actual_model).__name__}")
                return actual_model
                
            # Check if it's a pipeline
            elif hasattr(model, 'named_steps'):
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'predict') and not step_name.startswith(('scaler', 'encoder', 'imputer')):
                        self.logger.debug(f"Extracted from pipeline step '{step_name}': {type(step).__name__}")
                        return step
                
                # Fallback to last step
                step_names = list(model.named_steps.keys())
                if step_names:
                    final_step = model.named_steps[step_names[-1]]
                    return final_step
                    
            # Check for sklearn pipeline format
            elif hasattr(model, 'steps') and len(model.steps) > 0:
                final_step = model.steps[-1][1]
                self.logger.debug(f"Extracted from sklearn pipeline: {type(final_step).__name__}")
                return final_step
            
            # If it's already a raw estimator
            self.logger.debug(f"Model appears to be raw estimator: {type(model).__name__}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error extracting estimator: {e}")
            return model
    
    def _determine_explainer_type(self, model) -> str:
        """Determine best explainer type for model"""
        model_type = type(model).__name__.lower()
        
        # Tree-based models
        tree_keywords = ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient', 
                        'randomforest', 'extratrees', 'decisiontree']
        if any(keyword in model_type for keyword in tree_keywords):
            return "tree"
        
        # Linear models
        linear_keywords = ['linear', 'lasso', 'ridge', 'elastic', 'bayesianridge']
        if any(keyword in model_type for keyword in linear_keywords):
            return "linear"
        
        # Default to kernel explainer
        return "kernel"
    
    def _create_tree_explainer(self, model, background_data):
        """Create TreeExplainer for tree-based models"""
        try:
            if background_data is not None:
                explainer = shap.TreeExplainer(model, background_data)
            else:
                explainer = shap.TreeExplainer(model)
            self.logger.info("TreeExplainer created successfully")
            return explainer
        except Exception as e:
            self.logger.warning(f"TreeExplainer failed: {e}")
            return None
    
    def _create_linear_explainer(self, model, background_data):
        """Create LinearExplainer for linear models"""
        try:
            if background_data is not None:
                explainer = shap.LinearExplainer(model, background_data)
                self.logger.info("LinearExplainer created successfully")
                return explainer
        except Exception as e:
            self.logger.warning(f"LinearExplainer failed: {e}")
            return None
    
    def _create_kernel_explainer(self, model, background_data):
        """Create KernelExplainer as fallback"""
        try:
            if background_data is not None:
                # Use smaller sample for KernelExplainer (performance)
                kernel_sample = background_data[:min(PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE, len(background_data))]
                
                # Create prediction function
                def model_predict_func(X):
                    try:
                        if hasattr(model, 'predict'):
                            return model.predict(X)
                        else:
                            raise ValueError("Model has no predict method")
                    except Exception as e:
                        self.logger.error(f"Prediction error in KernelExplainer: {e}")
                        raise
                
                explainer = shap.KernelExplainer(model_predict_func, kernel_sample)
                self.logger.info("KernelExplainer created successfully")
                return explainer
        except Exception as e:
            self.logger.warning(f"KernelExplainer failed: {e}")
            return None