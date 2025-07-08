# shap_analysis/explainer_factory.py
"""
SHAP Explainer Factory - Creates and manages SHAP explainers with caching

This module provides a factory class for creating SHAP explainers with intelligent
caching and automatic model type detection. It handles the complexity of extracting
underlying estimators from wrapped models (like PyCaret) and selecting the most
appropriate SHAP explainer type.

Key features:
- Automatic model type detection and explainer selection
- Intelligent caching to avoid recreating explainers
- Support for tree-based, linear, and general models
- PyCaret model wrapper handling
- Memory-efficient KernelExplainer implementation

Migrated from shap_analysis_backup.py with complete implementation and enhanced
error handling and logging.
"""

import shap
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Optional, Callable, Any, Union

from constants import PipelineConstants


# Module-level constants
TREE_MODEL_KEYWORDS = [
    'forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient', 
    'randomforest', 'extratrees', 'decisiontree'
]

LINEAR_MODEL_KEYWORDS = [
    'linear', 'lasso', 'ridge', 'elastic', 'bayesianridge'
]

PIPELINE_STEP_EXCLUSIONS = {'scaler', 'encoder', 'imputer', 'transformer'}

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='shap')


class SHAPExplainerFactory:
    """
    Factory class for creating and managing SHAP explainers with caching.
    
    This class handles the creation of appropriate SHAP explainers based on model
    type detection. It provides intelligent caching to avoid expensive explainer
    recreation and supports various model wrapper formats including PyCaret.
    
    The factory automatically selects the best explainer type:
    - TreeExplainer for tree-based models (fastest, most accurate)
    - LinearExplainer for linear models (fast, exact)
    - KernelExplainer for other models (slower, model-agnostic)
    """
    
    def __init__(self):
        """Initialize explainer factory with empty cache."""
        self._explainer_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def create_explainer(
        self, 
        model_name: str, 
        get_trained_model_func: Callable,
        background_data: Any,
        sample_size: int = PipelineConstants.DEFAULT_SAMPLE_SIZE
    ) -> Optional[shap.Explainer]:
        """
        Create appropriate SHAP explainer for the specified model.
        
        This is the main entry point for explainer creation. It handles caching,
        model loading, estimator extraction, and explainer type selection.
        
        Args:
            model_name: Technical name of the model
            get_trained_model_func: Function to retrieve the trained model
            background_data: Background data for SHAP baseline
            sample_size: Sample size for caching key generation
            
        Returns:
            SHAP explainer object or None if creation fails
        """
        # Check cache first
        cache_key = self._generate_cache_key(model_name, sample_size)
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
            
            # Extract actual estimator from wrapper (e.g., PyCaret)
            actual_model = self._extract_pycaret_estimator(model)
            if actual_model is None:
                self.logger.error(f"Could not extract estimator from model")
                return None
            
            # Determine optimal explainer type
            explainer_type = self._determine_explainer_type(actual_model)
            self.logger.info(f"Creating {explainer_type}Explainer for {model_name}")
            
            # Create appropriate explainer
            explainer = self._create_explainer_by_type(
                explainer_type, actual_model, model, background_data
            )
            
            # Cache successful explainer
            if explainer is not None:
                self._cache_explainer(cache_key, explainer)
                self.logger.info(f"SHAP explainer created and cached for {model_name}")
            else:
                self.logger.warning(f"Failed to create any explainer for {model_name}")
            
            return explainer
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer for {model_name}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """
        Clear the explainer cache to free memory.
        
        This should be called periodically to prevent memory buildup,
        especially when working with many different models.
        """
        cache_size = len(self._explainer_cache)
        self._explainer_cache.clear()
        self.logger.info(f"SHAP explainer cache cleared ({cache_size} items removed)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary containing cache statistics and contents
        """
        return {
            'cached_models': list(self._explainer_cache.keys()),
            'cache_size': len(self._explainer_cache),
            'memory_usage_estimate': self._estimate_cache_memory_usage()
        }
    
    def remove_from_cache(self, model_name: str, sample_size: int = None) -> bool:
        """
        Remove specific model from cache.
        
        Args:
            model_name: Name of the model to remove
            sample_size: Optional sample size for specific cache entry
            
        Returns:
            True if item was removed, False if not found
        """
        if sample_size is not None:
            cache_key = self._generate_cache_key(model_name, sample_size)
            if cache_key in self._explainer_cache:
                del self._explainer_cache[cache_key]
                self.logger.info(f"Removed {cache_key} from explainer cache")
                return True
        else:
            # Remove all entries for this model
            keys_to_remove = [key for key in self._explainer_cache.keys() if model_name in key]
            for key in keys_to_remove:
                del self._explainer_cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Removed {len(keys_to_remove)} cache entries for {model_name}")
                return True
        
        return False
    
    def _get_cached_explainer(self, cache_key: str) -> Optional[shap.Explainer]:
        """
        Retrieve explainer from cache if available.
        
        Args:
            cache_key: Cache key for the explainer
            
        Returns:
            Cached explainer or None if not found
        """
        return self._explainer_cache.get(cache_key)
    
    def _cache_explainer(self, cache_key: str, explainer: shap.Explainer) -> None:
        """
        Store explainer in cache.
        
        Args:
            cache_key: Cache key for storage
            explainer: SHAP explainer to cache
        """
        self._explainer_cache[cache_key] = explainer
    
    def _generate_cache_key(self, model_name: str, sample_size: int) -> str:
        """
        Generate cache key for explainer storage.
        
        Args:
            model_name: Name of the model
            sample_size: Sample size used for background data
            
        Returns:
            String cache key
        """
        return f"{model_name}_{sample_size}"
    
    def _extract_pycaret_estimator(self, model) -> Optional[Any]:
        """
        Extract underlying estimator from wrapped models (PyCaret, pipelines).
        
        Args:
            model: Wrapped model object
            
        Returns:
            Underlying estimator or original model if extraction fails
        """
        try:
            # Check if it's a PyCaret model with _final_estimator
            if hasattr(model, '_final_estimator'):
                actual_model = model._final_estimator
                self.logger.debug(f"Extracted final estimator: {type(actual_model).__name__}")
                return actual_model
                
            # Check if it's a sklearn pipeline with named_steps
            elif hasattr(model, 'named_steps'):
                # Look for the actual estimator (not preprocessing steps)
                for step_name, step in model.named_steps.items():
                    if (hasattr(step, 'predict') and 
                        not any(exclusion in step_name.lower() for exclusion in PIPELINE_STEP_EXCLUSIONS)):
                        self.logger.debug(f"Extracted from pipeline step '{step_name}': {type(step).__name__}")
                        return step
                
                # Fallback to last step if no suitable step found
                step_names = list(model.named_steps.keys())
                if step_names:
                    final_step = model.named_steps[step_names[-1]]
                    self.logger.debug(f"Using final pipeline step: {type(final_step).__name__}")
                    return final_step
                    
            # Check for sklearn pipeline format with steps attribute
            elif hasattr(model, 'steps') and len(model.steps) > 0:
                final_step = model.steps[-1][1]  # (name, estimator) tuple
                self.logger.debug(f"Extracted from sklearn pipeline: {type(final_step).__name__}")
                return final_step
            
            # If none of the above, assume it's already a raw estimator
            self.logger.debug(f"Model appears to be raw estimator: {type(model).__name__}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error extracting estimator: {e}")
            return model  # Return original model as fallback
    
    def _determine_explainer_type(self, model) -> str:
        """
        Determine the best SHAP explainer type for the given model.
        
        Args:
            model: Model object to analyze
            
        Returns:
            String indicating explainer type ('tree', 'linear', or 'kernel')
        """
        model_type = type(model).__name__.lower()
        
        # Check for tree-based models (best performance with TreeExplainer)
        if any(keyword in model_type for keyword in TREE_MODEL_KEYWORDS):
            return "tree"
        
        # Check for linear models (exact explanations with LinearExplainer)
        if any(keyword in model_type for keyword in LINEAR_MODEL_KEYWORDS):
            return "linear"
        
        # Default to kernel explainer for unknown model types
        self.logger.debug(f"Unknown model type '{model_type}', using KernelExplainer")
        return "kernel"
    
    def _create_explainer_by_type(
        self, 
        explainer_type: str, 
        actual_model: Any, 
        full_model: Any, 
        background_data: Any
    ) -> Optional[shap.Explainer]:
        """
        Create explainer based on determined type.
        
        Args:
            explainer_type: Type of explainer to create
            actual_model: Extracted underlying model
            full_model: Original wrapped model (for kernel explainer)
            background_data: Background data for explainer
            
        Returns:
            Created explainer or None if creation fails
        """
        if explainer_type == "tree":
            return self._create_tree_explainer(actual_model, background_data)
        elif explainer_type == "linear":
            return self._create_linear_explainer(actual_model, background_data)
        else:
            return self._create_kernel_explainer(full_model, background_data)
    
    def _create_tree_explainer(self, model, background_data) -> Optional[shap.TreeExplainer]:
        """
        Create TreeExplainer for tree-based models.
        
        Args:
            model: Tree-based model
            background_data: Background data for explainer
            
        Returns:
            TreeExplainer or None if creation fails
        """
        try:
            if background_data is not None:
                explainer = shap.TreeExplainer(model, background_data)
            else:
                explainer = shap.TreeExplainer(model)
            
            self.logger.debug("TreeExplainer created successfully")
            return explainer
            
        except Exception as e:
            self.logger.warning(f"TreeExplainer creation failed: {e}")
            return None
    
    def _create_linear_explainer(self, model, background_data) -> Optional[shap.LinearExplainer]:
        """
        Create LinearExplainer for linear models.
        
        Args:
            model: Linear model
            background_data: Background data for explainer
            
        Returns:
            LinearExplainer or None if creation fails
        """
        try:
            if background_data is not None:
                explainer = shap.LinearExplainer(model, background_data)
                self.logger.debug("LinearExplainer created successfully")
                return explainer
            else:
                self.logger.warning("LinearExplainer requires background data")
                return None
                
        except Exception as e:
            self.logger.warning(f"LinearExplainer creation failed: {e}")
            return None
    
    def _create_kernel_explainer(self, model, background_data) -> Optional[shap.KernelExplainer]:
        """
        Create KernelExplainer as fallback for any model type.
        
        Args:
            model: Any model with predict method
            background_data: Background data for explainer
            
        Returns:
            KernelExplainer or None if creation fails
        """
        try:
            if background_data is None:
                self.logger.warning("KernelExplainer requires background data")
                return None
            
            # Use smaller sample for KernelExplainer to improve performance
            kernel_sample_size = min(
                PipelineConstants.KERNEL_EXPLAINER_SAMPLE_SIZE, 
                len(background_data)
            )
            kernel_sample = background_data[:kernel_sample_size]
            
            # Create prediction function with error handling
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
            self.logger.debug(f"KernelExplainer created with {kernel_sample_size} background samples")
            return explainer
            
        except Exception as e:
            self.logger.warning(f"KernelExplainer creation failed: {e}")
            return None
    
    def _estimate_cache_memory_usage(self) -> str:
        """
        Estimate memory usage of cached explainers.
        
        Returns:
            String description of estimated memory usage
        """
        # This is a rough estimate - actual memory usage depends on explainer type
        # and background data size
        cache_size = len(self._explainer_cache)
        if cache_size == 0:
            return "0 MB"
        
        # Rough estimate: 10-50 MB per cached explainer
        estimated_mb = cache_size * 25  # Average estimate
        
        if estimated_mb < 1024:
            return f"~{estimated_mb} MB"
        else:
            return f"~{estimated_mb / 1024:.1f} GB"


# Module-level utility functions
def validate_explainer_compatibility(model, explainer_type: str = None) -> Dict[str, Any]:
    """
    Validate if a model is compatible with SHAP explainers.
    
    Args:
        model: Model object to validate
        explainer_type: Optional specific explainer type to check
        
    Returns:
        Dictionary with compatibility information
    """
    result = {
        'compatible': False,
        'recommended_explainer': None,
        'issues': [],
        'model_type': type(model).__name__
    }
    
    try:
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            result['issues'].append("Model has no predict method")
            return result
        
        # Determine best explainer type
        factory = SHAPExplainerFactory()
        recommended_type = factory._determine_explainer_type(model)
        result['recommended_explainer'] = f"{recommended_type}Explainer"
        
        # Check specific compatibility
        if explainer_type:
            if explainer_type.lower() == recommended_type:
                result['compatible'] = True
            else:
                result['issues'].append(f"Requested {explainer_type}Explainer but {recommended_type}Explainer is recommended")
        else:
            result['compatible'] = True
        
    except Exception as e:
        result['issues'].append(f"Validation error: {str(e)}")
    
    return result


def get_supported_explainer_types() -> Dict[str, Dict[str, Any]]:
    """
    Get information about supported explainer types.
    
    Returns:
        Dictionary with explainer type information
    """
    return {
        'tree': {
            'name': 'TreeExplainer',
            'description': 'Fast and exact for tree-based models',
            'supported_models': TREE_MODEL_KEYWORDS,
            'performance': 'High',
            'accuracy': 'Exact'
        },
        'linear': {
            'name': 'LinearExplainer',
            'description': 'Fast and exact for linear models',
            'supported_models': LINEAR_MODEL_KEYWORDS,
            'performance': 'High',
            'accuracy': 'Exact'
        },
        'kernel': {
            'name': 'KernelExplainer',
            'description': 'Model-agnostic but slower',
            'supported_models': ['Any model with predict method'],
            'performance': 'Low to Medium',
            'accuracy': 'Approximate'
        }
    }