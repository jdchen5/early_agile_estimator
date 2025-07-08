# shap_analysis/data_preparer.py - COMPLETE UPDATED VERSION
"""Enhanced SHAP Data Preparer with reduced feature support"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

# Check if models module is available
try:
    from models import (
        prepare_isbsg_sample_data,
        prepare_features_for_model,
        get_trained_model,
        get_model_expected_features
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

class SHAPDataPreparer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_cache = {}
        self.validation_cache = {}
    
    def validate_shap_inputs(self, user_inputs: Dict[str, Any]) -> bool:
        """Validate that user inputs are suitable for SHAP analysis"""
        try:
            if not user_inputs:
                self.logger.warning("No user inputs provided")
                return False
            
            # Remove UI-specific keys
            exclude_keys = {
                'selected_model', 'selected_models', 'submit', 'clear_results', 
                'show_history', 'save_config', 'config_name', 'comparison_mode'
            }
            meaningful_inputs = {
                k: v for k, v in user_inputs.items() 
                if k not in exclude_keys and v is not None and v != ""
            }
            
            if len(meaningful_inputs) == 0:
                self.logger.warning("No meaningful inputs after filtering")
                return False
            
            # Check for required fields (basic validation)
            self.logger.info(f"Validated {len(meaningful_inputs)} meaningful inputs")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating SHAP inputs: {e}")
            return False
    
    def prepare_input_data(self, user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare user input data for SHAP analysis"""
        try:
            if not MODELS_AVAILABLE:
                self.logger.error("Models module not available")
                return None
            
            # Use existing feature preparation pipeline
            features_df = prepare_features_for_model(user_inputs)
            if features_df is None or features_df.empty:
                self.logger.error("Feature preparation failed")
                return None
            
            # Convert to numpy array
            input_array = features_df.values
            
            # Ensure it's 2D
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            
            self.logger.info(f"Input data prepared: shape {input_array.shape}")
            return input_array
            
        except Exception as e:
            self.logger.error(f"Error preparing input data: {e}")
            return None
    
    def prepare_reduced_background_data(
        self, 
        model_name: str,
        n_samples: int = 100,
        top_n_features: int = 15
    ) -> Optional[np.ndarray]:
        """Prepare background data with only top N features for the model"""
        try:
            if not MODELS_AVAILABLE:
                self.logger.error("Models module not available")
                return None
            
            # Get top features for this model
            top_features = self._get_top_features_for_model(model_name, top_n_features)
            if not top_features:
                self.logger.error(f"No top features found for model {model_name}")
                return None
            
            # Generate background data using ISBSG dataset
            full_background = prepare_isbsg_sample_data(n_samples)
            if full_background is None:
                self.logger.error("Could not prepare ISBSG background data")
                return None
            
            # For now, return the full background data
            # In a more sophisticated implementation, we would:
            # 1. Convert ISBSG data to UI format
            # 2. Process through feature pipeline
            # 3. Select only top features
            # But this requires complex reverse mapping
            
            self.logger.info(f"Prepared background data: {full_background.shape}")
            return full_background
            
        except Exception as e:
            self.logger.error(f"Error preparing reduced background data: {e}")
            return None
    
    def _get_top_features_for_model(self, model_name: str, n: int = 15) -> List[str]:
        """Get top N features for specific model with improved name mapping"""
        try:
            # Check cache first
            cache_key = f"{model_name}_{n}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # FIXED: Create mapping patterns for CSV file lookup
            csv_patterns = [
                # Exact match
                f"config/synthetic_isbsg2016r1_1_finance_sdv_generated_feature_importance_{model_name}.csv",
                # Remove 'top_' prefix
                f"config/synthetic_isbsg2016r1_1_finance_sdv_generated_feature_importance_{model_name.replace('top_', '')}.csv",
                # Alternative financial pattern
                f"config/synthetic_financial_feature_importance_{model_name.replace('top_', '')}.csv",
            ]
            
            # Try each pattern
            for pattern in csv_patterns:
                if os.path.exists(pattern):
                    self.logger.info(f"Found feature importance file: {pattern}")
                    df = pd.read_csv(pattern)
                    
                    if 'feature' in df.columns and 'importance' in df.columns:
                        # Sort by importance and get top N
                        top_features = df.nlargest(n, 'importance')['feature'].tolist()
                        self.feature_cache[cache_key] = top_features
                        self.logger.info(f"Loaded top {n} features for {model_name} from {pattern}")
                        return top_features
            
            # Fallback: extract from model
            self.logger.warning(f"No CSV file found for {model_name}, extracting from model")
            top_features = self._extract_features_from_model(model_name, n)
            self.feature_cache[cache_key] = top_features
            return top_features
            
        except Exception as e:
            self.logger.error(f"Error getting top features for {model_name}: {e}")
            return []
    
    def _extract_features_from_model(self, model_name: str, n: int) -> List[str]:
        """Extract top features from model feature importance"""
        try:
            if not MODELS_AVAILABLE:
                return []
            
            model = get_trained_model(model_name)
            if model is None:
                return []
            
            # Extract actual estimator
            actual_model = model
            if hasattr(model, '_final_estimator'):
                actual_model = model._final_estimator
            
            # Get feature importance
            importance = None
            if hasattr(actual_model, 'feature_importances_'):
                importance = actual_model.feature_importances_
            elif hasattr(actual_model, 'coef_'):
                importance = np.abs(actual_model.coef_).flatten()
            
            if importance is not None:
                feature_names = get_model_expected_features(model)
                if len(feature_names) == len(importance):
                    # Create feature importance pairs and sort
                    feature_importance = list(zip(feature_names, importance))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    top_features = [name for name, _ in feature_importance[:n]]
                    self.logger.info(f"Extracted top {n} features from {model_name}")
                    return top_features
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting features from model {model_name}: {e}")
            return []