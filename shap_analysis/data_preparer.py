# shap_analysis/data_preparer.py
"""Enhanced SHAP Data Preparer with reduced feature support"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

class SHAPDataPreparer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_cache = {}
        self.validation_cache = {}
    
    def prepare_reduced_background_data(
        self, 
        model_name: str,
        n_samples: int = 100,
        top_n_features: int = 15
    ) -> Optional[np.ndarray]:
        """Prepare background data with only top N features for the model"""
        try:
            # Get top features for this model
            top_features = self._get_top_features_for_model(model_name, top_n_features)
            if not top_features:
                self.logger.error(f"No top features found for model {model_name}")
                return None
            
            # Generate background data using existing method
            full_background = self.prepare_background_data(n_samples, model_name)
            if full_background is None:
                return None
            
            # Convert to DataFrame and select top features
            from models import get_model_expected_features, get_trained_model
            model = get_trained_model(model_name)
            all_features = get_model_expected_features(model)
            
            if len(all_features) == full_background.shape[1]:
                bg_df = pd.DataFrame(full_background, columns=all_features)
                # Select only top features that exist in the data
                available_top_features = [f for f in top_features if f in bg_df.columns]
                reduced_bg = bg_df[available_top_features].values
                
                self.logger.info(f"Reduced background data: {full_background.shape} → {reduced_bg.shape}")
                return reduced_bg
            
            return full_background  # Fallback to full data
            
        except Exception as e:
            self.logger.error(f"Error preparing reduced background data: {e}")
            return None
    
    def _get_top_features_for_model(self, model_name: str, n: int = 15) -> List[str]:
        """Get top N features for specific model"""
        try:
            # Check cache first
            cache_key = f"{model_name}_{n}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # Try to load from CSV file
            feature_file = f"config/synthetic_isbsg2016r1_1_finance_sdv_generated_feature_importance_{model_name}.csv"
            
            if os.path.exists(feature_file):
                df = pd.read_csv(feature_file)
                if 'feature' in df.columns and 'importance' in df.columns:
                    top_features = df.nlargest(n, 'importance')['feature'].tolist()
                    self.feature_cache[cache_key] = top_features
                    self.logger.info(f"Loaded top {n} features for {model_name} from file")
                    return top_features
            
            # Fallback: extract from model
            top_features = self._extract_features_from_model(model_name, n)
            self.feature_cache[cache_key] = top_features
            return top_features
            
        except Exception as e:
            self.logger.error(f"Error getting top features for {model_name}: {e}")
            return []
    
    def _extract_features_from_model(self, model_name: str, n: int) -> List[str]:
        """Extract top features from model feature importance"""
        try:
            from models import get_trained_model, get_model_expected_features
            
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