# shap_analysis/value_calculator.py
"""
SHAP Value Calculator - Calculates SHAP values and creates summaries
Migrated from shap_analysis_backup.py with complete implementation
"""

import numpy as np
import shap
import logging
from typing import Dict, List, Optional, Any, Union
from constants import PipelineConstants

class SHAPValueCalculator:
    """Calculates SHAP values and creates summaries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_shap_values(
        self, 
        explainer: shap.Explainer, 
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for input data"""
        try:
            if explainer is None:
                self.logger.error("Explainer is None")
                return None
            
            if input_data is None or input_data.size == 0:
                self.logger.error("Input data is None or empty")
                return None
            
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            self.logger.info(f"Calculating SHAP values for input shape: {input_data.shape}")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            if shap_values is None:
                self.logger.error("SHAP values calculation returned None")
                return None
            
            # Handle different SHAP value formats
            processed_values = self._handle_shap_value_formats(shap_values)
            
            self.logger.info(f"SHAP values calculated successfully: shape {processed_values.shape}")
            return processed_values
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def calculate_interaction_values(
        self,
        explainer: shap.Explainer,
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """Calculate SHAP interaction values (TreeExplainer only)"""
        try:
            if explainer is None or not hasattr(explainer, 'shap_interaction_values'):
                self.logger.info("Interaction values not available for this explainer type")
                return None
            
            if input_data is None or input_data.size == 0:
                self.logger.error("Input data is None or empty")
                return None
            
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            self.logger.info("Calculating SHAP interaction values...")
            
            interaction_values = explainer.shap_interaction_values(input_data)
            
            if interaction_values is None:
                self.logger.warning("Interaction values calculation returned None")
                return None
            
            # Extract first instance if needed
            if isinstance(interaction_values, list):
                result = interaction_values[0][0] if len(interaction_values) > 0 else None
            else:
                result = interaction_values[0] if len(interaction_values.shape) > 2 else interaction_values
            
            if result is not None:
                self.logger.info(f"Interaction values calculated: shape {result.shape}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating interaction values: {e}")
            return None
    
    def create_summary_data(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        user_inputs: Dict,
        top_n: int = PipelineConstants.TOP_N_FEATURES
    ) -> List[Dict]:
        """Create summary data for SHAP values display"""
        try:
            if shap_values is None or len(shap_values) == 0:
                self.logger.error("No SHAP values provided for summary")
                return []
            
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
            
            # Ensure we have matching dimensions
            min_len = min(len(shap_values), len(feature_names))
            shap_vals = shap_values[:min_len]
            names = feature_names[:min_len]
            
            summary_data = []
            for i, (name, shap_val) in enumerate(zip(names, shap_vals)):
                # Get input value safely
                input_value = user_inputs.get(name, 'N/A') if user_inputs else 'N/A'
                
                # Format input value
                if isinstance(input_value, (int, float)):
                    if input_value == int(input_value):
                        input_str = str(int(input_value))
                    else:
                        input_str = f"{input_value:.3f}"
                else:
                    input_str = str(input_value)
                
                summary_data.append({
                    'feature_name': name,
                    'display_name': self._clean_feature_name(name),
                    'shap_value': float(shap_val),
                    'input_value': input_str,
                    'abs_impact': abs(float(shap_val)),
                    'direction': 'Increases' if shap_val > 0 else 'Decreases',
                    'impact_magnitude': self._categorize_impact(abs(float(shap_val)))
                })
            
            # Sort by absolute impact
            summary_data.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            self.logger.info(f"Created summary for {len(summary_data)} features, returning top {top_n}")
            return summary_data[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error creating summary data: {e}")
            return []
    
    def analyze_feature_impacts(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze overall feature impacts"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return {}
            
            analysis = {
                'total_features': len(shap_values),
                'positive_impact': float(np.sum(shap_values[shap_values > 0])),
                'negative_impact': float(np.sum(shap_values[shap_values < 0])),
                'net_impact': float(np.sum(shap_values)),
                'max_positive_impact': float(np.max(shap_values)),
                'max_negative_impact': float(np.min(shap_values)),
                'mean_abs_impact': float(np.mean(np.abs(shap_values))),
                'impact_variance': float(np.var(shap_values))
            }
            
            # Find most impactful features
            max_impact_idx = np.argmax(np.abs(shap_values))
            if max_impact_idx < len(feature_names):
                analysis['most_impactful_feature'] = {
                    'name': feature_names[max_impact_idx],
                    'impact': float(shap_values[max_impact_idx])
                }
            
            self.logger.debug(f"Feature impact analysis completed: {len(shap_values)} features analyzed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in feature impact analysis: {e}")
            return {}
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = 10
    ) -> List[Dict]:
        """Get top N most impactful features"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return []
            
            # Create feature impact pairs
            feature_impacts = []
            min_len = min(len(shap_values), len(feature_names))
            
            for i in range(min_len):
                feature_impacts.append({
                    'name': feature_names[i],
                    'display_name': self._clean_feature_name(feature_names[i]),
                    'impact': float(shap_values[i]),
                    'abs_impact': abs(float(shap_values[i])),
                    'direction': 'positive' if shap_values[i] > 0 else 'negative'
                })
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            return feature_impacts[:n]
            
        except Exception as e:
            self.logger.error(f"Error getting top features: {e}")
            return []
    
    def calculate_baseline_impact(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate baseline impact statistics"""
        try:
            if shap_values is None or len(shap_values) == 0:
                return {}
            
            return {
                'baseline_sum': float(np.sum(shap_values)),
                'baseline_mean': float(np.mean(shap_values)),
                'baseline_std': float(np.std(shap_values)),
                'baseline_min': float(np.min(shap_values)),
                'baseline_max': float(np.max(shap_values)),
                'positive_count': int(np.sum(shap_values > 0)),
                'negative_count': int(np.sum(shap_values < 0)),
                'zero_count': int(np.sum(shap_values == 0))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating baseline impact: {e}")
            return {}
    
    # Private helper methods
    def _handle_shap_value_formats(self, shap_values) -> np.ndarray:
        """Handle different SHAP value formats consistently"""
        try:
            # Handle list format (from some explainers)
            if isinstance(shap_values, list):
                if len(shap_values) > 0:
                    result = shap_values[0]
                else:
                    return np.array([])
            else:
                result = shap_values
            
            # Handle 2D format (extract first instance)
            if hasattr(result, 'ndim') and result.ndim == 2:
                result = result[0]
            
            # Ensure it's a numpy array
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling SHAP value formats: {e}")
            return np.array([])
    
    def _clean_feature_name(self, name: str) -> str:
        """Clean feature name for display"""
        if not name:
            return "Unknown Feature"
        
        # Replace underscores with spaces and title case
        clean_name = name.replace('_', ' ').title()
        
        # Limit length for display
        if len(clean_name) > 25:
            clean_name = clean_name[:22] + "..."
        
        return clean_name
    
    def _categorize_impact(self, abs_impact: float) -> str:
        """Categorize impact magnitude"""
        if abs_impact > 1.0:
            return "High"
        elif abs_impact > 0.1:
            return "Medium"
        elif abs_impact > 0.01:
            return "Low"
        else:
            return "Minimal"