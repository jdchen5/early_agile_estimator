# shap_analysis/value_calculator.py
"""
SHAP Value Calculator - Calculates SHAP values and creates summaries

This module provides comprehensive SHAP value calculation and analysis functionality.
It handles the computation of SHAP values, interaction values, and generates structured
summaries for visualization and interpretation.

Key features:
- SHAP value calculation with robust error handling
- Interaction value computation for tree-based explainers
- Feature impact analysis and statistical summaries
- Top feature identification and ranking
- Display-ready data formatting with clean feature names
- Impact magnitude categorization for better interpretation

The calculator handles various SHAP value formats from different explainer types
and provides consistent output for downstream consumption by UI components.

Migrated from shap_analysis_backup.py with complete implementation and enhanced
analysis capabilities.
"""

import logging
import shap
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any

from constants import PipelineConstants


# Module-level constants
DEFAULT_TOP_N_FEATURES = 10
MAX_FEATURE_NAME_LENGTH = 25
FEATURE_NAME_TRUNCATE_SUFFIX = "..."

# Impact magnitude thresholds
IMPACT_THRESHOLDS = {
    'high': 1.0,
    'medium': 0.1,
    'low': 0.01
}

# Display formatting constants
NUMERIC_PRECISION = 3
INTEGER_THRESHOLD = 1e-10


class SHAPValueCalculator:
    """
    Calculates and analyzes SHAP values with comprehensive summary generation.
    
    This class handles the computation of SHAP values and interaction values,
    provides statistical analysis of feature impacts, and generates formatted
    summaries suitable for visualization and interpretation. It supports
    various SHAP explainer types and handles different output formats consistently.
    """
    
    def __init__(self):
        """Initialize value calculator with logging."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_shap_values(
        self, 
        explainer: shap.Explainer, 
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for given input data.
        
        This is the main entry point for SHAP value calculation. It handles
        input validation, format normalization, and error recovery.
        
        Args:
            explainer: SHAP explainer object
            input_data: Input data array (will be reshaped if 1D)
            feature_names: Optional list of feature names for logging
            
        Returns:
            SHAP values array or None if calculation fails
        """
        try:
            # Validate inputs
            if explainer is None:
                self.logger.error("Explainer is None")
                return None
            
            if input_data is None or input_data.size == 0:
                self.logger.error("Input data is None or empty")
                return None
            
            # Ensure input is 2D for SHAP compatibility
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
                self.logger.debug("Reshaped 1D input to 2D for SHAP calculation")
            
            self.logger.info(f"Calculating SHAP values for input shape: {input_data.shape}")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            if shap_values is None:
                self.logger.error("SHAP values calculation returned None")
                return None
            
            # Handle different SHAP value formats consistently
            processed_values = self._handle_shap_value_formats(shap_values)
            
            if processed_values is not None and len(processed_values) > 0:
                self.logger.info(f"SHAP values calculated successfully: shape {processed_values.shape}")
            else:
                self.logger.warning("SHAP values calculation produced empty result")
                return None
            
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
        """
        Calculate SHAP interaction values for feature interactions.
        
        Interaction values are only available for TreeExplainer and show how
        features interact with each other to influence predictions.
        
        Args:
            explainer: SHAP explainer object (must support interaction values)
            input_data: Input data array
            feature_names: Optional list of feature names for logging
            
        Returns:
            Interaction values matrix or None if not supported/calculation fails
        """
        try:
            # Check explainer compatibility
            if explainer is None or not hasattr(explainer, 'shap_interaction_values'):
                self.logger.info("Interaction values not available for this explainer type")
                return None
            
            # Validate input data
            if input_data is None or input_data.size == 0:
                self.logger.error("Input data is None or empty")
                return None
            
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            self.logger.info("Calculating SHAP interaction values...")
            
            # Calculate interaction values
            interaction_values = explainer.shap_interaction_values(input_data)
            
            if interaction_values is None:
                self.logger.warning("Interaction values calculation returned None")
                return None
            
            # Extract first instance for single prediction
            result = self._extract_interaction_instance(interaction_values)
            
            if result is not None:
                self.logger.info(f"Interaction values calculated: shape {result.shape}")
            else:
                self.logger.warning("Could not extract interaction values from result")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating interaction values: {e}")
            return None
    
    def create_summary_data(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        user_inputs: Dict[str, Any],
        top_n: int = PipelineConstants.TOP_N_FEATURES
    ) -> List[Dict[str, Any]]:
        """
        Create structured summary data for SHAP values display.
        
        Args:
            shap_values: Array of SHAP values
            feature_names: List of feature names
            user_inputs: Dictionary of user input values
            top_n: Number of top features to include in summary
            
        Returns:
            List of dictionaries containing feature summaries
        """
        try:
            if shap_values is None or len(shap_values) == 0:
                self.logger.error("No SHAP values provided for summary")
                return []
            
            # Generate feature names if not provided
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
                self.logger.debug("Generated default feature names")
            
            # Ensure matching dimensions
            min_len = min(len(shap_values), len(feature_names))
            if min_len < len(shap_values):
                self.logger.warning(f"Feature name count mismatch: truncating to {min_len} features")
            
            shap_vals = shap_values[:min_len]
            names = feature_names[:min_len]
            
            # Create summary entries
            summary_data = []
            for i, (name, shap_val) in enumerate(zip(names, shap_vals)):
                summary_entry = self._create_feature_summary_entry(name, shap_val, user_inputs)
                summary_data.append(summary_entry)
            
            # Sort by absolute impact (most important first)
            summary_data.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Return top N features
            result = summary_data[:top_n]
            self.logger.info(f"Created summary for {len(summary_data)} features, returning top {len(result)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating summary data: {e}")
            return []
    
    def analyze_feature_impacts(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of feature impacts.
        
        Args:
            shap_values: Array of SHAP values
            feature_names: List of feature names
            
        Returns:
            Dictionary containing impact analysis statistics
        """
        try:
            if shap_values is None or len(shap_values) == 0:
                self.logger.warning("No SHAP values provided for impact analysis")
                return {}
            
            # Calculate basic statistics
            positive_values = shap_values[shap_values > 0]
            negative_values = shap_values[shap_values < 0]
            
            analysis = {
                'total_features': len(shap_values),
                'positive_impact': float(np.sum(positive_values)) if len(positive_values) > 0 else 0.0,
                'negative_impact': float(np.sum(negative_values)) if len(negative_values) > 0 else 0.0,
                'net_impact': float(np.sum(shap_values)),
                'max_positive_impact': float(np.max(shap_values)),
                'max_negative_impact': float(np.min(shap_values)),
                'mean_abs_impact': float(np.mean(np.abs(shap_values))),
                'impact_variance': float(np.var(shap_values)),
                'impact_std': float(np.std(shap_values))
            }
            
            # Add feature count statistics
            analysis.update({
                'positive_feature_count': int(np.sum(shap_values > 0)),
                'negative_feature_count': int(np.sum(shap_values < 0)),
                'zero_feature_count': int(np.sum(np.abs(shap_values) < 1e-10)),
                'high_impact_count': int(np.sum(np.abs(shap_values) > IMPACT_THRESHOLDS['high'])),
                'medium_impact_count': int(np.sum(
                    (np.abs(shap_values) > IMPACT_THRESHOLDS['medium']) & 
                    (np.abs(shap_values) <= IMPACT_THRESHOLDS['high'])
                ))
            })
            
            # Find most impactful feature
            if len(feature_names) > 0:
                max_impact_idx = np.argmax(np.abs(shap_values))
                if max_impact_idx < len(feature_names):
                    analysis['most_impactful_feature'] = {
                        'name': feature_names[max_impact_idx],
                        'display_name': self._clean_feature_name(feature_names[max_impact_idx]),
                        'impact': float(shap_values[max_impact_idx]),
                        'abs_impact': float(np.abs(shap_values[max_impact_idx]))
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
        n: int = DEFAULT_TOP_N_FEATURES
    ) -> List[Dict[str, Any]]:
        """
        Get top N most impactful features with detailed information.
        
        Args:
            shap_values: Array of SHAP values
            feature_names: List of feature names
            n: Number of top features to return
            
        Returns:
            List of dictionaries containing top feature information
        """
        try:
            if shap_values is None or len(shap_values) == 0:
                self.logger.warning("No SHAP values provided for top features analysis")
                return []
            
            # Ensure we have feature names
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
            
            # Create feature impact entries
            feature_impacts = []
            min_len = min(len(shap_values), len(feature_names))
            
            for i in range(min_len):
                feature_impacts.append({
                    'name': feature_names[i],
                    'display_name': self._clean_feature_name(feature_names[i]),
                    'impact': float(shap_values[i]),
                    'abs_impact': abs(float(shap_values[i])),
                    'direction': 'positive' if shap_values[i] > 0 else 'negative',
                    'magnitude': self._categorize_impact(abs(float(shap_values[i]))),
                    'rank': i + 1  # Will be updated after sorting
                })
            
            # Sort by absolute impact (highest first)
            feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Update ranks after sorting
            for i, feature in enumerate(feature_impacts):
                feature['rank'] = i + 1
            
            # Return top N features
            result = feature_impacts[:n]
            self.logger.info(f"Retrieved top {len(result)} features from {len(feature_impacts)} total")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting top features: {e}")
            return []
    
    def calculate_baseline_impact(self, shap_values: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        Calculate baseline impact statistics for SHAP values.
        
        Args:
            shap_values: Array of SHAP values
            
        Returns:
            Dictionary containing baseline statistics
        """
        try:
            if shap_values is None or len(shap_values) == 0:
                self.logger.warning("No SHAP values provided for baseline calculation")
                return {}
            
            # Calculate comprehensive baseline statistics
            baseline_stats = {
                'baseline_sum': float(np.sum(shap_values)),
                'baseline_mean': float(np.mean(shap_values)),
                'baseline_median': float(np.median(shap_values)),
                'baseline_std': float(np.std(shap_values)),
                'baseline_var': float(np.var(shap_values)),
                'baseline_min': float(np.min(shap_values)),
                'baseline_max': float(np.max(shap_values)),
                'baseline_range': float(np.max(shap_values) - np.min(shap_values)),
                'positive_count': int(np.sum(shap_values > 0)),
                'negative_count': int(np.sum(shap_values < 0)),
                'zero_count': int(np.sum(np.abs(shap_values) < 1e-10)),
                'total_count': len(shap_values)
            }
            
            # Add percentile information
            baseline_stats.update({
                'percentile_25': float(np.percentile(shap_values, 25)),
                'percentile_75': float(np.percentile(shap_values, 75)),
                'percentile_90': float(np.percentile(shap_values, 90)),
                'percentile_95': float(np.percentile(shap_values, 95))
            })
            
            self.logger.debug("Baseline impact statistics calculated successfully")
            return baseline_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating baseline impact: {e}")
            return {}
    
    def generate_impact_insights(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        user_inputs: Dict[str, Any] = None
    ) -> List[str]:
        """
        Generate human-readable insights from SHAP analysis.
        
        Args:
            shap_values: Array of SHAP values
            feature_names: List of feature names
            user_inputs: Optional user input values for context
            
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            if shap_values is None or len(shap_values) == 0:
                return ["No SHAP values available for insight generation."]
            
            # Overall impact insight
            total_impact = np.sum(shap_values)
            positive_impact = np.sum(shap_values[shap_values > 0])
            negative_impact = np.sum(shap_values[shap_values < 0])
            
            insights.append(f"Overall prediction impact: {total_impact:.3f} "
                          f"(+{positive_impact:.3f} from positive features, "
                          f"{negative_impact:.3f} from negative features)")
            
            # Most impactful feature
            if len(feature_names) > 0:
                max_impact_idx = np.argmax(np.abs(shap_values))
                if max_impact_idx < len(feature_names):
                    feature_name = self._clean_feature_name(feature_names[max_impact_idx])
                    impact_value = shap_values[max_impact_idx]
                    direction = "increases" if impact_value > 0 else "decreases"
                    
                    insights.append(f"Most influential feature: '{feature_name}' "
                                  f"{direction} prediction by {abs(impact_value):.3f}")
            
            # Feature distribution insight
            high_impact_count = np.sum(np.abs(shap_values) > IMPACT_THRESHOLDS['high'])
            if high_impact_count > 0:
                insights.append(f"{high_impact_count} features have high impact (>{IMPACT_THRESHOLDS['high']})")
            
            # Balance insight
            pos_count = np.sum(shap_values > 0)
            neg_count = np.sum(shap_values < 0)
            if pos_count > neg_count * 2:
                insights.append("Prediction is driven primarily by positive factors")
            elif neg_count > pos_count * 2:
                insights.append("Prediction is driven primarily by negative factors")
            else:
                insights.append("Prediction shows balanced positive and negative factor influence")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating impact insights: {e}")
            return ["Error generating insights from SHAP analysis."]
    
    def _handle_shap_value_formats(self, shap_values) -> np.ndarray:
        """
        Handle different SHAP value formats consistently.
        
        Args:
            shap_values: SHAP values in various possible formats
            
        Returns:
            Standardized numpy array of SHAP values
        """
        try:
            # Handle list format (from some explainers)
            if isinstance(shap_values, list):
                if len(shap_values) > 0:
                    result = shap_values[0]
                else:
                    self.logger.warning("Empty SHAP values list")
                    return np.array([])
            else:
                result = shap_values
            
            # Handle 2D format (extract first instance for single prediction)
            if hasattr(result, 'ndim') and result.ndim == 2:
                if result.shape[0] > 0:
                    result = result[0]
                else:
                    self.logger.warning("Empty SHAP values array")
                    return np.array([])
            
            # Ensure it's a numpy array
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            # Validate result
            if result.size == 0:
                self.logger.warning("SHAP values conversion resulted in empty array")
                return np.array([])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling SHAP value formats: {e}")
            return np.array([])
    
    def _extract_interaction_instance(self, interaction_values) -> Optional[np.ndarray]:
        """
        Extract single instance from interaction values.
        
        Args:
            interaction_values: Interaction values from explainer
            
        Returns:
            Extracted interaction matrix or None
        """
        try:
            if isinstance(interaction_values, list):
                if len(interaction_values) > 0 and len(interaction_values[0]) > 0:
                    result = interaction_values[0][0]
                else:
                    return None
            else:
                if len(interaction_values.shape) > 2:
                    result = interaction_values[0]
                else:
                    result = interaction_values
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting interaction instance: {e}")
            return None
    
    def _create_feature_summary_entry(
        self, 
        feature_name: str, 
        shap_value: float, 
        user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a single feature summary entry.
        
        Args:
            feature_name: Name of the feature
            shap_value: SHAP value for the feature
            user_inputs: User input values dictionary
            
        Returns:
            Dictionary containing feature summary information
        """
        # Get and format input value
        input_value = user_inputs.get(feature_name, 'N/A') if user_inputs else 'N/A'
        input_str = self._format_input_value(input_value)
        
        return {
            'feature_name': feature_name,
            'display_name': self._clean_feature_name(feature_name),
            'shap_value': float(shap_value),
            'input_value': input_str,
            'abs_impact': abs(float(shap_value)),
            'direction': 'Increases' if shap_value > 0 else 'Decreases',
            'impact_magnitude': self._categorize_impact(abs(float(shap_value)))
        }
    
    def _format_input_value(self, input_value: Any) -> str:
        """
        Format input value for display.
        
        Args:
            input_value: Raw input value
            
        Returns:
            Formatted string representation
        """
        if isinstance(input_value, (int, float)):
            if abs(input_value - int(input_value)) < INTEGER_THRESHOLD:
                return str(int(input_value))
            else:
                return f"{input_value:.{NUMERIC_PRECISION}f}"
        else:
            return str(input_value)
    
    def _clean_feature_name(self, name: str) -> str:
        """
        Clean feature name for display purposes.
        
        Args:
            name: Raw feature name
            
        Returns:
            Cleaned feature name suitable for display
        """
        if not name:
            return "Unknown Feature"
        
        # Replace underscores with spaces and apply title case
        clean_name = name.replace('_', ' ').title()
        
        # Limit length for display and add ellipsis if needed
        if len(clean_name) > MAX_FEATURE_NAME_LENGTH:
            clean_name = clean_name[:MAX_FEATURE_NAME_LENGTH - len(FEATURE_NAME_TRUNCATE_SUFFIX)] + FEATURE_NAME_TRUNCATE_SUFFIX
        
        return clean_name
    
    def _categorize_impact(self, abs_impact: float) -> str:
        """
        Categorize impact magnitude for better interpretation.
        
        Args:
            abs_impact: Absolute impact value
            
        Returns:
            Impact category string
        """
        if abs_impact > IMPACT_THRESHOLDS['high']:
            return "High"
        elif abs_impact > IMPACT_THRESHOLDS['medium']:
            return "Medium"
        elif abs_impact > IMPACT_THRESHOLDS['low']:
            return "Low"
        else:
            return "Minimal"


# Module-level utility functions
def validate_shap_values(shap_values: np.ndarray) -> Dict[str, Any]:
    """
    Validate SHAP values array for common issues.
    
    Args:
        shap_values: SHAP values to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        if shap_values is None:
            validation['valid'] = False
            validation['issues'].append("SHAP values are None")
            return validation
        
        if not isinstance(shap_values, np.ndarray):
            validation['warnings'].append("SHAP values are not numpy array")
        
        if shap_values.size == 0:
            validation['valid'] = False
            validation['issues'].append("SHAP values array is empty")
            return validation
        
        # Check for problematic values
        if np.isnan(shap_values).any():
            validation['issues'].append("SHAP values contain NaN")
            validation['valid'] = False
        
        if np.isinf(shap_values).any():
            validation['issues'].append("SHAP values contain infinite values")
            validation['valid'] = False
        
        # Statistics
        validation['statistics'] = {
            'shape': shap_values.shape,
            'dtype': str(shap_values.dtype),
            'min': float(np.min(shap_values)),
            'max': float(np.max(shap_values)),
            'mean': float(np.mean(shap_values)),
            'std': float(np.std(shap_values))
        }
        
    except Exception as e:
        validation['valid'] = False
        validation['issues'].append(f"Validation error: {str(e)}")
    
    return validation


def compare_shap_values(
    values1: np.ndarray, 
    values2: np.ndarray, 
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compare two sets of SHAP values for differences.
    
    Args:
        values1: First set of SHAP values
        values2: Second set of SHAP values
        feature_names: Optional feature names for detailed comparison
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        if values1 is None or values2 is None:
            return {'error': 'One or both SHAP value arrays are None'}
        
        if len(values1) != len(values2):
            return {'error': f'Array length mismatch: {len(values1)} vs {len(values2)}'}
        
        # Calculate differences
        differences = values1 - values2
        abs_differences = np.abs(differences)
        
        comparison = {
            'mean_difference': float(np.mean(differences)),
            'mean_abs_difference': float(np.mean(abs_differences)),
            'max_difference': float(np.max(abs_differences)),
            'correlation': float(np.corrcoef(values1, values2)[0, 1]),
            'similar_features': int(np.sum(abs_differences < 0.01)),
            'different_features': int(np.sum(abs_differences >= 0.01))
        }
        
        # Feature-level comparison if names provided
        if feature_names and len(feature_names) == len(values1):
            feature_comparison = []
            for i, name in enumerate(feature_names):
                feature_comparison.append({
                    'feature': name,
                    'value1': float(values1[i]),
                    'value2': float(values2[i]),
                    'difference': float(differences[i]),
                    'abs_difference': float(abs_differences[i])
                })
            
            # Sort by absolute difference
            feature_comparison.sort(key=lambda x: x['abs_difference'], reverse=True)
            comparison['feature_comparison'] = feature_comparison[:10]  # Top 10 differences
        
        return comparison
        
    except Exception as e:
        return {'error': f'Comparison failed: {str(e)}'}