# shap_analysis/value_calculator.py
"""
SHAP Value Calculator - Calculates SHAP values and interactions
Single responsibility: SHAP value computation and formatting
"""

import numpy as np
import shap
from typing import Dict, List, Optional, Any
from constants import PipelineConstants

class SHAPValueCalculator:
    """Calculates SHAP values and creates summaries"""
    
    def calculate_shap_values(
        self, 
        explainer: shap.Explainer, 
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """Calculate SHAP values for input data"""
        
    def calculate_interaction_values(
        self,
        explainer: shap.Explainer,
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """Calculate SHAP interaction values (TreeExplainer only)"""
        
    def create_summary_data(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        user_inputs: Dict,
        top_n: int = PipelineConstants.TOP_N_FEATURES
    ) -> List[Dict]:
        """Create summary data for SHAP values display"""
        
    def analyze_feature_impacts(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze overall feature impacts"""
        
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = 10
    ) -> List[Dict]:
        """Get top N most impactful features"""
        
    def calculate_baseline_impact(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate baseline impact statistics"""