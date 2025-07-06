# shap_analysis/analysis_coordinator.py
"""
SHAP Analysis Coordinator - Orchestrates SHAP analysis workflow
Provides unified interface while delegating to specialized classes
"""

from .explainer_factory import SHAPExplainerFactory
from .data_preparer import SHAPDataPreparer
from .value_calculator import SHAPValueCalculator
from typing import Dict, List, Optional, Any, Callable
import logging

class SHAPAnalysisCoordinator:
    """Coordinates complete SHAP analysis workflow"""
    
    def __init__(self):
        self.explainer_factory = SHAPExplainerFactory()
        self.data_preparer = SHAPDataPreparer()
        self.calculator = SHAPValueCalculator()
    
    def run_instance_analysis(
        self,
        user_inputs: Dict[str, Any],
        model_name: str,
        get_trained_model_func: Callable,
        sample_size: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Run complete instance-specific SHAP analysis
        Returns structured result with explainer, values, and summary
        """
        try:
            # Step 1: Prepare background data
            background_data = self.data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                return {"error": "Failed to prepare background data"}
            
            # Step 2: Create explainer
            explainer = self.explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            if explainer is None:
                return {"error": "Failed to create SHAP explainer"}
            
            # Step 3: Prepare input data
            input_data = self.data_preparer.prepare_input_data(user_inputs)
            if input_data is None:
                return {"error": "Failed to prepare input data"}
            
            # Step 4: Calculate SHAP values
            shap_values = self.calculator.calculate_shap_values(explainer, input_data)
            if shap_values is None:
                return {"error": "Failed to calculate SHAP values"}
            
            # Step 5: Create summary
            feature_names = self._get_feature_names(user_inputs)
            summary = self.calculator.create_summary_data(shap_values, feature_names, user_inputs)
            
            return {
                "success": True,
                "explainer": explainer,
                "shap_values": shap_values,
                "feature_names": feature_names,
                "summary": summary,
                "model_name": model_name
            }
            
        except Exception as e:
            logging.error(f"SHAP instance analysis failed: {e}")
            return {"error": f"Analysis failed: {e}"}
    
    def _get_feature_names(self, user_inputs: Dict) -> List[str]:
        """Extract feature names from user inputs or model"""
        # Implementation to get proper feature names
        exclude_keys = {'selected_model', 'submit', 'clear_results', 'show_history'}
        return [k for k in user_inputs.keys() if k not in exclude_keys]