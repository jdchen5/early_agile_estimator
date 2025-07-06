# shap_analysis/analysis_coordinator.py
"""
SHAP Analysis Coordinator - Orchestrates complete SHAP analysis workflow
Combines DataPreparer, ExplainerFactory, and ValueCalculator
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from .data_preparer import SHAPDataPreparer
from .explainer_factory import SHAPExplainerFactory
from .value_calculator import SHAPValueCalculator

class SHAPAnalysisCoordinator:
    """Coordinates complete SHAP analysis workflow"""
    
    def __init__(self):
        self.data_preparer = SHAPDataPreparer()
        self.explainer_factory = SHAPExplainerFactory()
        self.calculator = SHAPValueCalculator()
        self.logger = logging.getLogger(__name__)
    
    def run_instance_analysis(
        self,
        user_inputs: Dict[str, Any],
        model_name: str,
        get_trained_model_func: Callable,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete instance-specific SHAP analysis
        Returns structured result with all analysis components
        """
        try:
            self.logger.info(f"Starting SHAP analysis for model: {model_name}")
            
            # Step 1: Validate inputs
            if not self.data_preparer.validate_shap_inputs(user_inputs):
                return {"success": False, "error": "Invalid user inputs provided"}
            
            # Step 2: Prepare background data
            self.logger.info("Preparing background data...")
            background_data = self.data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                return {"success": False, "error": "Failed to prepare background data"}
            
            # Step 3: Create explainer
            self.logger.info("Creating SHAP explainer...")
            explainer = self.explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            if explainer is None:
                return {"success": False, "error": "Failed to create SHAP explainer"}
            
            # Step 4: Prepare input data
            self.logger.info("Preparing input data...")
            input_data = self.data_preparer.prepare_input_data(user_inputs)
            if input_data is None:
                return {"success": False, "error": "Failed to prepare input data"}
            
            # Step 5: Calculate SHAP values
            self.logger.info("Calculating SHAP values...")
            feature_names = self._get_feature_names(user_inputs)
            shap_values = self.calculator.calculate_shap_values(explainer, input_data, feature_names)
            if shap_values is None:
                return {"success": False, "error": "Failed to calculate SHAP values"}
            
            # Step 6: Create analysis results
            self.logger.info("Creating analysis summary...")
            summary = self.calculator.create_summary_data(shap_values, feature_names, user_inputs)
            impact_analysis = self.calculator.analyze_feature_impacts(shap_values, feature_names)
            top_features = self.calculator.get_top_features(shap_values, feature_names, n=10)
            
            # Step 7: Calculate interaction values (if supported)
            interaction_values = self.calculator.calculate_interaction_values(explainer, input_data, feature_names)
            
            result = {
                "success": True,
                "model_name": model_name,
                "explainer": explainer,
                "shap_values": shap_values,
                "feature_names": feature_names,
                "summary": summary,
                "impact_analysis": impact_analysis,
                "top_features": top_features,
                "interaction_values": interaction_values,
                "background_data_shape": background_data.shape,
                "input_data_shape": input_data.shape
            }
            
            self.logger.info(f"SHAP analysis completed successfully for {len(feature_names)} features")
            return result
            
        except Exception as e:
            self.logger.error(f"SHAP instance analysis failed: {e}")
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    def run_what_if_analysis(
        self,
        base_inputs: Dict[str, Any],
        model_name: str,
        get_trained_model_func: Callable,
        param_name: str,
        param_values: List[float],
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run what-if analysis by varying a parameter
        """
        try:
            self.logger.info(f"Starting what-if analysis for parameter: {param_name}")
            
            # Prepare explainer once
            background_data = self.data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                return {"success": False, "error": "Failed to prepare background data"}
            
            explainer = self.explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            if explainer is None:
                return {"success": False, "error": "Failed to create explainer"}
            
            # Analyze each parameter value
            results = []
            for value in param_values:
                test_inputs = base_inputs.copy()
                test_inputs[param_name] = value
                
                # Quick analysis for this value
                input_data = self.data_preparer.prepare_input_data(test_inputs)
                if input_data is not None:
                    feature_names = self._get_feature_names(test_inputs)
                    shap_values = self.calculator.calculate_shap_values(explainer, input_data, feature_names)
                    
                    if shap_values is not None:
                        # Find parameter impact
                        param_impact = 0
                        if param_name in feature_names:
                            param_idx = feature_names.index(param_name)
                            if param_idx < len(shap_values):
                                param_impact = float(shap_values[param_idx])
                        
                        results.append({
                            'param_value': value,
                            'param_impact': param_impact,
                            'total_impact': float(sum(shap_values))
                        })
            
            return {
                "success": True,
                "parameter": param_name,
                "results": results,
                "analysis_count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"What-if analysis failed: {e}")
            return {"success": False, "error": f"What-if analysis failed: {str(e)}"}
    
    def run_scenario_comparison(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        model_name: str,
        get_trained_model_func: Callable,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Compare SHAP analysis across different scenarios
        """
        try:
            self.logger.info(f"Starting scenario comparison for {len(scenarios)} scenarios")
            
            # Prepare explainer once
            background_data = self.data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                return {"success": False, "error": "Failed to prepare background data"}
            
            explainer = self.explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            if explainer is None:
                return {"success": False, "error": "Failed to create explainer"}
            
            # Analyze each scenario
            scenario_results = {}
            for scenario_name, scenario_inputs in scenarios.items():
                try:
                    input_data = self.data_preparer.prepare_input_data(scenario_inputs)
                    if input_data is not None:
                        feature_names = self._get_feature_names(scenario_inputs)
                        shap_values = self.calculator.calculate_shap_values(explainer, input_data, feature_names)
                        
                        if shap_values is not None:
                            impact_analysis = self.calculator.analyze_feature_impacts(shap_values, feature_names)
                            scenario_results[scenario_name] = {
                                'shap_values': shap_values,
                                'impact_analysis': impact_analysis,
                                'feature_names': feature_names
                            }
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze scenario '{scenario_name}': {e}")
                    continue
            
            return {
                "success": True,
                "scenarios": scenario_results,
                "scenario_count": len(scenario_results)
            }
            
        except Exception as e:
            self.logger.error(f"Scenario comparison failed: {e}")
            return {"success": False, "error": f"Scenario comparison failed: {str(e)}"}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the SHAP analysis system"""
        try:
            sample_data_info = self.data_preparer.get_sample_data_info()
            cache_info = self.explainer_factory.get_cache_info()
            
            return {
                "data_preparer": {
                    "available": True,
                    "sample_data_info": sample_data_info
                },
                "explainer_factory": {
                    "available": True,
                    "cache_info": cache_info
                },
                "value_calculator": {
                    "available": True
                },
                "system_status": "Ready"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {"system_status": "Error", "error": str(e)}
    
    def clear_caches(self):
        """Clear all component caches"""
        try:
            self.explainer_factory.clear_cache()
            self.logger.info("All SHAP analysis caches cleared")
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
    
    def _get_feature_names(self, user_inputs: Dict) -> List[str]:
        """Extract feature names from user inputs"""
        exclude_keys = {
            'selected_model', 'selected_models', 'submit', 
            'clear_results', 'show_history', 'comparison_mode'
        }
        return [k for k in sorted(user_inputs.keys()) if k not in exclude_keys]