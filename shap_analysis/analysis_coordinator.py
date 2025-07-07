# shap_analysis/analysis_coordinator.py
"""
SHAP Analysis Coordinator - Orchestrates complete SHAP analysis workflow

This module coordinates the complete SHAP analysis workflow by combining
the DataPreparer, ExplainerFactory, and ValueCalculator components. It provides
high-level analysis methods and handles complex multi-step analysis scenarios.

Key responsibilities:
- Orchestrate multi-component SHAP analysis workflows
- Provide high-level analysis APIs for UI integration
- Handle complex analysis scenarios (what-if, scenario comparison)
- Coordinate error handling across components
- Manage analysis result formatting and insights generation
"""

import logging
from typing import Dict, List, Optional, Any, Callable

from .data_preparer import SHAPDataPreparer
from .explainer_factory import SHAPExplainerFactory
from .value_calculator import SHAPValueCalculator


# Module-level constants
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_TOP_FEATURES = 10
EXCLUDE_KEYS = {
    'selected_model', 'selected_models', 'submit', 
    'clear_results', 'show_history', 'comparison_mode'
}


class SHAPAnalysisCoordinator:
    """
    Coordinates complete SHAP analysis workflows.
    
    This class orchestrates the interaction between data preparation, explainer
    creation, and value calculation components to provide comprehensive SHAP
    analysis capabilities. It handles complex analysis scenarios and provides
    structured results for UI consumption.
    """
    
    def __init__(self):
        """Initialize coordinator with component instances."""
        self.data_preparer = SHAPDataPreparer()
        self.explainer_factory = SHAPExplainerFactory()
        self.calculator = SHAPValueCalculator()
        self.logger = logging.getLogger(__name__)
    
    def run_instance_analysis(
        self,
        user_inputs: Dict[str, Any],
        model_name: str,
        get_trained_model_func: Callable,
        sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Dict[str, Any]:
        """
        Run complete instance-specific SHAP analysis.
        
        This is the main entry point for comprehensive SHAP analysis of a single
        instance. It orchestrates all components to provide detailed analysis results.
        
        Args:
            user_inputs: Dictionary of user input features
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve the trained model
            sample_size: Number of background samples for SHAP baseline
            
        Returns:
            Dictionary containing structured analysis results with success status
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
            top_features = self.calculator.get_top_features(shap_values, feature_names, n=DEFAULT_TOP_FEATURES)
            
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
        sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Dict[str, Any]:
        """
        Run what-if analysis by varying a parameter across multiple values.
        
        Args:
            base_inputs: Base input feature dictionary
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve the trained model
            param_name: Name of the parameter to vary
            param_values: List of values to test for the parameter
            sample_size: Number of background samples for SHAP baseline
            
        Returns:
            Dictionary containing what-if analysis results
        """
        try:
            self.logger.info(f"Starting what-if analysis for parameter: {param_name}")
            
            # Prepare explainer once for efficiency
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
                        param_impact = self._find_parameter_impact(param_name, feature_names, shap_values)
                        
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
        sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Dict[str, Any]:
        """
        Compare SHAP analysis across different project scenarios.
        
        Args:
            scenarios: Dictionary mapping scenario names to input dictionaries
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve the trained model
            sample_size: Number of background samples for SHAP baseline
            
        Returns:
            Dictionary containing scenario comparison results
        """
        try:
            self.logger.info(f"Starting scenario comparison for {len(scenarios)} scenarios")
            
            # Prepare explainer once for efficiency
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
    
    def run_instance_specific_display(
        self, 
        user_inputs: Dict[str, Any], 
        model_name: str, 
        get_trained_model_func: Callable
    ) -> Dict[str, Any]:
        """
        Run analysis and return data formatted for UI display.
        
        Args:
            user_inputs: Dictionary of user input features
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve the trained model
            
        Returns:
            Dictionary containing formatted display data
        """
        analysis_result = self.run_instance_analysis(user_inputs, model_name, get_trained_model_func)
        if not analysis_result.get("success"):
            return analysis_result
        
        # Prepare display data
        return {
            "success": True,
            "charts": self._prepare_chart_data(analysis_result),
            "summary_table": analysis_result.get("summary", []),
            "insights": self._generate_insights(analysis_result)
        }
    
    def run_what_if_display(
        self, 
        user_inputs: Dict[str, Any], 
        model_name: str, 
        get_trained_model_func: Callable, 
        param_name: str, 
        param_values: List[float]
    ) -> Dict[str, Any]:
        """
        Run what-if analysis and return data formatted for display.
        
        Args:
            user_inputs: Dictionary of user input features
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve the trained model
            param_name: Name of the parameter to vary
            param_values: List of values to test for the parameter
            
        Returns:
            Dictionary containing formatted what-if display data
        """
        what_if_result = self.run_what_if_analysis(
            user_inputs, model_name, get_trained_model_func, param_name, param_values
        )
        if not what_if_result.get("success"):
            return what_if_result
        
        return {
            "success": True,
            "chart_data": what_if_result.get("results", []),
            "sensitivity_metrics": self._calculate_sensitivity(what_if_result["results"])
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the SHAP analysis system.
        
        Returns:
            Dictionary containing system status and component information
        """
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
    
    def clear_caches(self) -> None:
        """Clear all component caches to free memory."""
        try:
            self.explainer_factory.clear_cache()
            self.logger.info("All SHAP analysis caches cleared")
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
    
    def _get_feature_names(self, user_inputs: Dict[str, Any]) -> List[str]:
        """
        Extract feature names from user inputs, excluding UI control fields.
        
        Args:
            user_inputs: Dictionary of user inputs
            
        Returns:
            List of sorted feature names
        """
        return [k for k in sorted(user_inputs.keys()) if k not in EXCLUDE_KEYS]
    
    def _find_parameter_impact(
        self, 
        param_name: str, 
        feature_names: List[str], 
        shap_values: List[float]
    ) -> float:
        """
        Find the SHAP impact value for a specific parameter.
        
        Args:
            param_name: Name of the parameter to find
            feature_names: List of feature names
            shap_values: List of SHAP values corresponding to features
            
        Returns:
            SHAP impact value for the parameter, or 0 if not found
        """
        param_impact = 0
        if param_name in feature_names:
            param_idx = feature_names.index(param_name)
            if param_idx < len(shap_values):
                param_impact = float(shap_values[param_idx])
        return param_impact
    
    def _prepare_chart_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare analysis data for chart visualization.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            Dictionary containing chart-ready data
        """
        return {
            "impact_chart": analysis_result.get("top_features", [])[:DEFAULT_TOP_FEATURES],
            "summary_metrics": analysis_result.get("impact_analysis", {})
        }
    
    def _generate_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable insights from analysis results.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            List of insight strings for display
        """
        insights = []
        impact = analysis_result.get("impact_analysis", {})
        
        if impact.get("most_impactful_feature"):
            feature = impact["most_impactful_feature"]
            direction = "increases" if feature["impact"] > 0 else "decreases"
            insights.append(
                f"Most influential: '{feature['name']}' {direction} "
                f"prediction by {abs(feature['impact']):.3f}"
            )
        
        # Add additional insights based on impact analysis
        if impact.get("positive_impact", 0) > 0:
            insights.append(f"Total positive impact: {impact['positive_impact']:.3f}")
        
        if impact.get("negative_impact", 0) < 0:
            insights.append(f"Total negative impact: {impact['negative_impact']:.3f}")
        
        return insights
    
    def _calculate_sensitivity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate sensitivity metrics for what-if analysis results.
        
        Args:
            results: List of what-if analysis result dictionaries
            
        Returns:
            Dictionary containing sensitivity metrics
        """
        if not results:
            return {"sensitivity": 0, "error": "No results to analyze"}
        
        param_values = [r.get("param_value", 0) for r in results]
        impacts = [r.get("param_impact", 0) for r in results]
        
        if len(param_values) < 2:
            return {"sensitivity": 0, "error": "Need at least 2 data points"}
        
        param_range = max(param_values) - min(param_values)
        impact_range = max(impacts) - min(impacts)
        
        return {
            "sensitivity": impact_range / param_range if param_range > 0 else 0,
            "param_range": param_range,
            "impact_range": impact_range,
            "max_impact": max(impacts),
            "min_impact": min(impacts),
            "data_points": len(results)
        }


# Module-level utility functions
def create_default_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Create default project scenarios for comparison analysis.
    
    Returns:
        Dictionary mapping scenario names to input configurations
    """
    return {
        "Small Agile Project": {
            "project_prf_functional_size": 65,
            "tech_tf_primary_programming_language": "Python",
            "project_prf_max_team_size": 3,
            "project_prf_relative_size": "XS",
            "external_eef_industry_sector": "Financial"
        },
        "Medium Enterprise Project": {
            "project_prf_functional_size": 550,
            "tech_tf_primary_programming_language": "Java",
            "project_prf_max_team_size": 8,
            "project_prf_relative_size": "M",
            "external_eef_industry_sector": "Banking"
        },
        "Large Enterprise Project": {
            "project_prf_functional_size": 2000,
            "tech_tf_primary_programming_language": "C#",
            "project_prf_max_team_size": 15,
            "project_prf_relative_size": "L",
            "external_eef_industry_sector": "Insurance"
        }
    }


def validate_analysis_inputs(
    user_inputs: Dict[str, Any], 
    model_name: str, 
    get_trained_model_func: Callable
) -> Dict[str, Any]:
    """
    Validate inputs for SHAP analysis before processing.
    
    Args:
        user_inputs: Dictionary of user input features
        model_name: Name of the model to analyze
        get_trained_model_func: Function to retrieve the trained model
        
    Returns:
        Dictionary containing validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check user inputs
    if not user_inputs:
        validation_result["valid"] = False
        validation_result["errors"].append("No user inputs provided")
    
    # Check model name
    if not model_name or not isinstance(model_name, str):
        validation_result["valid"] = False
        validation_result["errors"].append("Invalid model name")
    
    # Check model function
    if not callable(get_trained_model_func):
        validation_result["valid"] = False
        validation_result["errors"].append("Model retrieval function is not callable")
    
    # Check for meaningful input features
    if user_inputs:
        meaningful_inputs = {
            k: v for k, v in user_inputs.items() 
            if k not in EXCLUDE_KEYS and v is not None and v != ""
        }
        if len(meaningful_inputs) == 0:
            validation_result["warnings"].append("No meaningful input features found")
    
    return validation_result