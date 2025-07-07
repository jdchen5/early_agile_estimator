# shap_analysis/ui_integration.py
"""
UI Integration Layer - Streamlit display functions

This module provides Streamlit-based user interface components for SHAP analysis.
It handles all visualization, user interaction, and display formatting for SHAP results.
"""

import logging
from typing import Dict, List, Optional, Callable, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .analysis_coordinator import SHAPAnalysisCoordinator


# Module-level constants
DEFAULT_CHART_WIDTH = 10
DEFAULT_CHART_HEIGHT = 6
MAX_FEATURES_DISPLAY = 10
DEFAULT_WHAT_IF_POINTS = 10


class SHAPUIIntegration:
    """
    Streamlit UI integration for SHAP analysis display and interaction.
    
    This class provides methods to display SHAP analysis results in Streamlit,
    including charts, tables, and interactive what-if analysis components.
    """
    
    def __init__(self):
        """Initialize UI integration with analysis coordinator."""
        self.coordinator = SHAPAnalysisCoordinator()
        self.logger = logging.getLogger(__name__)
    
    def display_instance_specific_shap(
        self, 
        user_inputs: Dict[str, Any], 
        model_name: str, 
        get_trained_model_func: Callable
    ) -> None:
        """
        Display instance-specific SHAP analysis results.
        
        Args:
            user_inputs: Dictionary of user input features
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve trained model
        """
        st.subheader("🎯 Your Prediction's Feature Impact")
        
        if not user_inputs:
            st.warning("Please make a prediction first to see SHAP analysis.")
            return
        
        try:
            with st.spinner("Creating SHAP analysis..."):
                result = self.coordinator.run_instance_specific_display(
                    user_inputs, model_name, get_trained_model_func
                )
            
            if not result.get("success"):
                st.error(f"SHAP analysis failed: {result.get('error')}")
                return
            
            # Display analysis components
            self._render_impact_chart(result.get("charts", {}))
            self._render_summary_table(result.get("summary_table", []))
            self._render_insights(result.get("insights", []))
            
        except Exception as e:
            self.logger.error(f"Error in instance-specific SHAP display: {e}")
            st.error("An error occurred during SHAP analysis. Please try again.")
    
    def display_what_if_shap_analysis(
        self, 
        user_inputs: Dict[str, Any], 
        model_name: str, 
        get_trained_model_func: Callable
    ) -> None:
        """
        Display interactive what-if SHAP analysis.
        
        Args:
            user_inputs: Dictionary of user input features
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve trained model
        """
        st.subheader("🔍 What-If SHAP Analysis")
        
        try:
            # Parameter selection interface
            numeric_params = self._get_numeric_parameters()
            if not numeric_params:
                st.warning("No numeric parameters available for what-if analysis.")
                return
            
            param_name = st.selectbox("Select parameter:", numeric_params)
            param_values = np.linspace(0, 100, DEFAULT_WHAT_IF_POINTS)  # Simplified
            
            with st.spinner("Running what-if analysis..."):
                result = self.coordinator.run_what_if_display(
                    user_inputs, model_name, get_trained_model_func, param_name, param_values
                )
            
            if result.get("success"):
                self._render_what_if_chart(result.get("chart_data", []))
            else:
                st.error(f"What-if analysis failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in what-if SHAP display: {e}")
            st.error("An error occurred during what-if analysis. Please try again.")
    
    def display_scenario_comparison(
        self, 
        scenarios: Dict[str, Dict[str, Any]], 
        model_name: str, 
        get_trained_model_func: Callable
    ) -> None:
        """
        Display scenario comparison analysis.
        
        Args:
            scenarios: Dictionary of scenario names to input dictionaries
            model_name: Name of the model to analyze
            get_trained_model_func: Function to retrieve trained model
        """
        st.subheader("📊 Scenario Comparison")
        
        if len(scenarios) < 2:
            st.warning("At least 2 scenarios required for comparison.")
            return
        
        try:
            with st.spinner("Comparing scenarios..."):
                result = self.coordinator.run_scenario_comparison(
                    scenarios, model_name, get_trained_model_func
                )
            
            if result.get("success"):
                self._render_scenario_comparison_results(result)
            else:
                st.error(f"Scenario comparison failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in scenario comparison display: {e}")
            st.error("An error occurred during scenario comparison. Please try again.")
    
    def _render_impact_chart(self, chart_data: Dict[str, Any]) -> None:
        """
        Render feature impact chart.
        
        Args:
            chart_data: Chart data dictionary containing impact information
        """
        if not chart_data:
            return
        
        features = chart_data.get("impact_chart", [])[:MAX_FEATURES_DISPLAY]
        if not features:
            st.info("No feature impact data available for visualization.")
            return
        
        try:
            names = [f.get("display_name", f.get("name", "Unknown")) for f in features]
            impacts = [f.get("impact", 0) for f in features]
            
            fig, ax = plt.subplots(figsize=(DEFAULT_CHART_WIDTH, DEFAULT_CHART_HEIGHT))
            colors = ['red' if x < 0 else 'blue' for x in impacts]
            ax.barh(names, impacts, color=colors)
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('Feature Impact Analysis')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error rendering impact chart: {e}")
            st.error("Could not render impact chart.")
    
    def _render_summary_table(self, summary_data: List[Dict[str, Any]]) -> None:
        """
        Render summary data table.
        
        Args:
            summary_data: List of summary dictionaries
        """
        if not summary_data:
            return
        
        try:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            self.logger.error(f"Error rendering summary table: {e}")
            st.error("Could not render summary table.")
    
    def _render_insights(self, insights: List[str]) -> None:
        """
        Render analysis insights.
        
        Args:
            insights: List of insight strings
        """
        if not insights:
            return
        
        st.write("**Key Insights:**")
        for insight in insights:
            if insight:  # Skip empty insights
                st.info(insight)
    
    def _render_what_if_chart(self, chart_data: List[Dict[str, Any]]) -> None:
        """
        Render what-if analysis chart.
        
        Args:
            chart_data: List of what-if analysis results
        """
        if not chart_data:
            st.info("No what-if analysis data available.")
            return
        
        try:
            param_values = [item.get("param_value", 0) for item in chart_data]
            impacts = [item.get("param_impact", 0) for item in chart_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=param_values,
                y=impacts,
                mode='lines+markers',
                name='Parameter Impact'
            ))
            
            fig.update_layout(
                title="What-If Analysis: Parameter Impact",
                xaxis_title="Parameter Value",
                yaxis_title="SHAP Impact",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Error rendering what-if chart: {e}")
            st.error("Could not render what-if chart.")
    
    def _render_scenario_comparison_results(self, results: Dict[str, Any]) -> None:
        """
        Render scenario comparison results.
        
        Args:
            results: Scenario comparison results dictionary
        """
        scenarios = results.get("scenarios", {})
        if not scenarios:
            st.info("No scenario comparison data available.")
            return
        
        # Create comparison summary
        comparison_data = []
        for scenario_name, scenario_data in scenarios.items():
            impact_analysis = scenario_data.get("impact_analysis", {})
            comparison_data.append({
                "Scenario": scenario_name,
                "Net Impact": impact_analysis.get("net_impact", 0),
                "Positive Impact": impact_analysis.get("positive_impact", 0),
                "Negative Impact": impact_analysis.get("negative_impact", 0)
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
    
    def _get_numeric_parameters(self) -> List[str]:
        """
        Get list of numeric parameters available for what-if analysis.
        
        Returns:
            List of numeric parameter names
        """
        # This could be enhanced to dynamically determine parameters
        # from the model or configuration
        return [
            "project_prf_functional_size",
            "project_prf_max_team_size",
            "process_pmf_docs",
            "people_prf_project_user_involvement"
        ]


# Module-level utility functions
def create_default_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Create default scenario configurations for comparison.
    
    Returns:
        Dictionary of default scenarios
    """
    return {
        "Small Project": {
            "project_prf_functional_size": 100,
            "project_prf_max_team_size": 3,
            "project_prf_relative_size": "XS"
        },
        "Medium Project": {
            "project_prf_functional_size": 500,
            "project_prf_max_team_size": 8,
            "project_prf_relative_size": "M"
        },
        "Large Project": {
            "project_prf_functional_size": 2000,
            "project_prf_max_team_size": 15,
            "project_prf_relative_size": "L"
        }
    }