# ui.py - with Multi-Model Support and Configuration Management
"""
Streamlit UI for ML Project Effort Estimator with Multi-Model Support and Advanced SHAP Analysis
This module provides a user interface for estimating project effort using machine learning models.
It includes form inputs, multi-model selection, prediction comparison, and comprehensive SHAP analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
from datetime import datetime
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from models import (
        predict_man_hours,
        list_available_models,
        check_required_models,
        get_feature_importance,
        get_model_display_name,
        get_model_display_name_from_config,
        get_trained_model,  # Add this function to get the actual model object
        prepare_input_data  # Add this function to prepare data for SHAP
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from models.py: {e}")
    MODELS_AVAILABLE = False
    
    # Define stub functions so UI doesn't crash
    def predict_with_training_features_optimized(inputs, model):
        return None
    def predict_man_hours_direct(inputs, model):
        return None
    def list_available_models():
        return []
    def check_required_models():
        return {"models_available": False}
    def get_trained_model(model_name):
        return None
    def prepare_input_data(inputs):
        return None

# Load merged configuration - will be called after load_yaml_config is defined
UI_INFO_CONFIG = {}
FIELDS = {}
TAB_ORG = {}
UI_BEHAVIOR = {}
FEATURE_IMPORTANCE_DISPLAY = {}
PREDICTION_THRESHOLDS = {}
DISPLAY_CONFIG = {}
IMPORTANT_TABS = "Important Features"
NICE_TABS = "Nice Features"
CONFIG_FOLDER = "config"
SHAP_ANALYSIS_FILE = f"{CONFIG_FOLDER}/shap_analysis.md"

# Minimal CSS for sidebar width only
def set_sidebar_width():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        'prediction_history': [],
        'comparison_results': [],
        'form_attempted': False,
        'prf_size_label2code': {},
        'prf_size_code2mid': {},
        'current_shap_values': None,
        'current_model_explainer': None,
        'last_prediction_inputs': None,
        # add new defaults here as needed
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default



# --- Configuration Loading ---
def load_yaml_config(path):
    """Load YAML configuration file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading YAML file {path}: {e}")
        return {}

# Load merged configuration after function is defined
UI_INFO_CONFIG = load_yaml_config("config/ui_info.yaml")
FIELDS = UI_INFO_CONFIG.get('fields', {})
TAB_ORG = UI_INFO_CONFIG.get('tab_organization', {})
UI_BEHAVIOR = UI_INFO_CONFIG.get('ui_behavior', {})
FEATURE_IMPORTANCE_DISPLAY = UI_INFO_CONFIG.get('feature_importance_display', {})
PREDICTION_THRESHOLDS = UI_INFO_CONFIG.get('prediction_thresholds', {})
DISPLAY_CONFIG = UI_INFO_CONFIG.get('display_config', {})

FEATURE_MAPPING = load_yaml_config("config/feature_mapping.yaml")
CATEGORICAL_MAPPING = FEATURE_MAPPING.get('categorical_features', {})

# --- SHAP Analysis Functions ---
def get_shap_explainer(model_name):
    """Get or create SHAP explainer for a model"""
    try:
        model = get_trained_model(model_name)
        if model is None:
            return None
        
        # Determine model type and create appropriate explainer
        model_type = type(model).__name__.lower()
        model_module = type(model).__module__.lower()
        
        # Tree-based models (use TreeExplainer)
        if any(tree_type in model_type for tree_type in ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient']):
            explainer = shap.TreeExplainer(model)
        
        # Linear models (use LinearExplainer)
        elif any(linear_type in model_type for linear_type in ['linear', 'lasso', 'ridge', 'elastic']):
            # For linear models, we need training data
            sample_data = prepare_sample_data()
            if sample_data is not None:
                explainer = shap.LinearExplainer(model, sample_data)
            else:
                # Fallback to KernelExplainer if no training data
                explainer = shap.KernelExplainer(model.predict, prepare_sample_data_small())
        
        # Neural networks and other models (use KernelExplainer)
        elif any(nn_type in model_type for nn_type in ['neural', 'mlp', 'perceptron']):
            sample_data = prepare_sample_data_small()  # Smaller sample for kernel explainer
            explainer = shap.KernelExplainer(model.predict, sample_data)
        
        # Default fallback - KernelExplainer (works with any model)
        else:
            sample_data = prepare_sample_data_small()
            explainer = shap.KernelExplainer(model.predict, sample_data)
        
        return explainer
        
    except Exception as e:
        st.error(f"Error creating SHAP explainer for {model_name}: {e}")
        return None

def prepare_sample_data(n_samples=100):
    """Prepare sample data for SHAP analysis - larger sample for LinearExplainer"""
    try:
        # This should ideally return actual training data
        # For now, create realistic sample data based on field configurations
        sample_data = []
        
        for _ in range(n_samples):
            sample_row = []
            for field_name, field_config in FIELDS.items():
                field_type = field_config.get('type', 'numeric')
                
                if field_type == 'numeric':
                    min_val = field_config.get('min', 1)
                    max_val = field_config.get('max', 100)
                    default_val = field_config.get('default', (min_val + max_val) / 2)
                    # Add some variation around default
                    value = np.random.normal(default_val, (max_val - min_val) * 0.2)
                    value = np.clip(value, min_val, max_val)
                    sample_row.append(value)
                elif field_type == 'categorical':
                    # For categorical, use encoded values (0, 1, 2, etc.)
                    options = get_field_options(field_name)
                    if options:
                        sample_row.append(np.random.randint(0, len(options)))
                    else:
                        sample_row.append(0)
                elif field_type == 'boolean':
                    sample_row.append(np.random.choice([0, 1]))
                else:
                    sample_row.append(0)  # Default fallback
            
            sample_data.append(sample_row)
        
        return np.array(sample_data)
    except Exception as e:
        st.warning(f"Could not prepare sample data: {e}")
        # Fallback to simple random data
        n_features = len(FIELDS) if FIELDS else 10
        return np.random.rand(n_samples, n_features)

def prepare_sample_data_small(n_samples=20):
    """Prepare smaller sample data for KernelExplainer (computationally expensive)"""
    return prepare_sample_data(n_samples)

def get_shap_values_for_input(user_inputs, model_name):
    """Get SHAP values for a specific input"""
    try:
        # Get the SHAP explainer for the model
        explainer = get_shap_explainer(model_name)
        if explainer is None:
            return None
        
        # Prepare the input data in the format expected by the model
        input_data = prepare_input_data(user_inputs)
        if input_data is None:
            return None
        
        # Ensure input_data is in the right shape (2D array for single prediction)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Calculate SHAP values
        try:
            shap_values = explainer.shap_values(input_data)
            
            # Handle different return formats from different explainer types
            if isinstance(shap_values, list):
                # For multi-class problems or some explainer types
                if len(shap_values) == 1:
                    return shap_values[0]  # Single class
                else:
                    return shap_values[0]  # Use first class for regression-like problems
            else:
                # For single output/regression problems
                return shap_values
                
        except Exception as e:
            # Fallback for explainers that might have different methods
            if hasattr(explainer, 'explain'):
                explanation = explainer.explain(input_data)
                if hasattr(explanation, 'values'):
                    return explanation.values
            raise e
            
    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")
        return None

def display_instance_specific_shap(user_inputs, model_name):
    """Display SHAP analysis for the current prediction"""
    st.subheader("🎯 Your Prediction's Feature Impact")
    
    if not user_inputs:
        st.warning("Please make a prediction first to see instance-specific SHAP analysis.")
        return
    
    shap_values = get_shap_values_for_input(user_inputs, model_name)
    if shap_values is None:
        st.error("Could not generate SHAP values for your input.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Waterfall Plot - Feature Contributions**")
        try:
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get feature names (you'll need to implement this)
            feature_names = get_feature_names_from_inputs(user_inputs)
            
            # Create waterfall-style plot
            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            
            # Sort by absolute value
            sorted_indices = np.argsort(np.abs(shap_vals))[-10:]  # Top 10 features
            sorted_values = shap_vals[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            colors = ['red' if x < 0 else 'blue' for x in sorted_values]
            ax.barh(range(len(sorted_values)), sorted_values, color=colors)
            ax.set_yticks(range(len(sorted_values)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('Feature Impact on Your Prediction')
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error creating waterfall plot: {e}")
    
    with col2:
        st.write("**Feature Impact Summary**")
        try:
            # Create summary table
            feature_names = get_feature_names_from_inputs(user_inputs)
            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            
            summary_data = []
            for i, (name, value) in enumerate(zip(feature_names, shap_vals)):
                summary_data.append({
                    'Feature': get_field_label(name),
                    'Input Value': user_inputs.get(name, 'N/A'),
                    'SHAP Impact': f"{value:.3f}",
                    'Effect': 'Increases' if value > 0 else 'Decreases'
                })
            
            # Sort by absolute impact
            summary_data.sort(key=lambda x: abs(float(x['SHAP Impact'])), reverse=True)
            
            summary_df = pd.DataFrame(summary_data[:10])  # Top 10
            st.dataframe(summary_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating impact summary: {e}")

def display_what_if_shap_analysis(user_inputs, model_name):
    """Interactive what-if analysis with SHAP"""
    st.subheader("🔍 What-If SHAP Analysis")
    
    if not user_inputs:
        st.warning("Please make a prediction first to enable what-if analysis.")
        return
    
    # Parameter selection
    numeric_params = get_what_if_parameters()
    
    if not numeric_params:
        st.warning("No numeric parameters available for what-if analysis.")
        return
    
    selected_param_label = st.selectbox(
        "Select parameter to analyze:",
        list(numeric_params.keys()),
        help="Choose which parameter to vary for sensitivity analysis"
    )
    
    selected_param = numeric_params[selected_param_label]
    
    # Get current value and create range
    current_value = user_inputs.get(selected_param)
    if current_value is None:
        st.warning(f"No value found for parameter: {selected_param}")
        return
    
    range_info = get_what_if_range_from_config(selected_param, current_value)
    if range_info is None:
        st.warning("Could not determine appropriate range for analysis.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create range of values
        num_points = st.slider("Number of analysis points:", 5, 20, 10)
        values = np.linspace(range_info['min'], range_info['max'], num_points)
        
        predictions = []
        shap_impacts = []
        
        progress_bar = st.progress(0)
        
        for i, val in enumerate(values):
            temp_inputs = user_inputs.copy()
            temp_inputs[selected_param] = val
            
            try:
                # Get prediction
                pred = predict_man_hours(temp_inputs, model_name)
                predictions.append(pred if pred is not None else 0)
                
                # Get SHAP value for this parameter
                shap_vals = get_shap_values_for_input(temp_inputs, model_name)
                if shap_vals is not None:
                    param_index = get_parameter_index(selected_param)
                    if param_index is not None:
                        shap_impact = shap_vals[0][param_index] if isinstance(shap_vals, list) else shap_vals[0][param_index]
                        shap_impacts.append(shap_impact)
                    else:
                        shap_impacts.append(0)
                else:
                    shap_impacts.append(0)
                    
            except Exception as e:
                predictions.append(0)
                shap_impacts.append(0)
            
            progress_bar.progress((i + 1) / len(values))
        
        progress_bar.empty()
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Prediction vs Parameter Value', 'SHAP Impact vs Parameter Value'),
            vertical_spacing=0.12
        )
        
        # Prediction plot
        fig.add_trace(
            go.Scatter(x=values, y=predictions, mode='lines+markers', name='Prediction'),
            row=1, col=1
        )
        
        # SHAP impact plot
        fig.add_trace(
            go.Scatter(x=values, y=shap_impacts, mode='lines+markers', name='SHAP Impact', line=dict(color='red')),
            row=2, col=1
        )
        
        # Highlight current value
        fig.add_vline(x=current_value, line_dash="dash", line_color="green", 
                      annotation_text="Current", row=1, col=1)
        fig.add_vline(x=current_value, line_dash="dash", line_color="green", 
                      annotation_text="Current", row=2, col=1)
        
        fig.update_layout(height=600, title_text=f"What-If Analysis: {selected_param_label}")
        fig.update_xaxes(title_text=selected_param_label, row=2, col=1)
        fig.update_yaxes(title_text="Hours", row=1, col=1)
        fig.update_yaxes(title_text="SHAP Value", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Analysis Summary**")
        
        # Summary statistics
        min_pred = min(predictions)
        max_pred = max(predictions)
        current_pred = predict_man_hours(user_inputs, model_name)
        
        st.metric("Current Prediction", f"{current_pred:.0f} hours")
        st.metric("Prediction Range", f"{min_pred:.0f} - {max_pred:.0f} hours")
        st.metric("Max Variation", f"{max_pred - min_pred:.0f} hours")
        
        if shap_impacts:
            avg_shap = np.mean(np.abs(shap_impacts))
            st.metric("Avg SHAP Impact", f"{avg_shap:.3f}")
        
        # Sensitivity analysis
        sensitivity = (max_pred - min_pred) / (range_info['max'] - range_info['min'])
        st.info(f"**Sensitivity:** {sensitivity:.1f} hours per unit change")
        
        # Show data table
        with st.expander("📋 View Analysis Data"):
            analysis_df = pd.DataFrame({
                selected_param_label: values,
                'Prediction (Hours)': predictions,
                'SHAP Impact': shap_impacts,
                'Difference from Current': [p - current_pred for p in predictions]
            })
            st.dataframe(analysis_df, use_container_width=True)

def display_scenario_comparison(user_inputs, model_name):
    """Compare SHAP values across different project scenarios"""
    st.subheader("📊 Scenario Comparison")
    
    # Define scenarios
    scenarios = {
        "Small Agile Project": {
            "project_prf_max_team_size": 3,
            "project_prf_functional_size": 5,
            "project_tech_primary_programming_language": "Python"
        },
        "Medium Enterprise Project": {
            "project_prf_max_team_size": 8,
            "project_prf_functional_size": 15,
            "project_tech_primary_programming_language": "Java"
        },
        "Large Enterprise Project": {
            "project_prf_max_team_size": 15,
            "project_prf_functional_size": 30,
            "project_tech_primary_programming_language": "C#"
        }
    }
    
    # Add current project as a scenario
    if user_inputs:
        scenarios["Your Current Project"] = user_inputs.copy()
    
    # Calculate predictions and SHAP values for each scenario
    scenario_results = {}
    
    for scenario_name, scenario_inputs in scenarios.items():
        try:
            # Merge with user inputs for missing values
            if scenario_name != "Your Current Project":
                full_inputs = user_inputs.copy() if user_inputs else {}
                full_inputs.update(scenario_inputs)
            else:
                full_inputs = scenario_inputs
            
            # Get prediction
            prediction = predict_man_hours(full_inputs, model_name)
            
            # Get SHAP values
            shap_values = get_shap_values_for_input(full_inputs, model_name)
            
            scenario_results[scenario_name] = {
                'prediction': prediction,
                'shap_values': shap_values,
                'inputs': full_inputs
            }
            
        except Exception as e:
            st.warning(f"Could not analyze scenario '{scenario_name}': {e}")
    
    if not scenario_results:
        st.error("Could not analyze any scenarios.")
        return
    
    # Display comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create comparison chart
        scenario_names = list(scenario_results.keys())
        predictions = [scenario_results[name]['prediction'] for name in scenario_names]
        
        fig = go.Figure(data=[
            go.Bar(x=scenario_names, y=predictions, 
                   text=[f"{p:.0f}h" for p in predictions],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title="Effort Predictions by Scenario",
            xaxis_title="Scenario",
            yaxis_title="Predicted Hours",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Scenario Details**")
        for name, results in scenario_results.items():
            with st.expander(f"{name}"):
                st.metric("Prediction", f"{results['prediction']:.0f} hours")
                
                # Show key parameters
                key_params = ['project_prf_max_team_size', 'project_prf_functional_size']
                for param in key_params:
                    if param in results['inputs']:
                        st.write(f"**{get_field_label(param)}:** {results['inputs'][param]}")
    
    # SHAP comparison heatmap
    st.write("**Feature Impact Comparison**")
    
    try:
        # Create SHAP comparison matrix
        feature_names = get_feature_names_from_inputs(user_inputs) if user_inputs else []
        
        if feature_names:
            shap_matrix = []
            for scenario_name in scenario_names:
                if scenario_results[scenario_name]['shap_values'] is not None:
                    shap_vals = scenario_results[scenario_name]['shap_values']
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    else:
                        shap_vals = shap_vals[0]
                    shap_matrix.append(shap_vals[:len(feature_names)])
                else:
                    shap_matrix.append([0] * len(feature_names))
            
            if shap_matrix:
                shap_df = pd.DataFrame(shap_matrix, 
                                     index=scenario_names, 
                                     columns=[get_field_label(name) for name in feature_names])
                
                # Show top features only (to avoid clutter)
                avg_abs_impact = shap_df.abs().mean().sort_values(ascending=False)
                top_features = avg_abs_impact.head(10).index
                
                fig = px.imshow(shap_df[top_features].T, 
                              aspect="auto",
                              color_continuous_scale="RdBu_r",
                              title="SHAP Values by Scenario (Top 10 Features)")
                
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not create SHAP comparison: {e}")

def display_feature_interactions(user_inputs, model_name):
    """Display feature interaction analysis"""
    st.subheader("🔗 Feature Interactions")
    
    if not user_inputs:
        st.warning("Please make a prediction first to analyze feature interactions.")
        return
    
    try:
        explainer = get_shap_explainer(model_name)
        if explainer is None:
            st.error("Could not create SHAP explainer for interaction analysis.")
            return
        
        # Prepare input data
        input_data = prepare_input_data(user_inputs)
        if input_data is None:
            st.error("Could not prepare input data for analysis.")
            return
        
        # Calculate interaction values (this might be computationally expensive)
        with st.spinner("Calculating feature interactions... This may take a moment."):
            try:
                # For tree-based models, we can get interaction values
                if hasattr(explainer, 'shap_interaction_values'):
                    interaction_values = explainer.shap_interaction_values(input_data.reshape(1, -1))
                else:
                    st.warning("Interaction analysis not available for this model type.")
                    return
                
            except Exception as e:
                st.error(f"Could not calculate interaction values: {e}")
                return
        
        feature_names = get_feature_names_from_inputs(user_inputs)
        
        if interaction_values is not None and len(feature_names) > 1:
            # Create interaction heatmap
            interaction_matrix = interaction_values[0]  # For single prediction
            
            # Limit to top features to avoid clutter
            n_features = min(15, len(feature_names))
            
            # Get feature importance to select top features
            main_effects = np.diagonal(interaction_matrix)
            top_indices = np.argsort(np.abs(main_effects))[-n_features:]
            
            selected_matrix = interaction_matrix[np.ix_(top_indices, top_indices)]
            selected_names = [get_field_label(feature_names[i]) for i in top_indices]
            
            fig = px.imshow(selected_matrix, 
                          x=selected_names, 
                          y=selected_names,
                          color_continuous_scale="RdBu_r",
                          title=f"Feature Interaction Matrix (Top {n_features} Features)")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show strongest interactions
            st.write("**Strongest Feature Interactions**")
            
            interactions = []
            for i in range(len(selected_matrix)):
                for j in range(i+1, len(selected_matrix)):
                    interaction_strength = abs(selected_matrix[i, j])
                    if interaction_strength > 0.001:  # Threshold for significance
                        interactions.append({
                            'Feature 1': selected_names[i],
                            'Feature 2': selected_names[j],
                            'Interaction Strength': interaction_strength,
                            'Effect': 'Positive' if selected_matrix[i, j] > 0 else 'Negative'
                        })
            
            if interactions:
                interaction_df = pd.DataFrame(interactions)
                interaction_df = interaction_df.sort_values('Interaction Strength', ascending=False)
                st.dataframe(interaction_df.head(10), use_container_width=True)
            else:
                st.info("No significant feature interactions detected.")
        
    except Exception as e:
        st.error(f"Error in feature interaction analysis: {e}")

# Helper functions for SHAP analysis
def get_feature_names_from_inputs(user_inputs):
    """Extract feature names from user inputs"""
    exclude_keys = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
    return [k for k in user_inputs.keys() if k not in exclude_keys]

def get_parameter_index(parameter_name):
    """Get the index of a parameter in the feature array"""
    # This should return the index of the parameter in your model's feature array
    # You'll need to implement this based on your feature ordering
    try:
        feature_names = list(FIELDS.keys())
        return feature_names.index(parameter_name)
    except ValueError:
        return None


# --- Field helper functions using merged config ---
def get_field_label(field_name):
    """Get display label for a field"""
    return FIELDS.get(field_name, {}).get("label", field_name.replace("_", " ").title())

def get_field_title(field_name):
    """Get title for a field"""
    return FIELDS.get(field_name, {}).get("title", get_field_label(field_name))

def get_field_help(field_name):
    """Get help text for a field"""
    return FIELDS.get(field_name, {}).get("help", "")

def get_field_options(field_name):
    opts = None
    raw_opts = None
     # Make sure the field exists in the mapping and is not None
    mapping = CATEGORICAL_MAPPING.get(field_name)
    if not mapping:
        return []

    raw_opts = mapping.get('options')
    if raw_opts is None:
        return []
    
    # Special handling for project_prf_relative_size if it's a dict or list of dicts
    if field_name == "project_prf_relative_size":
        # List of dicts style
        if isinstance(raw_opts, list) and raw_opts and isinstance(raw_opts[0], dict):
            opts = [v['label'] for v in raw_opts]
            # Save mappings in session state for later lookups
            st.session_state.prf_size_label2code = {v['label']: v['code'] for v in raw_opts}
            st.session_state.prf_size_code2mid = {v['code']: v['midpoint'] for v in raw_opts}
            # Save the entire option by code for future needs (e.g., min/max hour lookups)
            st.session_state.prf_size_code2full = {v.get('code', ''): v for v in raw_opts}
            print(f"DEBUG: relative size options = {opts}")
            return opts
        else:
            print(f"DEBUG: relative size options = {raw_opts}")
            return raw_opts  # fallback
    else:
        return raw_opts


def get_tab_organization():
    """Get tab organization from configuration"""
    return UI_INFO_CONFIG.get('tab_organization', {
        "Important Features": [],
        "Nice Features": []
    })

def get_ui_behavior():
    """Get UI behavior settings from configuration"""
    return UI_INFO_CONFIG.get('ui_behavior', {})

def get_feature_importance_display():
    """Get feature importance display settings from configuration"""
    return UI_INFO_CONFIG.get('feature_importance_display', {})

def get_prediction_thresholds():
    """Get prediction threshold settings from configuration"""
    return UI_INFO_CONFIG.get('prediction_thresholds', {})

def get_display_config():
    """Get display configuration settings"""
    return UI_INFO_CONFIG.get('display_config', {})

def get_what_if_parameters():
    """Get numeric parameters for what-if analysis"""
    return {get_field_label(f): f for f in FIELDS if FIELDS[f].get("type") == "numeric"}

def get_what_if_range_from_config(field_name, current_value=None):
    """Get appropriate range for what-if analysis from configuration"""
    field_config = FIELDS.get(field_name, {})
    
    if field_config.get('type') != 'numeric':
        return None
    
    # Get configured min/max
    config_min = field_config.get('min', 1)
    config_max = field_config.get('max', 100)
    
    # If current value is provided, create a range around it
    if current_value is not None:
        # Create range from 50% below to 200% above current value
        dynamic_min = max(config_min, current_value * 0.5)
        dynamic_max = min(config_max, current_value * 2.0)
        
        # Ensure we have at least a reasonable range
        if dynamic_max - dynamic_min < (config_max - config_min) * 0.1:
            # Fall back to config range if dynamic range is too small
            return {
                'min': config_min,
                'max': config_max,
                'current': current_value,
                'range_type': 'config_fallback'
            }
        
        return {
            'min': dynamic_min,
            'max': dynamic_max,
            'current': current_value,
            'range_type': 'dynamic'
        }
    
    # Default to config range
    return {
        'min': config_min,
        'max': config_max,
        'current': field_config.get('default', config_min),
        'range_type': 'config'
    }

def render_field(field_name, config, is_required=False):
    """Render a form field based on its configuration"""
    label = config.get("label", field_name)
    help_text = config.get("help", "")
    field_type = config.get("type", "text")
    value = config.get("default")
    field_value = None

    if is_required:
        label = f"{label} ⭐"

    if field_type == "numeric":
        min_val = config.get("min", 0)
        max_val = config.get("max", 9999)
        field_value = st.number_input(
            label, min_value=min_val, max_value=max_val, value=value, help=help_text, key=field_name
        )
    elif field_type == "categorical":
        #st.write("DEBUG: CATEGORICAL_MAPPING keys:", list(CATEGORICAL_MAPPING.keys()))
        #st.write("DEBUG: CATEGORICAL_MAPPING['project_prf_relative_size']:", CATEGORICAL_MAPPING.get("project_prf_relative_size"))

        options = get_field_options(field_name)
        default = config.get("default", options[0] if options else None)
        
        try:
            default_index = options.index(default)
        except (ValueError, IndexError):
            default_index = 0

        # For project_prf_relative_size, show label, store code in user_inputs
        if field_name == "project_prf_relative_size":
            # Defensive: ensure mapping exists
            if "prf_size_label2code" not in st.session_state:
                get_field_options(field_name)
            field_value_label = st.selectbox(
                label, options,
                index=default_index if options else None,
                help=help_text,
                key=field_name
            )
            # If user picks the empty or None, fallback to None or ""
            field_value = st.session_state.prf_size_label2code.get(field_value_label, None)
        else:
            field_value = st.selectbox(
                label, options,
                index=default_index if options else None,
                help=help_text,
                key=field_name
            )

    elif field_type == "boolean":
        field_value = st.checkbox(label, value=bool(value), help=help_text, key=field_name)
    else:
        field_value = st.text_input(label, value=str(value) if value else "", help=help_text, key=field_name)
    return field_value

# --- Main Sidebar Function ---
def sidebar_inputs():
    """Create sidebar inputs"""
    with st.sidebar:
        st.write("HELLO SIDEBAR!")  # Should always show at the very top!
        st.title("🔮 Project Parameters")
        st.info("Required fields (marked with ⭐)")
        user_inputs = {}

        # Get tab organization dynamically
        tab_org = get_tab_organization()
        
        tabs = st.tabs(list(tab_org.keys()))
        for idx, (tab_name, field_list) in enumerate(tab_org.items()):
            with tabs[idx]:
                for field_name in field_list:
                    config = FIELDS.get(field_name)
                    if not config:
                        st.warning(f"⚠️ Field '{field_name}' not configured.")
                        continue
                    
                    # Dynamically set required based on YAML, default False
                    is_required = config.get("mandatory", False)

                    # 👉 Dynamic default for function_size:
                    if field_name == "project_prf_functional_size":
                        # Get the selected relative size code (which must already be chosen in this session)
                        rel_code = user_inputs.get("project_prf_relative_size")  # or st.session_state.get("project_prf_relative_size")
                        if rel_code and rel_code in st.session_state.prf_size_code2mid:
                            config["default"] = st.session_state.prf_size_code2mid[rel_code]
                        else:
                            config["default"] = config.get("default", 5)  # fallback if none

                    field_value = render_field(field_name, config, is_required)
                    user_inputs[field_name] = field_value

        st.divider()
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_model = None
        selected_models = []
        
        try:
            model_status = check_required_models()
            if model_status.get("models_available", False):
                available_models = list_available_models()
                if available_models:
                    model_options = {m['display_name']: m['technical_name'] for m in available_models}
                    #st.write("DEBUG: Model mapping", model_options)
                    #st.write("DEBUG: available_models", available_models)
                    
                    # Support both single and multi-model selection
                    selection_mode = st.radio(
                        "Selection Mode",
                        ["Single Model", "Multiple Models"],
                        help="Choose single model for detailed analysis or multiple models for comparison"
                    )
                    
                    if selection_mode == "Single Model":
                        selected_display_name = st.selectbox(
                            "Choose ML Model",
                            list(model_options.keys()),
                            help="Select one model for prediction."
                        )
                        selected_model = model_options[selected_display_name]
                        selected_models = [selected_model]
                    else:
                        selected_display_names = st.multiselect(
                            "Choose ML Models",
                            list(model_options.keys()),
                            help="Select multiple models for comparison analysis."
                        )
                        selected_models = [model_options[name] for name in selected_display_names]
                        selected_model = selected_models[0] if selected_models else None
                    
                    if st.session_state.prediction_history:
                        st.info(f"📊 {len(st.session_state.prediction_history)} predictions made so far")
                else:
                    st.warning("⚠️ No trained models found")
            else:
                st.warning("⚠️ Models not available")
        except Exception as e:
            st.error(f"Model loading error: {e}")
            selected_model = None
            selected_models = []

        # Required field check using dynamic tab organization
        # Dynamically set required based on YAML, default False
        required_fields = [fname for fname, fdef in FIELDS.items() if fdef.get("mandatory", False)]
        missing_fields = []
        for field in required_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_label(field))

        if missing_fields and st.session_state.get('form_attempted'):
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")

        st.divider()
        predict_button = st.button(
            "🔮 Predict Effort",
            type="primary",
            use_container_width=True,
            disabled=len(missing_fields) > 0 or not selected_models
        )
        if predict_button:
            st.session_state['form_attempted'] = True

        # Prediction history management
        st.subheader("📈 Prediction History")
        col1, col2 = st.columns(2)
        with col1:
            clear_results = st.button(
                "🗑️ Clear History",
                use_container_width=True,
                help="Clear all previous predictions"
            )
        with col2:
            show_history = st.button(
                "📊 Show All",
                use_container_width=True,
                help="Show detailed prediction history"
            )

        # Save config
        st.subheader("💾 Save Configuration")
        config_name = st.text_input("Configuration Name", placeholder="e.g., Banking_Project_Template")
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.button("💾 Save Config", use_container_width=True, disabled=not config_name.strip())
        with col2:
            if st.button("📁 Load Config", use_container_width=True):
                configs_dir = "saved_configs"
                if os.path.exists(configs_dir):
                    config_files = [f.replace('.json', '') for f in os.listdir(configs_dir) if f.endswith('.json')]
                    if config_files:
                        st.info(f"Available configs: {', '.join(config_files)}")
                    else:
                        st.info("No saved configurations found")
                else:
                    st.info("No saved configurations found")

        if save_button and config_name.strip():
            save_current_configuration(user_inputs, config_name.strip())

        if clear_results:
            st.session_state.prediction_history = []
            st.session_state.comparison_results = []

        user_inputs["selected_model"] = selected_model
        user_inputs["selected_models"] = selected_models
        user_inputs["submit"] = predict_button
        user_inputs["clear_results"] = clear_results
        user_inputs["show_history"] = show_history


        return user_inputs

# --- Configuration Management ---
def save_current_configuration(user_inputs, config_name):
    """Save current configuration to file"""
    config = user_inputs.copy()
    config.pop('submit', None)
    config.pop('selected_models', None)
    config.pop('clear_results', None)
    config.pop('comparison_mode', None)
    config['saved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    configs_dir = "saved_configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    config_file = f'{configs_dir}/{config_name}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    st.success(f"✅ Configuration '{config_name}' saved!")

# --- Display Functions ---
def display_inputs(user_inputs, selected_models):
    """Display input parameters summary in a collapsible expander"""
    with st.expander("📋 Input Parameters Summary", expanded=False):
        exclude_keys = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
        items = [(get_field_label(k), v) for k, v in user_inputs.items() 
                if k not in exclude_keys and v is not None and v != ""]
        
        if items:
            # Group items for better display
            col1, col2 = st.columns(2)
            mid = len(items) // 2
            
            with col1:
                for param, value in items[:mid]:
                    st.text(f"**{param}:** {value}")
            with col2:
                for param, value in items[mid:]:
                    st.text(f"**{param}:** {value}")
            
            # Show selected models
            if selected_models:
                model_names = []
                for model in selected_models:
                    try:
                        model_display_name = get_model_display_name(model)
                        model_names.append(model_display_name)
                    except:
                        model_names.append(model)
                st.info(f"🤖 **Model(s):** {', '.join(model_names)}")
            
            # Show key parameters summary if configured
            display_config = get_display_config()
            key_params = display_config.get('key_parameters_for_summary', [])
            
            if key_params:
                st.markdown("**Key Parameters:**")
                key_items = [(get_field_label(k), user_inputs.get(k)) for k in key_params 
                           if k in user_inputs and user_inputs[k] is not None and user_inputs[k] != ""]
                
                if key_items:
                    for param, value in key_items:
                        st.text(f"• {param}: {value}")
        else:
            st.warning("No parameters to display")
            
        # Show configuration completeness
        total_fields = len(FIELDS)
        filled_fields = len([k for k in user_inputs.keys() 
                           if k not in exclude_keys and user_inputs.get(k) is not None and user_inputs.get(k) != ""])
        
        if total_fields > 0:
            completeness = (filled_fields / total_fields) * 100
            st.progress(completeness / 100)
            st.caption(f"Configuration completeness: {completeness:.1f}% ({filled_fields}/{total_fields} fields)")

def show_prediction(prediction, team_size, model_name, user_inputs=None):
    """Show prediction results with team breakdown and dynamic size-band warnings."""
    if prediction is None:
        st.error("Prediction failed. Please check your inputs and try again.")
        return
    
    st.subheader("🎯 Prediction Results")
    
    try:
        model_display_name = get_model_display_name(model_name)
        st.info(f"**Model Used:** {model_display_name}")
    except:
        st.info(f"**Model Used:** {model_name}")
    
    # Main prediction metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Effort", f"{prediction:.0f} hours")

    with col2:
        days = prediction / 8
        st.metric("📅 Working Days", f"{days:.1f} days")

    with col3:
        weeks = days / 5
        st.metric("📆 Working Weeks", f"{weeks:.1f} weeks")

    with col4:
        months = weeks / 4.33
        st.metric("🗓️ Months", f"{months:.1f} months")

    
    # --- Use dynamic thresholds based on selected relative size ---
    if user_inputs is not None and "project_prf_relative_size" in user_inputs:
        rel_code = user_inputs["project_prf_relative_size"]
        size_info = st.session_state.prf_size_code2full.get(rel_code, {})
        min_hour = size_info.get("minimumhour", 0)
        max_hour = size_info.get("maximumhour", None)

        if prediction < min_hour:
            st.warning(
                f"⚠️ The prediction ({prediction:.0f} hours) is **below** the expected minimum for this size ({min_hour} hours). "
                "Please check your inputs."
            )
        elif max_hour and prediction > max_hour:
            st.warning(
                f"⚠️ The prediction ({prediction:.0f} hours) is **above** the expected maximum for this size ({max_hour} hours). "
                "Consider breaking down the project or reviewing your parameters."
            )
    else:
        # Fallback to fixed thresholds
        low_threshold = PREDICTION_THRESHOLDS.get('low_prediction_warning', 10)
        high_threshold = PREDICTION_THRESHOLDS.get('high_prediction_warning', 192000)
        if prediction < low_threshold:
            st.warning(f"⚠️ Very low effort prediction ({prediction:.0f} hours). Please verify your inputs.")
        elif prediction > high_threshold:
            st.warning(f"⚠️ Very high effort prediction ({prediction:.0f} hours). Consider breaking down the project.")

def show_feature_importance(model_name, features_dict):
    """Display feature importance analysis"""
    try:
        feature_importance = get_feature_importance(model_name)
        if feature_importance is None:
            st.info("Feature importance analysis not available for this model.")
            return
        
        st.subheader("📊 Feature Importance Analysis")
        
        exclude_keys = {'selected_models', 'submit', 'clear_results', 'comparison_mode', 'selected_model', 'show_history'}
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
        
        # Get display settings
        max_features = FEATURE_IMPORTANCE_DISPLAY.get('max_features_shown', 15)
        precision = FEATURE_IMPORTANCE_DISPLAY.get('precision_decimals', 3)
        
        importance_data = []
        for i, name in enumerate(feature_names[:max_features]):
            if i < len(feature_importance):
                friendly_name = get_field_title(name)
                importance_data.append({
                    'Feature': friendly_name,
                    'Importance': abs(feature_importance[i])
                })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Display chart
            st.bar_chart(importance_df.set_index('Feature'))
            
            with st.expander("📋 View Detailed Importance Data"):
                st.dataframe(importance_df.round(precision), use_container_width=True)
        else:
            st.warning("No feature importance data available")
    
    except Exception as e:
        st.info(f"Feature importance analysis not available: {e}")

def perform_what_if_analysis(user_inputs, model_name, field_name, display_name):
    """Perform what-if analysis by varying a parameter"""
    try:
        field_config = FIELDS.get(field_name, {})
        if field_config.get('type') != 'numeric':
            st.warning("What-if analysis only available for numeric parameters")
            return
        
        current_val = user_inputs.get(field_name, field_config.get('default', 1))
        
        # Get appropriate range for analysis
        range_info = get_what_if_range_from_config(field_name, current_val)
        if range_info is None:
            st.warning("Could not determine appropriate range for what-if analysis")
            return
        
        min_val = range_info['min']
        max_val = range_info['max']
        
        # Show range information
        st.info(f"Analyzing {display_name} from {min_val:.1f} to {max_val:.1f} (Range type: {range_info['range_type']})")
        
        # Create range of values (more points for better resolution)
        num_points = 15
        values = np.linspace(min_val, max_val, num_points)
        predictions = []
        
        progress_bar = st.progress(0)
        for i, val in enumerate(values):
            temp_inputs = user_inputs.copy()
            temp_inputs[field_name] = val
            try:
                pred = predict_man_hours(temp_inputs, model_name)
                predictions.append(pred if pred is not None else 0)
            except:
                predictions.append(0)
            progress_bar.progress((i + 1) / len(values))
        
        progress_bar.empty()
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            display_name: values,
            'Predicted Hours': predictions
        })
        
        # Display chart
        st.line_chart(df.set_index(display_name))
        
        # Show current value and impact analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Value", f"{current_val}")
        with col2:
            min_pred = min(predictions)
            max_pred = max(predictions)
            st.metric("Prediction Range", f"{min_pred:.0f} - {max_pred:.0f} hours")
        with col3:
            impact = max_pred - min_pred if max_pred > min_pred else 0
            st.metric("Max Impact", f"{impact:.0f} hours")
        
        # Sensitivity analysis
        if impact > 0:
            sensitivity = impact / (max_val - min_val)
            st.info(f"**Sensitivity:** {sensitivity:.1f} hours per unit change in {display_name}")
        
        with st.expander("📋 View What-If Data"):
            df['Impact vs Current'] = df['Predicted Hours'] - predict_man_hours(user_inputs, model_name)
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in what-if analysis: {e}")

def show_prediction_history():
    """Display prediction history"""
    if not st.session_state.prediction_history:
        return
    
    st.subheader("📈 Prediction History")
    
    history_data = []
    
    try:
        for entry in st.session_state.prediction_history:
            # Safely extract model information
            model_technical = entry.get('model_technical', '')
            if not model_technical:
                model_technical = entry.get('model', 'Unknown Model')
            
            # Get display name with fallback
            try:
                # Import the function if not already imported
                from models import get_model_display_name_from_config
                model_display = get_model_display_name_from_config(model_technical)
            except Exception as e:
                # Fallback to the technical name or stored display name
                model_display = entry.get('model', model_technical)
                if not model_display:
                    model_display = model_technical
            
            # Build history entry
            history_entry = {
                'Timestamp': entry.get('timestamp', 'Unknown'),
                'Model': model_display,
                'Hours': f"{entry.get('prediction_hours', 0):.0f}",
                'Days': f"{entry.get('prediction_hours', 0)/8:.1f}"
            }
            history_data.append(history_entry)
        
        # Display the data
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prediction history to display")
            
    except Exception as e:
        st.error(f"Error displaying prediction history: {str(e)}")
        # Show debug information
        with st.expander("Debug Information"):
            st.write("Prediction history contents:")
            st.write(st.session_state.prediction_history)

def show_prediction_comparison_table():
    """Show comparison table if multiple predictions exist"""
    if len(st.session_state.prediction_history) <= 1:
        return
    
    st.subheader("🔍 Prediction Comparison")
    
    try:
        predictions = []
        models = []
        
        for entry in st.session_state.prediction_history:
            # Extract prediction safely
            prediction_hours = entry.get('prediction_hours', 0)
            predictions.append(prediction_hours)
            
            # Extract model name safely
            model_technical = entry.get('model_technical', '')
            if not model_technical:
                model_technical = entry.get('model', 'Unknown Model')
            
            # Get display name with fallback
            try:
                from models import get_model_display_name_from_config
                model_display = get_model_display_name_from_config(model_technical)
            except Exception:
                model_display = entry.get('model', model_technical)
                if not model_display:
                    model_display = model_technical
            
            models.append(model_display)
        
        # Create comparison data
        comparison_data = {
            'Model': models,
            'Hours': predictions,
            'Days': [p/8 for p in predictions]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Statistics
        if len(predictions) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{np.mean(predictions):.0f} hours")
            with col2:
                st.metric("Min", f"{np.min(predictions):.0f} hours")
            with col3:
                st.metric("Max", f"{np.max(predictions):.0f} hours")
            with col4:
                st.metric("Std Dev", f"{np.std(predictions):.0f} hours")
                
    except Exception as e:
        st.error(f"Error creating comparison table: {str(e)}")

def show_multiple_predictions(new_predictions, team_size):
    """Display results when multiple models are used"""
    if not new_predictions:
        st.warning("No predictions available")
        return
    
    st.subheader("🔍 Multi-Model Prediction Comparison")
    
    try:
        # Create comparison table
        comparison_data = []
        predictions_list = []
        
        for model_name, prediction in new_predictions.items():
            if prediction is not None:
                # Get display name with fallback
                try:
                    from models import get_model_display_name_from_config
                    model_display_name = get_model_display_name_from_config(model_name)
                except Exception:
                    model_display_name = model_name
                
                days = prediction / 8
                
                comparison_data.append({
                    'Model': model_display_name,
                    'Hours': f"{prediction:.0f}",
                    'Days': f"{days:.1f}",
                    'Weeks': f"{days/5:.1f}"
                })
                predictions_list.append(prediction)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Statistics summary
            if len(predictions_list) > 1:
                st.subheader("📊 Statistical Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average", f"{np.mean(predictions_list):.0f} hours")
                with col2:
                    st.metric("Minimum", f"{np.min(predictions_list):.0f} hours")
                with col3:
                    st.metric("Maximum", f"{np.max(predictions_list):.0f} hours")
                with col4:
                    std_dev = np.std(predictions_list)
                    st.metric("Std Deviation", f"{std_dev:.0f} hours")
                    
                # Variance warning
                if std_dev > np.mean(predictions_list) * 0.3:  # 30% threshold
                    st.warning("⚠️ High variance detected between models. Consider reviewing input parameters.")
        else:
            st.error("All predictions failed. Please check your inputs.")
            
    except Exception as e:
        st.error(f"Error displaying multiple predictions: {str(e)}")

def add_prediction_to_history(user_inputs, model_name, prediction, team_size):
    """Add prediction to session history - Fixed version"""
    if prediction is None:
        return
    
    try:
        # Get display name safely
        try:
            from models import get_model_display_name_from_config
            model_display_name = get_model_display_name_from_config(model_name)
        except Exception:
            model_display_name = model_name
        
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_display_name,  # Store display name
            'model_technical': model_name,  # Store technical name separately
            'prediction_hours': prediction,
            'team_size': team_size,
            'inputs': user_inputs.copy() if user_inputs else {}
        }
        
        st.session_state.prediction_history.append(history_entry)
        
    except Exception as e:
        st.error(f"Error adding prediction to history: {str(e)}")

def display_model_comparison():
    """Display model comparison analysis - Fixed version"""
    st.header("🤖 Model Comparison")
    
    if len(st.session_state.prediction_history) < 2:
        st.warning("⚠️ Please make predictions with at least 2 different models to enable comparison.")
        return
    
    try:
        # Group predictions by model
        model_predictions = {}
        
        for entry in st.session_state.prediction_history:
            # Use technical name for grouping to avoid display name inconsistencies
            model_name = entry.get('model_technical', entry.get('model', 'Unknown'))
            prediction_hours = entry.get('prediction_hours', 0)
            
            if model_name not in model_predictions:
                model_predictions[model_name] = []
            model_predictions[model_name].append(prediction_hours)
        
        # Create comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance Comparison")
            
            # Box plot for model predictions
            comparison_data = []
            for model, predictions in model_predictions.items():
                # Get display name for visualization
                try:
                    from models import get_model_display_name_from_config
                    display_name = get_model_display_name_from_config(model)
                except Exception:
                    display_name = model
                
                for pred in predictions:
                    comparison_data.append({
                        'Model': display_name,
                        'Prediction (Hours)': pred
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create box plot using plotly
                fig = px.box(comparison_df, x='Model', y='Prediction (Hours)',
                            title="Distribution of Predictions by Model")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Statistics")
            
            # Statistics table
            stats_data = []
            for model, predictions in model_predictions.items():
                if predictions:
                    try:
                        from models import get_model_display_name_from_config
                        display_name = get_model_display_name_from_config(model)
                    except Exception:
                        display_name = model
                    
                    stats_data.append({
                        'Model': display_name,
                        'Count': len(predictions),
                        'Mean': f"{np.mean(predictions):.0f}",
                        'Std Dev': f"{np.std(predictions):.0f}",
                        'Min': f"{np.min(predictions):.0f}",
                        'Max': f"{np.max(predictions):.0f}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
        # Additional analysis sections would go here...
        
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
        with st.expander("Debug Information"):
            st.write("Prediction history structure:")
            for i, entry in enumerate(st.session_state.prediction_history):
                st.write(f"Entry {i}: {entry}")

# Also fix the import at the top of ui.py - make sure you have this:
def get_model_display_name_safe(model_name):
    """Safe wrapper for getting model display name"""
    try:
        from models import get_model_display_name_from_config
        return get_model_display_name_from_config(model_name)
    except Exception as e:
        # Fallback to basic transformation
        return " ".join(word.capitalize() for word in model_name.split("_"))
        

def run_predictions(user_inputs, selected_models):
    """Run predictions for multiple models"""
    new_predictions = {}
    
    if isinstance(selected_models, str):
        selected_models = [selected_models]
    
    for model in selected_models:
        try:
            prediction = predict_man_hours(user_inputs, model)
            new_predictions[model] = prediction
            
            # Add to session state for this run
            team_size = user_inputs.get('project_prf_max_team_size', 5)
            add_prediction_to_history(user_inputs, model, prediction, team_size)
            
        except Exception as e:
            st.error(f"Error predicting with {model}: {str(e)}")
            new_predictions[model] = None
    
    return new_predictions

def display_prediction_results(selected_models, new_predictions, team_size, user_inputs, comparison_mode=False):
    """Display prediction results based on number of models and mode"""
    
    # Display current results
    if len(selected_models) == 1:
        # Single model - show detailed view
        model = selected_models[0]
        prediction = new_predictions.get(model)
        show_prediction(prediction, team_size, model, user_inputs)
    else:
        # Multiple models - show comparison
        show_multiple_predictions(new_predictions, team_size)
    
    # Show historical comparison if in comparison mode
    if comparison_mode and len(st.session_state.prediction_history) > len(selected_models):
        display_historical_comparison()

def display_historical_comparison():
    """Display historical comparison of predictions"""
    st.subheader("📈 Historical Prediction Comparison")
    
    if len(st.session_state.prediction_history) < 2:
        st.info("Need at least 2 predictions for historical comparison")
        return
    
    # Create timeline chart
    history_data = []
    for i, item in enumerate(st.session_state.prediction_history):
        history_data.append({
            'Prediction #': i + 1,
            'Model': item['model'],
            'Hours': item['prediction_hours'],
            'Timestamp': item['timestamp']
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Show line chart
    st.line_chart(history_df.set_index('Prediction #')['Hours'])
    
    # Show detailed table
    with st.expander("📋 View Historical Data"):
        st.dataframe(history_df, use_container_width=True)

def clear_prediction_results():
    """Clear all prediction results from session state"""
    st.session_state.prediction_history = []
    if 'comparison_results' in st.session_state:
        st.session_state.comparison_results = []
    if 'prediction_results' in st.session_state:
        st.session_state.prediction_results = []

def display_previous_results_summary():
    """Display summary of previous results"""
    if not st.session_state.prediction_history:
        return
    
    st.subheader("📊 Previous Predictions Summary")
    
    # Show last few predictions
    recent_predictions = st.session_state.prediction_history[-3:]  # Show last 3
    
    for item in recent_predictions:
        with st.expander(f"🔮 {item['model']} - {item['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hours", f"{item['prediction_hours']:.0f}")
            with col2:
                st.metric("Days", f"{item['prediction_hours']/8:.1f}")
            with col3:
                st.metric("Team Size", item['team_size'])
    
    # Summary statistics if multiple predictions
    if len(st.session_state.prediction_history) > 1:
        all_predictions = [item['prediction_hours'] for item in st.session_state.prediction_history]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(all_predictions))
        with col2:
            st.metric("Average Hours", f"{np.mean(all_predictions):.0f}")
        with col3:
            st.metric("Range", f"{np.min(all_predictions):.0f} - {np.max(all_predictions):.0f}")

def display_static_shap_analysis():
    """Display static SHAP analysis from file"""
    st.header("📈 Static SHAP Analysis - Model Feature Importance")

    try:
        with open(SHAP_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            shap_report_md = f.read()
        st.markdown(shap_report_md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load static SHAP analysis report: {e}")

def display_visualizations_and_analysis():
    """Display the Visualizations & Analysis tab content"""
    st.header("📊 Visualizations & Analysis")
    
    # Check if we have a recent prediction to analyze
    if not st.session_state.prediction_history:
        st.warning("⚠️ Please make at least one prediction first to enable detailed analysis.")
        return
    
    # Get the most recent prediction for analysis
    latest_prediction = st.session_state.prediction_history[-1]
    user_inputs = latest_prediction.get('inputs', {})
    model_name = latest_prediction.get('model_technical')
    
    if not user_inputs or not model_name:
        st.error("Cannot perform analysis - missing prediction data.")
        return
    
    # Create sub-tabs for different analyses
    analysis_tabs = st.tabs([
        "🎯 Instance-Specific SHAP", 
        "🔍 What-If Analysis", 
        "📊 Scenario Comparison",
        "🔗 Feature Interactions"
    ])
    
    with analysis_tabs[0]:
        display_instance_specific_shap(user_inputs, model_name)
    
    with analysis_tabs[1]:
        display_what_if_shap_analysis(user_inputs, model_name)
    
    with analysis_tabs[2]:
        display_scenario_comparison(user_inputs, model_name)
    
    with analysis_tabs[3]:
        display_feature_interactions(user_inputs, model_name)

def display_model_comparison():
    """Display model comparison analysis"""
    st.header("🤖 Model Comparison")
    
    if len(st.session_state.prediction_history) < 2:
        st.warning("⚠️ Please make predictions with at least 2 different models to enable comparison.")
        return
    
    # Group predictions by model
    model_predictions = {}
    for bulk_item in st.session_state.prediction_history:
        model_name = bulk_item.get('model_technical', bulk_item['model'])
        if model_name not in model_predictions:
            model_predictions[model_name] = []
        model_predictions[model_name].append(bulk_item['prediction_hours'])
    
    # Create comparison visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance Comparison")
        
        # Box plot for model predictions
        comparison_data = []
        for model, predictions in model_predictions.items():
            for pred in predictions:
                comparison_data.append({
                    'Model': get_model_display_name(model),
                    'Prediction (Hours)': pred
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create box plot using plotly
            fig = px.box(comparison_df, x='Model', y='Prediction (Hours)',
                        title="Distribution of Predictions by Model")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Model Statistics")
        
        # Statistics table
        stats_data = []
        for model, predictions in model_predictions.items():
            if predictions:
                stats_data.append({
                    'Model': get_model_display_name(model),
                    'Count': len(predictions),
                    'Mean': f"{np.mean(predictions):.0f}",
                    'Std Dev': f"{np.std(predictions):.0f}",
                    'Min': f"{np.min(predictions):.0f}",
                    'Max': f"{np.max(predictions):.0f}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    # Model agreement analysis
    st.subheader("🎯 Model Agreement Analysis")
    
    if len(model_predictions) >= 2:
        # Find predictions made with same inputs across different models
        input_signatures = {}
        for history_item in st.session_state.prediction_history:
            # Create a signature from input parameters
            inputs = history_item.get('inputs', {})
            exclude_keys = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history'}
            signature = tuple(sorted([(k, v) for k, v in inputs.items() if k not in exclude_keys]))
            
            if signature not in input_signatures:
                input_signatures[signature] = {}
            
            model_name = history_item.get('model_technical', history_item['model'])
            input_signatures[signature][model_name] = history_item['prediction_hours']
        
        # Find cases where multiple models predicted same inputs
        multi_model_cases = {sig: models for sig, models in input_signatures.items() if len(models) >= 2}
        
        if multi_model_cases:
            agreement_data = []
            for signature, model_preds in multi_model_cases.items():
                predictions = list(model_preds.values())
                variance = np.var(predictions)
                agreement_score = 1 / (1 + variance/np.mean(predictions)**2) if np.mean(predictions) > 0 else 0
                
                agreement_data.append({
                    'Input Set': f"Case {len(agreement_data) + 1}",
                    'Models': len(model_preds),
                    'Predictions': ', '.join([f"{p:.0f}h" for p in predictions]),
                    'Variance': f"{variance:.0f}",
                    'Agreement Score': f"{agreement_score:.3f}"
                })
            
            if agreement_data:
                st.dataframe(pd.DataFrame(agreement_data), use_container_width=True)
                
                avg_agreement = np.mean([float(row['Agreement Score']) for row in agreement_data])
                if avg_agreement > 0.8:
                    st.success(f"✅ High model agreement (avg: {avg_agreement:.3f})")
                elif avg_agreement > 0.6:
                    st.warning(f"⚠️ Moderate model agreement (avg: {avg_agreement:.3f})")
                else:
                    st.error(f"❌ Low model agreement (avg: {avg_agreement:.3f}) - consider reviewing inputs")
        else:
            st.info("No cases found where multiple models predicted the same inputs.")

# --- About Section Function ---
def about_section():
    """Display about section with tool information"""
    st.markdown("""
    ### About This Tool
    
    The **ML Project Effort Estimator** is a machine learning-powered tool designed to help project managers, 
    developers, and teams estimate the effort required for software development projects.
    
    #### Key Features:
    - **Multiple ML Models**: Compare predictions from different machine learning models
    - **Comprehensive Parameters**: Considers project size, team composition, technology stack, and organizational factors
    - **Interactive Interface**: User-friendly form with real-time validation and feedback
    - **Advanced SHAP Analysis**: Deep insights into feature importance and model behavior
    - **Configuration Management**: Save and load project configurations for reuse
    - **Historical Tracking**: Keep track of previous predictions for comparison
    
    #### How It Works:
    1. **Data Input**: Enter project parameters including team size, technology stack, and project characteristics
    2. **Model Selection**: Choose one or multiple ML models for prediction/comparison
    3. **ML Prediction**: The tool applies trained machine learning models to generate effort estimates
    4. **Results Analysis**: View the predicted effort in hours, days, and per-person breakdowns
    5. **Deep Analysis**: Understand which factors most influence your project's effort estimate using SHAP
    
    #### Advanced Analysis Features:
    - **Instance-Specific SHAP**: See how each feature impacts your specific prediction
    - **What-If Analysis**: Understand sensitivity to parameter changes
    - **Scenario Comparison**: Compare different project types and approaches
    - **Feature Interactions**: Discover how features work together
    - **Model Comparison**: Analyze agreement and variance between different models
    
    #### Best Practices:
    - Provide accurate team size and project complexity information
    - Select multiple models to get a range of estimates and validate consistency
    - Use the visualization tools to understand model behavior
    - Save configurations for similar future projects
    - Consider the tool's predictions as guidance alongside expert judgment
    
    #### Model Information:
    The underlying models are trained on historical project data and consider factors such as:
    - Project size and complexity
    - Team size and composition
    - Technology stack and architecture
    - Industry sector and organization type
    - Development approach and methodology
    
    For technical support or questions, please refer to the documentation or contact the development team.
    """)

# --- Main Application Function ---
def main():
    """Main application function with full multi-model support and enhanced SHAP analysis"""

    # Set sidebar width
    set_sidebar_width()
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.title("🔮 ML Agile Software Project Effort Estimator")
    st.markdown("Get accurate effort estimates using machine learning models trained on historical project data.")
    
    #st.write("DEBUG (main): Model mapping", model_options)
    #st.write("DEBUG (main): available_models", list_available_models())
    try:
        # Get user inputs from sidebar
        user_inputs = sidebar_inputs()
        
        # Handle clear results
        if user_inputs.get('clear_results', False):
            clear_prediction_results()
            st.rerun()

        # --- Add tab navigation for main content ---
        main_tabs = st.tabs(["🔮 Estimator", "📊 Visualisations & Analysis", "🤖 Model Comparison", "📈 Static SHAP Analysis", "❓ Help"])

        with main_tabs[0]:  # Estimator tab
            if user_inputs.get('submit', False):
                selected_model = user_inputs.get('selected_model')
                selected_models = user_inputs.get('selected_models', [])
                
                if selected_model:
                    # Display input summary
                    display_inputs(user_inputs, [selected_model])
                    st.divider()
                    
                    # Run prediction(s)
                    with st.spinner("Calculating estimation..."):
                        try:
                            if len(selected_models) <= 1:
                                # Single model workflow
                                prediction = predict_man_hours(user_inputs, selected_model)
                                team_size = user_inputs.get('project_prf_max_team_size', 5)
                                
                                # Show current prediction
                                show_prediction(prediction, team_size, selected_model, user_inputs)
                                
                                # Add to history
                                add_prediction_to_history(user_inputs, selected_model, prediction, team_size)
                                
                            else:
                                # Multi-model workflow
                                new_predictions = run_predictions(user_inputs, selected_models)
                                team_size = user_inputs.get('project_prf_max_team_size', 5)
                                comparison_mode = user_inputs.get('comparison_mode', False)
                                
                                # Display results
                                display_prediction_results(selected_models, new_predictions, team_size, user_inputs, comparison_mode)
                            
                            # Show history and comparisons
                            show_prediction_history()
                            show_prediction_comparison_table()
                            
                            # Show feature importance
                            st.divider()
                            show_feature_importance(selected_model, user_inputs)
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                
                else:
                    st.warning("⚠️ Please select a model to make predictions")
            else:
                # Welcome screen
                st.info("👈 **Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate.")
                
                # Show previous results summary if any
                display_previous_results_summary()

        with main_tabs[1]:  # Visualisations & Analysis tab
            display_visualizations_and_analysis()

        with main_tabs[2]:  # Model Comparison tab
            display_model_comparison()

        with main_tabs[3]:  # Static SHAP Analysis tab
            display_static_shap_analysis()

        with main_tabs[4]:  # Help tab            
            with st.expander("ℹ️ How to Use This Tool"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with ⭐ in the "Required Fields" tab
                2. **Optional Parameters** - Add more details in the "Optional Fields" tab for better accuracy  
                3. **Select Model** - Choose a model for prediction
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                5. **Analyze Results** - Use the Visualizations & Analysis tab for deep insights
                6. **Save Configuration** - Save your parameter settings for future use
                
                ### New Analysis Features
                - **Instance-Specific SHAP**: See how each feature impacts YOUR specific prediction
                - **What-If Analysis**: Understand how changing parameters affects predictions
                - **Scenario Comparison**: Compare your project against typical project types
                - **Feature Interactions**: Discover how features work together
                - **Model Comparison**: Analyze agreement between different models
                
                ### Multi-Model Features
                - **Single Model**: Detailed prediction view with full analysis
                - **Multiple Models**: Comparison table with statistics
                - **Compare Mode**: Keep previous results for historical comparison
                - **Clear Results**: Remove all stored predictions
                
                ### Features
                - **Detailed Predictions**: Hours, days, and per-person breakdowns
                - **Prediction History**: Track and compare multiple predictions
                - **Advanced SHAP Analysis**: Deep understanding of feature importance
                - **Interactive Visualizations**: Dynamic charts and plots
                - **Configuration Save/Load**: Reuse settings for similar projects
                
                ### Tips for Better Estimates
                - Fill in as many relevant fields as possible
                - Use realistic team sizes and project characteristics
                - Explore the Visualizations & Analysis tab after making predictions
                - Compare multiple predictions to understand variability
                - Save configurations for similar future projects
                - Use what-if analysis to understand parameter sensitivity
                
                ### Troubleshooting
                - Ensure all required fields (⭐) are completed
                - Check that models are available in the dropdown
                - Make at least one prediction to enable analysis features
                - Review field help text for guidance on values
                - Use "Clear History" if you want to start fresh
                """)
            
            # About section
            with st.expander("📖 About This Tool"):
                about_section()
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration files and model setup.")


if __name__ == "__main__":
    main()