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
import traceback
from constants import FileConstants, UIConstants, PipelineConstants
from config_loader import ConfigLoader

from shap_analysis import (
    # Keep existing backward compatibility imports
    display_optimized_shap_analysis,  # NEW: Main function to replace old SHAP calls
    get_shap_explainer_optimized,     # NEW: For any direct explainer needs
    clear_explainer_cache,            # Keep existing
    get_cache_info   
)

# ---------------- CONFIG & HISTORY HELPERS ----------------

def make_current_config_json(user_inputs, config_name, selected_model, prediction):
    """
    Return a JSON string for downloading the current configuration (inputs, model, prediction, metadata).
    """
    config = user_inputs.copy()
    exclude_keys = {'submit', 'selected_models', 'clear_results', 'comparison_mode', 'selected_model', 'show_history'}
    for key in exclude_keys:
        config.pop(key, None)
    config['picked_model'] = selected_model
    config['predicted_effort_hours'] = prediction
    config['saved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    config['_metadata'] = {
        'config_name': config_name,
        'saved_date': config['saved_date'],
        'app_version': '1.0',
        'description': f'ML Project Effort Estimator configuration: {config_name}'
    }
    return json.dumps(config, indent=2, default=str)

def make_history_json():
    """
    Return a JSON string of the full prediction history with export metadata.
    """
    history = st.session_state.get('prediction_history', [])
    export = {
        "_metadata": {
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "record_count": len(history),
            "app_version": "1.0"
        },
        "prediction_history": history
    }
    return json.dumps(export, indent=2, default=str)

# -------------- MODEL IMPORTS AND FALLBACKS ----------------


try:
    from models import (
        predict_man_hours,
        list_available_models,
        check_required_models,
        get_feature_importance,
        get_model_display_name,
        get_model_display_name_from_config,
        get_trained_model,  # Add this function to get the actual model object
        prepare_input_data,  # Add this function to prepare data for SHAP
        prepare_features_for_model,
        load_model_display_names,  # Add this function to load model display names
        load_preprocessing_pipeline  # Add this function to load the preprocessing pipeline
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

print("🔍 DEBUG: ui.py execution started")
print("🔍 DEBUG: About to load configurations...")

# --------------------- CONFIG LOADING ---------------------


ui_config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.UI_INFO_FILE)
UI_INFO_CONFIG = ConfigLoader.load_yaml_config(ui_config_path)
if UI_INFO_CONFIG is None:
    UI_INFO_CONFIG = {}
    print(f"⚠️ Warning: Could not load UI configuration from {ui_config_path}. Using empty config.")
print("🔍 DEBUG: UI_INFO_CONFIG loaded successfully")

FIELDS = UI_INFO_CONFIG.get('fields', {})
print("🔍 DEBUG: FIELDS loaded successfully")

TAB_ORG = UI_INFO_CONFIG.get('tab_organization', {})
print("🔍 DEBUG: TAB_ORG loaded successfully")

UI_BEHAVIOR = UI_INFO_CONFIG.get('ui_behavior', {})
FEATURE_IMPORTANCE_DISPLAY = UI_INFO_CONFIG.get('feature_importance_display', {})
PREDICTION_THRESHOLDS = UI_INFO_CONFIG.get('prediction_thresholds', {})
DISPLAY_CONFIG = UI_INFO_CONFIG.get('display_config', {})

feature_mapping_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.FEATURE_MAPPING_FILE)
FEATURE_MAPPING = ConfigLoader.load_yaml_config(feature_mapping_path)
if FEATURE_MAPPING is None:
    FEATURE_MAPPING = {}
    print(f"⚠️ Warning: Could not load feature mapping from {feature_mapping_path}. Using empty mapping.")

CATEGORICAL_MAPPING = FEATURE_MAPPING.get('categorical_features', {})

IMPORTANT_TABS = "Important Features"
NICE_TABS = "Nice Features"

print("🔍 DEBUG: About to define functions...")

def set_sidebar_width():
    """Minimal CSS for sidebar width only"""
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

# ---------------- FIELD & UI HELPERS  ----------------
# E.g. get_field_label, get_field_options, get_tab_organization, render_field, etc.


# ui.py - Replace the 4 SHAP functions with new implementations

def display_instance_specific_shap(user_inputs, model_name):
    """UPDATED: Use new optimized SHAP system"""
    # Route to new system
    display_optimized_shap_analysis(user_inputs, model_name, get_trained_model)

def display_what_if_shap_analysis(user_inputs, model_name):
    """UPDATED: Enhanced what-if analysis using new system"""
    st.subheader("🔍 What-If SHAP Analysis")
    
    if not user_inputs:
        st.warning("Please make a prediction first to enable what-if analysis.")
        return
    
    # Use new system for better performance
    from shap_analysis import SHAPAnalysisCoordinator
    coordinator = SHAPAnalysisCoordinator()
    
    try:
        # Get numeric parameters for analysis
        numeric_params = {get_field_label(f): f for f in FIELDS if FIELDS[f].get('type') == 'numeric'}
        
        if not numeric_params:
            st.warning("No numeric parameters available for what-if analysis.")
            return
        
        selected_param_label = st.selectbox("Select parameter to analyze:", list(numeric_params.keys()))
        selected_param = numeric_params[selected_param_label]
        
        # Get current value and create range
        current_value = user_inputs.get(selected_param, 100)
        param_config = FIELDS.get(selected_param, {})
        min_val = param_config.get('min', max(1, current_value * 0.1))
        max_val = param_config.get('max', current_value * 3)
        
        num_points = st.slider("Number of analysis points:", 5, 20, 10)
        values = np.linspace(min_val, max_val, num_points)
        
        # Enhanced analysis using new coordinator
        with st.spinner("Running what-if analysis with optimized SHAP..."):
            predictions = []
            progress_bar = st.progress(0)
            
            for i, val in enumerate(values):
                temp_inputs = user_inputs.copy()
                temp_inputs[selected_param] = val
                
                # Use coordinator for consistent analysis
                result = coordinator.run_reduced_instance_analysis(
                    temp_inputs, model_name, get_trained_model, 15, 50
                )
                
                if result.get('success'):
                    # Extract prediction from SHAP analysis or fallback to direct prediction
                    pred = predict_man_hours(temp_inputs, model_name)
                    predictions.append(pred if pred is not None else 0)
                else:
                    predictions.append(0)
                
                progress_bar.progress((i + 1) / len(values))
            
            progress_bar.empty()
        
        # Display results (keep existing visualization code)
        if predictions:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create visualization
                fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Prediction vs {selected_param_label}'])
                fig.add_trace(go.Scatter(x=values, y=predictions, mode='lines+markers', name='Prediction'))
                fig.add_vline(x=current_value, line_dash="dash", line_color="green", annotation_text="Current")
                fig.update_layout(height=400, title_text=f"What-If Analysis: {selected_param_label}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Analysis Summary**")
                current_pred = predict_man_hours(user_inputs, model_name)
                min_pred, max_pred = min(predictions), max(predictions)
                
                st.metric("Current Prediction", f"{current_pred:.0f} hours")
                st.metric("Prediction Range", f"{min_pred:.0f} - {max_pred:.0f} hours")
                st.metric("Max Variation", f"{max_pred - min_pred:.0f} hours")
                
                if max_val > min_val:
                    sensitivity = (max_pred - min_pred) / (max_val - min_val)
                    st.info(f"**Sensitivity:** {sensitivity:.1f} hours per unit change")
        
    except Exception as e:
        st.error(f"Error in what-if analysis: {e}")

def display_scenario_comparison(user_inputs, model_name):
    """UPDATED: Enhanced scenario comparison using new system"""
    st.subheader("📊 Scenario Comparison")
    
    if not user_inputs:
        st.warning("Please make a prediction first to enable scenario comparison.")
        return
    
    # Use new coordinator for consistent analysis
    from shap_analysis import SHAPAnalysisCoordinator
    coordinator = SHAPAnalysisCoordinator()
    
    try:
        # Define scenarios (keep existing scenarios)
        scenarios = {
            "Small Agile Project": {
                "project_prf_functional_size": 65,
                "tech_tf_primary_programming_language": "Python",
                "project_prf_max_team_size": 3,
                "project_prf_relative_size": "XS"
            },
            "Medium Enterprise Project": {
                "project_prf_functional_size": 550,
                "tech_tf_primary_programming_language": "Java", 
                "project_prf_max_team_size": 8,
                "project_prf_relative_size": "M"
            },
            "Large Enterprise Project": {
                "project_prf_functional_size": 2000,
                "tech_tf_primary_programming_language": "C#",
                "project_prf_max_team_size": 15,
                "project_prf_relative_size": "L"
            },
            "Your Current Project": user_inputs.copy()
        }
        
        # Enhanced scenario analysis
        with st.spinner("Analyzing scenarios with optimized SHAP..."):
            scenario_results = {}
            
            for scenario_name, scenario_inputs in scenarios.items():
                if scenario_name != "Your Current Project":
                    full_inputs = user_inputs.copy()
                    full_inputs.update(scenario_inputs)
                else:
                    full_inputs = scenario_inputs
                
                # Use coordinator for comprehensive analysis
                result = coordinator.run_reduced_instance_analysis(
                    full_inputs, model_name, get_trained_model, 15, 50
                )
                
                if result.get('success'):
                    prediction = predict_man_hours(full_inputs, model_name)
                    scenario_results[scenario_name] = {
                        'prediction': prediction,
                        'shap_analysis': result,
                        'inputs': full_inputs
                    }
        
        # Display enhanced results (keep existing visualization but add SHAP insights)
        if scenario_results:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Scenario comparison chart
                scenario_names = list(scenario_results.keys())
                predictions = [scenario_results[name]['prediction'] for name in scenario_names]
                colors = ['red' if name == "Your Current Project" else 'lightblue' for name in scenario_names]
                
                fig = go.Figure(data=[go.Bar(x=scenario_names, y=predictions, marker_color=colors,
                                           text=[f"{p:.0f}h" for p in predictions], textposition='auto')])
                fig.update_layout(title="Effort Predictions by Scenario", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Enhanced Scenario Analysis**")
                for name, results in scenario_results.items():
                    with st.expander(f"📋 {name}"):
                        if results['prediction']:
                            st.metric("Hours", f"{results['prediction']:.0f}")
                            st.metric("Days", f"{results['prediction']/8:.1f}")
                            
                            # Add SHAP insights
                            shap_data = results.get('shap_analysis', {})
                            if shap_data.get('success'):
                                validation = shap_data.get('validation', {})
                                st.info(f"SHAP Analysis: {validation.get('estimated_accuracy', 'Good')}")
                                st.caption(f"Features analyzed: {shap_data.get('feature_count', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error in scenario comparison: {e}")

def display_feature_interactions(user_inputs, model_name):
    """UPDATED: Enhanced feature interactions using new system"""
    st.subheader("🔗 Feature Interactions")
    
    if not user_inputs:
        st.warning("Please make a prediction first to analyze feature interactions.")
        return
    
    # Use new coordinator
    from shap_analysis import SHAPAnalysisCoordinator
    coordinator = SHAPAnalysisCoordinator()
    
    try:
        with st.spinner("Analyzing feature interactions with optimized SHAP..."):
            # Get full SHAP analysis
            result = coordinator.run_reduced_instance_analysis(
                user_inputs, model_name, get_trained_model, 15, 50
            )
            
            if not result.get('success'):
                st.error(f"SHAP analysis failed: {result.get('error')}")
                return
            
            explainer = result.get('explainer')
            if not explainer or not hasattr(explainer, 'shap_interaction_values'):
                st.warning("Feature interaction analysis is only available for tree-based models.")
                return
            
            # Calculate interaction values using the new system
            from shap_analysis.value_calculator import SHAPValueCalculator
            calculator = SHAPValueCalculator()
            
            input_data = coordinator.data_preparer.prepare_input_data(user_inputs)
            if input_data is None:
                st.error("Could not prepare input data for interaction analysis.")
                return
            
            interaction_values = calculator.calculate_interaction_values(explainer, input_data)
            
            if interaction_values is not None:
                # Enhanced visualization using top features from analysis
                top_features = result.get('feature_names', [])[:15]  # Limit to top 15
                
                if len(top_features) >= 2:
                    # Create interaction heatmap
                    n_features = min(15, len(top_features))
                    selected_matrix = interaction_values[:n_features, :n_features]
                    display_names = [f.replace('_', ' ').title()[:15] for f in top_features[:n_features]]
                    
                    fig = px.imshow(selected_matrix, x=display_names, y=display_names,
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
                            if interaction_strength > 0.001:
                                interactions.append({
                                    'Feature 1': display_names[i],
                                    'Feature 2': display_names[j], 
                                    'Interaction Strength': f"{interaction_strength:.4f}",
                                    'Effect': 'Positive' if selected_matrix[i, j] > 0 else 'Negative'
                                })
                    
                    if interactions:
                        interactions.sort(key=lambda x: float(x['Interaction Strength']), reverse=True)
                        interaction_df = pd.DataFrame(interactions[:10])
                        st.dataframe(interaction_df, use_container_width=True)
                        
                        st.info("""
                        **Enhanced Interaction Analysis:**
                        - Analysis uses top 15 most important features for better focus
                        - Positive interactions: Features work together to increase prediction
                        - Negative interactions: Features counteract each other
                        """)
                    else:
                        st.info("No significant feature interactions detected above the threshold.")
                else:
                    st.warning("Need at least 2 features for interaction analysis.")
            else:
                st.warning("Could not calculate interaction values for this model.")
    
    except Exception as e:
        st.error(f"Error in feature interaction analysis: {e}")


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
        label = f"{label} {UIConstants.REQUIRED_FIELD_MARKER}"

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

# ------------------- MAIN SIDEBAR FUNCTION ----------------------

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
                    is_required = config.get("mandatory", False)
                    if field_name == "project_prf_functional_size":
                        rel_code = user_inputs.get("project_prf_relative_size")
                        if rel_code and rel_code in st.session_state.prf_size_code2mid:
                            config["default"] = st.session_state.prf_size_code2mid[rel_code]
                        else:
                            config["default"] = config.get("default", 5)
                    field_value = render_field(field_name, config, is_required)
                    user_inputs[field_name] = field_value

        st.divider()
        st.subheader("🤖 Model Selection")
        selected_model = None
        selected_models = []

        
        try:
            model_status = check_required_models()
            if model_status.get("models_available", False):
                available_models = list_available_models()
                if available_models:
                    model_options = {m['display_name']: m['technical_name'] for m in available_models}
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
        

        # Add this instead temporarily:
        #t.warning("⚠️ Model selection temporarily disabled for testing")
        #selected_model = "test_model"
        #selected_models = ["test_model"]

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

        # --- SAVE & LOAD SECTION ---
        st.divider()
        st.subheader("💾 Download or Load Configurations")
        config_name = st.text_input("Configuration Name", placeholder="e.g., Banking_Project_Template")
        prediction = st.session_state.get('latest_prediction', None)
        history = st.session_state.get('prediction_history', [])
        #st.write("DEBUG: prediction_history count:", len(st.session_state['prediction_history']))
        #st.write(st.session_state['prediction_history'])


        col1, col2 = st.columns(2)
        with col1:
            if config_name.strip():
                if prediction is not None:
                    config_json = make_current_config_json(user_inputs, config_name, selected_model, prediction)
                    st.download_button(
                        label="💾 Export Config",
                        data=config_json,
                        file_name=f"{config_name.strip().replace(' ', '_')}_config.json",
                        mime="application/json",
                        help="Download the current configuration and its prediction",
                        use_container_width=True
                    )
                else:
                    st.warning("Please make a prediction before saving the configuration.")
            else:
                st.button("💾 Export Config", disabled=True, use_container_width=True,
                          help="Enter a configuration name first")

        with col2:
            if history:
                history_json = make_history_json()
                st.download_button(
                    label="📥 Export History",
                    data=history_json,
                    file_name="prediction_history.json",
                    mime="application/json",
                    help="Download all historical predictions",
                    use_container_width=True
                )
            else:
                st.button("📥 Export History", disabled=True, use_container_width=True,
                          help="No prediction history to download yet.")

        st.divider()
        st.markdown("### 📂 Upload Configuration or History")
        col1, col2 = st.columns(2)

        with col1:
            uploaded_config = st.file_uploader(
                "Upload Config",
                type=['json'],
                key="config_upload",
                help="Upload a previously saved single configuration file"
            )
            if uploaded_config is not None:
                config_data = json.load(uploaded_config)
                metadata = config_data.get('_metadata', {})
                config_fields = [k for k in config_data if not k.startswith('_')]
                st.info(
                    f"Config name: {metadata.get('config_name', 'N/A')}, "
                    f"Saved: {metadata.get('saved_date', 'N/A')}, "
                    f"Fields: {len(config_fields)}"
                )
                with st.expander("Preview uploaded config"):
                    st.json({k: config_data[k] for k in config_fields[:5]})

                col_apply, col_cancel = st.columns(2)
                apply_clicked = col_apply.button("Apply Config", key="apply_config")
                cancel_clicked = col_cancel.button("Cancel", key="cancel_config")

                if apply_clicked:
                    # Backup relevant current config fields
                    st.session_state['_backup_config'] = {
                        k: st.session_state.get(k)
                        for k in config_fields if k in st.session_state
                    }
                    # Apply new config values
                    for field_name in config_fields:
                        if field_name in FIELDS:
                            st.session_state[field_name] = config_data[field_name]
                    st.success("Configuration applied! You can restore the previous config below.")
                    #st.rerun()
                elif cancel_clicked:
                    st.info("Config upload cancelled.")

        # Show restore if backup exists
        if st.session_state.get('_backup_config'):
            if st.button("Restore Previous Config", key="restore_backup_config"):
                for k, v in st.session_state['_backup_config'].items():
                    st.session_state[k] = v
                st.success("Previous config restored!")
                del st.session_state['_backup_config']
                #st.rerun()

        with col2:
            uploaded_history = st.file_uploader(
                "Upload History",
                type=['json'],
                key="history_upload",
                help="Upload a previously downloaded prediction history file"
            )
            if uploaded_history is not None:
                history_data = json.load(uploaded_history)
                prediction_history = history_data.get('prediction_history', [])
                st.warning(
                    f"Uploading will REPLACE your current history "
                    f"({len(st.session_state.get('prediction_history', []))} records) "
                    f"with {len(prediction_history)} uploaded records."
                )
                st.write("Preview of uploaded history (first 3 records):")
                st.json(prediction_history[:3])
                if st.button("Replace History"):
                    st.session_state['prediction_history'] = prediction_history
                    st.success("Prediction history replaced!")
                    #st.rerun()

        # Clear results button
        if clear_results:
            st.session_state.prediction_history = []
            st.session_state.comparison_results = []


        user_inputs["selected_model"] = selected_model
        user_inputs["selected_models"] = selected_models
        user_inputs["submit"] = predict_button
        user_inputs["clear_results"] = clear_results
        user_inputs["show_history"] = show_history

        return user_inputs

# --------------------- REMAINDER OF UI AND ANALYSIS FUNCTIONS --------------------
# SHAP, feature, prediction, and visualization functions here.

def load_configuration_from_data(config_data):
    """Load configuration data into session state"""
    try:
        # Extract metadata if present
        metadata = config_data.pop('_metadata', {})
        config_name = metadata.get('config_name', 'Loaded Configuration')
        saved_date = metadata.get('saved_date', 'Unknown')
        
        st.info(f"Loading configuration: {config_name} (saved: {saved_date})")
        
        # Apply configuration to session state
        # This will update the form fields when the page reruns
        for field_name, field_value in config_data.items():
            if field_name in FIELDS:  # Only load valid fields
                st.session_state[field_name] = field_value
        
        return True
        
    except Exception as e:
        st.error(f"Error applying configuration: {e}")
        return False

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

def show_prediction(prediction, model_name, user_inputs=None):
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
        days = prediction / UIConstants.HOURS_PER_DAY
        st.metric("📅 Working Days", f"{days:.1f} days")

    with col3:
        weeks = days / UIConstants.DAYS_PER_WEEK
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
        
        # Get display name for the model
        try:
            model_display_name = get_model_display_name_safe(model_name)
        except Exception:
            model_display_name = model_name

        st.subheader(f"📊 Feature Importance Analysis (Model: {model_display_name})")

        
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
                'Days': f"{entry.get('prediction_hours', 0)/UIConstants.HOURS_PER_DAY:.1f}"
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
            'Days': [p/UIConstants.HOURS_PER_DAY for p in predictions]
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

def show_multiple_predictions(new_predictions):
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
                
                days = prediction / UIConstants.HOURS_PER_DAY
                
                comparison_data.append({
                    'Model': model_display_name,
                    'Hours': f"{prediction:.0f}",
                    'Days': f"{days:.1f}",
                    'Weeks': f"{days/UIConstants.DAYS_PER_WEEK:.1f}"
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

def add_prediction_to_history(user_inputs, model_name, prediction):
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
            #'team_size': team_size,
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
            #team_size = user_inputs.get('project_prf_max_team_size', 5)
            add_prediction_to_history(user_inputs, model, prediction)
            #st.rerun()
            
        except Exception as e:
            st.error(f"Error predicting with {model}: {str(e)}")
            new_predictions[model] = None
    
    return new_predictions

def display_prediction_results(selected_models, new_predictions, user_inputs, comparison_mode=False):
    """Display prediction results based on number of models and mode"""
    
    # Display current results
    if len(selected_models) == 1:
        # Single model - show detailed view
        model = selected_models[0]
        prediction = new_predictions.get(model)
        show_prediction(prediction, model, user_inputs)
    else:
        # Multiple models - show comparison
        show_multiple_predictions(new_predictions)
    
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
            col1, col2= st.columns(2)
            with col1:
                st.metric("Hours", f"{item['prediction_hours']:.0f}")
            with col2:
                st.metric("Days", f"{item['prediction_hours']/UIConstants.HOURS_PER_DAY:.1f}")
    
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

# Use coordinator for complex analysis

def display_advanced_shap_analysis(user_inputs, model_name):
    """New function using coordinator for better analysis"""

    try:
        from shap_analysis import SHAPAnalysisCoordinator
        coordinator = SHAPAnalysisCoordinator()
    
        # Run complete analysis with structured results
        result = coordinator.run_instance_analysis(
            user_inputs, model_name, get_trained_model
        )
        
        if result.get("success"):
            # Extract data from structured result
            shap_values = result.get("shap_values")
            feature_names = result.get("feature_names", [])
            
            # Use existing display logic (simplified)
            st.write("**SHAP Analysis Results**")
            st.write(f"Analysis completed for {len(feature_names)} features")
            
            # You can expand this to show actual SHAP visualization
            if len(shap_values) > 0:
                st.success("SHAP values calculated successfully")
        else:
            st.error(f"SHAP analysis failed: {result.get('error')}")

    except ImportError:
        st.warning("Advanced SHAP analysis not available yet")
        return
    
    


def display_static_shap_analysis():
    """Display static SHAP analysis from file"""
    st.header("📈 Static SHAP Analysis - Model Feature Importance")

    try:
        with open(FileConstants.SHAP_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            shap_report_md = f.read()
        st.markdown(shap_report_md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load static SHAP analysis report: {e}")


def display_visualizations_and_analysis():
    """Display the Visualizations & Analysis tab content - UPDATED for new SHAP system"""
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
    
    # Enhanced info about the new system
    st.info("🚀 **Enhanced SHAP Analysis**: Now using optimized feature analysis for ~4x faster performance with 85-90% accuracy of full analysis.")
    
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
            #st.rerun()

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
                                st.session_state['latest_prediction'] = prediction
                                #team_size = user_inputs.get('project_prf_max_team_size', 5)
                                
                                # Show current prediction
                                show_prediction(prediction, selected_model, user_inputs)
                                
                                # Add to history
                                add_prediction_to_history(user_inputs, selected_model, prediction)
                                #st.rerun()
                                
                            else:
                                # Multi-model workflow
                                new_predictions = run_predictions(user_inputs, selected_models)
                                #team_size = user_inputs.get('project_prf_max_team_size', 5)
                                comparison_mode = user_inputs.get('comparison_mode', False)
                                
                                # Display results
                                display_prediction_results(selected_models, new_predictions, user_inputs, comparison_mode)
                            
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
            #display_visualizations_and_analysis()
            st.info("🔧 SHAP analysis temporarily disabled for performance")

        with main_tabs[2]:  # Model Comparison tab
            display_model_comparison()

        with main_tabs[3]:  # Static SHAP Analysis tab
            display_static_shap_analysis()

        with main_tabs[4]:  # Help tab            
            with st.expander("ℹ️ How to Use This Tool"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with {UIConstants.REQUIRED_FIELD_MARKER} in the "Required Fields" tab
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
                - Ensure all required fields ({UIConstants.REQUIRED_FIELD_MARKER}) are completed
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