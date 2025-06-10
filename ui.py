# ui.py - with Multi-Model Support and Configuration Management
"""
Streamlit UI for ML Project Effort Estimator with Multi-Model Support
This module provides a user interface for estimating project effort using machine learning models.
It includes form inputs, multi-model selection, prediction comparison, and feature importance analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
from datetime import datetime
try:
    from models import (
        predict_man_hours,
        list_available_models,
        check_required_models,
        get_feature_importance,
        get_model_display_name
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

# Load merged configuration - will be called after load_yaml_config is defined
UI_INFO_CONFIG = {}
FIELDS = {}
TAB_ORG = {}
UI_BEHAVIOR = {}
FEATURE_IMPORTANCE_DISPLAY = {}
PREDICTION_THRESHOLDS = {}
DISPLAY_CONFIG = {}

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
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []

# --- Configuration Loading ---
def load_yaml_config(path):
    """Load YAML configuration file with error handling"""
    try:
        with open(path, "r") as f:
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

def get_tab_organization():
    """Get tab organization from configuration"""
    return UI_INFO_CONFIG.get('tab_organization', {
        "Required Fields": [],
        "Optional Fields": []
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
        options = config.get("options", [])
        field_value = st.selectbox(label, options, index=0 if options else None, help=help_text, key=field_name)
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
                    is_required = tab_name == "Required Fields"
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
        required_fields = [f for tab, fields in tab_org.items() if tab == "Required Fields" for f in fields]
        missing_fields = []
        for field in required_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_label(field))

        if missing_fields:
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")

        st.divider()
        predict_button = st.button(
            "🔮 Predict Effort",
            type="primary",
            use_container_width=True,
            disabled=len(missing_fields) > 0 or not selected_models
        )

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

def show_prediction(prediction, team_size, model_name):
    """Show prediction results with team breakdown"""
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total Effort",
            value=f"{prediction:.0f} hours",
            help="Total estimated development effort in hours"
        )
    
    with col2:
        days = prediction / 8
        st.metric(
            label="📅 Working Days", 
            value=f"{days:.1f} days",
            help="Estimated working days (8 hours per day)"
        )
    
    with col3:
        per_person = prediction / team_size if team_size > 0 else prediction
        st.metric(
            label="👤 Per Person",
            value=f"{per_person:.0f} hours",
            help=f"Hours per team member (team size: {team_size})"
        )
    
    # Additional breakdown
    col1, col2 = st.columns(2)
    with col1:
        weeks = days / 5
        st.metric("📆 Working Weeks", f"{weeks:.1f} weeks")
    
    with col2:
        months = weeks / 4.33
        st.metric("🗓️ Months", f"{months:.1f} months")
    
    # Warnings based on thresholds
    low_threshold = PREDICTION_THRESHOLDS.get('low_prediction_warning', 10)
    high_threshold = PREDICTION_THRESHOLDS.get('high_prediction_warning', 5000)
    
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

def add_prediction_to_history(user_inputs, model_name, prediction, team_size):
    """Add prediction to session history"""
    if prediction is None:
        return
    
    try:
        model_display_name = get_model_display_name(model_name)
    except:
        model_display_name = model_name
    
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_display_name,
        'model_technical': model_name,
        'prediction_hours': prediction,
        'team_size': team_size,
        'inputs': user_inputs.copy()
    }
    
    st.session_state.prediction_history.append(history_entry)

def show_prediction_history():
    """Display prediction history"""
    if not st.session_state.prediction_history:
        return
    
    st.subheader("📈 Prediction History")
    
    history_data = []
    for entry in st.session_state.prediction_history:
        history_data.append({
            'Timestamp': entry['timestamp'],
            'Model': get_model_display_name(entry.get('model_technical', entry['model'])),
            'Hours': f"{entry['prediction_hours']:.0f}",
            'Days': f"{entry['prediction_hours']/8:.1f}",
            'Team Size': entry['team_size']
        })
    
    if history_data:
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)

def show_prediction_comparison_table():
    """Show comparison table if multiple predictions exist"""
    if len(st.session_state.prediction_history) <= 1:
        return
    
    st.subheader("🔍 Prediction Comparison")
    
    predictions = [entry['prediction_hours'] for entry in st.session_state.prediction_history]
    models = [get_model_display_name(entry.get('model_technical', entry['model'])) for entry in st.session_state.prediction_history]
    
    comparison_data = {
        'Model': models,
        'Hours': predictions,
        'Days': [p/8 for p in predictions]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{np.mean(predictions):.0f} hours")
    with col2:
        st.metric("Min", f"{np.min(predictions):.0f} hours")
    with col3:
        st.metric("Max", f"{np.max(predictions):.0f} hours")
    with col4:
        if len(predictions) > 1:
            st.metric("Std Dev", f"{np.std(predictions):.0f} hours")

def show_multiple_predictions(new_predictions, team_size):
    """Display results when multiple models are used"""
    if not new_predictions:
        st.warning("No predictions available")
        return
    
    st.subheader("🔍 Multi-Model Prediction Comparison")
    
    # Create comparison table
    comparison_data = []
    predictions_list = []
    
    for model_name, prediction in new_predictions.items():
        if prediction is not None:
            try:
                model_display_name = get_model_display_name(model_name)
            except:
                model_display_name = model_name
            
            days = prediction / 8
            per_person = prediction / team_size if team_size > 0 else prediction
            
            comparison_data.append({
                'Model': model_display_name,
                'Hours': f"{prediction:.0f}",
                'Days': f"{days:.1f}",
                'Per Person': f"{per_person:.0f}",
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

def display_prediction_results(selected_models, new_predictions, team_size, comparison_mode=False):
    """Display prediction results based on number of models and mode"""
    
    # Display current results
    if len(selected_models) == 1:
        # Single model - show detailed view
        model = selected_models[0]
        prediction = new_predictions.get(model)
        show_prediction(prediction, team_size, model)
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
    for i, entry in enumerate(st.session_state.prediction_history):
        history_data.append({
            'Prediction #': i + 1,
            'Model': entry['model'],
            'Hours': entry['prediction_hours'],
            'Timestamp': entry['timestamp']
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
    
    for entry in recent_predictions:
        with st.expander(f"🔮 {entry['model']} - {entry['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hours", f"{entry['prediction_hours']:.0f}")
            with col2:
                st.metric("Days", f"{entry['prediction_hours']/8:.1f}")
            with col3:
                st.metric("Team Size", entry['team_size'])
    
    # Summary statistics if multiple predictions
    if len(st.session_state.prediction_history) > 1:
        all_predictions = [entry['prediction_hours'] for entry in st.session_state.prediction_history]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(all_predictions))
        with col2:
            st.metric("Average Hours", f"{np.mean(all_predictions):.0f}")
        with col3:
            st.metric("Range", f"{np.min(all_predictions):.0f} - {np.max(all_predictions):.0f}")

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
    - **Feature Importance**: Shows which factors most influence the effort estimation
    - **Configuration Management**: Save and load project configurations for reuse
    - **Historical Tracking**: Keep track of previous predictions for comparison
    
    #### How It Works:
    1. **Data Input**: Enter project parameters including team size, technology stack, and project characteristics
    2. **Model Selection**: Choose one or multiple ML models for prediction/comparison
    3. **ML Prediction**: The tool applies trained machine learning models to generate effort estimates
    4. **Results Analysis**: View the predicted effort in hours, days, and per-person breakdowns
    5. **Insights**: Understand which factors most influence your project's effort estimate
    
    #### Multi-Model Features:
    - **Single Model**: Detailed prediction view with full analysis
    - **Multiple Models**: Comparison table with statistics (average, min, max, standard deviation)
    - **Compare Mode**: Keep previous results for historical comparison
    - **Variance Detection**: Warnings when models disagree significantly
    
    #### Best Practices:
    - Provide accurate team size and project complexity information
    - Select multiple models to get a range of estimates and validate consistency
    - Use comparison mode to see how parameter changes affect predictions
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
    """Main application function with full multi-model support"""
    
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
        
        # Main content area
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
                        # Support both single and multi-model workflows
                        if len(selected_models) <= 1:
                            # Single model workflow
                            prediction = predict_man_hours(user_inputs, selected_model)
                            team_size = user_inputs.get('project_prf_max_team_size', 5)
                            
                            # Show current prediction
                            show_prediction(prediction, team_size, selected_model)
                            
                            # Add to history
                            add_prediction_to_history(user_inputs, selected_model, prediction, team_size)
                            
                        else:
                            # Multi-model workflow
                            new_predictions = run_predictions(user_inputs, selected_models)
                            team_size = user_inputs.get('project_prf_max_team_size', 5)
                            comparison_mode = user_inputs.get('comparison_mode', False)
                            
                            # Display results
                            display_prediction_results(selected_models, new_predictions, team_size, comparison_mode)
                        
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
            
            # Help section
            with st.expander("ℹ️ How to Use This Tool"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with ⭐ in the "Required Fields" tab
                2. **Optional Parameters** - Add more details in the "Optional Fields" tab for better accuracy  
                3. **Select Model** - Choose a model for prediction
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                5. **Save Configuration** - Save your parameter settings for future use
                
                ### Multi-Model Features
                - **Single Model**: Detailed prediction view with full analysis
                - **Multiple Models**: Comparison table with statistics
                - **Compare Mode**: Keep previous results for historical comparison
                - **Clear Results**: Remove all stored predictions
                
                ### Features
                - **Detailed Predictions**: Hours, days, and per-person breakdowns
                - **Prediction History**: Track and compare multiple predictions
                - **Feature Importance**: See which factors matter most
                - **What-If Analysis**: Understand parameter sensitivity
                - **Configuration Save/Load**: Reuse settings for similar projects
                
                ### Tips for Better Estimates
                - Fill in as many relevant fields as possible
                - Use realistic team sizes and project characteristics
                - Compare multiple predictions to understand variability
                - Save configurations for similar future projects
                - Review variance between models - high variance may indicate parameter issues
                
                ### Troubleshooting
                - Ensure all required fields (⭐) are completed
                - Check that models are available in the dropdown
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