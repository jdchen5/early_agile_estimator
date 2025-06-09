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


# --- Configuration Loading ---
def load_yaml_config(path):
    """Load YAML configuration file with error handling"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading YAML file {path}: {e}")
        return {}


# --- New field-centric helpers ---
def get_field_label(field_name, FIELDS):
    return FIELDS[field_name].get("label", field_name.replace("_", " ").title())

def get_field_title(field_name, FIELDS):
    return FIELDS[field_name].get("title", get_field_label(field_name, FIELDS))

def get_field_help(field_name, FIELDS):
    return FIELDS[field_name].get("help", "")

def get_what_if_parameters(FIELDS):
    return {get_field_label(f, FIELDS): f for f in FIELDS if FIELDS[f].get("type") == "numeric"}


def render_field(field_name, config, is_required=False):
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

def sidebar_inputs(FIELDS, TAB_ORG):
    with st.sidebar:
        st.title("🔮 Project Parameters")
        st.info("Required fields (marked with ⭐)")
        user_inputs = {}

        tabs = st.tabs(list(TAB_ORG.keys()))
        for idx, (tab_name, field_list) in enumerate(TAB_ORG.items()):
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
        try:
            model_status = check_required_models()
            if model_status.get("models_available", False):
                available_models = list_available_models()
                if available_models:
                    model_options = {m['display_name']: m['technical_name'] for m in available_models}
                    selected_display_name = st.selectbox(
                        "Choose ML Model",
                        list(model_options.keys()),
                        help="Select one model for prediction. You can run multiple predictions sequentially."
                    )
                    selected_model = model_options[selected_display_name]
                    if st.session_state.prediction_history:
                        st.info(f"📊 {len(st.session_state.prediction_history)} predictions made so far")
                else:
                    st.warning("⚠️ No trained models found")
            else:
                st.warning("⚠️ Models not available")
        except Exception as e:
            st.error(f"Model loading error: {e}")
            selected_model = None

        # Required field check
        required_fields = [f for tab, fields in TAB_ORG.items() if tab == "Required Fields" for f in fields]
        missing_fields = []
        for field in required_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_label(field, FIELDS))

        if missing_fields:
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")

        st.divider()
        predict_button = st.button(
            "🔮 Predict Effort",
            type="primary",
            use_container_width=True,
            disabled=len(missing_fields) > 0 or not selected_model
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

        user_inputs["selected_model"] = selected_model
        user_inputs["selected_models"] = [selected_model] if selected_model else []
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
        items = [(get_field_label(k, FIELDS), v) for k, v in user_inputs.items() if k not in exclude_keys and v is not None and v != ""]
        if items:
            col1, col2 = st.columns(2)
            mid = len(items) // 2
            with col1:
                for param, value in items[:mid]:
                    st.text(f"**{param}:** {value}")
            with col2:
                for param, value in items[mid:]:
                    st.text(f"**{param}:** {value}")
            if selected_models:
                model_names = []
                for model in selected_models:
                    try:
                        model_display_name = get_model_display_name(model)
                        model_names.append(model_display_name)
                    except:
                        model_names.append(model)
                st.info(f"🤖 **Model(s):** {', '.join(model_names)}")
        else:
            st.warning("No parameters to display")

def show_feature_importance(selected_models, features_dict):
    """Display feature importance analysis for multiple models"""
    if not selected_models:
        return
    
    try:
        if len(selected_models) > 1:
            st.subheader("📊 Feature Importance Comparison")
            
            importance_data = {}
            exclude_keys = {'selected_models', 'submit', 'clear_results', 'comparison_mode'}
            feature_names = [k for k in features_dict.keys() if k not in exclude_keys]

            if isinstance(selected_models, str):
                selected_models = [selected_models]
            assert isinstance(selected_models, list), "selected_models must be a list"

            for model in selected_models:
                try:
                    feature_importance = get_feature_importance(model)
                    if feature_importance is not None:
                        display_name = get_model_display_name(model)
                        importance_data[display_name] = feature_importance[:len(feature_names)]
                except:
                    continue
            
            if importance_data:
                comparison_df = pd.DataFrame(importance_data, 
                                           index=[get_field_title(name) for name in feature_names[:len(list(importance_data.values())[0])]])
                
                st.bar_chart(comparison_df)
                
                with st.expander("📋 View Detailed Comparison Data"):
                    st.dataframe(comparison_df.round(3), use_container_width=True)
        
        else:
            model = selected_models[0]
            feature_importance = get_feature_importance(model)
            if feature_importance is not None:
                st.subheader("📊 Feature Importance Analysis")
                
                exclude_keys = {'selected_models', 'submit', 'clear_results', 'comparison_mode'}
                feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
                
                importance_data = []
                for i, name in enumerate(feature_names[:10]):
                    if i < len(feature_importance):
                        friendly_name = get_field_title(name)
                        importance_data.append({
                            'Feature': friendly_name,
                            'Importance': abs(feature_importance[i])
                        })
                
                if importance_data:
                    importance_df = pd.DataFrame(importance_data)
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
                    
                    with st.expander("📋 View Detailed Importance Data"):
                        st.dataframe(importance_df.round(3), use_container_width=True)
    
    except Exception as e:
        st.info(f"Feature importance analysis not available: {e}")

def display_prediction_results(selected_models, new_predictions, team_size, comparison_mode):
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
    if comparison_mode and len(st.session_state.prediction_results) > len(selected_models):
        display_historical_comparison()

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
    """Main application function with multi-model support"""
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.title("🔮 ML Project Effort Estimator")
    st.markdown("Get accurate effort estimates using machine learning models trained on historical project data.")
    
    try:
        # Get user inputs from sidebar
        user_inputs = sidebar_inputs()
        
        # Handle clear results
        if user_inputs.get('clear_results', False):
            clear_prediction_results()
            st.rerun()
        
        # Main content area
        if user_inputs.get('submit', False):
            selected_models = user_inputs.get('selected_models', [])
            
            if selected_models:
                # Display input summary
                display_inputs(user_inputs, selected_models)
                st.divider()
                
                # Run predictions
                new_predictions = run_predictions(user_inputs, selected_models)
                team_size = user_inputs.get('project_prf_max_team_size', 5)
                comparison_mode = user_inputs.get('comparison_mode', False)
                
                # Display results
                display_prediction_results(selected_models, new_predictions, team_size, comparison_mode)
                
                # Show feature analysis if predictions successful
                successful_predictions = {k: v for k, v in new_predictions.items() if v is not None}
                if successful_predictions:
                    st.divider()
                    show_feature_importance(list(successful_predictions.keys()), user_inputs)
            
            else:
                st.warning("⚠️ Please select at least one model to make predictions")
        
        else:
            # Welcome screen
            st.info("👈 **Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate.")
            
            # Show previous results summary
            display_previous_results_summary()
            
            # Help section
            with st.expander("ℹ️ How to Use This Tool"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with ⭐ in the "Required" tab
                2. **Optional Parameters** - Add more details in the "Optional" tab for better accuracy  
                3. **Select Models** - Choose one or more models for prediction/comparison
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                5. **Save Configuration** - Save your parameter settings for future use
                
                ### Multi-Model Features
                - **Single Model**: Detailed prediction view with full analysis
                - **Multiple Models**: Comparison table with statistics
                - **Compare Mode**: Keep previous results for historical comparison
                - **Clear Results**: Remove all stored predictions
                
                ### Tips for Better Estimates
                - Select multiple models to get a range of estimates
                - Use comparison mode to see how different parameters affect predictions
                - Review variance between models - high variance may indicate parameter issues
                - Save configurations for similar future projects
                
                ### Troubleshooting
                - Use "Check Models" if no models appear in the dropdown
                - Check the Debug Info section for technical details
                - Ensure all required fields are filled before predicting
                """)
            
            # About section
            with st.expander("📖 About This Tool"):
                about_section()
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration files and model setup.")

if __name__ == "__main__":

    main()