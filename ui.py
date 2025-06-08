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
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 350px !important;  /* Change this value */
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
    except FileNotFoundError:
        return create_default_config(path)
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file {path}: {e}")
        return {}

# --- Default Configuration Creation TOBEREMOVED---
def create_default_config(path):
    """Create default configuration if file doesn't exist"""
    if "feature_mapping" in path:
        default_config = {
            "categorical_features": {
                "external_eef_industry_sector": {"options": ["Banking", "Healthcare", "Technology", "Retail", "Manufacturing"]},
                "external_eef_organisation_type": {"options": ["SME", "Large Enterprise", "Startup", "Government"]},
                "project_prf_application_type": {"options": ["Web Application", "Mobile App", "Desktop Software", "API/Service"]},
                "project_prf_development_type": {"options": ["New Development", "Enhancement", "Maintenance", "Migration"]},
                "tech_tf_architecture": {"options": ["Monolithic", "Microservices", "SOA", "Serverless"]},
                "tech_tf_development_platform": {"options": ["Cloud", "On-Premise", "Hybrid"]},
                "tech_tf_language_type": {"options": ["Object-Oriented", "Functional", "Procedural", "Scripting"]},
                "tech_tf_primary_programming_language": {"options": ["Java", "Python", "JavaScript", "C#", "PHP", "Ruby"]},
                "project_prf_relative_size": {"options": ["XS (Extra Small)", "S (Small)", "M (Medium)", "L (Large)", "XL (Extra Large)"]}
            },
            "numeric_features": ["project_prf_year_of_project", "project_prf_max_team_size"],
            "special_cases": {
                "team_size_group": {
                    "input_key": "project_prf_team_size_group",
                    "options": ["1", "2-3", "4-5", "6-10", "11+"]
                }
            }
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(default_config, f)
        return default_config
    return {}

def load_configs():
    """Load all configuration files"""
    ui_config_path = "config/ui_config.yaml" # UI fields mapping between display and feature names and other UI settings
    feature_config_path = "config/feature_mapping.yaml"  # Feature mapping for categorical and numeric fields
    help_config_path = "config/field_help.yaml"  # Help text and titles for fields
    
    feature_config = load_yaml_config(feature_config_path)
    ui_config = load_yaml_config(ui_config_path) or {}
    help_config = load_yaml_config(help_config_path) or {}
    
    return feature_config, ui_config, help_config

FEATURE_CONFIG, UI_CONFIG, HELP_CONFIG = load_configs()

# --- Helper Functions ---
def get_field_help(field_name):
    """Get help text for a field"""
    field_help = HELP_CONFIG.get("field_help", {})
    if field_name in field_help:
        return field_help[field_name].get("help", "")
    return None

def get_field_title(field_name):
    """Get display title for a field"""
    field_help = HELP_CONFIG.get("field_help", {})
    if field_name in field_help:
        return field_help[field_name].get("title", get_field_label(field_name))
    return get_field_label(field_name)

def get_field_label(field_name):
    """Convert field name to readable label"""
    field_labels = UI_CONFIG.get("field_labels", {})
    if field_name in field_labels:
        return field_labels[field_name]
    
    # Default conversion
    return field_name.replace('_', ' ').replace('prf', '').replace('eef', '').replace('tf', '').title().strip()

def is_mandatory_field(field_name):
    """Check if field is mandatory based on tab configuration"""
    # Try to get from tab configuration first
    tab_config = UI_CONFIG.get("tab_configuration") or HELP_CONFIG.get("tab_configuration")
    
    if tab_config:
        # Check if field is in "Required Fields" tab
        required_tab = tab_config.get("Required Fields", {})
        required_fields = required_tab.get("fields", [])
        return field_name in required_fields
    else:
        # Fallback to hardcoded list
        mandatory_fields = [
            "project_prf_year_of_project",
            "external_eef_industry_sector", 
            "external_eef_organisation_type",
            "project_prf_team_size_group",
            "project_prf_max_team_size",
            "project_prf_relative_size",
            "project_prf_application_type",
            "project_prf_development_type",
            "tech_tf_architecture",
            "tech_tf_development_platform",
            "tech_tf_language_type",
            "tech_tf_primary_programming_language"
        ]
        return field_name in mandatory_fields

def get_team_size_group_from_max(max_team_size):
    """Calculate team size group from max team size"""
    if max_team_size <= 1:
        return "1"
    elif max_team_size <= 3:
        return "2-3"
    elif max_team_size <= 5:
        return "4-5"
    elif max_team_size <= 10:
        return "6-10"
    else:
        return "11+"

def get_numeric_field_config(field_name):
    """Get configuration for numeric fields"""
    if "year" in field_name.lower():
        return {"min": 2015, "max": 2030, "default": 2024}
    elif "team" in field_name.lower() and "size" in field_name.lower():
        return {"min": 1, "max": 100, "default": 5}
    return {"min": 1, "max": 1000, "default": 10}

# --- Field Rendering Functions ---
def render_field(field_name, is_required=False):
    """Render a single form field """
    label = get_field_title(field_name)
    help_text = get_field_help(field_name)
    
    # Add required indicator to label
    if is_required:
        label = f"{label} ⭐"
    
    field_value = None
    
    # Handle different field types
    if field_name in FEATURE_CONFIG.get("numeric_features", []):
        config = get_numeric_field_config(field_name)
        field_value = st.number_input(
            label,
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"],
            help=help_text,
            key=field_name
        )
    
    elif field_name in FEATURE_CONFIG.get("categorical_features", {}):
        options = FEATURE_CONFIG["categorical_features"][field_name].get("options", [])
        if options:
            field_value = st.selectbox(
                label,
                options,
                index=0,
                help=help_text,
                key=field_name,
                placeholder="Choose an option"
            )
        else:
            field_value = st.text_input(
                label, 
                help=help_text, 
                key=field_name
            )
    
    elif field_name == "project_prf_team_size_group":
        # Auto-calculated field - display only
        max_team_size = field_name
        team_size_group = get_team_size_group_from_max(max_team_size)
        field_value = st.selectbox(
            label,
            ["1", "2-3", "4-5", "6-10", "11+"],
            index=4,  # Default to "6-10"
            help="Team size group (will be auto-calculated in final version)"
        )
    
    else:
        # Default field handling
        field_value = st.text_input(
            label, 
            help=help_text
        )
    
    return field_value

def get_tab_organization():
    """Organize fields into tabs using tab_configuration from YAML files"""
    
    # Try to get tab configuration from UI_CONFIG first, then HELP_CONFIG
    tab_config = UI_CONFIG.get("tab_configuration") or HELP_CONFIG.get("tab_configuration")
    
    if tab_config:
        # Use the tab configuration from YAML file
        return tab_config
    else:
        # Fallback to default configuration if not found in YAML
        st.warning("⚠️ Tab configuration not found in YAML files, using default")
        
        # Get all available fields from configuration
        all_available_fields = set()
        all_available_fields.update(FEATURE_CONFIG.get("numeric_features", []))
        all_available_fields.update(FEATURE_CONFIG.get("categorical_features", {}).keys())
        
        # Add special case fields
        for group_config in FEATURE_CONFIG.get("special_cases", {}).values():
            if "input_key" in group_config:
                all_available_fields.add(group_config["input_key"])
        
        # Default mandatory fields
        mandatory_field_names = [
            "project_prf_year_of_project",
            "external_eef_industry_sector", 
            "external_eef_organisation_type",
            "project_prf_team_size_group",
            "project_prf_max_team_size",
            "project_prf_relative_size",
            "project_prf_application_type",
            "project_prf_development_type",
            "tech_tf_architecture",
            "tech_tf_development_platform",
            "tech_tf_language_type",
            "tech_tf_primary_programming_language"
        ]
        
        required_fields = [f for f in mandatory_field_names if f in all_available_fields]
        optional_fields = [f for f in all_available_fields if f not in mandatory_field_names]
        
        return {
            "Required Fields": {
                "fields": required_fields,
                "description": "Essential parameters for accurate estimation"
            },
            "Optional Fields": {
                "fields": optional_fields,
                "description": "Additional parameters for improved accuracy"
            }
        }

# --- Display Functions ---
def display_inputs(user_inputs, selected_model):
    """Display input parameters summary in a collapsible expander"""
    with st.expander("📋 Input Parameters Summary", expanded=False):
        display_params = {}
        exclude_keys = {'selected_model', 'submit', 'clear_results', 'show_history'}
        
        for key, value in user_inputs.items():
            if key not in exclude_keys and value is not None and value != "":
                label = get_field_title(key)
                display_params[label] = str(value)
        
        if display_params:
            col1, col2 = st.columns(2)
            items = list(display_params.items())
            mid_point = len(items) // 2
            
            with col1:
                for param, value in items[:mid_point]:
                    st.text(f"**{param}:** {value}")
            
            with col2:
                for param, value in items[mid_point:]:
                    st.text(f"**{param}:** {value}")
            
            if selected_model:
                try:
                    model_display_name = get_model_display_name(selected_model)
                    st.info(f"🤖 **Selected Model:** {model_display_name}")
                except:
                    st.info(f"🤖 **Selected Model:** {selected_model}")
        else:
            st.warning("No parameters to display")


# --- Session State Management ---
def initialize_session_state():
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'prediction_counter' not in st.session_state:
        st.session_state.prediction_counter = 0

def clear_prediction_results():
    """Clear all stored prediction results"""
    st.session_state.prediction_history = []
    st.session_state.prediction_counter = 0

def add_prediction_to_history(user_inputs, model_name, prediction, team_size):
    """Add a single prediction to the history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.prediction_counter += 1
    
    prediction_entry = {
        'id': st.session_state.prediction_counter,
        'model': model_name,
        'prediction': prediction,
        'timestamp': timestamp,
        'team_size': team_size,
        'inputs': user_inputs.copy()  # Store inputs for comparison
    }
    
    st.session_state.prediction_history.append(prediction_entry)
    return prediction_entry

# --- Prediction Management Functions ---
def run_predictions(user_inputs, selected_models):
    """Run predictions for multiple models and return results dictionary"""
    if isinstance(selected_models, str):
        selected_models = [selected_models]  # Convert single model to list
    
    predictions = {}
    team_size = user_inputs.get('project_prf_max_team_size', 5)
    
    for model in selected_models:
        try:
            model_display_name = get_model_display_name(model)
        except:
            model_display_name = model
        
        with st.spinner(f"🤔 Analyzing with {model_display_name}..."):
            try:
                prediction = predict_man_hours(user_inputs, model, use_preprocessing_pipeline=True)
                
                if prediction is not None:
                    # Add to history
                    add_prediction_to_history(user_inputs, model, prediction, team_size)
                    predictions[model] = prediction
                    st.success(f"✅ {model_display_name} completed")
                else:
                    st.error(f"❌ {model_display_name} failed")
                    predictions[model] = None
                    
            except Exception as e:
                st.error(f"Error with {model_display_name}: {e}")
                predictions[model] = None
    
    return predictions

def run_single_prediction(user_inputs, selected_model):
    """Run prediction for a single model and add to history"""
    team_size = user_inputs.get('project_prf_max_team_size', 5)
    
    with st.spinner(f"🤔 Analyzing project parameters with {get_model_display_name(selected_model)}..."):
        try:
            prediction = predict_man_hours(user_inputs, selected_model, use_preprocessing_pipeline=True)
            
            if prediction is not None:
                # Add to history
                prediction_entry = add_prediction_to_history(user_inputs, selected_model, prediction, team_size)
                st.success(f"✅ Prediction #{prediction_entry['id']} completed successfully!")
                return prediction
            else:
                st.error(f"❌ Prediction failed for {get_model_display_name(selected_model)}")
                return None
                
        except Exception as e:
            st.error(f"Prediction error for {get_model_display_name(selected_model)}: {e}")
            return None

def show_prediction(prediction, team_size, model_name=None):
    """Display prediction results for a single model"""
    
    if prediction is None:
        st.error(f"❌ Prediction Failed {f'({model_name})' if model_name else ''}")
        return
    
    try:
        if model_name:
            try:
                display_name = get_model_display_name(model_name)
                st.subheader(f"🎯 {display_name} Results")
            except:
                st.subheader(f"🎯 {model_name} Results")
        else:
            st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Effort", 
                value=f"{prediction:.1f} hours",
                help="Estimated total project effort"
            )
        
        with col2:
            if team_size and team_size > 0:
                per_person = prediction / team_size
                st.metric(
                    label="Per Person", 
                    value=f"{per_person:.1f} hours",
                    help="Effort per team member"
                )
            else:
                st.metric(label="Per Person", value="N/A")
        
        with col3:
            days = prediction / 8
            st.metric(
                label="Work Days", 
                value=f"{days:.1f} days",
                help="Estimated work days (8 hours/day)"
            )
        
        if prediction < 1:
            st.warning("⚠️ Very low estimate - please verify inputs")
        elif prediction > 10000:
            st.warning("⚠️ Very high estimate - please verify inputs")
        else:
            st.success("✅ Prediction completed successfully")
    
    except Exception as e:
        st.error(f"Display error: {e}")

def show_multiple_predictions(predictions_dict, team_size):
    """Display multiple model predictions in comparison format"""
    st.subheader("🔄 Model Comparison Results")
    
    valid_predictions = {k: v for k, v in predictions_dict.items() if v is not None}
    
    if not valid_predictions:
        st.error("❌ All predictions failed")
        return
    
    # Statistics
    prediction_values = list(valid_predictions.values())
    
    if len(prediction_values) > 1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{np.mean(prediction_values):.1f} hrs")
        with col2:
            st.metric("Min", f"{np.min(prediction_values):.1f} hrs")
        with col3:
            st.metric("Max", f"{np.max(prediction_values):.1f} hrs")
        with col4:
            std_dev = np.std(prediction_values)
            st.metric("Std Dev", f"{std_dev:.1f} hrs")
        
        if std_dev > np.mean(prediction_values) * 0.3:
            st.warning("⚠️ High variance between models - consider reviewing parameters")
    
    # Detailed results table
    results_data = []
    for model, prediction in valid_predictions.items():
        try:
            model_display_name = get_model_display_name(model)
        except:
            model_display_name = model
        
        per_person = prediction / team_size if team_size > 0 else 0
        days = prediction / 8
        
        results_data.append({
            'Model': model_display_name,
            'Total Hours': f"{prediction:.1f}",
            'Per Person': f"{per_person:.1f}" if per_person > 0 else "N/A",
            'Work Days': f"{days:.1f}",
            'Status': "✅ Success"
        })
    
    # Add failed predictions
    failed_predictions = {k: v for k, v in predictions_dict.items() if v is None}
    for model, _ in failed_predictions.items():
        try:
            model_display_name = get_model_display_name(model)
        except:
            model_display_name = model
        
        results_data.append({
            'Model': model_display_name,
            'Total Hours': "N/A",
            'Per Person': "N/A",
            'Work Days': "N/A",
            'Status': "❌ Failed"
        })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)


def show_prediction_history():
    """Display all predictions in chronological order"""
    if not st.session_state.prediction_history:
        return
    
    st.subheader("📈 Prediction History")
    st.write(f"Total predictions made: {len(st.session_state.prediction_history)}")
    
    # Show summary statistics if multiple predictions
    if len(st.session_state.prediction_history) > 1:
        predictions = [entry['prediction'] for entry in st.session_state.prediction_history if entry['prediction']]
        if predictions:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{np.mean(predictions):.1f} hrs")
            with col2:
                st.metric("Min", f"{np.min(predictions):.1f} hrs")
            with col3:
                st.metric("Max", f"{np.max(predictions):.1f} hrs")
            with col4:
                std_dev = np.std(predictions)
                st.metric("Std Dev", f"{std_dev:.1f} hrs")
            
            if std_dev > np.mean(predictions) * 0.3:
                st.warning("⚠️ High variance between predictions - different models or parameters may be affecting results")
    
    # Display each prediction in chronological order
    for i, entry in enumerate(reversed(st.session_state.prediction_history)):  # Most recent first
        prediction_number = len(st.session_state.prediction_history) - i
        
        try:
            model_display_name = get_model_display_name(entry['model'])
        except:
            model_display_name = entry['model']
        
        # Create an expandable section for each prediction
        with st.expander(f"🎯 Prediction #{entry['id']} - {model_display_name} at {entry['timestamp']}", expanded=(i == 0)):
            
            if entry['prediction']:
                # Show prediction metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Effort", 
                        value=f"{entry['prediction']:.1f} hours"
                    )
                
                with col2:
                    if entry['team_size'] and entry['team_size'] > 0:
                        per_person = entry['prediction'] / entry['team_size']
                        st.metric(
                            label="Per Person", 
                            value=f"{per_person:.1f} hours"
                        )
                    else:
                        st.metric(label="Per Person", value="N/A")
                
                with col3:
                    days = entry['prediction'] / 8
                    st.metric(
                        label="Work Days", 
                        value=f"{days:.1f} days"
                    )
                
                # Show key parameters used for this prediction
                if st.checkbox(f"Show parameters for prediction #{entry['id']}", key=f"params_{entry['id']}"):
                    key_params = {
                        'Team Size': entry['inputs'].get('project_prf_max_team_size', 'N/A'),
                        'Industry': entry['inputs'].get('external_eef_industry_sector', 'N/A'),
                        'Project Size': entry['inputs'].get('project_prf_relative_size', 'N/A'),
                        'Architecture': entry['inputs'].get('tech_tf_architecture', 'N/A'),
                        'Language': entry['inputs'].get('tech_tf_primary_programming_language', 'N/A')
                    }
                    
                    st.write("**Key Parameters:**")
                    for param, value in key_params.items():
                        st.write(f"- {param}: {value}")
            else:
                st.error("❌ This prediction failed")

def show_prediction_comparison_table():
    """Show a comparison table of all predictions"""
    if len(st.session_state.prediction_history) < 2:
        return
    
    st.subheader("📊 Prediction Comparison Table")
    
    comparison_data = []
    for entry in st.session_state.prediction_history:
        if entry['prediction']:
            try:
                model_display_name = get_model_display_name(entry['model'])
            except:
                model_display_name = entry['model']
            
            per_person = entry['prediction'] / entry['team_size'] if entry['team_size'] > 0 else 0
            days = entry['prediction'] / 8
            
            comparison_data.append({
                'Prediction #': entry['id'],
                'Time': entry['timestamp'],
                'Model': model_display_name,
                'Total Hours': f"{entry['prediction']:.1f}",
                'Per Person': f"{per_person:.1f}" if per_person > 0 else "N/A",
                'Work Days': f"{days:.1f}",
                'Team Size': entry['team_size']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Option to download as CSV
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison as CSV",
            data=csv,
            file_name=f"prediction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_feature_importance(selected_model, features_dict):
    """Display feature importance analysis for the selected model"""
    if not selected_model:
        return
    
    try:
        feature_importance = get_feature_importance(selected_model)
        if feature_importance is not None:
            st.subheader("📊 Feature Importance Analysis")
            
            exclude_keys = {'selected_model', 'submit', 'clear_results', 'show_history'}
            feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
            
            importance_data = []
            for i, name in enumerate(feature_names[:10]):  # Top 10
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
            else:
                st.info("No feature importance data available to display")
        else:
            st.info("Feature importance not available for this model")
    
    except Exception as e:
        st.info(f"Feature importance analysis not available: {e}")
        
def display_historical_comparison():
    """Display historical comparison of all predictions with trend analysis"""
    if len(st.session_state.prediction_history) < 2:
        return
    
    st.subheader("📈 Historical Prediction Trends")
    
    # Prepare data for trend analysis
    valid_predictions = [entry for entry in st.session_state.prediction_history if entry['prediction']]
    
    if len(valid_predictions) < 2:
        st.info("Need at least 2 successful predictions for trend analysis")
        return
    
    # Create trend dataframe
    trend_data = []
    for entry in valid_predictions:
        try:
            model_display_name = get_model_display_name(entry['model'])
        except:
            model_display_name = entry['model']
        
        trend_data.append({
            'Prediction_ID': entry['id'],
            'Timestamp': entry['timestamp'], 
            'Model': model_display_name,
            'Hours': entry['prediction'],
            'Team_Size': entry['team_size'],
            'Hours_Per_Person': entry['prediction'] / entry['team_size'] if entry['team_size'] > 0 else 0
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Display trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Total Hours Trend**")
        chart_data = trend_df.set_index('Prediction_ID')['Hours']
        st.line_chart(chart_data)
    
    with col2:
        st.write("**Hours Per Person Trend**")
        chart_data = trend_df.set_index('Prediction_ID')['Hours_Per_Person']
        st.line_chart(chart_data)
    
    # Model performance comparison
    if len(trend_df['Model'].unique()) > 1:
        st.write("**Model Performance Comparison**")
        model_stats = trend_df.groupby('Model')['Hours'].agg(['mean', 'std', 'count']).round(2)
        model_stats.columns = ['Average Hours', 'Standard Deviation', 'Prediction Count']
        st.dataframe(model_stats)
        
        # Show model consistency warning
        high_variance_models = model_stats[model_stats['Standard Deviation'] > model_stats['Average Hours'] * 0.3]
        if not high_variance_models.empty:
            st.warning(f"⚠️ High variance detected in models: {', '.join(high_variance_models.index.tolist())}")
    
    # Recent vs older predictions comparison
    if len(valid_predictions) >= 4:
        recent_count = len(valid_predictions) // 2
        recent_predictions = valid_predictions[-recent_count:]
        older_predictions = valid_predictions[:-recent_count]
        
        recent_avg = np.mean([p['prediction'] for p in recent_predictions])
        older_avg = np.mean([p['prediction'] for p in older_predictions])
        
        st.write("**Recent vs Historical Comparison**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recent Average", f"{recent_avg:.1f} hrs", 
                     delta=f"{recent_avg - older_avg:.1f} hrs")
        with col2:
            st.metric("Historical Average", f"{older_avg:.1f} hrs")
        with col3:
            trend_direction = "📈" if recent_avg > older_avg else "📉" if recent_avg < older_avg else "➡️"
            percentage_change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
            st.metric("Trend", f"{trend_direction} {percentage_change:.1f}%")
    
    # Parameter impact analysis
    st.write("**Parameter Impact Analysis**")
    
    # Group by key parameters and show averages
    if len(valid_predictions) >= 3:
        param_analysis = {}
        
        # Analyze team size impact
        team_size_groups = {}
        for entry in valid_predictions:
            team_size = entry['team_size']
            size_group = get_team_size_group_from_max(team_size)
            if size_group not in team_size_groups:
                team_size_groups[size_group] = []
            team_size_groups[size_group].append(entry['prediction'])
        
        if len(team_size_groups) > 1:
            st.write("*Team Size Impact:*")
            for size_group, predictions in team_size_groups.items():
                avg_hours = np.mean(predictions)
                st.write(f"- {size_group} people: {avg_hours:.1f} hours average ({len(predictions)} predictions)")
        
        # Analyze industry impact if available
        industry_groups = {}
        for entry in valid_predictions:
            industry = entry['inputs'].get('external_eef_industry_sector', 'Unknown')
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(entry['prediction'])
        
        if len(industry_groups) > 1:
            st.write("*Industry Sector Impact:*")
            for industry, predictions in industry_groups.items():
                if len(predictions) >= 2:  # Only show if we have multiple data points
                    avg_hours = np.mean(predictions)
                    st.write(f"- {industry}: {avg_hours:.1f} hours average ({len(predictions)} predictions)")
    
    # Detailed breakdown table
    with st.expander("📋 Detailed Historical Data"):
        st.dataframe(trend_df, use_container_width=True)
        
        # Option to download historical data
        csv = trend_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Historical Data as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
def display_previous_results_summary():
    """Display summary of previous results on welcome screen"""
    if st.session_state.prediction_history:
        st.subheader("📋 Previous Predictions")
        
        recent_predictions = st.session_state.prediction_history[-3:]  # Show last 3
        for entry in reversed(recent_predictions):
            try:
                model_display_name = get_model_display_name(entry['model'])
            except:
                model_display_name = entry['model']
            
            if entry['prediction']:
                st.info(f"🎯 #{entry['id']}: {model_display_name} → {entry['prediction']:.1f} hours at {entry['timestamp']}")
            else:
                st.error(f"❌ #{entry['id']}: {model_display_name} → Failed at {entry['timestamp']}")
        
        if len(st.session_state.prediction_history) > 3:
            st.write(f"... and {len(st.session_state.prediction_history) - 3} more predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Predictions"):
                clear_prediction_results()
                st.rerun()
        with col2:
            if st.button("📊 View All Predictions"):
                st.session_state.show_all_predictions = True
                st.rerun()

# --- Main Sidebar Function ---
def sidebar_inputs():
    """Create ultra-compact sidebar with form inputs and SINGLE model selection"""
    with st.sidebar:
        st.title("🔮 Project Parameters")
        
        st.info("Required fields (marked with ⭐)")
        
        user_inputs = {}
        tab_organization = get_tab_organization()
        
        tab_names = list(tab_organization.keys())
        tabs = st.tabs(tab_names)
        
        for tab_idx, (tab_name, tab_config) in enumerate(tab_organization.items()):
            with tabs[tab_idx]:
                description = tab_config.get("description", "")
                if description:
                    st.caption(description)
                
                fields = tab_config.get("fields", [])
                for field_name in fields:
                    if field_name in FEATURE_CONFIG.get("categorical_features", {}) or \
                       field_name in FEATURE_CONFIG.get("numeric_features", []) or \
                       field_name in [sc.get("input_key") for sc in FEATURE_CONFIG.get("special_cases", {}).values()]:
                        
                        is_required = is_mandatory_field(field_name)
                        field_value = render_field(field_name, is_required=is_required)
                        user_inputs[field_name] = field_value
                    else:
                        st.warning(f"⚠️ Field '{field_name}' not configured")
        
        st.divider()
        
        # SINGLE Model Selection (back to original approach)
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
                    
                    # Show prediction history count
                    if st.session_state.prediction_history:
                        st.info(f"📊 {len(st.session_state.prediction_history)} predictions made so far")
                else:
                    st.warning("⚠️ No trained models found")
            else:
                st.warning("⚠️ Models not available")
        except Exception as e:
            st.error(f"Model loading error: {e}")
            selected_model = "default_model"
        
        # Validation
        mandatory_fields = [f for f in user_inputs.keys() if is_mandatory_field(f)]
        missing_fields = []
        for field in mandatory_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_title(field))
        
        if missing_fields:
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")
        
        st.divider()
        
        # Action buttons
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
        
        # Save Configuration Section
        st.subheader("💾 Save Configuration")
        config_name = st.text_input(
            "Configuration Name",
            placeholder="e.g., Banking_Project_Template",
            help="Enter a name to save current parameter settings"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.button(
                "💾 Save Config",
                use_container_width=True,
                disabled=not config_name.strip(),
                help="Save current parameters for future use"
            )
        
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
        
        if st.button("Check Models", use_container_width=True):
            try:
                model_status = check_required_models()
                if model_status.get("models_available", False):
                    st.success("✅ Models available")
                else:
                    st.error("❌ Models not available")
            except Exception as e:
                st.error(f"❌ Error: {e}")

        with st.expander("🐛 Debug Info"):
            st.json({
                "Total Fields": len(user_inputs),
                "Missing Required": len(missing_fields),
                "Selected Model": selected_model,
                "Prediction History": len(st.session_state.prediction_history),
                "Field Values": {k: str(v)[:50] for k, v in user_inputs.items() if v}
            })
        
        user_inputs["selected_model"] = selected_model
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
        display_params = {}
        exclude_keys = {'selected_models', 'submit', 'clear_results', 'comparison_mode'}
        
        for key, value in user_inputs.items():
            if key not in exclude_keys and value is not None and value != "":
                label = get_field_title(key)
                display_params[label] = str(value)
        
        if display_params:
            col1, col2 = st.columns(2)
            items = list(display_params.items())
            mid_point = len(items) // 2
            
            with col1:
                for param, value in items[:mid_point]:
                    st.text(f"**{param}:** {value}")
            
            with col2:
                for param, value in items[mid_point:]:
                    st.text(f"**{param}:** {value}")
            
            if selected_models:
                model_names = []
                for model in selected_models:
                    try:
                        model_display_name = get_model_display_name(model)
                        model_names.append(model_display_name)
                    except:
                        model_names.append(model)
                
                if len(model_names) == 1:
                    st.info(f"🤖 **Model:** {model_names[0]}")
                else:
                    st.info(f"🤖 **Models:** {', '.join(model_names)}")
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