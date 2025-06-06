# ui.py - Completely Fixed Ultra-Compact Version
"""
Fixed Streamlit UI with ultra-compact form design and proper field handling
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
from datetime import datetime
from models import (
    # Core prediction functions
    predict_with_training_features_optimized,
    predict_man_hours_direct,
    
    # Feature creation
    create_training_compatible_features,
    
    # Model management
    list_available_models,
    check_required_models,
    
    # Diagnostics (optional)
    diagnose_model_file,
    fix_model_loading_issues
)

st.set_page_config(
    page_title="ML Project Effort Estimator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            index=1,  # Default to "2-3"
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
# --- About Section Function ---
def about_section():
    """Display about section with tool information"""
    st.markdown("""
    ### About This Tool
    
    The **ML Project Effort Estimator** is a machine learning-powered tool designed to help project managers, 
    developers, and teams estimate the effort required for software development projects.
    
    #### Key Features:
    - **Machine Learning Models**: Uses trained ML models to predict effort based on project characteristics
    - **Comprehensive Parameters**: Considers project size, team composition, technology stack, and organizational factors
    - **Interactive Interface**: User-friendly form with real-time validation and feedback
    - **Feature Importance**: Shows which factors most influence the effort estimation
    - **Configuration Management**: Save and load project configurations for reuse
    
    #### How It Works:
    1. **Data Input**: Enter project parameters including team size, technology stack, and project characteristics
    2. **ML Prediction**: The tool applies trained machine learning models to generate effort estimates
    3. **Results Analysis**: View the predicted effort in hours, days, and per-person breakdowns
    4. **Insights**: Understand which factors most influence your project's effort estimate
    
    #### Best Practices:
    - Provide accurate team size and project complexity information
    - Select the most appropriate technology stack options
    - Use historical project data to validate estimates
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

# --- Main Sidebar Function ---
def sidebar_inputs():
    """Create ultra-compact sidebar with form inputs"""
    with st.sidebar:
        st.title("🔮 Project Parameters")
        
        # Instructions
        st.info("Required fields (marked with ⭐)")
        
        user_inputs = {}
        tab_organization = get_tab_organization()
        
        # Create tabs
        tab_names = list(tab_organization.keys())
        tabs = st.tabs(tab_names)
        
        for tab_idx, (tab_name, tab_config) in enumerate(tab_organization.items()):
            with tabs[tab_idx]:
                # Tab description
                description = tab_config.get("description", "")
                if description:
                    st.caption(description)
                
                # Render fields
                fields = tab_config.get("fields", [])
                for field_name in fields:
                    if field_name in FEATURE_CONFIG.get("categorical_features", {}) or \
                       field_name in FEATURE_CONFIG.get("numeric_features", []) or \
                       field_name in [sc.get("input_key") for sc in FEATURE_CONFIG.get("special_cases", {}).values()]:
                        
                        is_required = is_mandatory_field(field_name)
                        field_value = render_field(field_name, is_required=is_required)
                        user_inputs[field_name] = field_value
                    else:
                        # Handle missing field configurations
                        st.warning(f"⚠️ Field '{field_name}' not configured")
        
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
                        help="Select machine learning model for prediction"
                    )
                    selected_model = model_options[selected_display_name]
                else:
                    st.warning("⚠️ No trained models found")
            else:
                st.warning("⚠️ Models not available")
        except Exception as e:
            st.error(f"Model loading error: {e}")
            selected_model = "default_model"  # Fallback
        
        # Validation
        mandatory_fields = [f for f in user_inputs.keys() if is_mandatory_field(f)]
        missing_fields = []
        for field in mandatory_fields:
            value = user_inputs.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(get_field_title(field))
        
        if missing_fields:
            st.error(f"⚠️ Missing required fields: {', '.join(missing_fields)}")
        
        # Action buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            predict_button = st.button(
                "🔮 Predict Effort",
                type="primary",
                use_container_width=True,
                disabled=len(missing_fields) > 0 or not selected_model
            )
        
        with col2:
            save_button = st.button(
                "💾 Save Config",
                use_container_width=True
            )
        
        # Configuration saving
        if save_button:
            config_name = st.text_input(
                "Configuration Name", 
                placeholder="Enter a name for this configuration"
            )
            if config_name.strip():
                save_current_configuration(user_inputs, config_name.strip())
        
        # Diagnostics section
        st.divider()
        st.subheader("🔧 Diagnostics")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Models", use_container_width=True):
                try:
                    fix_model_loading_issues()
                    st.success("✅ Models checked")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("Test Model", use_container_width=True):
                try:
                    if create_test_model_file():
                        st.success("✅ Test model created")
                    else:
                        st.error("❌ Creation failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Debug information
        with st.expander("🐛 Debug Info"):
            st.json({
                "Total Fields": len(user_inputs),
                "Missing Required": len(missing_fields),
                "Selected Model": selected_model,
                "Field Values": {k: str(v)[:50] for k, v in user_inputs.items() if v}
            })
        
        # Prepare return data
        user_inputs["selected_model"] = selected_model
        user_inputs["submit"] = predict_button
        
        return user_inputs

# --- Configuration Management: save the current form input state to a JSON files of user configs for future reuse ---
def save_current_configuration(user_inputs, config_name):
    """Save current configuration to file"""
    config = user_inputs.copy()
    config.pop('submit', None)
    config['saved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    configs_dir = "saved_configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    config_file = f'{configs_dir}/{config_name}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    st.success(f"✅ Configuration '{config_name}' saved!")

# --- Display Functions ---
def display_inputs(user_inputs, selected_model):
    """Display input parameters summary"""
    st.subheader("📋 Input Parameters Summary")
    
    # Filter and display parameters
    display_params = {}
    exclude_keys = {'selected_model', 'submit'}
    
    for key, value in user_inputs.items():
        if key not in exclude_keys and value is not None and value != "":
            label = get_field_title(key)
            display_params[label] = str(value)
    
    if display_params:
        # Display in columns for better layout
        col1, col2 = st.columns(2)
        items = list(display_params.items())
        mid_point = len(items) // 2
        
        with col1:
            for param, value in items[:mid_point]:
                st.text(f"**{param}:** {value}")
        
        with col2:
            for param, value in items[mid_point:]:
                st.text(f"**{param}:** {value}")
        
        # Display selected model
        if selected_model:
            try:
                model_display_name = get_model_display_name(selected_model)
                st.info(f"🤖 **Model:** {model_display_name}")
            except:
                st.info(f"🤖 **Model:** {selected_model}")
    else:
        st.warning("No parameters to display")

def show_prediction(prediction, team_size):
    """Display prediction results"""
    st.subheader("🎯 Prediction Results")
    
    if prediction is None:
        st.error("❌ Prediction Failed")
        st.info("Check diagnostics section for troubleshooting")
        return
    
    try:
        # Main prediction display in columns
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
        
        # Additional insights
        if prediction < 1:
            st.warning("⚠️ Very low estimate - please verify inputs")
        elif prediction > 10000:
            st.warning("⚠️ Very high estimate - please verify inputs")
        else:
            st.success("✅ Prediction completed successfully")
    
    except Exception as e:
        st.error(f"Display error: {e}")

# --- Feature Analysis ---
def show_feature_importance(selected_model, features_dict):
    """Display feature importance analysis" with horizontal bar chart and data table."""
    if not selected_model:
        return
    
    try:
        feature_importance = get_feature_importance(selected_model)
        if feature_importance is not None:
            st.subheader("📊 Feature Importance Analysis")
            
            # Prepare data
            exclude_keys = {'selected_model', 'submit'}
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
                
                # Display as chart
                st.bar_chart(importance_df.set_index('Feature'))
                
                # Data table
                with st.expander("📋 View Detailed Importance Data"):
                    st.dataframe(importance_df.round(3), use_container_width=True)
    
    except Exception as e:
        st.info(f"Feature importance analysis not available: {e}")

# --- Main Application ---
def main():
    """Main application function"""
    
    # Main header
    st.title("🔮 ML Project Effort Estimator")
    st.markdown("Get accurate effort estimates using machine learning models trained on historical project data.")
    
    try:
        # Get user inputs from sidebar
        user_inputs = sidebar_inputs()
        
        # Main content area
        if user_inputs.get('submit', False):
            selected_model = user_inputs.get('selected_model')
            
            if selected_model:
                # Display input summary
                display_inputs(user_inputs, selected_model)
                
                st.divider()
                
                # Make prediction
                with st.spinner("🤔 Analyzing project parameters and generating prediction..."):
                    try:
                        prediction = predict_man_hours(user_inputs, selected_model, use_preprocessing_pipeline=True)
                        team_size = user_inputs.get('project_prf_max_team_size', 5)
                    except Exception as e:
                        prediction = None
                        st.error(f"Prediction error: {e}")
                
                # Show prediction results
                show_prediction(prediction, team_size)
                
                # Show feature analysis if prediction successful
                if prediction is not None:
                    st.divider()
                    show_feature_importance(selected_model, user_inputs)
            
            else:
                st.warning("⚠️ Please select a model to make predictions")
        
        else:
            # Welcome screen
            st.info("👈 **Get Started:** Fill in the project parameters in the sidebar and click 'Predict Effort' to get your estimate.")
            
            # Help section
            with st.expander("ℹ️ How to Use This Tool"):
                st.markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** - Complete all fields marked with ⭐ in the "Required" tab
                2. **Optional Parameters** - Add more details in the "Optional" tab for better accuracy  
                3. **Select Model** - Choose from available ML models
                4. **Get Prediction** - Click 'Predict Effort' to see your estimate
                
                ### Tips for Better Estimates
                - Provide accurate team size information
                - Select the most appropriate technology stack
                - Choose realistic project complexity levels
                - Review historical similar projects if available
                
                ### Troubleshooting
                - Use "Check Models" if no models appear in the dropdown
                - Use "Test Model" to create a sample model for testing
                - Check the Debug Info section for technical details
                """)
            
            # About section
            with st.expander("📖 About This Tool"):
                st.markdown("""
                This tool uses machine learning models trained on historical project data to estimate 
                software development effort. It considers factors like:
                
                - **Project Characteristics**: Size, complexity, type
                - **Team Factors**: Size, experience, composition  
                - **Technology Stack**: Programming languages, architecture, platform
                - **Organizational Context**: Industry sector, company type
                
                The predictions should be used as guidance alongside expert judgment and 
                historical project experience.
                """)
    
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration files and model setup.")

if __name__ == "__main__":
    main()