# ui.py
"""
- Launch the Streamlit App: Run streamlit run main.py and Use the 
  "Check for required models" option to verify your models are detected properly. Should be only model allowed at one time.
- Explore Additional Features: Save/load configurations for frequently used project settings, Use the what-if analysis to 
  see how changing parameters affects estimates and Review feature importance to understand which factors have the biggest impact
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import yaml
from datetime import datetime
from models import list_available_models, get_feature_importance, check_required_models, get_model_display_name

# --- Helper functions ---
def safe_import_pipeline_function(func_name):
    """Safely import pipeline functions with fallbacks"""
    try:
        from pipeline import load_preprocessing_pipeline, validate_pipeline_compatibility
        if func_name == 'load_preprocessing_pipeline':
            return load_preprocessing_pipeline
        elif func_name == 'validate_pipeline_compatibility':
            return validate_pipeline_compatibility
        else:
            return None
    except ImportError:
        return None

def get_pipeline_status():
    """Get pipeline status with safe imports"""
    load_pipeline_func = safe_import_pipeline_function('load_preprocessing_pipeline')
    if load_pipeline_func is None:
        return None
    return load_pipeline_func()



# --- Compact sidebar CSS ---
st.markdown("""
<style>
section[data-testid="stSidebar"] .stForm .stFormItem, 
div[data-testid="stForm"] > div > div {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] .stForm label, 
section[data-testid="stSidebar"] .stForm .stRadio, 
section[data-testid="stSidebar"] .stForm .stSelectbox, 
section[data-testid="stSidebar"] .stForm .stNumberInput {
    margin-bottom: 0 !important;
}
.main .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# --- Config Loaders ---
def load_yaml_config(path):
    """Load any YAML configuration file"""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file {path}: {e}")
        return {}

def load_configs():
    """Load all configuration files"""
    # Try to load UI config first to get paths
    ui_config = load_yaml_config("config/ui_config.yaml")
    
    # Get paths from UI config or use defaults
    feature_config_path = ui_config.get("app_settings", {}).get("default_config_path", "config/feature_mapping.yaml")
    
    feature_config = load_yaml_config(feature_config_path)
    
    return feature_config, ui_config

FEATURE_CONFIG, UI_CONFIG = load_configs()

def get_team_size_group_from_config(max_team_size):
    """Dynamically determine team size group based on YAML config options"""
    team_size_config = FEATURE_CONFIG.get("special_cases", {}).get("team_size_group", {})
    options = team_size_config.get("options", [])
    
    # Parse the options to create ranges
    for option in options:
        if option == "Missing":
            continue
        elif option.isdigit():
            if max_team_size == int(option):
                return option
        elif "-" in option:
            try:
                start, end = map(int, option.split("-"))
                if start <= max_team_size <= end:
                    return option
            except ValueError:
                continue
        elif option.endswith("+"):
            try:
                threshold = int(option[:-1])
                if max_team_size >= threshold:
                    return option
            except ValueError:
                continue
    
    return "Missing"

def get_numeric_field_config(field_name):
    """Get configuration for numeric fields from UI config"""
    numeric_configs = UI_CONFIG.get("numeric_field_config", {})
    
    # Return specific config if exists, otherwise default
    if field_name in numeric_configs:
        return numeric_configs[field_name]
    
    # Fallback defaults
    return {"min": 1, "max": 10, "default": 1, "input_type": "number_input"}

def get_field_label(field_name):
    """Get human-readable labels from UI config"""
    field_labels = UI_CONFIG.get("field_labels", {})
    
    # Return configured label or generate from field name
    return field_labels.get(field_name, field_name.replace('_', ' ').title())

def get_tab_organization():
    """Get tab organization from UI config"""
    return UI_CONFIG.get("tab_organization", {
        "Basic": [],
        "Technical": [],
        "Advanced": []
    })

def get_ui_behavior():
    """Get UI behavior settings from config"""
    return UI_CONFIG.get("ui_behavior", {
        "multiselect_threshold": 10,
        "radio_threshold": 4,
        "selectbox_threshold": 8
    })

def render_field(field_name, config_section, config_data):
    """Dynamically render form fields based on YAML config"""
    label = get_field_label(field_name)
    ui_behavior = get_ui_behavior()
    
    if config_section == "numeric_features":
        field_config = get_numeric_field_config(field_name)
        input_type = field_config.get("input_type", "number_input")
        
        if input_type == "slider":
            return st.slider(label, field_config["min"], field_config["max"], field_config["default"])
        else:
            return st.number_input(label, field_config["min"], field_config["max"], field_config["default"])
    
    elif config_section == "categorical_features":
        options = config_data.get("options", [])
        if len(options) > ui_behavior["multiselect_threshold"]:
            return st.multiselect(label, options)
        else:
            return st.selectbox(label, options)
    
    elif config_section == "one_hot_features":
        mapping = config_data.get("mapping", {})
        options = list(mapping.keys())
        
        if len(options) <= ui_behavior["radio_threshold"]:
            return st.radio(label, options)
        elif len(options) <= ui_behavior["selectbox_threshold"]:
            return st.selectbox(label, options)
        else:
            return st.multiselect(label, options)
    
    elif config_section == "special_cases":
        if field_name == "project_prf_team_size_group":
            # This will be auto-calculated, just display it
            return None
        else:
            options = config_data.get("options", [])
            return st.selectbox(label, options)
    
    elif config_section == "binary_features":
        mapping = config_data.get("mapping", {})
        options = list(mapping.keys())
        return st.radio(label, options)
    
    return None

def sidebar_inputs():
    model_status = check_required_models()
    tab_organization = get_tab_organization()
    user_inputs = {}
    
    with st.sidebar:
        with st.form("estimation_form"):
            # Create tabs dynamically
            tabs = st.tabs(list(tab_organization.keys()))
            
            for tab_idx, (tab_name, field_list) in enumerate(tab_organization.items()):
                with tabs[tab_idx]:
                    if tab_name != "Advanced":
                        st.header(f"{tab_name} Information")
                    
                    for field_name in field_list:
                        # Find field in config
                        field_found = False
                        
                        # Check numeric features
                        if field_name in FEATURE_CONFIG.get("numeric_features", []):
                            user_inputs[field_name] = render_field(field_name, "numeric_features", {})
                            field_found = True
                        
                        # Check categorical features
                        elif field_name in FEATURE_CONFIG.get("categorical_features", {}):
                            config_data = FEATURE_CONFIG["categorical_features"][field_name]
                            user_inputs[field_name] = render_field(field_name, "categorical_features", config_data)
                            field_found = True
                        
                        # Check one-hot features
                        else:
                            for group_name, group_config in FEATURE_CONFIG.get("one_hot_features", {}).items():
                                if group_config.get("input_key") == field_name:
                                    user_inputs[field_name] = render_field(field_name, "one_hot_features", group_config)
                                    field_found = True
                                    break
                        
                        # Check special cases
                        if not field_found:
                            for group_name, group_config in FEATURE_CONFIG.get("special_cases", {}).items():
                                if group_config.get("input_key") == field_name:
                                    if field_name == "project_prf_team_size_group":
                                        # Auto-calculate based on max team size
                                        max_team_size = user_inputs.get("project_prf_max_team_size", 5)
                                        team_size_group = get_team_size_group_from_config(max_team_size)
                                        user_inputs[field_name] = team_size_group
                                        st.write(f"**{get_field_label(field_name)}:** {team_size_group}")
                                    else:
                                        user_inputs[field_name] = render_field(field_name, "special_cases", group_config)
                                    field_found = True
                                    break
                        
                        # Check binary features
                        if not field_found:
                            for group_name, group_config in FEATURE_CONFIG.get("binary_features", {}).items():
                                if group_config.get("input_key") == field_name:
                                    user_inputs[field_name] = render_field(field_name, "binary_features", group_config)
                                    field_found = True
                                    break
            
            # Model Selection (always in Advanced tab)
            with tabs[-1]:  # Last tab (Advanced)
                st.header("Model Selection")
                selected_model = None
                if model_status["models_available"]:
                    available_models = list_available_models()
                    if available_models:
                        model_options = {model['display_name']: model['technical_name'] for model in available_models}
                        selected_display_name = st.selectbox("Select Prediction Model", list(model_options.keys()), key="model_selectbox")
                        selected_model = model_options[selected_display_name]
                    else:
                        st.warning("No trained models found. Please add trained models to the 'models' directory.")
                else:
                    st.warning("No trained models found. Please create or add trained models.")

            col1, col2 = st.columns(2)
            submit = col1.form_submit_button("Predict Man-Hours")
            save_config = col2.form_submit_button("Save Config")
            config_name = None
            if save_config:
                config_name = st.text_input("Enter a name for this configuration:")

            user_inputs["selected_model"] = selected_model
            user_inputs["submit"] = submit

            # Save config logic
            if save_config and selected_model:
                if config_name:
                    save_current_configuration(user_inputs, config_name)
                else:
                    st.warning("Please enter a name for your configuration above and resubmit.")

            if submit or save_config:
                return create_feature_dict_from_config(user_inputs, FEATURE_CONFIG)
            return {'selected_model': None, 'submit': False}

def create_feature_dict_from_config(user_inputs, config):
    """Create feature dictionary dynamically from config"""
    features = {}
    
    # Process numeric features
    for key in config.get("numeric_features", []):
        features[key] = user_inputs.get(key, 0)
    
    # Process categorical features
    for key, meta in config.get("categorical_features", {}).items():
        val = user_inputs.get(key, None)
        if isinstance(val, list):
            features[key] = ";".join(val) if val else ""
        else:
            features[key] = val if val else ""
    
    # Process one-hot features
    for group, mapping in config.get("one_hot_features", {}).items():
        input_key = mapping["input_key"]
        input_value = user_inputs.get(input_key, [])
        
        if isinstance(input_value, list):
            selected_values = set(input_value)
        else:
            selected_values = {input_value} if input_value else set()
        
        for label, feat_key in mapping["mapping"].items():
            features[feat_key] = int(label in selected_values)
    
    # Process special cases
    for group, spec in config.get("special_cases", {}).items():
        input_key = spec["input_key"]
        input_value = user_inputs.get(input_key, "")
        
        if "output_keys" in spec:
            for label, feat_key in spec["output_keys"].items():
                features[feat_key] = int(input_value == label)
    
    # Process binary features
    for group, mapping in config.get("binary_features", {}).items():
        input_key = mapping["input_key"]
        input_value = user_inputs.get(input_key, "")
        for label, feat_key in mapping["mapping"].items():
            features[feat_key] = int(input_value == label)
    
    features["selected_model"] = user_inputs.get("selected_model")
    features["submit"] = user_inputs.get("submit", False)
    return features

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_current_configuration(user_inputs, config_name):
    config = user_inputs.copy()
    config.pop('submit', None)
    config['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    configs_dir = UI_CONFIG.get("app_settings", {}).get("configs_directory", "configs")
    ensure_dir(configs_dir)
    
    with open(f'{configs_dir}/{config_name}.json', 'w') as f:
        json.dump(config, f, default=str)
    st.success(f"Configuration '{config_name}' saved successfully!")

def load_saved_configurations():
    configs_dir = UI_CONFIG.get("app_settings", {}).get("configs_directory", "configs")
    configs = {}
    if not os.path.exists(configs_dir):
        return configs
    for filename in os.listdir(configs_dir):
        if filename.endswith('.json'):
            config_name = os.path.splitext(filename)[0]
            try:
                with open(f'{configs_dir}/{filename}', 'r') as f:
                    configs[config_name] = json.load(f)
            except Exception:
                pass
    return configs

def load_configuration(config_name):
    configs_dir = UI_CONFIG.get("app_settings", {}).get("configs_directory", "configs")
    try:
        with open(f'{configs_dir}/{config_name}.json', 'r') as f:
            return json.load(f)
    except Exception:
        st.error(f"Failed to load configuration '{config_name}'")
        return None

def display_inputs(user_inputs, selected_model):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Input Parameters Summary")
        
        # Get key parameters from config
        display_config = UI_CONFIG.get("display_config", {})
        key_param_fields = display_config.get("key_parameters_for_summary", [])
        
        key_params = {}
        for field in key_param_fields:
            label = get_field_label(field)
            key_params[label] = user_inputs.get(field, 'N/A')
        
        input_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in key_params.items()])
        st.dataframe(input_df, use_container_width=True)
        
        if selected_model:
            model_display_name = get_model_display_name(selected_model)
            st.write(f"📊 Selected Model: **{model_display_name}**")
        else:
            st.write("📊 Selected Model: **None**")
    return col2

def create_feature_name_mapping():
    """Dynamically create feature name mapping from config"""
    mapping = {}
    
    # Add numeric features
    for feature in FEATURE_CONFIG.get("numeric_features", []):
        mapping[feature] = get_field_label(feature)
    
    # Add one-hot features
    for group, config in FEATURE_CONFIG.get("one_hot_features", {}).items():
        for label, feat_key in config.get("mapping", {}).items():
            # Create more descriptive names
            base_name = group.replace('_', ' ').title()
            mapping[feat_key] = f"{base_name}: {label}"
    
    # Add special case features
    for group, config in FEATURE_CONFIG.get("special_cases", {}).items():
        if "output_keys" in config:
            base_name = group.replace('_', ' ').title()
            for label, feat_key in config["output_keys"].items():
                mapping[feat_key] = f"{base_name}: {label}"
    
    return mapping

def show_feature_importance(selected_model, features_dict, st):
    if not selected_model:
        st.info("No model selected for feature importance analysis.")
        return
    
    feature_importance = get_feature_importance(selected_model)
    if feature_importance is not None:
        st.subheader("Feature Importance")
        exclude_keys = {'selected_model', 'submit'}
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
        
        # Use dynamic feature name mapping
        feature_name_mapping = create_feature_name_mapping()
        
        # Get display config
        importance_config = UI_CONFIG.get("feature_importance_display", {})
        max_features = importance_config.get("max_features_shown", 15)
        chart_size = importance_config.get("chart_size", {"width": 10, "height": 8})
        precision = importance_config.get("precision_decimals", 3)
        
        importance_data = []
        for i, name in enumerate(feature_names[:min(len(feature_importance), max_features)]):
            if i < len(feature_importance):
                friendly_name = feature_name_mapping.get(name, name.replace('_', ' ').title())
                importance_data.append({
                    'Feature': friendly_name,
                    'Importance': abs(feature_importance[i])
                })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(chart_size["width"], chart_size["height"]))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    label_x_pos = width * 1.01
                    format_str = f'{{:.{precision}f}}'
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, format_str.format(width), va='center')
            
            ax.set_xlabel('Relative Importance')
            ax.set_title(f'Top Feature Importance - {get_model_display_name(selected_model)}')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(importance_df.round(precision), use_container_width=True)
        else:
            st.info("No feature importance data available.")
    else:
        model_display_name = get_model_display_name(selected_model)
        st.info(f"Feature importance is not available for {model_display_name}. This might be because the model doesn't support feature importance or there was an error retrieving it.")

def show_prediction(col2, prediction, team_size):
    # Get prediction thresholds from config
    thresholds = UI_CONFIG.get("prediction_thresholds", {"low_prediction_warning": 1, "high_prediction_warning": 10000})
    
    with col2:
        st.subheader("Prediction Result")
        if prediction is None:
            st.error("Failed to make prediction. Please check logs for details.")
            return
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='text-align:center;'>Estimated Effort</h3>
            <h1 style='text-align:center; color:#1f77b4; font-size:2.5rem;'>{prediction:.2f} Man-Hours</h1>
        </div>
        """, unsafe_allow_html=True)
        hours = int(prediction)
        days = hours // 8
        per_person = prediction / team_size if team_size > 0 else prediction
        st.markdown("### Timeline Breakdown")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Calendar Time", f"{days}d", help="Estimated calendar duration assuming full team availability")
        with metrics_col2:
            st.metric("Per Person", f"{per_person:.2f}h", help="Average effort per team member in hours")
        
        # Use configurable thresholds
        if prediction < thresholds["low_prediction_warning"]:
            st.warning("This prediction seems unusually low. Consider reviewing your inputs.")
        elif prediction > thresholds["high_prediction_warning"]:
            st.warning("This prediction seems unusually high. Consider reviewing your inputs.")

def about_section():
    st.markdown("---")
    with st.expander("About this Estimator", expanded=False):
        st.subheader("Machine Learning for Project Estimation")
        st.write("""
        This application uses machine learning models to predict the effort required for projects.
        The models have been trained on historical project data to provide early estimations based on
        key project parameters. These estimations can help in project planning and resource allocation.
        
        ### How it Works
        The estimator dynamically processes input parameters based on the configured feature mapping
        and uses the selected machine learning model to predict the required effort in man-hours.
        
        ### Input Categories
        The system organizes inputs into configurable categories based on the UI configuration file.
        All field labels, organization, and behavior can be customized through YAML configuration files.
        """)

def tips_section():
    with st.expander("Tips for Accurate Estimation", expanded=False):
        st.markdown("""
        ### Tips for Getting Accurate Estimations
        
        1. **Provide Complete Information**
           - Fill out all relevant fields for the most accurate predictions
           - Use realistic values based on your project's actual requirements
        
        2. **Consider Project Context**
           - Factor in your team's specific experience and capabilities
           - Account for any unique constraints or requirements
        
        3. **Review and Validate**
           - Compare estimates with historical data from similar projects
           - Use estimates as starting points, not absolute values
        
        4. **Iterative Refinement**
           - Update estimates as project requirements become clearer
           - Consider re-estimating at key project milestones
        """)
        st.info("Remember that these estimations are meant to be starting points. Always review and adjust based on your team's specific context and historical performance.")