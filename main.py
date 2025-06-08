import streamlit as st

# THIS MUST BE FIRST, before any other Streamlit call!
st.set_page_config(
    page_title="ML Agile Project Effort Estimator", 
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style and session helpers
from ui import set_sidebar_width, initialize_session_state

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Import core functions from models
from models import (
    load_model, 
    predict_man_hours, 
    get_feature_importance, 
    get_model_display_name,
    get_expected_feature_names_from_model,
    check_required_models
)

# Import UI functions
from ui import (
    sidebar_inputs, 
    display_inputs, 
    show_prediction, 
    about_section, 
    show_feature_importance,
    load_yaml_config,
    get_field_label,
    add_prediction_to_history,
    show_prediction_history,
    show_prediction_comparison_table
)

# Try to import pipeline-related functions with fallbacks
try:
    from models import (
        check_preprocessing_pipeline_compatibility,
        get_preprocessing_pipeline_info,
        test_feature_pipeline_integration,
        get_feature_statistics
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    
    def check_preprocessing_pipeline_compatibility():
        return {"compatible": False, "error": "Pipeline module not available"}
    
    def get_preprocessing_pipeline_info():
        return {"available": False, "error": "Pipeline module not available"}
    
    def test_feature_pipeline_integration(feature_dict):
        return {"success": False, "errors": ["Pipeline module not available"]}
    
    def get_feature_statistics():
        return {"error": "Pipeline module not available"}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configurations
UI_CONFIG = load_yaml_config("config/ui_config.yaml")
FEATURE_CONFIG = load_yaml_config("config/feature_mapping.yaml")

def add_custom_css():
    st.markdown("""
    <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1 {color: #1E88E5;}
        .stTabs [data-baseweb="tab-list"] {gap: 2px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-left: 10px; padding-right: 10px;
        }
        .stTabs [aria-selected="true"] {background-color: #E3F2FD;}
    </style>
    """, unsafe_allow_html=True)

def show_disclaimer():
    with st.expander("⚠️ Disclaimer"):
        st.markdown("""
        The predictions provided by this application are estimates based on machine learning models trained on sample data.
        These estimations should be used as a starting point and always be reviewed by experienced project managers.
        Actual project timelines may vary based on factors not captured in this model.
        """)

def check_dependencies():
    """Check if all required dependencies are available"""
    status = {
        "models_module": True,  # We know this works
        "pipeline_module": PIPELINE_AVAILABLE,
        "ui_module": True,  # We know this works
        "config_files": True,  # We know this works
    }
    return status

def main():
    # --- Style and Session ---
    add_custom_css()
    set_sidebar_width()  # Call after set_page_config
    initialize_session_state()

    st.title("⏱️ Machine Learning for Early Estimation in Agile Projects")
    st.markdown("""
    This application helps project managers and team leads estimate the effort required for
    agile software projects using machine learning models trained on historical project data.
    """)
    st.markdown("---")
    
    # --- User Inputs from Sidebar ---
    try:
        user_inputs = sidebar_inputs()
        selected_model = user_inputs.get('selected_model', None)
        selected_models = [selected_model] if selected_model else []

        submit = user_inputs.get('submit', False)
        save_config = user_inputs.get('save_config', False)
        config_name = user_inputs.get('config_name', '')
        team_size = user_inputs.get('project_prf_max_team_size', 5)
    except Exception as e:
        st.error(f"Error in sidebar inputs: {str(e)}")
        logger.error(f"Sidebar error: {str(e)}")
        return

    tab_results, tab_viz, tab_help = st.tabs(["Estimation Results", "Visualization", "Help & Documentation"])

    with tab_results:
        try:
            display_inputs(user_inputs, [selected_model] if selected_model else [])


            if submit and selected_model:
                with st.spinner("Calculating estimation..."):
                    try:
                        # Run the prediction
                        prediction = predict_man_hours(user_inputs, selected_model)
                        # Show current prediction (with model name)
                        show_prediction(prediction, team_size, selected_model)
                        # Save to history
                        add_prediction_to_history(user_inputs, selected_model, prediction, team_size)
                        # Show ALL predictions (with model names, results, and timestamps)
                        show_prediction_history()
                        # (Optional) Show comparison table for quick view if more than 1 prediction
                        show_prediction_comparison_table()
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        logger.error(f"Prediction error: {str(e)}")
            elif submit and not selected_model:
                st.error("Please select a model before making a prediction.")
            else:
                st.info("Click the 'Predict Effort' button to see the estimation result.")

        except Exception as e:
            st.error(f"Error in results tab: {str(e)}")
            logger.error(f"Results tab error: {str(e)}")

    with tab_viz:
        try:
            if submit and selected_model:
                # Show feature importance
                show_feature_importance(selected_model, user_inputs)
                
                st.subheader("What-If Analysis")
                st.write("See how changing one parameter affects the estimation:")
                
                # Get available parameters dynamically from configuration
                from ui import get_what_if_parameters, perform_what_if_analysis
                what_if_params = get_what_if_parameters()
                
                if what_if_params:
                    what_if_param = st.selectbox("Select parameter to vary", list(what_if_params.keys()))
                    if st.button("Run What-If Analysis"):
                        with st.spinner("Running what-if analysis..."):
                            try:
                                perform_what_if_analysis(
                                    user_inputs, 
                                    selected_model, 
                                    what_if_params[what_if_param], 
                                    what_if_param
                                )
                            except Exception as e:
                                st.error(f"Error during what-if analysis: {str(e)}")
                                logger.error(f"What-if analysis error: {str(e)}")
                else:
                    st.warning("No numeric parameters available for what-if analysis.")
                    
            else:
                st.info("Make a prediction first to see visualizations and what-if analysis.")
        except Exception as e:
            st.error(f"Error in visualization tab: {str(e)}")
            logger.error(f"Visualization tab error: {str(e)}")

    with tab_help:
        try:
            about_section()
            show_disclaimer()
            
            # Add configuration info
            with st.expander("📋 Configuration Information"):
                st.subheader("Loaded Configurations")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Feature Configuration:**")
                    st.write(f"- Numeric features: {len(FEATURE_CONFIG.get('numeric_features', []))}")
                    st.write(f"- Categorical features: {len(FEATURE_CONFIG.get('categorical_features', {}))}")
                    st.write(f"- One-hot features: {len(FEATURE_CONFIG.get('one_hot_features', {}))}")
                    st.write(f"- Special cases: {len(FEATURE_CONFIG.get('special_cases', {}))}")
                with col2:
                    st.write("**UI Configuration:**")
                    st.write(f"- Field labels: {len(UI_CONFIG.get('field_labels', {}))}")
                    st.write(f"- Tab organization: {len(UI_CONFIG.get('tab_organization', {}))}")
                    st.write(f"- Numeric field configs: {len(UI_CONFIG.get('numeric_field_config', {}))}")
                # Show current field organization
                if st.checkbox("Show Field Organization"):
                    tab_org = UI_CONFIG.get('tab_organization', {})
                    for tab_name, fields in tab_org.items():
                        st.write(f"**{tab_name} Tab:** {', '.join(fields)}")
            
            # Add dependency status
            with st.expander("🔧 System Status"):
                st.subheader("Module Status")
                deps = check_dependencies()
                for module, status in deps.items():
                    if status:
                        st.success(f"✅ {module.replace('_', ' ').title()}")
                    else:
                        st.warning(f"⚠️ {module.replace('_', ' ').title()}")
                # Model status
                st.subheader("Model Status")
                try:
                    model_status = check_required_models()
                    if model_status["models_available"]:
                        st.success(f"✅ Found {len(model_status['found_models'])} model(s)")
                        for model in model_status['found_models']:
                            st.write(f"- {model['display_name']}")
                    else:
                        st.warning("⚠️ No trained models found")
                        st.info("Use the diagnostics tools in the sidebar to create a test model")
                except Exception as e:
                    st.error(f"Error checking models: {e}")
            
            # Add preprocessing pipeline info (only if available)
            if PIPELINE_AVAILABLE:
                with st.expander("🔧 Preprocessing Pipeline Information"):
                    st.subheader("Pipeline Status")
                    try:
                        pipeline_info = get_preprocessing_pipeline_info()
                        if pipeline_info.get("available", False):
                            st.success("✅ Preprocessing pipeline is available")
                            st.write(f"**Pipeline Steps:** {pipeline_info.get('step_count', 0)}")
                            if st.checkbox("Show Pipeline Details"):
                                steps = pipeline_info.get("steps", [])
                                for i, step in enumerate(steps, 1):
                                    st.write(f"{i}. {step}")
                            # Check compatibility
                            try:
                                compat_info = check_preprocessing_pipeline_compatibility()
                                if compat_info.get("compatible", False):
                                    st.success("✅ Pipeline is compatible with current configuration")
                                else:
                                    st.warning("⚠️ Pipeline compatibility issues detected")
                                    for rec in compat_info.get("recommendations", []):
                                        st.write(f"- {rec}")
                            except Exception as e:
                                st.warning(f"Could not check pipeline compatibility: {str(e)}")
                        else:
                            st.warning("⚠️ No preprocessing pipeline found")
                            st.info("Models will use basic feature processing. For best results, train models with preprocessing pipeline.")
                            if "error" in pipeline_info:
                                st.error(f"Error: {pipeline_info['error']}")
                    except Exception as e:
                        st.error(f"Error checking pipeline status: {str(e)}")
                        st.info("Pipeline integration may not be available. Basic feature processing will be used.")
            else:
                with st.expander("⚠️ Pipeline Information"):
                    st.warning("Preprocessing pipeline module not available")
                    st.info("The application will use basic feature processing. All core functionality should still work.")
        except Exception as e:
            st.error(f"Error in help tab: {str(e)}")
            logger.error(f"Help tab error: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred in the application: {str(e)}")
        logger.exception("An error occurred in the application:")
