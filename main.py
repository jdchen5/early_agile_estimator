# main.py 

import streamlit as st

# THIS MUST BE FIRST, before any other Streamlit call!
st.set_page_config(
    page_title="Agile Project Estimator", 
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
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

# Import UI functions with fallbacks for enhanced functions
try:
    from ui import (
        sidebar_inputs, 
        display_inputs, 
        show_prediction, 
        about_section, 
        tips_section, 
        show_feature_importance,
        load_yaml_config,
        get_field_label,
        enhanced_main,              # New enhanced function
        enhanced_sidebar_inputs,    # New enhanced function
        safe_display_inputs,        # New safe function
        safe_show_prediction        # New safe function
    )
    ENHANCED_UI_AVAILABLE = True
except ImportError as e:
    # Fallback to basic UI functions if enhanced ones aren't available
    from ui import (
        sidebar_inputs, 
        display_inputs, 
        show_prediction, 
        about_section, 
        tips_section, 
        show_feature_importance,
        load_yaml_config,
        get_field_label
    )
    ENHANCED_UI_AVAILABLE = False
    print(f"Enhanced UI features not available: {e}")

# Try to import diagnostic functions with fallbacks
try:
    from models import (
        diagnose_model_file,
        create_test_model_file,
        fix_model_loading_issues
    )
    DIAGNOSTICS_AVAILABLE = True
except ImportError as e:
    DIAGNOSTICS_AVAILABLE = False
    print(f"Diagnostic functions not available: {e}")
    
    # Create dummy functions to prevent errors
    def diagnose_model_file(model_name):
        return {"error": "Diagnostics not available"}
    
    def create_test_model_file():
        return False
    
    def fix_model_loading_issues():
        pass

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

MODEL_SCALER = 'standard_scaler'

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

def get_what_if_range_from_config(param_key, current_value):
    """Get what-if analysis ranges from configuration or use smart defaults"""
    # Get numeric field config to determine appropriate ranges
    numeric_config = UI_CONFIG.get("numeric_field_config", {})
    
    if param_key in numeric_config:
        field_config = numeric_config[param_key]
        min_val = field_config.get("min", 1)
        max_val = field_config.get("max", 100)
        
        # Create range around current value, bounded by config limits
        range_size = min(50, (max_val - min_val) // 10)
        start_val = max(min_val, current_value - range_size)
        end_val = min(max_val, current_value + range_size)
        
        return np.linspace(start_val, end_val, 11)
    
    # Fallback defaults for specific fields
    if param_key == "project_prf_year_of_project":
        return np.arange(max(2015, current_value-5), min(2031, current_value+6))
    elif param_key == "project_prf_functional_size":
        return np.linspace(max(1, current_value-500), current_value+500, 11)
    elif param_key == "project_prf_max_team_size":
        return np.linspace(max(1, current_value-5), current_value+5, 11)
    else:
        # Generic range for other numeric fields
        return np.linspace(max(0, current_value-2), current_value+2, 11)

def get_what_if_parameters():
    """Get available parameters for what-if analysis from configuration"""
    # Get all numeric features from config
    numeric_features = FEATURE_CONFIG.get("numeric_features", [])
    
    # Create dictionary of parameter labels to keys
    what_if_params = {}
    for feature in numeric_features:
        label = get_field_label(feature)
        what_if_params[label] = feature
    
    return what_if_params

def perform_what_if_analysis(user_inputs, selected_model, param_key, param_label):
    """Perform what-if analysis for a given parameter"""
    try:
        current_value = user_inputs.get(param_key, 0)
        
        # Get what-if range from configuration
        what_if_values = get_what_if_range_from_config(param_key, current_value)
        
        predictions = []
        for val in what_if_values:
            modified_features = user_inputs.copy()
            modified_features[param_key] = val
            prediction = predict_man_hours(modified_features, selected_model)
            predictions.append(prediction if prediction is not None else 0)
        
        # Filter out invalid predictions
        valid_predictions = [(val, pred) for val, pred in zip(what_if_values, predictions) 
                            if pred is not None and pred > 0]
        
        if not valid_predictions:
            st.error("All predictions failed. Please check your model and input parameters.")
            return
        
        valid_values, valid_preds = zip(*valid_predictions)
        what_if_df = pd.DataFrame({
            param_label: valid_values, 
            "Estimated Man-Hours": valid_preds
        })
        
        # Get chart configuration from UI config
        chart_config = UI_CONFIG.get("feature_importance_display", {}).get("chart_size", {"width": 10, "height": 8})
        
        # Plot
        fig, ax = plt.subplots(figsize=(chart_config["width"], 6))
        ax.plot(what_if_df[param_label], what_if_df["Estimated Man-Hours"], 
                marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(param_label)
        ax.set_ylabel("Estimated Man-Hours")
        
        model_display_name = get_model_display_name(selected_model)
        ax.set_title(f"Impact of {param_label} on Estimation\nUsing {model_display_name}")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Show current estimation as a red line
        current_pred = predict_man_hours(user_inputs, selected_model)
        if current_pred is not None:
            ax.axhline(y=current_pred, color='r', linestyle='--',
                       label=f"Current Estimation ({current_pred:.2f} man-hours)")
            ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display data table with appropriate precision
        precision = UI_CONFIG.get("feature_importance_display", {}).get("precision_decimals", 2)
        st.dataframe(what_if_df.round(precision), use_container_width=True)
        
        st.markdown(f"""
        ### Interpretation
        The graph above shows how the estimated man-hours change when varying **{param_label}** 
        while keeping all other parameters constant. This analysis was performed using the 
        **{model_display_name}** model.
        
        **Current Value**: {current_value}  
        **Range Analyzed**: {min(valid_values):.1f} to {max(valid_values):.1f}  
        **Current Prediction**: {current_pred:.2f} man-hours (if available)
        """)
        
    except Exception as e:
        st.error(f"Error in what-if analysis: {str(e)}")
        logger.error(f"What-if analysis error: {str(e)}")

def validate_user_inputs(user_inputs):
    """Validate user inputs against configuration constraints"""
    errors = []
    warnings = []
    
    # Check numeric fields against their constraints
    numeric_config = UI_CONFIG.get("numeric_field_config", {})
    for field_name, config in numeric_config.items():
        if field_name in user_inputs:
            value = user_inputs[field_name]
            min_val = config.get("min", 0)
            max_val = config.get("max", float('inf'))
            
            if value < min_val:
                errors.append(f"{get_field_label(field_name)} must be at least {min_val}")
            elif value > max_val:
                warnings.append(f"{get_field_label(field_name)} is above recommended maximum of {max_val}")
    
    # Check for required fields
    required_fields = ["project_prf_functional_size", "project_prf_max_team_size"]
    for field in required_fields:
        if field not in user_inputs or user_inputs[field] <= 0:
            errors.append(f"{get_field_label(field)} is required and must be greater than 0")
    
    # Basic pipeline test (only if submission attempted and function is available)
    if user_inputs.get('submit', False) and PIPELINE_AVAILABLE:
        try:
            pipeline_test = test_feature_pipeline_integration(user_inputs)
            if not pipeline_test.get("success", False):
                for error in pipeline_test.get("errors", []):
                    warnings.append(f"Pipeline issue: {error}")
            for warning in pipeline_test.get("warnings", []):
                warnings.append(warning)
        except Exception as e:
            # Don't fail validation if pipeline test fails
            logger.warning(f"Could not validate preprocessing pipeline: {str(e)}")
    
    return errors, warnings

def check_dependencies():
    """Check if all required dependencies are available"""
    status = {
        "models_module": True,  # We know this works from debug
        "pipeline_module": PIPELINE_AVAILABLE,
        "ui_module": True,  # We know this works from debug
        "config_files": True,  # We know this works from debug
        "enhanced_ui": ENHANCED_UI_AVAILABLE,
        "diagnostics": DIAGNOSTICS_AVAILABLE
    }
    
    return status

def main():
    """Main function with enhanced UI support"""
    add_custom_css()

    st.title("⏱️ Machine Learning for Early Estimation in Agile Projects")
    st.markdown("""
    This application helps project managers and team leads estimate the effort required for
    agile software projects using machine learning models trained on historical project data.
    """)
    st.markdown("---")
    
    # Check dependencies first
    deps_status = check_dependencies()
    if not deps_status["models_module"]:
        st.error("Critical dependency missing: models module. Please check your installation.")
        return
    
    # Use enhanced UI if available, otherwise fallback to original
    if ENHANCED_UI_AVAILABLE:
        try:
            # Use the enhanced main function from ui.py
            enhanced_main()
            return
        except Exception as e:
            st.error(f"Enhanced UI failed: {e}. Falling back to basic UI.")
            logger.error(f"Enhanced UI error: {e}")
    
    # Fallback to original main function
    original_main()

def original_main():
    """Original main function as fallback"""
    # Get user inputs from sidebar
    try:
        # Use enhanced sidebar if available, otherwise use original
        if ENHANCED_UI_AVAILABLE:
            try:
                user_inputs = enhanced_sidebar_inputs()
            except:
                user_inputs = sidebar_inputs()
        else:
            user_inputs = sidebar_inputs()
            
        selected_model = user_inputs.get('selected_model')
        submit = user_inputs.get('submit', False)
        team_size = user_inputs.get('project_prf_max_team_size', 5)
    except Exception as e:
        st.error(f"Error in sidebar inputs: {str(e)}")
        logger.error(f"Sidebar error: {str(e)}")
        return

    # Validate inputs if submission attempted
    if submit:
        try:
            errors, warnings = validate_user_inputs(user_inputs)
            
            if errors:
                for error in errors:
                    st.error(error)
                submit = False  # Prevent submission with errors
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
        except Exception as e:
            st.warning(f"Input validation error: {str(e)}")

    tab_results, tab_viz, tab_help = st.tabs(["Estimation Results", "Visualization", "Help & Documentation"])

    with tab_results:
        try:
            # Use safe display if available, otherwise use original
            if ENHANCED_UI_AVAILABLE:
                try:
                    col2 = safe_display_inputs(user_inputs, selected_model)
                except:
                    col2 = display_inputs(user_inputs, selected_model)
            else:
                col2 = display_inputs(user_inputs, selected_model)
            
            if submit and selected_model:
                with st.spinner("Calculating estimation..."):
                    try:
                        # Use the dict directly for prediction - models.py will handle preprocessing
                        prediction = predict_man_hours(user_inputs, selected_model)
                        
                        # Use safe show prediction if available
                        if ENHANCED_UI_AVAILABLE:
                            try:
                                safe_show_prediction(col2, prediction, team_size)
                            except:
                                show_prediction(col2, prediction, team_size)
                        else:
                            show_prediction(col2, prediction, team_size)
                        
                        # Log successful prediction
                        logger.info(f"Successful prediction: {prediction} man-hours using model {selected_model}")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        logger.error(f"Prediction error: {str(e)}")
                        # Show additional debug info
                        with st.expander("Debug Information"):
                            st.write(f"**Model**: {selected_model}")
                            st.write(f"**User inputs keys**: {list(user_inputs.keys())}")
                            st.write(f"**Error type**: {type(e).__name__}")
                            st.write(f"**Error message**: {str(e)}")
                            
            elif submit and not selected_model:
                col2.error("Please select a model before making a prediction.")
            else:
                col2.info("Click the 'Predict Man-Hours' button to see the estimation result.")
        except Exception as e:
            st.error(f"Error in results tab: {str(e)}")
            logger.error(f"Results tab error: {str(e)}")

    with tab_viz:
        try:
            if submit and selected_model:
                # Show feature importance
                show_feature_importance(selected_model, user_inputs, st)
                
                st.subheader("What-If Analysis")
                st.write("See how changing one parameter affects the estimation:")
                
                # Get available parameters dynamically from configuration
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
                    
            elif not selected_model:
                st.info("Please select a model first to see visualizations and what-if analysis.")
            else:
                st.info("Make a prediction first to see visualizations and what-if analysis.")
        except Exception as e:
            st.error(f"Error in visualization tab: {str(e)}")
            logger.error(f"Visualization tab error: {str(e)}")

    with tab_help:
        try:
            about_section()
            tips_section()
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
        
        # Show debug info
        with st.expander("Debug Information"):
            st.write("**Error Details:**")
            import traceback
            st.code(traceback.format_exc())