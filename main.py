# main.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from models import load_model, predict_man_hours, get_feature_importance, get_model_display_name
from ui import sidebar_inputs, display_inputs, show_prediction, about_section, tips_section, show_feature_importance

import sys
print("Python executable:", sys.executable)

# Constants
MODEL_SCALER = 'standard_scaler'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Agile Project Estimator", 
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 10px;
        padding-right: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)

# App title and header
st.title("⏱️ Machine Learning for Early Estimation in Agile Projects")
st.markdown("""
This application helps project managers and team leads estimate the effort required for
agile software projects using machine learning models trained on historical project data.
""")
st.markdown("---")

# Function to create a disclaimer
def show_disclaimer():
    with st.expander("⚠️ Disclaimer"):
        st.markdown("""
        The predictions provided by this application are estimates based on machine learning models trained on sample data.
        These estimations should be used as a starting point and always be reviewed by experienced project managers.
        Actual project timelines may vary based on factors not captured in this model.
        """)

# Main application logic
def main():
    try:
        # Sidebar form inputs - now returns a dictionary of all features
        user_inputs = sidebar_inputs()
        
        # Extract key values for display and analysis
        selected_model = user_inputs.get('selected_model')
        submit = user_inputs.get('submit', False)
        team_size = user_inputs.get('project_prf_max_team_size', 5)
        
        # Create tabs for different sections
        tab_results, tab_viz, tab_help = st.tabs(["Estimation Results", "Visualization", "Help & Documentation"])

        with tab_results:
            # Display input parameters
            col2 = display_inputs(user_inputs, selected_model)
            
            # Perform prediction
            if submit and selected_model:
                with st.spinner("Calculating estimation..."):
                    model = load_model(selected_model)
                    scaler = load_model(MODEL_SCALER)

                    print("Loaded model type:", type(model))   # <-- for debugging
                    if scaler is not None:
                        print("Scaler loaded successfully.")
                    else:
                        print("Scaler not found, using default scaling.")
                    print("Loaded scaler type:", type(scaler)) # <-- for debugging

                    if model is not None:
                        prediction = predict_man_hours(user_inputs, selected_model)
                        show_prediction(col2, prediction, team_size)
                    else:
                        model_display_name = get_model_display_name(selected_model)
                        st.error(f"Required model '{model_display_name}' not found. Please make sure your trained models are in the models folder.")
            elif submit and not selected_model:
                col2.error("Please select a model before making a prediction.")
            else:
                col2.info("Click the 'Predict Man-Hours' button to see the estimation result.")

        with tab_viz:
            if submit and selected_model:
                # Show feature importance
                show_feature_importance(selected_model, user_inputs, st)
                
                # Show what-if analysis
                st.subheader("What-If Analysis")
                st.write("See how changing one parameter affects the estimation:")
                
                # Define key parameters for what-if analysis
                key_params = {
                    "Project Year": "project_prf_year_of_project",
                    "Functional Size": "project_prf_functional_size", 
                    "Max Team Size": "project_prf_max_team_size",
                    "Documentation": "process_pmf_docs",
                    "Tools Used": "tech_tf_tools_used",
                    "Personnel Changes": "people_prf_personnel_changes"
                }
                
                what_if_param = st.selectbox(
                    "Select parameter to vary",
                    list(key_params.keys())
                )
                
                param_key = key_params[what_if_param]

                # Create ranges for different parameters
                current_value = user_inputs.get(param_key, 0)
                
                if param_key == "project_prf_year_of_project":
                    what_if_values = np.arange(2015, 2026)
                elif param_key == "project_prf_functional_size":
                    what_if_values = np.linspace(max(1, current_value-500), current_value+500, 11)
                elif param_key == "project_prf_max_team_size":
                    what_if_values = np.linspace(max(1, current_value-5), current_value+5, 11)
                else:
                    # For other numerical parameters, create a reasonable range
                    what_if_values = np.linspace(max(0, current_value-50), current_value+50, 11)
                
                # Get the model
                model = load_model(selected_model)
                
                if model is not None:
                    predictions = []
                    
                    for val in what_if_values:
                        # Create a copy of features and modify the selected parameter
                        modified_features = user_inputs.copy()
                        modified_features[param_key] = val
                        
                        # Make prediction
                        prediction = predict_man_hours(modified_features, selected_model)
                        predictions.append(prediction if prediction is not None else 0)
                    
                    # Filter out None predictions
                    valid_predictions = [(val, pred) for val, pred in zip(what_if_values, predictions) if pred is not None and pred > 0]
                    
                    if valid_predictions:
                        # Create a DataFrame for the results
                        valid_values, valid_preds = zip(*valid_predictions)
                        what_if_df = pd.DataFrame({
                            what_if_param: valid_values,
                            "Estimated Man-Hours": valid_preds
                        })
                        
                        # Plot the results
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(what_if_df[what_if_param], what_if_df["Estimated Man-Hours"], marker='o', linewidth=2, markersize=6)
                        ax.set_xlabel(what_if_param)
                        ax.set_ylabel("Estimated Man-Hours")
                        
                        # Get model display name for title
                        model_display_name = get_model_display_name(selected_model)
                        ax.set_title(f"Impact of {what_if_param} on Estimation\nUsing {model_display_name}")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Add horizontal line for current prediction
                        current_pred = predict_man_hours(user_inputs, selected_model)
                        if current_pred is not None:
                            ax.axhline(y=current_pred, color='r', linestyle='--', 
                                       label=f"Current Estimation ({current_pred:.2f} man-hours)")
                            # Add legend
                            ax.legend()
                        else:
                            st.warning("Current prediction failed, so reference line is not shown.")
                        
                        # Improve layout
                        plt.tight_layout()
                        
                        # Show the plot
                        st.pyplot(fig)
                        
                        # Show the data table
                        st.dataframe(what_if_df.round(2), use_container_width=True)
                        
                        # Add interpretation
                        st.markdown(f"""
                        ### Interpretation

                        The graph above shows how the estimated man-hours change when varying **{what_if_param}** 
                        while keeping all other parameters constant. This analysis was performed using the 
                        **{model_display_name}** model.
                        
                        This can help you understand which factors most strongly influence your project timeline.
                        """)
                    else:
                        st.error("All predictions failed. Please check your model and input parameters.")
                else:
                    model_display_name = get_model_display_name(selected_model) if selected_model else "Unknown"
                    st.error(f"Model '{model_display_name}' not found. Please check your model files.")
            elif not selected_model:
                st.info("Please select a model first to see visualizations and what-if analysis.")
            else:
                st.info("Make a prediction first to see visualizations and what-if analysis.")

        with tab_help:
            # Additional sections
            about_section()
            tips_section()
            show_disclaimer()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in the application:")

# Run the main application
if __name__ == "__main__":
    main()