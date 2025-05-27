# main.py


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from models import load_model, predict_man_hours, get_feature_importance
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
        # Sidebar form inputs
        (complexity, team_experience, num_requirements, team_size, 
         tech_complexity, selected_model, submit) = sidebar_inputs()

        # Make a features dict -- KEYS MUST MATCH your model's CSV column names!
        features = {
            "project_prf_complexity": complexity,
            "project_prf_team_experience": team_experience,
            "project_prf_num_requirements": num_requirements,
            "project_prf_max_team_size": team_size,
            "tech_tf_tech_complexity": tech_complexity
        }
        
        # Create tabs for different sections
        tab_results, tab_viz, tab_help = st.tabs(["Estimation Results", "Visualization", "Help & Documentation"])

        with tab_results:
            # Display input parameters
            col2 = display_inputs(complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model)
            
            # Perform prediction
            if submit:
                with st.spinner("Calculating estimation..."):
                    model = load_model(selected_model)
                    scaler = load_model("scaler")

                    print("Loaded model type:", type(model))   # <-- for debugging
                    if scaler is not None:
                        print("Scaler loaded successfully.")
                    else:
                        print("Scaler not found, using default scaling.")
                    print("Loaded scaler type:", type(scaler)) # <-- for debugging

                    if model is not None:
                        features = np.array([complexity, team_experience, num_requirements, team_size, tech_complexity])
                        prediction = predict_man_hours(features, selected_model)
                        show_prediction(col2, prediction, team_size)
                    else:
                        st.error("Required model not found. Please make sure your trained models are in the models folder.")
            else:
                col2.info("Click the 'Predict Man-Hours' button to see the estimation result.")

        with tab_viz:
            if submit:
                # Show feature importance
                features = np.array([complexity, team_experience, num_requirements, team_size, tech_complexity])
                show_feature_importance(selected_model, features, st)
                
                # Show what-if analysis
                st.subheader("What-If Analysis")
                st.write("See how changing one parameter affects the estimation:")
                
                what_if_param = st.selectbox(
                    "Select parameter to vary",
                    ["Project Complexity", "Team Experience", "Number of Requirements", "Team Size", "Technology Stack Complexity"]
                )
                
                param_key_map = {
                    "Project Complexity": "project_prf_complexity",
                    "Team Experience": "project_prf_team_experience",
                    "Number of Requirements": "project_prf_num_requirements",
                    "Team Size": "project_prf_max_team_size",
                    "Technology Stack Complexity": "tech_tf_tech_complexity"
                }
                param_key = param_key_map[what_if_param]

                # Create ranges for different parameters
                ranges = {
                    "project_prf_complexity": np.linspace(1, 5, 5),
                    "project_prf_team_experience": np.linspace(1, 5, 5),
                    "project_prf_num_requirements": np.linspace(max(1, num_requirements-50), num_requirements+50, 11),
                    "project_prf_max_team_size": np.linspace(max(1, team_size-5), team_size+5, 11),
                    "tech_tf_tech_complexity": np.linspace(1, 5, 5)
                }
                what_if_values = ranges[param_key]
                
                # Get the model and scaler
                model = load_model(selected_model)
                scaler = load_model(MODEL_SCALER)
                
                if model is not None:
                    # Make predictions for each value in the range
                    what_if_values = ranges[param_key]
                    predictions = []
                    
                    for val in what_if_values:
                        # Create a copy of features and modify the selected parameter
                        modified_features = features.copy()
                        modified_features[param_key] = val
                        
                        # Make prediction
                        prediction = predict_man_hours(modified_features, selected_model)
                        predictions.append(prediction if prediction is not None else 0)
                    
                    # Filter out None predictions
                    valid_predictions = [(val, pred) for val, pred in zip(what_if_values, predictions) if pred is not None]
                    
                    if valid_predictions:
                        # Create a DataFrame for the results
                        valid_values, valid_preds = zip(*valid_predictions)
                        what_if_df = pd.DataFrame({
                            what_if_param: valid_values,
                            "Estimated Man-Hours": valid_preds
                        })
                        
                        # Plot the results
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(what_if_df[what_if_param], what_if_df["Estimated Man-Months"], marker='o')
                        ax.set_xlabel(what_if_param)
                        ax.set_ylabel("Estimated Man-Months")
                        ax.set_title(f"Impact of {what_if_param} on Estimation")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Add horizontal line for current prediction
                        current_pred = predict_man_hours(features, selected_model)
                        if current_pred is not None:
                            ax.axhline(y=current_pred, color='r', linestyle='--', 
                                       label=f"Current Estimation ({current_pred:.2f} man-hours)")
                            # Add legend
                            ax.legend()
                        else:
                            st.warning("Current prediction failed, so reference line is not shown.")
                        
                        # Show the plot
                        st.pyplot(fig)
                        
                        # Show the data table
                        st.dataframe(what_if_df.round(2), use_container_width=True)
                        
                        # Add interpretation
                        st.markdown(f"""
                        ### Interpretation

                        The graph above shows how the estimated man-months change when varying **{what_if_param}** 
                        while keeping all other parameters constant. This can help you understand which factors 
                        most strongly influence your project timeline.
                        """)
                    else:
                        st.error("All predictions failed. Please check your model and input parameters.")
                else:
                    st.error("Model not found. Please check your model files.")
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