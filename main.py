# main.py


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from models import create_sample_models, load_model, predict_man_months, get_feature_importance
from ui import sidebar_inputs, display_inputs, show_prediction, about_section, tips_section, show_feature_importance

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
        # Sidebar form input
        (complexity, team_experience, num_requirements, team_size, 
         tech_complexity, selected_model, create_models, submit) = sidebar_inputs()

        # Create sample models if requested
        if create_models:
            with st.spinner("Creating sample models..."):
                if create_sample_models():
                    st.success("Sample models created successfully!")
                else:
                    st.error("Failed to create sample models. Check logs for details.")

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

                    if model is not None and scaler is not None:
                        features = np.array([complexity, team_experience, num_requirements, team_size, tech_complexity])
                        prediction = predict_man_months(features, model, scaler)
                        show_prediction(col2, prediction, team_size)
                    else:
                        st.error("Required model or scaler not found. Try creating sample models first.")
            else:
                col2.info("Click the 'Predict Man-Months' button to see the estimation result.")

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
                
                param_index = {
                    "Project Complexity": 0,
                    "Team Experience": 1,
                    "Number of Requirements": 2,
                    "Team Size": 3,
                    "Technology Stack Complexity": 4
                }[what_if_param]
                
                # Create ranges for different parameters
                ranges = {
                    0: np.linspace(1, 5, 5),  # Project Complexity
                    1: np.linspace(1, 5, 5),  # Team Experience
                    2: np.linspace(max(1, num_requirements-50), num_requirements+50, 11),  # Number of Requirements
                    3: np.linspace(max(1, team_size-5), team_size+5, 11),  # Team Size
                    4: np.linspace(1, 5, 5)   # Tech Complexity
                }
                
                # Get the model and scaler
                model = load_model(selected_model)
                scaler = load_model("scaler")
                
                if model is not None and scaler is not None:
                    # Make predictions for each value in the range
                    what_if_values = ranges[param_index]
                    predictions = []
                    
                    for val in what_if_values:
                        # Create a copy of features and modify the selected parameter
                        modified_features = features.copy()
                        modified_features[param_index] = val
                        
                        # Make prediction
                        prediction = predict_man_months(modified_features, model, scaler)
                        predictions.append(prediction)
                    
                    # Create a DataFrame for the results
                    what_if_df = pd.DataFrame({
                        what_if_param: what_if_values,
                        "Estimated Man-Months": predictions
                    })
                    
                    # Plot the results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(what_if_df[what_if_param], what_if_df["Estimated Man-Months"], marker='o')
                    ax.set_xlabel(what_if_param)
                    ax.set_ylabel("Estimated Man-Months")
                    ax.set_title(f"Impact of {what_if_param} on Estimation")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add horizontal line for current prediction
                    ax.axhline(y=predict_man_months(features, model, scaler), color='r', linestyle='--', 
                               label=f"Current Estimation ({predict_man_months(features, model, scaler):.2f} man-months)")
                    
                    # Add legend
                    ax.legend()
                    
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
                    st.warning("Please make a prediction first to see visualizations.")
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