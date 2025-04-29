# ui.py
import streamlit as st
import numpy as np
import pandas as pd
from models import load_model, predict_man_months

model_options = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor"
}

def sidebar_inputs():
    with st.sidebar.form("estimation_form"):
        st.header("Project Information")

        complexity = st.slider("Project Complexity (1-5)", 1, 5, 3)
        team_experience = st.slider("Team Experience (1-5)", 1, 5, 3)
        num_requirements = st.number_input("Number of Requirements", 1, 500, 20)
        team_size = st.number_input("Team Size", 1, 100, 5)
        tech_complexity = st.slider("Technology Stack Complexity (1-5)", 1, 5, 3)

        st.header("Model Selection")
        selected_model = st.selectbox("Select Prediction Model", list(model_options.keys()), 
                                      format_func=lambda x: model_options[x])

        create_models = st.checkbox("Create sample models (for testing)")
        submit = st.form_submit_button("Predict Man-Months")

    return complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model, create_models, submit

def display_inputs(complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Input Parameters")
        input_df = pd.DataFrame({
            "Parameter": ["Project Complexity", "Team Experience", "Number of Requirements", 
                          "Team Size", "Technology Stack Complexity"],
            "Value": [complexity, team_experience, num_requirements, team_size, tech_complexity]
        })
        st.table(input_df)
        st.write(f"Selected Model: **{model_options[selected_model]}**")
    return col2

def show_prediction(col2, prediction, team_size):
    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
            <h3 style='text-align:center;'>Estimated Effort</h3>
            <h1 style='text-align:center; color:#1f77b4;'>{prediction:.2f} Man-Months</h1>
        </div>
        """, unsafe_allow_html=True)

        months = int(prediction)
        remaining_days = int((prediction - months) * 30)

        st.write(f"⏱️ Approximately: {months} months and {remaining_days} days")
        st.write(f"👨‍💻 With team size of {team_size}: {(prediction / team_size):.2f} months per person")

def about_section():
    st.markdown("---")
    st.subheader("About this Estimator")
    st.write("""
    This application uses machine learning models to predict the effort required for agile projects.
    The models have been trained on historical project data to provide early estimations based on
    key project parameters. These estimations can help in project planning and resource allocation.
    """)

def tips_section():
    st.subheader("Tips for Accurate Estimation")
    st.write("""
    - Be realistic about project complexity
    - Consider team experience with similar projects
    - Break down requirements to ensure accurate counting
    - Factor in technical complexity appropriately
    - Validate predictions against your historical data
    """)
