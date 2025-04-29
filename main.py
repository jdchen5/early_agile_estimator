# main.py
import streamlit as st
import numpy as np
from models import create_sample_models, load_model, predict_man_months
from ui import sidebar_inputs, display_inputs, show_prediction, about_section, tips_section

st.set_page_config(page_title="Agile Project Estimator", layout="wide")
st.title("Machine Learning for Early Estimation in Agile Projects")
st.markdown("---")

# Sidebar form input
(complexity, team_experience, num_requirements, team_size, 
 tech_complexity, selected_model, create_models, submit) = sidebar_inputs()

if create_models:
    create_sample_models()

# Display input parameters
col2 = display_inputs(complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model)

# Perform prediction
if submit:
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

# Additional sections
about_section()
tips_section()
