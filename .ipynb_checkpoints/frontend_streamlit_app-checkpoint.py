# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set page title and configure layout
st.set_page_config(page_title="Agile Project Estimator", layout="wide")

# Application title
st.title("Machine Learning for Early Estimation in Agile Projects")
st.markdown("---")

# Create sample models (in a real application, you'd load pre-trained models)
def create_sample_models():
    # Create a directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create and save sample models if they don't exist
    if not os.path.exists('models/linear_regression.pkl'):
        X = np.random.rand(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5 * X[:, 3] + 0.5 * X[:, 4] + np.random.randn(100) * 0.5
        
        # Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        with open('models/linear_regression.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        
        # Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

# Function to load models
def load_model(model_name):
    with open(f'models/{model_name}.pkl', 'rb') as f:
        return pickle.load(f)

# Create sample models
create_sample_models()

# Function to make predictions
def predict_man_months(features, model_name):
    # Load the selected model and scaler
    model = load_model(model_name)
    scaler = load_model('scaler')
    
    # Scale the features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Ensure prediction is not negative
    return max(0, prediction)

# Sidebar for project information input
st.sidebar.header("Project Information")

# Project complexity (1-5)
complexity = st.sidebar.slider(
    "Project Complexity (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    help="How complex is the project? 1 = Simple, 5 = Very Complex"
)

# Team experience (1-5)
team_experience = st.sidebar.slider(
    "Team Experience (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    help="How experienced is your team? 1 = Beginner, 5 = Expert"
)

# Number of requirements
num_requirements = st.sidebar.number_input(
    "Number of Requirements",
    min_value=1,
    max_value=500,
    value=20,
    help="How many requirements are there in the project?"
)

# Team size
team_size = st.sidebar.number_input(
    "Team Size",
    min_value=1,
    max_value=100,
    value=5,
    help="How many team members will work on this project?"
)

# Technology stack complexity (1-5)
tech_complexity = st.sidebar.slider(
    "Technology Stack Complexity (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    help="How complex is the technology stack? 1 = Simple, 5 = Very Complex"
)

# Model selection
st.sidebar.header("Model Selection")
model_options = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor"
}
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# Prediction button
predict_button = st.sidebar.button("Predict Man-Months", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Parameters")
    
    # Display the input parameters in a table
    input_data = {
        "Parameter": ["Project Complexity", "Team Experience", "Number of Requirements", 
                     "Team Size", "Technology Stack Complexity"],
        "Value": [complexity, team_experience, num_requirements, team_size, tech_complexity]
    }
    input_df = pd.DataFrame(input_data)
    st.table(input_df)

    # Show the selected model
    st.write(f"Selected Model: **{model_options[selected_model]}**")

with col2:
    st.subheader("Prediction Result")
    
    if predict_button:
        # Create a features array from inputs
        features = np.array([complexity, team_experience, num_requirements, team_size, tech_complexity])
        
        # Make the prediction
        prediction = predict_man_months(features, selected_model)
        
        # Display the prediction
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
            <h3 style='text-align:center;'>Estimated Effort</h3>
            <h1 style='text-align:center; color:#1f77b4;'>{prediction:.2f} Man-Months</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Display equivalent time representations
        months = int(prediction)
        remaining_days = int((prediction - months) * 30)
        
        st.write(f"⏱️ Approximately: {months} months and {remaining_days} days")
        st.write(f"👨‍💻 With team size of {team_size}: {(prediction / team_size):.2f} months per person")
    else:
        st.info("Click the 'Predict Man-Months' button to see the estimation result.")

# Add information about the application
st.markdown("---")
st.subheader("About this Estimator")
st.write("""
This application uses machine learning models to predict the effort required for agile projects.
The models have been trained on historical project data to provide early estimations based on
key project parameters. These estimations can help in project planning and resource allocation.
""")

# Tips for accurate estimation
st.subheader("Tips for Accurate Estimation")
st.write("""
- Be realistic about project complexity
- Consider team experience with similar projects
- Break down requirements to ensure accurate counting
- Factor in technical complexity appropriately
- Validate predictions against your historical data
""")