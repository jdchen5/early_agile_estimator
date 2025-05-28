# ui.py
"""
- Launch the Streamlit App: Run streamlit run main.py and Use the 
  "Check for required models" option to verify your models are detected properly. Should be only model allowed at one time.
- Explore Additional Features: Save/load configurations for frequently used project settings, Use the what-if analysis to 
  see how changing parameters affects estimates and Review feature importance to understand which factors have the biggest impact
"""

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
from datetime import datetime
from models import list_available_models, get_feature_importance, check_required_models, get_model_display_name, get_model_technical_name

def sidebar_inputs():
    """Create input form in the sidebar."""
    # Check if we have any models available
    model_status = check_required_models()
    
    with st.sidebar:
        st.title("Project Estimator")
        
        # Create tabs in the sidebar
        tab1, tab2 = st.tabs(["Basic", "Advanced"])
        
        with tab1:
            with st.form("estimation_form"):
                st.header("Project Information")
                
                complexity = st.slider(
                    "Project Complexity (1-5)", 
                    min_value=1, 
                    max_value=5, 
                    value=3,
                    help="Higher values indicate more complex projects"
                )
                
                team_experience = st.slider(
                    "Team Experience (1-5)", 
                    min_value=1, 
                    max_value=5, 
                    value=3,
                    help="Higher values indicate more experienced teams"
                )
                
                num_requirements = st.number_input(
                    "Number of Requirements", 
                    min_value=1, 
                    max_value=500, 
                    value=20,
                    help="Total number of user stories or requirements"
                )
                
                team_size = st.number_input(
                    "Team Size", 
                    min_value=1, 
                    max_value=100, 
                    value=5,
                    help="Number of full-time team members"
                )
                
                tech_complexity = st.slider(
                    "Technology Stack Complexity (1-5)", 
                    min_value=1, 
                    max_value=5, 
                    value=3,
                    help="Higher values indicate more complex technology stack"
                )

                st.header("Model Selection")
                
                # Check if we have any models available
                if model_status["models_available"]:
                    # Get available models with display names
                    available_models = list_available_models()
                    
                    if available_models:
                        # Create options mapping for selectbox (display_name -> technical_name)
                        model_options = {model['display_name']: model['technical_name'] for model in available_models}
                        
                        # Select model using display names
                        selected_display_name = st.selectbox(
                            "Select Prediction Model", 
                            options=list(model_options.keys()),
                            help="Choose which trained model to use for estimation"
                        )
                        
                        # Get the technical name for the selected display name
                        selected_model = model_options[selected_display_name]
                    else:
                        st.warning("No trained models found. Please add trained models to the 'models' directory.")
                        selected_model = None
                        selected_display_name = None
                else:
                    st.warning("No trained models found. Please create sample models or add trained models to the 'models' directory.")
                    selected_model = None
                    selected_display_name = None
                
                # Buttons
                col1, col2 = st.columns(2)
                submit = col1.form_submit_button("Predict Man-Hours")
                save_config = col2.form_submit_button("Save Config")
        
        with tab2:
            st.header("Saved Configurations")
            configs = load_saved_configurations()
            
            if configs:
                selected_config = st.selectbox(
                    "Choose a saved configuration",
                    options=list(configs.keys()),
                    format_func=lambda x: f"{x} ({configs[x]['date']})"
                )
                
                load_config = st.button("Load Selected Config")
                if load_config and selected_config:
                    st.session_state.config_to_load = selected_config
                    st.rerun()
            else:
                st.info("No saved configurations found. You can save configurations in the Basic tab.")
                
            # Add model information
            st.header("Model Information")
            if model_status["models_available"]:
                available_models = list_available_models()
                model_display_names = [model['display_name'] for model in available_models]
                st.success(f"Found {len(available_models)} trained models:")
                for model in available_models:
                    st.write(f"• {model['display_name']}")
                
                if model_status["scaler_available"]:
                    st.info("Feature scaler is available for normalization.")
                else:
                    st.warning("No feature scaler found. Models will use raw feature values.")
            else:
                st.error("No trained models found. Please create sample models or add trained models to the 'models' directory.")
                
            # Option to view models folder
            if st.button("Check for Required Models"):
                st.session_state.check_models = True
                st.rerun()
    
    # Handle loading saved configuration
    if hasattr(st.session_state, 'config_to_load'):
        config_name = st.session_state.config_to_load
        loaded_config = load_configuration(config_name)
        if loaded_config:
            complexity = loaded_config.get('complexity', 3)
            team_experience = loaded_config.get('team_experience', 3)
            num_requirements = loaded_config.get('num_requirements', 20)
            team_size = loaded_config.get('team_size', 5)
            tech_complexity = loaded_config.get('tech_complexity', 3)
            saved_model = loaded_config.get('selected_model')
            
            # Make sure the selected model is available
            if model_status["models_available"]:
                available_technical_names = [model['technical_name'] for model in list_available_models()]
                if saved_model in available_technical_names:
                    selected_model = saved_model
                    selected_display_name = get_model_display_name(saved_model)
                else:
                    # Use first available model if saved model is not available
                    available_models = list_available_models()
                    if available_models:
                        selected_model = available_models[0]['technical_name']
                        selected_display_name = available_models[0]['display_name']
                        st.warning(f"The model specified in the configuration is not available. Using {selected_display_name} instead.")
            
            # Clear the session state to prevent reloading
            del st.session_state.config_to_load
            
            # Show success message
            st.success(f"Configuration '{config_name}' loaded successfully!")
    
    # Check if models should be displayed
    if hasattr(st.session_state, 'check_models'):
        st.subheader("Available Models")
        if model_status["models_available"]:
            available_models = list_available_models()
            for model in available_models:
                st.write(f"✅ {model['display_name']} ({model['technical_name']})")
        else:
            st.error("No models found in the 'models' directory.")
        
        # Display scaler information
        if model_status["scaler_available"]:
            st.write("✅ Feature scaler available")
        else:
            st.write("❌ No feature scaler found")
        
        # Remove flag from session state
        del st.session_state.check_models
    
    # Handle saving configuration
    if save_config and selected_model:
        save_current_configuration(
            complexity, team_experience, num_requirements, 
            team_size, tech_complexity, selected_model
        )

    return complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model, submit

def save_current_configuration(complexity, team_experience, num_requirements, 
                              team_size, tech_complexity, selected_model):
    """Save current configuration to a file."""
    config_name = st.text_input("Enter a name for this configuration:")
    
    if not config_name:
        st.warning("Please enter a name for your configuration.")
        return
    
    config = {
        'complexity': complexity,
        'team_experience': team_experience,
        'num_requirements': num_requirements,
        'team_size': team_size,
        'tech_complexity': tech_complexity,
        'selected_model': selected_model,  # Save technical name
        'date': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # Make sure the configs directory exists
    if not os.path.exists('configs'):
        os.makedirs('configs')
    
    # Save to file
    with open(f'configs/{config_name}.json', 'w') as f:
        json.dump(config, f)
    
    st.success(f"Configuration '{config_name}' saved successfully!")

def load_saved_configurations():
    """Load all saved configurations."""
    configs = {}
    
    if not os.path.exists('configs'):
        return configs
    
    for filename in os.listdir('configs'):
        if filename.endswith('.json'):
            config_name = os.path.splitext(filename)[0]
            try:
                with open(f'configs/{filename}', 'r') as f:
                    config = json.load(f)
                configs[config_name] = config
            except:
                pass
    
    return configs

def load_configuration(config_name):
    """Load a specific configuration by name."""
    try:
        with open(f'configs/{config_name}.json', 'r') as f:
            return json.load(f)
    except:
        st.error(f"Failed to load configuration '{config_name}'")
        return None

def display_inputs(complexity, team_experience, num_requirements, team_size, tech_complexity, selected_model):
    """Display the input parameters in a formatted way."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Create a more visually appealing display for inputs
        input_df = pd.DataFrame({
            "Parameter": ["Project Complexity", "Team Experience", "Number of Requirements", 
                         "Team Size", "Technology Stack Complexity"],
            "Value": [complexity, team_experience, num_requirements, team_size, tech_complexity],
            "Description": [
                "Higher value = more complex",
                "Higher value = more experienced",
                "Total count of requirements",
                "Number of team members",
                "Higher value = more complex tech"
            ]
        })
        
        # Format the table for better readability
        st.dataframe(input_df, use_container_width=True)
        
        # Display the selected model with icon and friendly name
        if selected_model:
            model_display_name = get_model_display_name(selected_model)
            st.write(f"📊 Selected Model: **{model_display_name}**")
        else:
            st.write("📊 Selected Model: **None**")
    
    return col2

def show_feature_importance(selected_model, features_dict, st):
    """Display feature importance if available."""
    if not selected_model:
        st.info("No model selected for feature importance analysis.")
        return
        
    feature_importance = get_feature_importance(selected_model)
    
    if feature_importance is not None:
        st.subheader("Feature Importance")
        
        # Get feature names from the features_dict keys
        feature_names = list(features_dict.keys())
        
        # Map technical feature names to user-friendly names
        feature_name_mapping = {
            "project_prf_complexity": "Project Complexity",
            "project_prf_team_experience": "Team Experience",
            "project_prf_num_requirements": "Number of Requirements",
            "project_prf_max_team_size": "Team Size",
            "tech_tf_tech_complexity": "Technology Stack Complexity"
        }
        
        # Create friendly names list
        friendly_names = [feature_name_mapping.get(name, name) for name in feature_names]
        
        # Ensure we have the right number of importance values
        if len(feature_importance) >= len(friendly_names):
            importance_values = feature_importance[:len(friendly_names)]
        else:
            # Pad with zeros if we have fewer importance values than features
            importance_values = list(feature_importance) + [0] * (len(friendly_names) - len(feature_importance))
        
        # Create a DataFrame for the feature importance values
        importance_df = pd.DataFrame({
            'Feature': friendly_names,
            'Importance': np.abs(importance_values)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:  # Only add label if there's a value
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                       va='center')
        
        ax.set_xlabel('Relative Importance')
        ax.set_title(f'Feature Importance - {get_model_display_name(selected_model)}')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Display the data table
        st.dataframe(importance_df.round(4), use_container_width=True)
    else:
        model_display_name = get_model_display_name(selected_model)
        st.info(f"Feature importance is not available for {model_display_name}. This might be because the model doesn't support feature importance or there was an error retrieving it.")

def show_prediction(col2, prediction, team_size):
    """Display the prediction results in a visually appealing way."""
    with col2:
        st.subheader("Prediction Result")
        
        if prediction is None:
            st.error("Failed to make prediction. Please check logs for details.")
            return
        
        # Create a visually appealing display for the prediction
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='text-align:center;'>Estimated Effort</h3>
            <h1 style='text-align:center; color:#1f77b4; font-size:2.5rem;'>{prediction:.2f} Man-Hours</h1>
        </div>
        """, unsafe_allow_html=True)

        # Calculate calendar time and per-person effort
        hours = int(prediction)
        days = hours // 8  # Assuming 8-hour workdays
        per_person = prediction / team_size

        # Display additional metrics
        st.markdown("### Timeline Breakdown")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric(
                label="Calendar Time", 
                value=f"{days}d",
                help="Estimated calendar duration assuming full team availability"
            )
        
        with metrics_col2:
            st.metric(
                label="Per Person", 
                value=f"{per_person:.2f}h",
                help="Average effort per team member in hours"
            )
        
        # Display warning for potentially inaccurate predictions
        if prediction < 1:
            st.warning("This prediction seems unusually low. Consider reviewing your inputs.")
        elif prediction > 10000:
            st.warning("This prediction seems unusually high. Consider reviewing your inputs.")

def about_section():
    """Display information about the application."""
    st.markdown("---")
    
    with st.expander("About this Estimator", expanded=False):
        st.subheader("Machine Learning for Agile Project Estimation")
        
        st.write("""
        This application uses machine learning models to predict the effort required for agile projects.
        The models have been trained on historical project data to provide early estimations based on
        key project parameters. These estimations can help in project planning and resource allocation.
        
        ### How it Works
        
        The estimator uses the following input parameters:
        
        - **Project Complexity**: Overall complexity of the project scope
        - **Team Experience**: Experience level of the team with similar projects
        - **Number of Requirements**: Count of user stories or requirements
        - **Team Size**: Number of full-time team members
        - **Technology Stack Complexity**: Complexity of the technology being used
        
        The selected machine learning model processes these inputs to predict the required effort in man-hours.
        """)

def tips_section():
    """Display tips for accurate estimation."""
    with st.expander("Tips for Accurate Estimation", expanded=False):
        st.markdown("""
        ### Tips for Getting Accurate Estimations

        1. **Project Complexity**
           - Rate 1-2 for simple projects with well-understood requirements
           - Rate 3 for moderate complexity with some uncertainty
           - Rate 4-5 for highly complex projects with significant unknowns

        2. **Team Experience**
           - Rate 1-2 for teams new to the domain or technology
           - Rate 3 for teams with moderate experience in similar projects
           - Rate 4-5 for highly experienced teams who have done similar work

        3. **Requirements Analysis**
           - Count only well-defined requirements
           - Break down epics into smaller stories when possible
           - Consider using story points as a proxy for requirements count

        4. **Team Size Considerations**
           - Larger teams may increase coordination overhead
           - Consider the "mythical man-month" effect
           - Ensure your team size is appropriate for the project scope

        5. **Technology Complexity**
           - Rate 1-2 for familiar, stable technology stacks
           - Rate 3 for mixed familiar/new technologies
           - Rate 4-5 for cutting-edge or highly specialized technologies
        """)

        st.info("Remember that these estimations are meant to be starting points. Always review and adjust based on your team's specific context and historical performance.")