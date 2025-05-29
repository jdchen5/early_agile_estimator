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
        tab1, tab2, tab3 = st.tabs(["Basic", "Technical", "Advanced"])
        
        with tab1:
            with st.form("estimation_form"):
                st.header("Project Information")
                
                # Basic project parameters
                project_year = st.number_input(
                    "Project Year", 
                    min_value=2015, 
                    max_value=2030, 
                    value=2024,
                    help="Year when the project is being developed"
                )
                
                industry_sector = st.selectbox(
                    "Industry Sector",
                    options=["Banking", "Insurance", "Manufacturing", "Government", "Healthcare", "Telecommunications", "Other"],
                    index=0,
                    help="Industry sector of the organization"
                )
                
                organisation_type = st.selectbox(
                    "Organisation Type",
                    options=["Commercial", "Government", "Non-Profit", "Academic"],
                    index=0,
                    help="Type of organization developing the project"
                )
                
                application_type = st.selectbox(
                    "Application Type",
                    options=["Management Information System", "Process Control", "Scientific", "Embedded", "Web Application", "Mobile Application"],
                    index=0,
                    help="Primary type of application being developed"
                )
                
                functional_size = st.number_input(
                    "Functional Size (Function Points)", 
                    min_value=1, 
                    max_value=10000, 
                    value=100,
                    help="Size of the application in function points"
                )
                
                max_team_size = st.number_input(
                    "Maximum Team Size", 
                    min_value=1, 
                    max_value=100, 
                    value=5,
                    help="Maximum number of people on the team during development"
                )
                
                # Application group (radio buttons for mutually exclusive options)
                st.subheader("Application Group")
                application_group = st.radio(
                    "Primary Application Category",
                    options=["Business Application", "Infrastructure Software", "Mathematically Intensive", "Real-time Application"],
                    index=0,
                    help="Primary category that best describes your application"
                )
                
                # Development type
                development_type = st.radio(
                    "Development Type",
                    options=["New Development", "Re-development"],
                    index=0,
                    help="Whether this is new development or redevelopment of existing system"
                )
                
                # Relative size
                relative_size = st.selectbox(
                    "Relative Project Size",
                    options=["XXS (Extra Extra Small)", "XS (Extra Small)", "S (Small)", "M1 (Medium 1)", "M2 (Medium 2)", "L (Large)"],
                    index=2,
                    help="Relative size of the project compared to other projects in your organization"
                )

        with tab2:
            st.header("Technical Information")
            
            # Development platform
            development_platform = st.selectbox(
                "Development Platform",
                options=["PC", "Multi-platform", "Mainframe (MR)", "Proprietary"],
                index=0,
                help="Primary development platform"
            )
            
            # Programming language type
            language_type = st.selectbox(
                "Programming Language Type",
                options=["3GL (Third Generation)", "4GL (Fourth Generation)", "5GL (Fifth Generation)"],
                index=0,
                help="Generation of primary programming language"
            )
            
            # Primary programming language
            primary_language = st.selectbox(
                "Primary Programming Language",
                options=["Java", "C", "C*", "JavaScript", "ABAP", "Oracle", "PL/I", "Proprietary Agile Platform"],
                index=0,
                help="Main programming language used in development"
            )
            
            # Architecture
            architecture = st.selectbox(
                "System Architecture",
                options=["Stand-alone", "Client-Server", "Multi-tier with Web Interface"],
                index=1,
                help="Primary system architecture"
            )
            
            # Client-server
            client_server = st.radio(
                "Client-Server Architecture",
                options=["Yes", "No"],
                index=0,
                help="Does the system use client-server architecture?"
            )
            
            # Web development
            web_development = st.radio(
                "Web Development",
                options=["Yes", "No"],
                index=0,
                help="Is this a web-based application?"
            )
            
            # DBMS used
            dbms_used = st.radio(
                "Database Management System Used",
                options=["Yes", "No"],
                index=0,
                help="Does the project use a database management system?"
            )
            
            # Tools used
            tools_used = st.slider(
                "Development Tools Sophistication (1-5)", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Level of sophistication of development tools used"
            )

        with tab3:
            st.header("Process & People")
            
            # Documentation
            docs = st.slider(
                "Documentation Level (1-5)", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Level of documentation required/produced"
            )
            
            # Personnel changes
            personnel_changes = st.slider(
                "Personnel Changes (1-5)", 
                min_value=1, 
                max_value=5, 
                value=2,
                help="Expected level of personnel turnover during project"
            )
            
            # Development methodology
            development_methodology = st.selectbox(
                "Development Methodology",
                options=[
                    "Agile Development/Iterative", 
                    "Agile Development/Scrum",
                    "Agile Development/JAD/Multifunctional Teams",
                    "Agile Development/PSP/Unified Process",
                    "Agile Development/Unified Process"
                ],
                index=1,
                help="Primary development methodology used"
            )
            
            # Team size group (automatically determined but can be overridden)
            team_size_group_options = ["2", "3-4", "5-8", "9-14", "21-30", "41-50", "61-70", "Missing"]
            
            # Auto-determine team size group based on max_team_size
            if max_team_size == 2:
                default_team_group_idx = 0
            elif 3 <= max_team_size <= 4:
                default_team_group_idx = 1
            elif 5 <= max_team_size <= 8:
                default_team_group_idx = 2
            elif 9 <= max_team_size <= 14:
                default_team_group_idx = 3
            elif 21 <= max_team_size <= 30:
                default_team_group_idx = 4
            elif 41 <= max_team_size <= 50:
                default_team_group_idx = 5
            elif 61 <= max_team_size <= 70:
                default_team_group_idx = 6
            else:
                default_team_group_idx = 7  # Missing
            
            team_size_group = st.selectbox(
                "Team Size Group",
                options=team_size_group_options,
                index=default_team_group_idx,
                help="Categorical grouping of team size (auto-determined but can be overridden)"
            )
            
            # Cost currency
            cost_currency = st.selectbox(
                "Cost Currency",
                options=["US Dollar", "Canadian Dollar", "European Euro"],
                index=0,
                help="Currency used for project costing"
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
        
        # Additional tabs for configurations and model info
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
            st.info("No saved configurations found. You can save configurations in the tabs above.")
            
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
    
    # Create feature dictionary for the model
    user_inputs = create_feature_dict(
        project_year, industry_sector, organisation_type, application_type,
        functional_size, max_team_size, docs, tools_used, personnel_changes,
        application_group, development_type, development_platform, language_type,
        primary_language, relative_size, team_size_group, development_methodology,
        architecture, client_server, web_development, dbms_used, cost_currency,
        selected_model, submit
    )
    
    # Handle loading saved configuration
    if hasattr(st.session_state, 'config_to_load'):
        config_name = st.session_state.config_to_load
        loaded_config = load_configuration(config_name)
        if loaded_config:
            # Update user_inputs with loaded values
            user_inputs.update(loaded_config)
            
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
        save_current_configuration(user_inputs)

    return user_inputs

def create_feature_dict(project_year, industry_sector, organisation_type, application_type,
                       functional_size, max_team_size, docs, tools_used, personnel_changes,
                       application_group, development_type, development_platform, language_type,
                       primary_language, relative_size, team_size_group, development_methodology,
                       architecture, client_server, web_development, dbms_used, cost_currency,
                       selected_model, submit):
    """Create a dictionary with all features required by the model."""
    
    # Initialize all features to 0 (for one-hot encoded features)
    features = {
        'project_prf_year_of_project': project_year,
        'external_eef_industry_sector': industry_sector,
        'external_eef_organisation_type': organisation_type,
        'project_prf_application_type': application_type,
        'project_prf_functional_size': functional_size,
        'project_prf_max_team_size': max_team_size,
        'process_pmf_docs': docs,
        'tech_tf_tools_used': tools_used,
        'people_prf_personnel_changes': personnel_changes,
        
        # Application group (one-hot encoded)
        'project_prf_application_group_business_application': 1 if application_group == "Business Application" else 0,
        'project_prf_application_group_infrastructure_software': 1 if application_group == "Infrastructure Software" else 0,
        'project_prf_application_group_mathematically_intensive_application': 1 if application_group == "Mathematically Intensive" else 0,
        'project_prf_application_group_real_time_application': 1 if application_group == "Real-time Application" else 0,
        
        # Development type (one-hot encoded)
        'project_prf_development_type_new_development': 1 if development_type == "New Development" else 0,
        'project_prf_development_type_re_development': 1 if development_type == "Re-development" else 0,
        
        # Development platform (one-hot encoded)
        'tech_tf_development_platform_mr': 1 if development_platform == "Mainframe (MR)" else 0,
        'tech_tf_development_platform_multi': 1 if development_platform == "Multi-platform" else 0,
        'tech_tf_development_platform_pc': 1 if development_platform == "PC" else 0,
        'tech_tf_development_platform_proprietary': 1 if development_platform == "Proprietary" else 0,
        
        # Language type (one-hot encoded)
        'tech_tf_language_type_4GL': 1 if language_type == "4GL (Fourth Generation)" else 0,
        'tech_tf_language_type_5GL': 1 if language_type == "5GL (Fifth Generation)" else 0,
        
        # Primary programming language (one-hot encoded)
        'tech_tf_primary_programming_language_abap': 1 if primary_language == "ABAP" else 0,
        'tech_tf_primary_programming_language*c*': 1 if primary_language == "C*" else 0,
        'tech_tf_primary_programming_language_c': 1 if primary_language == "C" else 0,
        'tech_tf_primary_programming_language_java': 1 if primary_language == "Java" else 0,
        'tech_tf_primary_programming_language_javascript': 1 if primary_language == "JavaScript" else 0,
        'tech_tf_primary_programming_language_oracle': 1 if primary_language == "Oracle" else 0,
        'tech_tf_primary_programming_language_pl_i': 1 if primary_language == "PL/I" else 0,
        'tech_tf_primary_programming_language_proprietary_agile_platform': 1 if primary_language == "Proprietary Agile Platform" else 0,
        
        # Relative size (one-hot encoded)
        'project_prf_relative_size_l': 1 if relative_size.startswith("L") else 0,
        'project_prf_relative_size_m1': 1 if relative_size.startswith("M1") else 0,
        'project_prf_relative_size_m2': 1 if relative_size.startswith("M2") else 0,
        'project_prf_relative_size_s': 1 if relative_size.startswith("S") else 0,
        'project_prf_relative_size_xs': 1 if relative_size.startswith("XS") and not relative_size.startswith("XXS") else 0,
        'project_prf_relative_size_xxs': 1 if relative_size.startswith("XXS") else 0,
        
        # Team size group (one-hot encoded)
        'project_prf_team_size_group_2': 1 if team_size_group == "2" else 0,
        'project_prf_team_size_group_21_30': 1 if team_size_group == "21-30" else 0,
        'project_prf_team_size_group_3_4': 1 if team_size_group == "3-4" else 0,
        'project_prf_team_size_group_41_50': 1 if team_size_group == "41-50" else 0,
        'project_prf_team_size_group_5_8': 1 if team_size_group == "5-8" else 0,
        'project_prf_team_size_group_61_70': 1 if team_size_group == "61-70" else 0,
        'project_prf_team_size_group_9_14': 1 if team_size_group == "9-14" else 0,
        'project_prf_team_size_group_Missing': 1 if team_size_group == "Missing" else 0,
        
        # Development methodologies (one-hot encoded)
        'process_pmf_development_methodologies_agile_developmentiterative': 1 if development_methodology == "Agile Development/Iterative" else 0,
        'process_pmf_development_methodologies_agile_developmentjoint_application_development_jadmultifunctional_teams': 1 if development_methodology == "Agile Development/JAD/Multifunctional Teams" else 0,
        'process_pmf_development_methodologies_agile_developmentpersonal_software_process_pspunified_process': 1 if development_methodology == "Agile Development/PSP/Unified Process" else 0,
        'process_pmf_development_methodologies_agile_developmentscrum': 1 if development_methodology == "Agile Development/Scrum" else 0,
        'process_pmf_development_methodologies_agile_developmentunified_process': 1 if development_methodology == "Agile Development/Unified Process" else 0,
        
        # Architecture (one-hot encoded)
        'tech_tf_architecture_client_server': 1 if architecture == "Client-Server" else 0,
        'tech_tf_architecture_multi_tier_with_web_public_interface': 1 if architecture == "Multi-tier with Web Interface" else 0,
        'tech_tf_architecture_stand_alone': 1 if architecture == "Stand-alone" else 0,
        
        # Client-server (one-hot encoded)
        'tech_tf_client_server_no': 1 if client_server == "No" else 0,
        'tech_tf_client_server_yes': 1 if client_server == "Yes" else 0,
        
        # Web development (one-hot encoded)
        'tech_tf_web_development_tech_tf_web_development': 1 if web_development == "Yes" else 0,
        'tech_tf_web_development_web': 1 if web_development == "Yes" else 0,
        
        # DBMS used (one-hot encoded)
        'tech_tf_dbms_used_tech_tf_dbms_used': 1 if dbms_used == "Yes" else 0,
        'tech_tf_dbms_used_yes': 1 if dbms_used == "Yes" else 0,
        
        # Cost currency (one-hot encoded)
        'project_prf_cost_currency_canadadollar': 1 if cost_currency == "Canadian Dollar" else 0,
        'project_prf_cost_currency_europeaneuro': 1 if cost_currency == "European Euro" else 0,
        
        # Model selection and submission
        'selected_model': selected_model,
        'submit': submit
    }
    
    return features

def save_current_configuration(user_inputs):
    """Save current configuration to a file."""
    config_name = st.text_input("Enter a name for this configuration:")
    
    if not config_name:
        st.warning("Please enter a name for your configuration.")
        return
    
    # Create a copy of user_inputs without the submit flag
    config = user_inputs.copy()
    config.pop('submit', None)  # Remove submit flag
    config['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
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

def display_inputs(user_inputs, selected_model):
    """Display the input parameters in a formatted way."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters Summary")
        
        # Create a summary of key parameters
        key_params = {
            "Project Year": user_inputs.get('project_prf_year_of_project', 'N/A'),
            "Functional Size": user_inputs.get('project_prf_functional_size', 'N/A'),
            "Max Team Size": user_inputs.get('project_prf_max_team_size', 'N/A'),
            "Industry Sector": user_inputs.get('external_eef_industry_sector', 'N/A'),
            "Application Type": user_inputs.get('project_prf_application_type', 'N/A'),
            "Documentation Level": user_inputs.get('process_pmf_docs', 'N/A'),
            "Tools Used": user_inputs.get('tech_tf_tools_used', 'N/A'),
            "Personnel Changes": user_inputs.get('people_prf_personnel_changes', 'N/A')
        }
        
        # Create DataFrame for display
        input_df = pd.DataFrame([
            {"Parameter": k, "Value": v} for k, v in key_params.items()
        ])
        
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
        
        # Get feature names from the features_dict keys (exclude non-feature keys)
        exclude_keys = {'selected_model', 'submit'}
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
        
        # Map technical feature names to user-friendly names
        feature_name_mapping = {
            "project_prf_year_of_project": "Project Year",
            "project_prf_functional_size": "Functional Size",
            "project_prf_max_team_size": "Max Team Size",
            "external_eef_industry_sector": "Industry Sector",
            "external_eef_organisation_type": "Organisation Type",
            "project_prf_application_type": "Application Type",
            "process_pmf_docs": "Documentation Level",
            "tech_tf_tools_used": "Tools Used",
            "people_prf_personnel_changes": "Personnel Changes",
            "project_prf_application_group_business_application": "Business Application",
            "project_prf_application_group_infrastructure_software": "Infrastructure Software",
            "project_prf_development_type_new_development": "New Development",
            "project_prf_development_type_re_development": "Re-development",
            "tech_tf_development_platform_pc": "PC Platform",
            "tech_tf_language_type_4GL": "4GL Language",
            "tech_tf_primary_programming_language_java": "Java Language",
            "project_prf_relative_size_m1": "Medium Size (M1)",
            "process_pmf_development_methodologies_agile_developmentscrum": "Scrum Methodology"
        }
        
        # Create friendly names list, keeping only top features
        importance_data = []
        for i, name in enumerate(feature_names[:min(len(feature_importance), 15)]):  # Show top 15 features
            if i < len(feature_importance):
                friendly_name = feature_name_mapping.get(name, name.replace('_', ' ').title())
                importance_data.append({
                    'Feature': friendly_name,
                    'Importance': abs(feature_importance[i])
                })
        
        if importance_data:
            # Create a DataFrame for the feature importance values
            importance_df = pd.DataFrame(importance_data)
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                if width > 0:  # Only add label if there's a value
                    label_x_pos = width * 1.01
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                           va='center')
            
            ax.set_xlabel('Relative Importance')
            ax.set_title(f'Top Feature Importance - {get_model_display_name(selected_model)}')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            
            # Display the data table
            st.dataframe(importance_df.round(4), use_container_width=True)
        else:
            st.info("No feature importance data available.")
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