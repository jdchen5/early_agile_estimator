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
import yaml
from datetime import datetime
from models import list_available_models, get_feature_importance, check_required_models, get_model_display_name

# --- Config Loader ---
def load_feature_mapping_config(path="config/feature_mapping.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

FEATURE_CONFIG = load_feature_mapping_config()

def get_team_size_group(max_team_size):
    if max_team_size == 2: return "2"
    elif 3 <= max_team_size <= 4: return "3-4"
    elif 5 <= max_team_size <= 8: return "5-8"
    elif 9 <= max_team_size <= 14: return "9-14"
    elif 21 <= max_team_size <= 30: return "21-30"
    elif 41 <= max_team_size <= 50: return "41-50"
    elif 61 <= max_team_size <= 70: return "61-70"
    else: return "Missing"

def sidebar_inputs():
    model_status = check_required_models()
    with st.sidebar:
        with st.form("estimation_form"):
            tab1, tab2, tab3 = st.tabs(["Basic", "Technical", "Advanced"])
            with tab1:
                # All your tab 1 input widgets here
                project_year = st.number_input("Project Year", 2015, 2030, 2024)
                industry_sector = st.selectbox("Industry Sector",
                    FEATURE_CONFIG["categorical_features"]["external_eef_industry_sector"]["options"])
                organisation_type = st.selectbox("Organisation Type",
                    FEATURE_CONFIG["categorical_features"]["external_eef_organisation_type"]["options"])
                application_type = st.selectbox("Application Type",
                    FEATURE_CONFIG["categorical_features"]["project_prf_application_type"]["options"])
                functional_size = st.number_input("Functional Size (Function Points)", 1, 10000, 100)
                max_team_size = st.number_input("Maximum Team Size", 1, 100, 5)
                st.subheader("Application Group")
                application_group = st.radio("Primary Application Category",
                    list(FEATURE_CONFIG["one_hot_features"]["application_group"]["mapping"].keys()))
                development_type = st.radio("Development Type",
                    list(FEATURE_CONFIG["one_hot_features"]["development_type"]["mapping"].keys()))
                relative_size = st.selectbox("Relative Project Size",
                    list(FEATURE_CONFIG["one_hot_features"]["relative_size"]["mapping"].keys()), 2)

            # --- Tab 2: Technical ---
            with tab2:
                st.header("Technical Information")
                development_platform = st.selectbox("Development Platform",
                    list(FEATURE_CONFIG["one_hot_features"]["development_platform"]["mapping"].keys()))
                language_type = st.selectbox("Programming Language Type",
                    ["3GL (Third Generation)"] +
                    list(FEATURE_CONFIG["one_hot_features"]["language_type"]["mapping"].keys()))
                primary_language = st.selectbox("Primary Programming Language",
                    list(FEATURE_CONFIG["one_hot_features"]["primary_language"]["mapping"].keys()))
                architecture = st.selectbox("System Architecture",
                    list(FEATURE_CONFIG["one_hot_features"]["architecture"]["mapping"].keys()), 1)
                client_server = st.radio("Client-Server Architecture",
                    list(FEATURE_CONFIG["one_hot_features"]["client_server"]["mapping"].keys()))
                web_development = st.radio("Web Development",
                    list(FEATURE_CONFIG["one_hot_features"]["web_development"]["mapping"].keys()))
                dbms_used = st.radio("Database Management System Used",
                    list(FEATURE_CONFIG["one_hot_features"]["dbms_used"]["mapping"].keys()))
                tools_used = st.slider("Development Tools Sophistication (1-5)", 1, 5, 3)

            # --- Tab 3: Process & People ---
            with tab3:
                st.header("Process & People")
                docs = st.slider("Documentation Level (1-5)", 1, 5, 3)
                personnel_changes = st.slider("Personnel Changes (1-5)", 1, 5, 2)
                development_methodology = st.selectbox("Development Methodology",
                    list(FEATURE_CONFIG["special_cases"]["development_methodology"]["mapping"].keys()))
                team_size_group = st.selectbox("Team Size Group",
                    FEATURE_CONFIG["special_cases"]["team_size_group"]["options"],
                    index=FEATURE_CONFIG["special_cases"]["team_size_group"]["options"].index(get_team_size_group(max_team_size)))
                cost_currency = st.selectbox("Cost Currency",
                    list(FEATURE_CONFIG["one_hot_features"]["cost_currency"]["mapping"].keys()))
        
            # AFTER all tabs, always visible:
            st.markdown("---")
            st.header("Model Selection")
            selected_model = None
            if model_status["models_available"]:
                available_models = list_available_models()
                if available_models:
                    model_options = {model['display_name']: model['technical_name'] for model in available_models}
                    selected_display_name = st.selectbox("Select Prediction Model", list(model_options.keys()))
                    selected_model = model_options[selected_display_name]
                else:
                    st.warning("No trained models found. Please add trained models to the 'models' directory.")
            else:
                st.warning("No trained models found. Please create or add trained models.")


            col1, col2 = st.columns(2)
            submit = col1.form_submit_button("Predict Man-Hours")
            save_config = col2.form_submit_button("Save Config")
            config_name = None
            if save_config:
                config_name = st.text_input("Enter a name for this configuration:")

            # Build and return your input dict as before!
            user_inputs = {
                "project_prf_year_of_project": project_year,
                "external_eef_industry_sector": industry_sector,
                "external_eef_organisation_type": organisation_type,
                "project_prf_application_type": application_type,
                "project_prf_functional_size": functional_size,
                "project_prf_max_team_size": max_team_size,
                "process_pmf_docs": docs,
                "tech_tf_tools_used": tools_used,
                "people_prf_personnel_changes": personnel_changes,
                "application_group": application_group,
                "development_type": development_type,
                "development_platform": development_platform,
                "language_type": language_type,
                "primary_language": primary_language,
                "relative_size": relative_size,
                "team_size_group": team_size_group,
                "development_methodology": development_methodology,
                "architecture": architecture,
                "client_server": client_server,
                "web_development": web_development,
                "dbms_used": dbms_used,
                "cost_currency": cost_currency,
                "selected_model": selected_model,
                "submit": submit
            }

            if submit or save_config:
                return create_feature_dict_from_config(user_inputs, FEATURE_CONFIG)
            # Always return a dict!
            return {'selected_model': None, 'submit': False}


def create_feature_dict_from_config(user_inputs, config):
    features = {}
    for key in config.get("numeric_features", []):
        features[key] = user_inputs.get(key, 0)
    for key, meta in config.get("categorical_features", {}).items():
        features[key] = user_inputs.get(key, "")
    for group, mapping in config.get("one_hot_features", {}).items():
        input_value = user_inputs.get(mapping["input_key"], "")
        for label, feat_key in mapping["mapping"].items():
            features[feat_key] = int(input_value == label)
    for group, spec in config.get("special_cases", {}).items():
        input_value = user_inputs.get(spec["input_key"], "")
        if "mapping" in spec:
            for label, feat_key in spec["mapping"].items():
                features[feat_key] = int(input_value == label)
        if "output_keys" in spec:
            for label, feat_key in spec["output_keys"].items():
                features[feat_key] = int(input_value == label)
    features["selected_model"] = user_inputs.get("selected_model")
    features["submit"] = user_inputs.get("submit", False)
    return features

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_current_configuration(user_inputs, config_name):
    config = user_inputs.copy()
    config.pop('submit', None)
    config['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ensure_dir('configs')
    with open(f'configs/{config_name}.json', 'w') as f:
        json.dump(config, f)
    st.success(f"Configuration '{config_name}' saved successfully!")

def load_saved_configurations():
    configs = {}
    if not os.path.exists('configs'):
        return configs
    for filename in os.listdir('configs'):
        if filename.endswith('.json'):
            config_name = os.path.splitext(filename)[0]
            try:
                with open(f'configs/{filename}', 'r') as f:
                    configs[config_name] = json.load(f)
            except Exception:
                pass
    return configs

def load_configuration(config_name):
    try:
        with open(f'configs/{config_name}.json', 'r') as f:
            return json.load(f)
    except Exception:
        st.error(f"Failed to load configuration '{config_name}'")
        return None

def display_inputs(user_inputs, selected_model):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Input Parameters Summary")
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
        input_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in key_params.items()])
        st.dataframe(input_df, use_container_width=True)
        if selected_model:
            model_display_name = get_model_display_name(selected_model)
            st.write(f"📊 Selected Model: **{model_display_name}**")
        else:
            st.write("📊 Selected Model: **None**")
    return col2

def show_feature_importance(selected_model, features_dict, st):
    if not selected_model:
        st.info("No model selected for feature importance analysis.")
        return
    feature_importance = get_feature_importance(selected_model)
    if feature_importance is not None:
        st.subheader("Feature Importance")
        exclude_keys = {'selected_model', 'submit'}
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
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
        importance_data = []
        for i, name in enumerate(feature_names[:min(len(feature_importance), 15)]):
            if i < len(feature_importance):
                friendly_name = feature_name_mapping.get(name, name.replace('_', ' ').title())
                importance_data.append({
                    'Feature': friendly_name,
                    'Importance': abs(feature_importance[i])
                })
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    label_x_pos = width * 1.01
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')
            ax.set_xlabel('Relative Importance')
            ax.set_title(f'Top Feature Importance - {get_model_display_name(selected_model)}')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(importance_df.round(4), use_container_width=True)
        else:
            st.info("No feature importance data available.")
    else:
        model_display_name = get_model_display_name(selected_model)
        st.info(f"Feature importance is not available for {model_display_name}. This might be because the model doesn't support feature importance or there was an error retrieving it.")

def show_prediction(col2, prediction, team_size):
    with col2:
        st.subheader("Prediction Result")
        if prediction is None:
            st.error("Failed to make prediction. Please check logs for details.")
            return
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='text-align:center;'>Estimated Effort</h3>
            <h1 style='text-align:center; color:#1f77b4; font-size:2.5rem;'>{prediction:.2f} Man-Hours</h1>
        </div>
        """, unsafe_allow_html=True)
        hours = int(prediction)
        days = hours // 8
        per_person = prediction / team_size
        st.markdown("### Timeline Breakdown")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Calendar Time", f"{days}d", help="Estimated calendar duration assuming full team availability")
        with metrics_col2:
            st.metric("Per Person", f"{per_person:.2f}h", help="Average effort per team member in hours")
        if prediction < 1:
            st.warning("This prediction seems unusually low. Consider reviewing your inputs.")
        elif prediction > 10000:
            st.warning("This prediction seems unusually high. Consider reviewing your inputs.")

def about_section():
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
