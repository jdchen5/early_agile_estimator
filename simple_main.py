# simple_main.py - Simplified version for debugging

import streamlit as st
import os

# THIS MUST BE FIRST, before any other Streamlit call!
st.set_page_config(
    page_title="Agile Project Estimator", 
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_files():
    """Check if required files exist"""
    files_status = {}
    
    required_files = [
        "models.py",
        "ui.py", 
        "pipeline.py",
        "config/feature_mapping.yaml",
        "config/ui_config.yaml"
    ]
    
    for file in required_files:
        files_status[file] = os.path.exists(file)
    
    return files_status

def test_imports():
    """Test importing modules step by step"""
    import_results = {}
    
    # Test basic imports
    try:
        import yaml
        import pandas as pd
        import numpy as np
        import logging
        import json
        import pickle
        from typing import Dict, List, Optional, Union, Any, Tuple
        import_results["basic_imports"] = "✅ Success"
    except Exception as e:
        import_results["basic_imports"] = f"❌ Error: {e}"
    
    # Test models import
    try:
        from models import list_available_models, predict_man_hours
        import_results["models_import"] = "✅ Success"
    except Exception as e:
        import_results["models_import"] = f"❌ Error: {e}"
    
    # Test UI import
    try:
        from ui import sidebar_inputs
        import_results["ui_import"] = "✅ Success"
    except Exception as e:
        import_results["ui_import"] = f"❌ Error: {e}"
    
    # Test pipeline import (optional)
    try:
        from pipeline import load_preprocessing_pipeline
        import_results["pipeline_import"] = "✅ Success"
    except Exception as e:
        import_results["pipeline_import"] = f"⚠️ Warning: {e}"
    
    return import_results

def main():
    st.title("🔧 Agile Project Estimator - Debug Mode")
    
    st.subheader("System Check")
    
    # Check files
    st.write("### File Status")
    files_status = check_files()
    for file, exists in files_status.items():
        if exists:
            st.success(f"✅ {file}")
        else:
            st.error(f"❌ {file} - Missing")
    
    # Check imports
    st.write("### Import Status")
    import_results = test_imports()
    for module, status in import_results.items():
        if "✅" in status:
            st.success(f"{module}: {status}")
        elif "⚠️" in status:
            st.warning(f"{module}: {status}")
        else:
            st.error(f"{module}: {status}")
    
    # Show working directory
    st.write("### Current Working Directory")
    st.code(os.getcwd())
    
    # List files in current directory
    st.write("### Files in Current Directory")
    try:
        files = [f for f in os.listdir(".") if f.endswith(('.py', '.yaml', '.yml', '.json'))]
        for file in sorted(files):
            st.write(f"📄 {file}")
    except Exception as e:
        st.error(f"Error listing files: {e}")
    
    # Try to run basic functionality
    st.write("### Basic Functionality Test")
    
    if all(status for status in files_status.values()):
        st.success("All required files found!")
        
        if import_results.get("models_import", "").startswith("✅"):
            try:
                from models import list_available_models
                models = list_available_models()
                st.write(f"Found {len(models)} models")
                for model in models:
                    st.write(f"- {model.get('display_name', 'Unknown')}")
            except Exception as e:
                st.error(f"Error listing models: {e}")
        
        if import_results.get("ui_import", "").startswith("✅"):
            st.success("UI module can be imported successfully")
            
            # Try to load configs
            try:
                from ui import load_yaml_config
                feature_config = load_yaml_config("config/feature_mapping.yaml")
                ui_config = load_yaml_config("config/ui_config.yaml")
                
                st.write(f"Feature config loaded: {len(feature_config)} sections")
                st.write(f"UI config loaded: {len(ui_config)} sections")
                
            except Exception as e:
                st.error(f"Error loading configs: {e}")
    else:
        st.warning("Some required files are missing. Please check your setup.")
    
    # Button to try full app
    if st.button("Try to Load Full Application"):
        try:
            # Import everything we need
            from models import predict_man_hours, list_available_models
            from ui import sidebar_inputs, display_inputs
            
            st.success("✅ All imports successful! You can now try running the full application.")
            st.info("Run: streamlit run main.py")
            
        except Exception as e:
            st.error(f"❌ Full import failed: {e}")
            st.write("**Error details:**")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()