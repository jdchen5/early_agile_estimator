# debug_ui.py - Helper to debug UI issues

import streamlit as st

def debug_imports():
    """Debug function to check which imports are working"""
    import_status = {}
    
    try:
        from models import list_available_models
        import_status["models.list_available_models"] = "✅ Success"
    except Exception as e:
        import_status["models.list_available_models"] = f"❌ Error: {e}"
    
    try:
        from models import predict_man_hours
        import_status["models.predict_man_hours"] = "✅ Success"
    except Exception as e:
        import_status["models.predict_man_hours"] = f"❌ Error: {e}"
    
    try:
        from pipeline import load_preprocessing_pipeline
        import_status["pipeline.load_preprocessing_pipeline"] = "✅ Success"
    except Exception as e:
        import_status["pipeline.load_preprocessing_pipeline"] = f"❌ Error: {e}"
    
    try:
        from models import check_preprocessing_pipeline_compatibility
        import_status["models.check_preprocessing_pipeline_compatibility"] = "✅ Success"
    except Exception as e:
        import_status["models.check_preprocessing_pipeline_compatibility"] = f"❌ Error: {e}"
    
    return import_status

def show_debug_info():
    """Show debug information in Streamlit"""
    with st.expander("🐛 Debug Information"):
        st.subheader("Import Status")
        
        import_status = debug_imports()
        
        for module_func, status in import_status.items():
            if "✅" in status:
                st.success(f"{module_func}: {status}")
            else:
                st.error(f"{module_func}: {status}")
        
        st.subheader("Python Path")
        import sys
        for i, path in enumerate(sys.path):
            st.write(f"{i}: {path}")
        
        st.subheader("Current Working Directory")
        import os
        st.write(os.getcwd())
        
        st.subheader("Files in Current Directory")
        try:
            files = os.listdir(".")
            for file in sorted(files):
                if file.endswith(('.py', '.yaml', '.yml', '.json')):
                    st.write(f"📄 {file}")
        except Exception as e:
            st.error(f"Error listing files: {e}")

if __name__ == "__main__":
    st.title("Debug UI Helper")
    show_debug_info()