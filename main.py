import streamlit as st

# THIS MUST BE FIRST, before any other Streamlit call!
st.set_page_config(
    page_title="Machine Learning Agile Software Project Effort Estimator", 
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import logging
import sys
import traceback
from constants import LoggingConstants

# Logging setup
logging.basicConfig(
    level=getattr(logging, LoggingConstants.DEFAULT_LOG_LEVEL),
    format=LoggingConstants.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LoggingConstants.APP_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point - delegates to ui.py main function"""
    try:
        # Import and call the main function from ui.py
        from ui import main as ui_main
        ui_main()
        
    except ImportError as e:
        st.error("❌ Failed to import UI module")
        st.error(f"Error: {str(e)}")
        logger.error(f"Import error: {str(e)}")
        
        st.markdown("""
        ### Troubleshooting Steps:
        1. Ensure `ui.py` is in the same directory as `main.py`
        2. Check that all required dependencies are installed
        3. Verify configuration files are present in the `config/` directory
        4. Check the application logs for detailed error information
        """)
        
    except Exception as e:
        st.error("❌ An unexpected error occurred")
        st.error(f"Error: {str(e)}")
        logger.exception("Unexpected error in main application")
        
        # Show traceback in expander for debugging
        with st.expander("🔧 Technical Details (for debugging)"):
            st.code(traceback.format_exc())
        
        st.markdown("""
        ### Recovery Options:
        1. Refresh the page to restart the application
        2. Check the application logs for more details
        3. Verify all configuration files are properly formatted
        4. Ensure all required models and dependencies are available
        """)

if __name__ == "__main__":
    main()