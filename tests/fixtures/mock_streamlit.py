# fixtures/mock_streamlit.py - Streamlit mocking utilities
"""
Mock Streamlit components for testing UI logic without full Streamlit server
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import pandas as pd

class MockStreamlit:
    """Mock Streamlit interface for testing"""
    
    def __init__(self):
        self.session_state = {}
        self.components = {}
        self.calls = []
    
    def reset(self):
        """Reset mock state"""
        self.session_state.clear()
        self.components.clear()
        self.calls.clear()
    
    def selectbox(self, label, options, index=0, **kwargs):
        """Mock selectbox that returns predefined values"""
        self.calls.append(('selectbox', label, options))
        return options[index] if options else None
    
    def number_input(self, label, min_value=None, max_value=None, value=None, **kwargs):
        """Mock number input"""
        self.calls.append(('number_input', label, value))
        return value or min_value or 0
    
    def checkbox(self, label, value=False, **kwargs):
        """Mock checkbox"""
        self.calls.append(('checkbox', label, value))
        return value
    
    def button(self, label, **kwargs):
        """Mock button - returns False by default unless configured"""
        self.calls.append(('button', label))
        return self.components.get(f"button_{label}", False)
    
    def configure_button_click(self, label: str, clicked: bool = True):
        """Configure a button to simulate click"""
        self.components[f"button_{label}"] = clicked

@pytest.fixture
def mock_streamlit():
    """Provide mock Streamlit interface"""
    mock_st = MockStreamlit()
    
    with patch('streamlit.selectbox', mock_st.selectbox), \
         patch('streamlit.number_input', mock_st.number_input), \
         patch('streamlit.checkbox', mock_st.checkbox), \
         patch('streamlit.button', mock_st.button), \
         patch('streamlit.session_state', mock_st.session_state):
        yield mock_st

class UITestHelper:
    """Helper for testing UI components"""
    
    @staticmethod
    def create_filled_form(mock_st: MockStreamlit, ui_inputs: Dict[str, Any]):
        """Simulate a filled form with given inputs"""
        mock_st.reset()
        
        # Configure form fields to return test values
        for field, value in ui_inputs.items():
            if isinstance(value, bool):
                mock_st.components[f"checkbox_{field}"] = value
            elif isinstance(value, (int, float)):
                mock_st.components[f"number_{field}"] = value
            else:
                mock_st.components[f"select_{field}"] = value
    
    @staticmethod
    def simulate_predict_button_click(mock_st: MockStreamlit):
        """Simulate clicking the predict button"""
        mock_st.configure_button_click("🔮 Predict Effort", True)

@pytest.fixture
def ui_test_helper():
    """UI testing helper instance"""
    return UITestHelper()