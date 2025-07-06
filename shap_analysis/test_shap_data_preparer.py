# test_shap_data_preparer.py
"""
Unit tests for SHAPDataPreparer
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from shap_analysis.data_preparer import SHAPDataPreparer

class TestSHAPDataPreparer(unittest.TestCase):
    
    def setUp(self):
        self.preparer = SHAPDataPreparer()
    
    def test_validate_shap_inputs_valid(self):
        """Test validation with valid inputs"""
        valid_inputs = {
            'project_prf_functional_size': 100,
            'project_prf_max_team_size': 5,
            'selected_model': 'test_model'  # Should be excluded
        }
        self.assertTrue(self.preparer.validate_shap_inputs(valid_inputs))
    
    def test_validate_shap_inputs_invalid(self):
        """Test validation with invalid inputs"""
        # Empty inputs
        self.assertFalse(self.preparer.validate_shap_inputs({}))
        
        # Only excluded fields
        invalid_inputs = {'selected_model': 'test', 'submit': True}
        self.assertFalse(self.preparer.validate_shap_inputs(invalid_inputs))
    
    def test_validate_shap_inputs_required_fields(self):
        """Test validation with required fields"""
        inputs = {'field1': 'value1'}
        
        # Missing required field
        self.assertFalse(self.preparer.validate_shap_inputs(inputs, ['field2']))
        
        # Has required field
        self.assertTrue(self.preparer.validate_shap_inputs(inputs, ['field1']))
    
    @patch('shap_analysis.data_preparer.prepare_features_for_model')
    def test_prepare_input_data_success(self, mock_prepare):
        """Test successful input data preparation"""
        # Mock successful preparation
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.values = np.array([[1, 2, 3]])
        mock_prepare.return_value = mock_df
        
        user_inputs = {'project_prf_functional_size': 100}
        result = self.preparer.prepare_input_data(user_inputs)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 3))
    
    @patch('shap_analysis.data_preparer.prepare_features_for_model')
    def test_prepare_input_data_failure(self, mock_prepare):
        """Test input data preparation failure"""
        # Mock failed preparation
        mock_prepare.return_value = None
        
        user_inputs = {'project_prf_functional_size': 100}
        result = self.preparer.prepare_input_data(user_inputs)
        
        self.assertIsNone(result)
    
    def test_create_realistic_ui_inputs(self):
        """Test realistic UI input generation"""
        inputs = self.preparer._create_realistic_ui_inputs()
        
        # Check required fields are present
        required_fields = ['project_prf_functional_size', 'project_prf_max_team_size']
        for field in required_fields:
            self.assertIn(field, inputs)
        
        # Check realistic ranges
        self.assertBetween(inputs['project_prf_max_team_size'], 3, 15)
        self.assertGreater(inputs['project_prf_functional_size'], 0)
    
    def test_get_sample_data_info(self):
        """Test sample data info retrieval"""
        info = self.preparer.get_sample_data_info()
        
        # Should always have fallback options
        self.assertIn('synthetic_fallback', info)
        self.assertTrue(info['synthetic_fallback'])
        self.assertIn('recommended_source', info)
    
    def assertBetween(self, value, min_val, max_val):
        """Helper assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

if __name__ == '__main__':
    unittest.main()