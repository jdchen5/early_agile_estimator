# test_shap_explainer_factory.py
"""
Unit tests for SHAPExplainerFactory
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from shap_analysis.explainer_factory import SHAPExplainerFactory

class TestSHAPExplainerFactory(unittest.TestCase):
    
    def setUp(self):
        self.factory = SHAPExplainerFactory()
    
    def test_cache_operations(self):
        """Test cache functionality"""
        # Initially empty
        self.assertEqual(len(self.factory._explainer_cache), 0)
        
        # Add to cache
        mock_explainer = MagicMock()
        self.factory._explainer_cache["test_model_100"] = mock_explainer
        
        # Check cache info
        cache_info = self.factory.get_cache_info()
        self.assertEqual(cache_info['cache_size'], 1)
        self.assertIn("test_model_100", cache_info['cached_models'])
        
        # Clear cache
        self.factory.clear_cache()
        self.assertEqual(len(self.factory._explainer_cache), 0)
    
    def test_determine_explainer_type(self):
        """Test explainer type determination"""
        # Mock tree-based model
        mock_tree_model = MagicMock()
        mock_tree_model.__class__.__name__ = "RandomForestRegressor"
        self.assertEqual(self.factory._determine_explainer_type(mock_tree_model), "tree")
        
        # Mock linear model
        mock_linear_model = MagicMock()
        mock_linear_model.__class__.__name__ = "LinearRegression"
        self.assertEqual(self.factory._determine_explainer_type(mock_linear_model), "linear")
        
        # Mock unknown model
        mock_unknown_model = MagicMock()
        mock_unknown_model.__class__.__name__ = "SomeUnknownModel"
        self.assertEqual(self.factory._determine_explainer_type(mock_unknown_model), "kernel")
    
    def test_extract_pycaret_estimator(self):
        """Test PyCaret estimator extraction"""
        # Test with _final_estimator attribute
        mock_pycaret_model = MagicMock()
        mock_estimator = MagicMock()
        mock_pycaret_model._final_estimator = mock_estimator
        
        result = self.factory._extract_pycaret_estimator(mock_pycaret_model)
        self.assertEqual(result, mock_estimator)
        
        # Test with named_steps (pipeline)
        mock_pipeline = MagicMock()
        mock_step = MagicMock()
        mock_step.predict = MagicMock()
        mock_pipeline.named_steps = {"estimator": mock_step}
        
        result = self.factory._extract_pycaret_estimator(mock_pipeline)
        self.assertEqual(result, mock_step)
        
        # Test with raw estimator
        mock_raw_model = MagicMock()
        result = self.factory._extract_pycaret_estimator(mock_raw_model)
        self.assertEqual(result, mock_raw_model)
    
    @patch('shap.TreeExplainer')
    def test_create_tree_explainer_success(self, mock_tree_explainer):
        """Test successful TreeExplainer creation"""
        mock_model = MagicMock()
        background_data = np.random.rand(10, 5)
        mock_explainer = MagicMock()
        mock_tree_explainer.return_value = mock_explainer
        
        result = self.factory._create_tree_explainer(mock_model, background_data)
        
        self.assertEqual(result, mock_explainer)
        mock_tree_explainer.assert_called_once_with(mock_model, background_data)
    
    @patch('shap.TreeExplainer')
    def test_create_tree_explainer_failure(self, mock_tree_explainer):
        """Test TreeExplainer creation failure"""
        mock_model = MagicMock()
        background_data = np.random.rand(10, 5)
        mock_tree_explainer.side_effect = Exception("SHAP error")
        
        result = self.factory._create_tree_explainer(mock_model, background_data)
        
        self.assertIsNone(result)
    
    @patch('shap.LinearExplainer')
    def test_create_linear_explainer_success(self, mock_linear_explainer):
        """Test successful LinearExplainer creation"""
        mock_model = MagicMock()
        background_data = np.random.rand(10, 5)
        mock_explainer = MagicMock()
        mock_linear_explainer.return_value = mock_explainer
        
        result = self.factory._create_linear_explainer(mock_model, background_data)
        
        self.assertEqual(result, mock_explainer)
        mock_linear_explainer.assert_called_once_with(mock_model, background_data)
    
    @patch('shap.KernelExplainer')
    def test_create_kernel_explainer_success(self, mock_kernel_explainer):
        """Test successful KernelExplainer creation"""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([1, 2, 3]))
        background_data = np.random.rand(10, 5)
        mock_explainer = MagicMock()
        mock_kernel_explainer.return_value = mock_explainer
        
        result = self.factory._create_kernel_explainer(mock_model, background_data)
        
        self.assertEqual(result, mock_explainer)
        self.assertTrue(mock_kernel_explainer.called)
    
    def test_create_explainer_with_cache(self):
        """Test explainer creation with caching"""
        # Setup
        mock_explainer = MagicMock()
        self.factory._explainer_cache["test_model_100"] = mock_explainer
        
        mock_get_model = MagicMock()
        background_data = np.random.rand(10, 5)
        
        # Test cached explainer is returned
        result = self.factory.create_explainer("test_model", mock_get_model, background_data, 100)
        
        self.assertEqual(result, mock_explainer)
        # get_model should not be called since we used cache
        mock_get_model.assert_not_called()

if __name__ == '__main__':
    unittest.main()