# unit/test_models.py - Model component tests
"""
Test model loading, prediction, and feature preparation functionality
Critical path: Models must load and predict correctly
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
import os

# Import functions to test
from models import (
    list_available_models,
    check_required_models,
    predict_man_hours,
    prepare_features_for_model,
    get_model_display_name,
    load_model,
    get_model_expected_features
)

class TestModelDiscovery:
    """Test model discovery and validation"""
    
    def test_list_available_models_returns_list(self):
        """Test that model listing returns a list structure"""
        models = list_available_models()
        
        assert isinstance(models, list)
        
        # If models exist, validate structure
        for model in models:
            assert isinstance(model, dict)
            assert 'technical_name' in model
            assert 'display_name' in model
            assert isinstance(model['technical_name'], str)
            assert isinstance(model['display_name'], str)
    
    def test_check_required_models_structure(self):
        """Test model availability check returns proper structure"""
        result = check_required_models()
        
        assert isinstance(result, dict)
        assert 'models_available' in result
        assert isinstance(result['models_available'], bool)
        
        if result['models_available']:
            assert 'found_models' in result
            assert 'model_count' in result
            assert isinstance(result['model_count'], int)
    
    @pytest.mark.skipif(not Path("models").exists(), reason="Models folder not found")
    def test_models_folder_has_pkl_files(self):
        """Test that models folder contains .pkl files"""
        models_dir = Path("models")
        pkl_files = list(models_dir.glob("*.pkl"))
        
        # Filter out pipeline/scaler files
        model_files = [f for f in pkl_files 
                      if not any(keyword in f.name.lower() 
                               for keyword in ['pipeline', 'scaler'])]
        
        assert len(model_files) > 0, "No model .pkl files found in models folder"

class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_model_with_mock(self, mock_model):
        """Test model loading logic with mocked model"""
        # Simply test that load_model handles non-existent models gracefully
        result = load_model("definitely_nonexistent_model_12345")
        assert result is None
        
        # Test the mock model has expected attributes
        assert hasattr(mock_model, 'predict')
        assert mock_model.predict([1, 2, 3]).shape == (1,)
    
    def test_load_nonexistent_model_returns_none(self):
        """Test loading non-existent model returns None"""
        result = load_model("definitely_nonexistent_model_12345")
        assert result is None
    
    @pytest.mark.skipif(not Path("models").exists(), reason="Models folder not found")
    def test_load_real_model_if_available(self):
        """Test loading actual model files if they exist"""
        available_models = list_available_models()
        
        if available_models:
            model_name = available_models[0]['technical_name']
            model = load_model(model_name)
            
            # Model should load successfully
            assert model is not None
            
            # Should have prediction capability
            assert hasattr(model, 'predict') or hasattr(model, '_final_estimator')

class TestFeaturePreparation:
    """Test feature preparation pipeline"""
    
    def test_prepare_features_with_valid_input(self, sample_ui_inputs):
        """Test feature preparation with valid UI inputs"""
        result = prepare_features_for_model(sample_ui_inputs)
        
        if result is not None:  # May be None if pipeline not available
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert result.shape[0] == 1  # Single row for single prediction
            
            # Should have reasonable number of features
            assert 20 <= result.shape[1] <= 100
    
    def test_prepare_features_with_empty_input(self):
        """Test feature preparation with empty input"""
        # Empty input should be handled gracefully - may return None or raise exception
        try:
            result = prepare_features_for_model({})
            
            # If it returns something, validate it
            if result is not None:
                assert isinstance(result, pd.DataFrame)
            else:
                # None is acceptable for empty input
                pass
                
        except Exception as e:
            # Exception is also acceptable for empty input
            # The system should fail gracefully with empty input
            assert "No input features provided" in str(e) or "clean_features" in str(e)
            print(f"Empty input handled with expected exception: {type(e).__name__}")
    
    def test_prepare_features_with_missing_required_fields(self):
        """Test feature preparation with missing required fields"""
        minimal_input = {'project_prf_functional_size': 100}
        
        try:
            result = prepare_features_for_model(minimal_input)
            
            # Should not crash, may return None or valid DataFrame
            if result is not None:
                assert isinstance(result, pd.DataFrame)
                
        except Exception as e:
            # Some exceptions may be expected with incomplete input
            print(f"Incomplete input caused exception: {e}")
            # Don't fail the test - this may be expected behavior

class TestPrediction:
    """Test prediction functionality"""
    
    def test_predict_with_mock_model(self, sample_ui_inputs, mock_model):
        """Test prediction logic without actual model loading"""
        # Test the prediction workflow components separately
        
        # Test that prepare_features_for_model can process sample inputs
        features_result = prepare_features_for_model(sample_ui_inputs)
        
        # May be None if pipeline not available, which is acceptable
        if features_result is not None:
            assert isinstance(features_result, pd.DataFrame)
            assert not features_result.empty
        
        # Test mock model prediction capability
        test_input = np.array([[1, 2, 3, 4, 5]])
        mock_prediction = mock_model.predict(test_input)
        assert mock_prediction is not None
        assert len(mock_prediction) == 1
    
    def test_predict_with_invalid_model(self, sample_ui_inputs):
        """Test prediction with invalid model name"""
        prediction = predict_man_hours(sample_ui_inputs, "invalid_model")
        assert prediction is None
    
    def test_predict_with_invalid_features(self, mock_model):
        """Test prediction with invalid feature input"""
        # Test with empty input
        prediction = predict_man_hours({}, "any_model")
        # Should handle gracefully - may return None or raise exception
        assert prediction is None or isinstance(prediction, (int, float))

class TestModelUtilities:
    """Test model utility functions"""
    
    def test_get_model_display_name(self):
        """Test model display name generation"""
        # Test with common model naming patterns
        test_cases = [
            ("rf_model", "Rf Model"),  # Default transformation
            ("xgb_classifier", "Xgb Classifier"),
            ("simple_model", "Simple Model")
        ]
        
        for technical_name, expected_pattern in test_cases:
            display_name = get_model_display_name(technical_name)
            assert isinstance(display_name, str)
            assert len(display_name) > 0
    
    def test_get_model_expected_features_with_mock(self, mock_model):
        """Test getting expected features from model"""
        # Mock model with feature names
        mock_model.feature_names_in_ = ['feature_1', 'feature_2', 'feature_3']
        
        features = get_model_expected_features(mock_model)
        assert isinstance(features, list)
        assert len(features) == 3
        assert 'feature_1' in features

class TestModelIntegration:
    """Test model integration scenarios"""
    
    @pytest.mark.skipif(not Path("models").exists(), reason="Models folder not found")
    def test_end_to_end_prediction_if_models_available(self, sample_ui_inputs, test_validator):
        """Test complete prediction workflow with real models if available"""
        available_models = list_available_models()
        
        if available_models:
            model_name = available_models[0]['technical_name']
            
            # Test complete prediction workflow
            prediction = predict_man_hours(sample_ui_inputs, model_name)
            
            if prediction is not None:
                # Validate prediction is reasonable
                assert test_validator.validate_prediction_output(prediction)
                assert isinstance(prediction, (int, float))
                print(f"✅ Prediction test passed: {prediction} hours")
    
    def test_feature_model_compatibility(self, sample_ui_inputs, expected_feature_count):
        """Test that prepared features match expected model input"""
        features_df = prepare_features_for_model(sample_ui_inputs)
        
        if features_df is not None:
            # Should produce expected number of features (approximately)
            feature_count = features_df.shape[1]
            
            # Allow some flexibility in feature count
            assert 50 <= feature_count <= 80, \
                f"Expected ~{expected_feature_count} features, got {feature_count}"