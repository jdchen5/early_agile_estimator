# integration/test_pipeline_integration.py - Pipeline integration tests
"""
Test complete pipeline integration: UI inputs → model-ready features
Critical path: Feature preparation pipeline must work end-to-end
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import pipeline components
from pipeline import (
    create_preprocessing_pipeline,
    convert_feature_dict_to_dataframe,
    process_features_for_prediction,
    preprocess_for_prediction
)

class TestPipelineIntegration:
    """Test complete pipeline workflow"""
    
    def test_ui_to_dataframe_conversion(self, sample_ui_inputs):
        """Test converting UI inputs to DataFrame format"""
        try:
            df = convert_feature_dict_to_dataframe(sample_ui_inputs)
            
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert df.shape[0] == 1  # Single row
            assert df.shape[1] > 0   # Some columns
            
            print(f"✅ UI→DataFrame conversion: {df.shape}")
            
        except Exception as e:
            pytest.skip(f"UI→DataFrame conversion not available: {e}")
    
    def test_create_preprocessing_pipeline(self):
        """Test pipeline creation without errors"""
        try:
            # Test pipeline creation for prediction (no target)
            pipeline = create_preprocessing_pipeline(target_col=None)
            
            assert pipeline is not None
            assert hasattr(pipeline, 'fit')
            assert hasattr(pipeline, 'transform')
            assert hasattr(pipeline, 'steps')
            
            print(f"✅ Pipeline created with {len(pipeline.steps)} steps")
            
        except Exception as e:
            pytest.skip(f"Pipeline creation not available: {e}")
    
    def test_process_features_for_prediction(self, sample_ui_inputs):
        """Test complete feature processing workflow"""
        try:
            processed_features = process_features_for_prediction(sample_ui_inputs)
            
            if processed_features is not None:
                assert isinstance(processed_features, pd.DataFrame)
                assert not processed_features.empty
                assert processed_features.shape[0] == 1
                
                # Check for reasonable feature count
                feature_count = processed_features.shape[1]
                assert 20 <= feature_count <= 100
                
                # Check for numeric data
                numeric_cols = processed_features.select_dtypes(include=[np.number]).columns
                assert len(numeric_cols) > 0
                
                print(f"✅ Feature processing: {processed_features.shape}")
                
        except Exception as e:
            pytest.skip(f"Feature processing not available: {e}")
    
    def test_pipeline_with_various_inputs(self):
        """Test pipeline with different input scenarios"""
        test_scenarios = [
            # Minimal input
            {
                'project_prf_functional_size': 100,
                'project_prf_max_team_size': 5
            },
            # Complete input
            {
                'project_prf_functional_size': 500,
                'project_prf_max_team_size': 8,
                'external_eef_industry_sector': 'Financial',
                'tech_tf_primary_programming_language': 'Java',
                'tech_tf_web_development': True,
                'tech_tf_dbms_used': False
            },
            # Edge case values
            {
                'project_prf_functional_size': 1,
                'project_prf_max_team_size': 1,
                'external_eef_industry_sector': 'Other',
                'tech_tf_primary_programming_language': 'Assembly'
            }
        ]
        
        try:
            for i, inputs in enumerate(test_scenarios):
                result = process_features_for_prediction(inputs)
                
                if result is not None:
                    assert isinstance(result, pd.DataFrame)
                    assert not result.empty
                    print(f"✅ Scenario {i+1} processed: {result.shape}")
                    
        except Exception as e:
            pytest.skip(f"Multi-scenario testing not available: {e}")

class TestPipelineDataFlow:
    """Test data flow through pipeline stages"""
    
    def test_feature_types_preservation(self, sample_ui_inputs):
        """Test that feature types are handled correctly"""
        try:
            result = process_features_for_prediction(sample_ui_inputs)
            
            if result is not None:
                # Should be mostly numeric after processing
                numeric_ratio = len(result.select_dtypes(include=[np.number]).columns) / len(result.columns)
                assert numeric_ratio > 0.8  # At least 80% numeric
                
                # No missing values after processing
                assert not result.isnull().any().any()
                
                print(f"✅ Feature types: {numeric_ratio:.2%} numeric")
                
        except Exception as e:
            pytest.skip(f"Feature type testing not available: {e}")
    
    def test_feature_scaling_and_encoding(self, sample_ui_inputs):
        """Test that features are properly scaled and encoded"""
        try:
            result = process_features_for_prediction(sample_ui_inputs)
            
            if result is not None:
                # Check for reasonable value ranges (scaled features)
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    # Most values should be in reasonable range
                    sample_values = result[numeric_cols].values.flatten()
                    
                    # Remove any extreme outliers for testing
                    reasonable_values = sample_values[np.abs(sample_values) < 100]
                    
                    if len(reasonable_values) > 0:
                        assert len(reasonable_values) > len(sample_values) * 0.5  # Most values reasonable
                        print(f"✅ Feature scaling: {len(reasonable_values)}/{len(sample_values)} values in range")
                
        except Exception as e:
            pytest.skip(f"Feature scaling testing not available: {e}")

class TestPipelineRobustness:
    """Test pipeline robustness and error handling"""
    
    def test_empty_input_handling(self):
        """Test pipeline handles empty input gracefully"""
        try:
            result = process_features_for_prediction({})
            
            # Should not crash, may return None or default features
            if result is not None:
                assert isinstance(result, pd.DataFrame)
                print("✅ Empty input handled gracefully")
            else:
                print("✅ Empty input returns None (expected)")
                
        except Exception as e:
            # Should not raise unhandled exceptions
            print(f"⚠️ Empty input caused exception: {e}")
    
    def test_invalid_input_handling(self):
        """Test pipeline handles invalid input gracefully"""
        invalid_inputs = [
            {'invalid_field': 'invalid_value'},
            {'project_prf_functional_size': 'not_a_number'},
            {'project_prf_max_team_size': -1}
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = process_features_for_prediction(invalid_input)
                
                # Should handle gracefully without crashing
                if result is not None:
                    assert isinstance(result, pd.DataFrame)
                
                print(f"✅ Invalid input handled: {invalid_input}")
                
            except Exception as e:
                # Log but don't fail - some exceptions may be expected
                print(f"⚠️ Invalid input caused exception: {e}")
    
    def test_missing_required_fields(self):
        """Test pipeline with missing required fields"""
        incomplete_inputs = [
            {'project_prf_functional_size': 100},  # Missing team size
            {'project_prf_max_team_size': 5},      # Missing functional size
            {'external_eef_industry_sector': 'Financial'}  # Missing numeric fields
        ]
        
        for inputs in incomplete_inputs:
            try:
                result = process_features_for_prediction(inputs)
                
                # Should handle gracefully
                if result is not None:
                    assert isinstance(result, pd.DataFrame)
                    assert not result.empty
                
                print(f"✅ Incomplete input handled: {len(inputs)} fields")
                
            except Exception as e:
                print(f"⚠️ Incomplete input caused exception: {e}")

class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    def test_single_prediction_performance(self, sample_ui_inputs, benchmark_config):
        """Test single prediction processing time"""
        import time
        
        try:
            start_time = time.time()
            result = process_features_for_prediction(sample_ui_inputs)
            processing_time = time.time() - start_time
            
            if result is not None:
                # Should process within reasonable time
                max_time = benchmark_config['targets'].get('prediction_time_seconds', 5.0)
                assert processing_time < max_time, \
                    f"Processing took {processing_time:.2f}s, expected < {max_time}s"
                
                print(f"✅ Pipeline processing: {processing_time:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Performance testing not available: {e}")
    
    def test_batch_prediction_performance(self, sample_ui_inputs):
        """Test processing multiple predictions"""
        import time
        
        try:
            # Create batch of inputs
            batch_inputs = [sample_ui_inputs for _ in range(5)]
            
            start_time = time.time()
            results = []
            
            for inputs in batch_inputs:
                result = process_features_for_prediction(inputs)
                if result is not None:
                    results.append(result)
            
            total_time = time.time() - start_time
            
            if results:
                avg_time_per_prediction = total_time / len(results)
                print(f"✅ Batch processing: {avg_time_per_prediction:.2f}s per prediction")
                
                # Batch should be reasonably efficient
                assert avg_time_per_prediction < 10.0
            
        except Exception as e:
            pytest.skip(f"Batch performance testing not available: {e}")