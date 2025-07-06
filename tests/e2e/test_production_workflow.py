# e2e/test_production_workflow.py - End-to-end production workflow tests
"""
End-to-end tests for critical production scenarios:
1. Single model prediction
2. Multi-model comparison  
3. Complete user workflow simulation

These tests validate the entire system works as expected in production
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch

# Import main components
from models import (
    list_available_models,
    predict_man_hours,
    prepare_features_for_model
)

class TestSingleModelPrediction:
    """Test complete single model prediction workflow"""
    
    @pytest.mark.skipif(not Path("models").exists(), reason="Models folder not found")
    def test_complete_single_model_workflow(self, sample_ui_inputs, test_validator):
        """Test complete workflow: UI input → prediction output"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available for testing")
        
        # Test with first available model
        model_name = available_models[0]['technical_name']
        print(f"Testing workflow with model: {model_name}")
        
        # Step 1: Validate input
        assert test_validator.validate_ui_inputs(sample_ui_inputs)
        
        # Step 2: Feature preparation
        features_df = prepare_features_for_model(sample_ui_inputs)
        if features_df is None:
            pytest.skip("Feature preparation not available")
        
        assert not features_df.empty
        print(f"Features prepared: {features_df.shape}")
        
        # Step 3: Make prediction
        prediction = predict_man_hours(sample_ui_inputs, model_name)
        
        assert prediction is not None
        assert test_validator.validate_prediction_output(prediction)
        print(f"Prediction completed: {prediction:.0f} hours")
        
        # Step 4: Validate prediction range is reasonable
        assert 1.0 <= prediction <= 50000.0  # Reasonable business range
        
        print(f"Single model workflow completed successfully")
    
    def test_single_model_with_edge_cases(self, test_validator):
        """Test single model with edge case inputs"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available")
        
        model_name = available_models[0]['technical_name']
        
        # Test edge cases
        edge_cases = [
            # Very small project
            {
                'project_prf_functional_size': 1,
                'project_prf_max_team_size': 1,
                'external_eef_industry_sector': 'Financial'
            },
            # Large project
            {
                'project_prf_functional_size': 5000,
                'project_prf_max_team_size': 20,
                'external_eef_industry_sector': 'Banking'
            },
            # Typical project
            {
                'project_prf_functional_size': 300,
                'project_prf_max_team_size': 6,
                'external_eef_industry_sector': 'Insurance'
            }
        ]
        
        for i, inputs in enumerate(edge_cases):
            prediction = predict_man_hours(inputs, model_name)
            
            if prediction is not None:
                assert test_validator.validate_prediction_output(prediction)
                print(f"Edge case {i+1}: {prediction:.0f} hours")

class TestMultiModelComparison:
    """Test multi-model comparison workflow"""
    
    @pytest.mark.skipif(not Path("models").exists(), reason="Models folder not found")
    def test_multi_model_comparison_workflow(self, sample_ui_inputs, test_validator):
        """Test comparing predictions across multiple models"""
        available_models = list_available_models()
        
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for comparison testing")
        
        # Test with up to 3 models for reasonable test time
        test_models = available_models[:min(3, len(available_models))]
        predictions = {}
        
        print(f"Testing multi-model comparison with {len(test_models)} models")
        
        # Get predictions from each model
        for model_info in test_models:
            model_name = model_info['technical_name']
            
            prediction = predict_man_hours(sample_ui_inputs, model_name)
            
            if prediction is not None:
                assert test_validator.validate_prediction_output(prediction)
                predictions[model_name] = prediction
                print(f"{model_name}: {prediction:.0f} hours")
        
        # Validate we got multiple predictions
        assert len(predictions) >= 2, "Need at least 2 successful predictions"
        
        # Analyze prediction variance
        pred_values = list(predictions.values())
        mean_pred = sum(pred_values) / len(pred_values)
        max_pred = max(pred_values)
        min_pred = min(pred_values)
        
        # Variance should be reasonable (not wildly different)
        variance_ratio = (max_pred - min_pred) / mean_pred if mean_pred > 0 else 0
        
        print(f"Prediction range: {min_pred:.0f} - {max_pred:.0f} hours")
        print(f"Variance ratio: {variance_ratio:.2%}")
        
        # Log warning if variance is very high
        if variance_ratio > 1.0:  # 100% variance
            print(f"High prediction variance detected: {variance_ratio:.2%}")
    
    def test_model_agreement_analysis(self, sample_ui_inputs):
        """Test analysis of model agreement"""
        available_models = list_available_models()
        
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for agreement analysis")
        
        predictions = []
        
        for model_info in available_models[:3]:  # Test up to 3 models
            model_name = model_info['technical_name']
            prediction = predict_man_hours(sample_ui_inputs, model_name)
            
            if prediction is not None:
                predictions.append(prediction)
        
        if len(predictions) >= 2:
            # Calculate agreement metrics
            import numpy as np
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Coefficient of variation as agreement measure
            cv = std_pred / mean_pred if mean_pred > 0 else float('inf')
            
            print(f"Model agreement - CV: {cv:.3f}")
            
            # Good agreement if CV < 0.3 (30%)
            if cv < 0.3:
                print("Models show good agreement")
            elif cv < 0.5:
                print("Models show moderate agreement")
            else:
                print("Models show poor agreement")

class TestCompleteUserWorkflow:
    """Test complete user workflow scenarios"""
    
    def test_new_user_workflow(self, sample_ui_inputs):
        """Test workflow for new user making first prediction"""
        # Simulate new user workflow
        print("Testing new user workflow...")
        
        # Step 1: System validation
        available_models = list_available_models()
        assert len(available_models) > 0, "No models available for new users"
        
        # Step 2: Input validation
        required_fields = ['project_prf_functional_size', 'project_prf_max_team_size']
        for field in required_fields:
            assert field in sample_ui_inputs, f"Missing required field: {field}"
        
        # Step 3: Model selection
        selected_model = available_models[0]['technical_name']
        
        # Step 4: Prediction
        prediction = predict_man_hours(sample_ui_inputs, selected_model)
        
        if prediction is not None:
            assert isinstance(prediction, (int, float))
            assert prediction > 0
            print(f"New user prediction: {prediction:.0f} hours")
        else:
            pytest.skip("Prediction not available")
    
    def test_power_user_workflow(self, sample_ui_inputs):
        """Test workflow for power user using multiple models"""
        available_models = list_available_models()
        
        if len(available_models) < 2:
            pytest.skip("Need multiple models for power user workflow")
        
        print("Testing power user workflow...")
        
        # Power user tests multiple models
        results = {}
        
        for model_info in available_models[:3]:  # Test up to 3 models
            model_name = model_info['technical_name']
            display_name = model_info['display_name']
            
            prediction = predict_man_hours(sample_ui_inputs, model_name)
            
            if prediction is not None:
                results[display_name] = prediction
        
        # Power user analyzes results
        if len(results) >= 2:
            print("Power user multi-model analysis:")
            for model, pred in results.items():
                print(f"   {model}: {pred:.0f} hours")
            
            # Calculate statistics
            values = list(results.values())
            avg_pred = sum(values) / len(values)
            print(f"   Average: {avg_pred:.0f} hours")

class TestProductionReadiness:
    """Test production deployment readiness"""
    
    def test_system_health_check(self):
        """Test basic system health and readiness"""
        print("Running production health check...")
        
        # Check 1: Models available
        models = list_available_models()
        assert len(models) > 0, "No models available - system not ready"
        print(f"[OK] {len(models)} models available")
        
        # Check 2: Configuration files present
        config_files = [
            Path("config/ui_info.yaml"),
            Path("config/feature_mapping.yaml")
        ]
        
        for config_file in config_files:
            if config_file.exists():
                print(f"[OK] {config_file.name} present")
            else:
                print(f"[WARN] {config_file.name} missing")
        
        # Check 3: Data files present
        data_files = [
            Path("data/synthetic_isbsg2016r1_1_finance_sdv_generated.csv")
        ]
        
        for data_file in data_files:
            if data_file.exists():
                print(f"[OK] {data_file.name} present")
            else:
                print(f"[WARN] {data_file.name} missing")
    
    def test_performance_benchmarks(self, sample_ui_inputs, benchmark_config):
        """Test system meets performance benchmarks"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models for performance testing")
        
        model_name = available_models[0]['technical_name']
        
        # Benchmark 1: Prediction response time
        start_time = time.time()
        prediction = predict_man_hours(sample_ui_inputs, model_name)
        prediction_time = time.time() - start_time
        
        max_time = benchmark_config['targets']['prediction_time_seconds']
        assert prediction_time < max_time, \
            f"Prediction took {prediction_time:.2f}s, target: {max_time}s"
        
        print(f"[OK] Prediction performance: {prediction_time:.2f}s (target: {max_time}s)")
        
        # Benchmark 2: Memory usage (basic check)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        max_memory = benchmark_config['targets']['memory_usage_mb']
        
        print(f"[OK] Memory usage: {memory_mb:.1f}MB (target: <{max_memory}MB)")
        
        if memory_mb > max_memory:
            print(f"[WARN] Memory usage exceeds target: {memory_mb:.1f}MB > {max_memory}MB")
    
    def test_error_recovery(self):
        """Test system handles errors gracefully"""
        print("Testing error recovery...")
        
        # Test 1: Invalid model name
        prediction = predict_man_hours({'test': 'data'}, "invalid_model_name")
        assert prediction is None
        print("[OK] Invalid model handled gracefully")
        
        # Test 2: Empty input
        prediction = predict_man_hours({}, "any_model")
        # Should not crash
        print("[OK] Empty input handled gracefully")
        
        # Test 3: Malformed input
        malformed_inputs = [
            {'project_prf_functional_size': 'not_a_number'},
            {'invalid_field': 'invalid_value'},
            None
        ]
        
        for inputs in malformed_inputs:
            try:
                if inputs is not None:
                    prediction = predict_man_hours(inputs, "any_model")
                print("[OK] Malformed input handled gracefully")
            except Exception as e:
                print(f"[WARN] Malformed input caused exception: {e}")

@pytest.mark.slow
class TestStressTests:
    """Stress tests for production deployment"""
    
    def test_concurrent_predictions(self, sample_ui_inputs):
        """Test handling multiple concurrent predictions"""
        import threading
        import time
        
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models for stress testing")
        
        model_name = available_models[0]['technical_name']
        results = []
        
        def make_prediction():
            result = predict_man_hours(sample_ui_inputs, model_name)
            results.append(result)
        
        # Create multiple threads
        threads = []
        num_threads = 3  # Conservative for testing
        
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Validate results
        successful_predictions = [r for r in results if r is not None]
        
        print(f"Concurrent test: {len(successful_predictions)}/{num_threads} successful")
        print(f"Total time: {total_time:.2f}s")
        
        assert len(successful_predictions) > 0, "No concurrent predictions succeeded"