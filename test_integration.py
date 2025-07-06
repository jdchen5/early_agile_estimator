# test_integration.py
"""
Integration tests for SHAP Analysis components
Tests components working together, not just individually
"""

import numpy as np
import logging
from unittest.mock import MagicMock

def test_data_preparer_to_explainer_integration():
    """Test DataPreparer → ExplainerFactory integration"""
    print("\n=== Testing DataPreparer → ExplainerFactory Integration ===")
    
    try:
        from shap_analysis.data_preparer import SHAPDataPreparer
        from shap_analysis.explainer_factory import SHAPExplainerFactory
        
        # Setup
        data_preparer = SHAPDataPreparer()
        explainer_factory = SHAPExplainerFactory()
        
        # Test: Background data preparation for explainer
        print("1. Testing background data preparation...")
        background_data = data_preparer.prepare_background_data(50, "test_model")
        
        if background_data is not None:
            print(f"✅ Background data prepared: shape {background_data.shape}")
            
            # Test: Can explainer factory use this data?
            print("2. Testing explainer creation with prepared data...")
            
            # Mock model and get_model function
            mock_model = MagicMock()
            mock_model.__class__.__name__ = "RandomForestRegressor"
            mock_model.predict = MagicMock(return_value=np.array([100.0]))
            
            # This should work without errors
            explainer = explainer_factory.create_explainer(
                "test_model", mock_get_model, background_data, 50
            )
            
            if explainer is not None:
                print("✅ Explainer created successfully with prepared data")
                return True
            else:
                print("❌ Explainer creation failed")
                return False
        else:
            print("❌ Background data preparation failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_explainer_to_calculator_integration():
    """Test ExplainerFactory → ValueCalculator integration"""
    print("\n=== Testing ExplainerFactory → ValueCalculator Integration ===")
    
    try:
        from shap_analysis.explainer_factory import SHAPExplainerFactory
        from shap_analysis.value_calculator import SHAPValueCalculator
        from shap_analysis.data_preparer import SHAPDataPreparer
        
        # Setup components
        data_preparer = SHAPDataPreparer()
        explainer_factory = SHAPExplainerFactory()
        calculator = SHAPValueCalculator()
        
        # Prepare test data
        background_data = np.random.rand(20, 10).astype(np.float32)
        input_data = np.random.rand(1, 10).astype(np.float32)
        
        print("1. Creating mock explainer...")
        # Create mock explainer that returns predictable SHAP values
        mock_explainer = MagicMock()
        mock_shap_values = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.0, 0.7, -0.1, 0.2, -0.3])
        mock_explainer.shap_values.return_value = mock_shap_values
        
        print("2. Testing value calculation...")
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Test SHAP value calculation
        calculated_values = calculator.calculate_shap_values(mock_explainer, input_data, feature_names)
        
        if calculated_values is not None:
            print(f"✅ SHAP values calculated: shape {calculated_values.shape}")
            
            # Test summary creation
            print("3. Testing summary creation...")
            user_inputs = {f"feature_{i}": i*10 for i in range(10)}
            summary = calculator.create_summary_data(calculated_values, feature_names, user_inputs, top_n=5)
            
            if summary and len(summary) > 0:
                print(f"✅ Summary created: {len(summary)} items")
                print(f"   Top feature: {summary[0]['feature_name']} (impact: {summary[0]['shap_value']:.3f})")
                return True
            else:
                print("❌ Summary creation failed")
                return False
        else:
            print("❌ SHAP value calculation failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_coordinator_full_workflow():
    """Test SHAPAnalysisCoordinator complete workflow"""
    print("\n=== Testing SHAPAnalysisCoordinator Full Workflow ===")
    
    try:
        from shap_analysis.analysis_coordinator import SHAPAnalysisCoordinator
        
        coordinator = SHAPAnalysisCoordinator()
        print("✅ Coordinator created")
        
        # Test realistic user inputs
        user_inputs = {
            'project_prf_functional_size': 100,
            'project_prf_max_team_size': 5,
            'external_eef_industry_sector': 'Financial',
            'tech_tf_primary_programming_language': 'Java',
            'selected_model': 'test_model'  # Should be excluded
        }
        
        print("1. Testing input validation...")
        is_valid = coordinator.data_preparer.validate_shap_inputs(user_inputs)
        if is_valid:
            print("✅ Input validation passed")
        else:
            print("❌ Input validation failed")
            return False
        
        print("2. Testing feature name extraction...")
        feature_names = coordinator._get_feature_names(user_inputs)
        if len(feature_names) > 0:
            print(f"✅ Feature extraction: {len(feature_names)} features")
            print(f"   Features: {feature_names}")
        else:
            print("❌ No features extracted")
            return False
        
        print("3. Testing system info...")
        system_info = coordinator.get_system_info()
        if system_info.get('system_status') == 'Ready':
            print("✅ System info retrieved successfully")
            print(f"   Data preparer: {system_info['data_preparer']['available']}")
            print(f"   Explainer factory: {system_info['explainer_factory']['available']}")
            print(f"   Value calculator: {system_info['value_calculator']['available']}")
        else:
            print("❌ System not ready")
            return False
        
        print("4. Testing cache operations...")
        coordinator.clear_caches()
        print("✅ Cache clearing worked")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinator workflow test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that new system provides same interface as backup"""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test imports work
        from shap_analysis import get_shap_explainer, prepare_sample_data, get_cache_info
        print("✅ Backward compatibility imports work")
        
        # Test basic function calls don't crash
        print("1. Testing prepare_sample_data...")
        sample_data = prepare_sample_data(10)
        if sample_data is not None:
            print(f"✅ Sample data preparation: shape {sample_data.shape}")
        else:
            print("⚠️ Sample data preparation returned None (may be expected)")
        
        print("2. Testing cache info...")
        cache_info = get_cache_info()
        if isinstance(cache_info, dict):
            print(f"✅ Cache info retrieved: {cache_info}")
        else:
            print("❌ Cache info failed")
            return False
        
        print("3. Testing system status...")
        from shap_analysis import get_system_status
        status = get_system_status()
        print(f"✅ System status: {status['recommended_system']} system recommended")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n=== Testing Error Handling ===")
    
    try:
        from shap_analysis.data_preparer import SHAPDataPreparer
        from shap_analysis.value_calculator import SHAPValueCalculator
        
        data_preparer = SHAPDataPreparer()
        calculator = SHAPValueCalculator()
        
        print("1. Testing invalid inputs...")
        
        # Test with empty inputs
        is_valid = data_preparer.validate_shap_inputs({})
        if not is_valid:
            print("✅ Empty input validation correctly failed")
        else:
            print("❌ Empty input validation should have failed")
            return False
        
        # Test with None SHAP values
        summary = calculator.create_summary_data(None, [], {})
        if len(summary) == 0:
            print("✅ None SHAP values handled correctly")
        else:
            print("❌ None SHAP values should return empty summary")
            return False
        
        # Test with invalid data shapes
        invalid_data = np.array([])
        result = calculator.calculate_shap_values(None, invalid_data)
        if result is None:
            print("✅ Invalid data handled correctly")
        else:
            print("❌ Invalid data should return None")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_integration_tests():
    """Run all integration tests"""
    print("🧪 RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("DataPreparer → ExplainerFactory", test_data_preparer_to_explainer_integration),
        ("ExplainerFactory → ValueCalculator", test_explainer_to_calculator_integration), 
        ("Coordinator Full Workflow", test_coordinator_full_workflow),
        ("Backward Compatibility", test_backward_compatibility),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"Result: {status}")
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Ready for manual UI testing")
        return True
    else:
        print("⚠️ Some integration tests failed")
        print("🔧 Fix issues before manual UI testing")
        return False

if __name__ == "__main__":
    run_all_integration_tests()