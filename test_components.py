# test_components.py - Simple component testing
"""
Individual testing of SHAP components
"""

def test_data_preparer():
    """Test SHAPDataPreparer individually"""
    try:
        from shap_analysis.data_preparer import SHAPDataPreparer
        
        preparer = SHAPDataPreparer()
        print("✅ SHAPDataPreparer imported successfully")
        
        # Test validation
        valid_inputs = {'project_prf_functional_size': 100, 'project_prf_max_team_size': 5}
        is_valid = preparer.validate_shap_inputs(valid_inputs)
        print(f"✅ Input validation works: {is_valid}")
        
        # Test sample data info
        info = preparer.get_sample_data_info()
        print(f"✅ Sample data info: {info.get('recommended_source', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAPDataPreparer test failed: {e}")
        return False

def test_explainer_factory():
    """Test SHAPExplainerFactory individually"""
    try:
        from shap_analysis.explainer_factory import SHAPExplainerFactory
        
        factory = SHAPExplainerFactory()
        print("✅ SHAPExplainerFactory imported successfully")
        
        # Test cache operations
        cache_info = factory.get_cache_info()
        print(f"✅ Cache operations work: {cache_info}")
        
        # Test model type detection
        class MockRandomForest:
            pass
        explainer_type = factory._determine_explainer_type(MockRandomForest())
        print(f"✅ Model type detection: {explainer_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAPExplainerFactory test failed: {e}")
        return False

def test_value_calculator():
    """Test SHAPValueCalculator individually"""
    try:
        from shap_analysis.value_calculator import SHAPValueCalculator
        import numpy as np
        
        calculator = SHAPValueCalculator()
        print("✅ SHAPValueCalculator imported successfully")
        
        # Test with mock data
        mock_shap_values = np.array([0.5, -0.3, 0.1, -0.8, 0.2])
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        user_inputs = {'feature_1': 100, 'feature_2': 'test'}
        
        # Test summary creation
        summary = calculator.create_summary_data(mock_shap_values, feature_names, user_inputs, top_n=3)
        print(f"✅ Summary creation works: {len(summary)} items")
        
        # Test impact analysis
        analysis = calculator.analyze_feature_impacts(mock_shap_values, feature_names)
        print(f"✅ Impact analysis works: {len(analysis)} metrics")
        
        # Test baseline calculation
        baseline = calculator.calculate_baseline_impact(mock_shap_values)
        print(f"✅ Baseline calculation works: {baseline.get('baseline_sum', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAPValueCalculator test failed: {e}")
        return False

def test_analysis_coordinator():
    """Test SHAPAnalysisCoordinator individually"""
    try:
        from shap_analysis.analysis_coordinator import SHAPAnalysisCoordinator
        
        coordinator = SHAPAnalysisCoordinator()
        print("✅ SHAPAnalysisCoordinator imported successfully")
        
        # Test system info
        system_info = coordinator.get_system_info()
        print(f"✅ System info works: {system_info.get('system_status', 'Unknown')}")
        
        # Test cache operations
        coordinator.clear_caches()
        print("✅ Cache clearing works")
        
        # Test feature name extraction
        test_inputs = {'feature_1': 100, 'selected_model': 'test'}
        feature_names = coordinator._get_feature_names(test_inputs)
        print(f"✅ Feature extraction works: {len(feature_names)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAPAnalysisCoordinator test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Individual Component Testing ===")
    
    print("\n1. Testing SHAPDataPreparer...")
    test_data_preparer()
    
    print("\n2. Testing SHAPExplainerFactory...")
    test_explainer_factory()

    print("\n3. Testing SHAPValueCalculator...")
    test_value_calculator()
    
    print("\n4. Testing SHAPAnalysisCoordinator...")
    test_analysis_coordinator()    
    
    print("\n=== Testing Complete ===")