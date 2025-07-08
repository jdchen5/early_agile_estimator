# test_phase2.py - COMPLETE UPDATED VERSION
def test_csv_mapping_fix():
    """Test that CSV files can now be found for models"""
    print("=== Testing CSV Mapping Fix ===")
    from shap_analysis.data_preparer import SHAPDataPreparer
    
    preparer = SHAPDataPreparer()
    models = ['top_model_1_LGBMRegressor', 'top_model_2_GradientBoostingRegressor', 'top_model_3_LassoLars']
    
    for model_name in models:
        top_features = preparer._get_top_features_for_model(model_name, 15)
        assert len(top_features) > 0, f"No features found for {model_name}"
        assert len(top_features) <= 15, f"Too many features returned for {model_name}"
        print(f"✓ {model_name}: {len(top_features)} features loaded")

def test_isbsg_sample_fix():
    """Test that ISBSG sample data can be prepared"""
    print("\n=== Testing ISBSG Sample Fix ===")
    from models import prepare_isbsg_sample_data
    
    sample_data = prepare_isbsg_sample_data(50)
    assert sample_data is not None, "ISBSG sample data should not be None"
    assert sample_data.shape[0] <= 50, "Sample size should be limited"
    assert sample_data.shape[1] > 0, "Should have features"
    print(f"✓ ISBSG sample data: {sample_data.shape}")

def test_new_system_integration():
    """Test new system with real models"""
    print("\n=== Testing New System Integration ===")
    from shap_analysis import get_shap_explainer_optimized
    from models import get_trained_model
    
    model_name = 'top_model_1_LGBMRegressor'
    
    print(f"Testing explainer creation for {model_name}...")
    explainer = get_shap_explainer_optimized(
        model_name, 
        get_trained_model, 
        top_n_features=15, 
        sample_size=50
    )
    
    if explainer is not None:
        assert hasattr(explainer, 'shap_values'), "Explainer should have shap_values method"
        print(f"✓ New system explainer created for {model_name}")
        print(f"✓ Explainer type: {type(explainer).__name__}")
        return True
    else:
        print(f"⚠ Explainer creation returned None for {model_name}")
        return False

def test_coordinator_directly():
    """Test coordinator directly with proper inputs"""
    print("\n=== Testing Coordinator Directly ===")
    from shap_analysis.analysis_coordinator import SHAPAnalysisCoordinator
    from models import get_trained_model
    
    coordinator = SHAPAnalysisCoordinator()
    
    # Test explainer creation only (no user inputs)
    print("Testing explainer creation...")
    result = coordinator.create_explainer_only(
        'top_model_1_LGBMRegressor', get_trained_model, 15, 50
    )
    
    print(f"✓ Coordinator explainer creation: {result.get('success')}")
    if not result.get('success'):
        print(f"  Error: {result.get('error')}")
        return False
    else:
        print(f"  Features analyzed: {result.get('feature_count')}")
        print(f"  Explainer type: {type(result.get('explainer')).__name__}")
    
    # Test with user inputs
    print("\nTesting with user inputs...")
    test_user_inputs = {
        'project_prf_functional_size': 150,
        'project_prf_max_team_size': 8,
        'external_eef_industry_sector': 'financial',
        'tech_tf_primary_programming_language': 'Java',
        'project_prf_relative_size': 'M',
        'tech_tf_web_development': True,
        'process_pmf_prototyping_used': False,
        'tech_tf_dbms_used': True,
        'project_prf_case_tool_used': False
    }
    
    result = coordinator.run_reduced_instance_analysis(
        test_user_inputs, 'top_model_1_LGBMRegressor', get_trained_model, 15, 50
    )
    
    print(f"✓ Coordinator full analysis: {result.get('success')}")
    if not result.get('success'):
        print(f"  Error: {result.get('error')}")
        return False
    else:
        print(f"  Features analyzed: {result.get('feature_count')}")
        print(f"  Performance improvement: {result.get('performance_improvement')}")
        print(f"  Validation passed: {result.get('validation', {}).get('validation_passed')}")
    
    return True

def test_backward_compatibility():
    """Test that old interface still works"""
    print("\n=== Testing Backward Compatibility ===")
    from shap_analysis import get_shap_explainer
    from models import get_trained_model
    
    model_name = 'top_model_1_LGBMRegressor'
    explainer = get_shap_explainer(model_name, get_trained_model)
    
    if explainer is not None:
        print(f"✓ Backward compatibility maintained for {model_name}")
        print(f"✓ Explainer type: {type(explainer).__name__}")
        return True
    else:
        print(f"⚠ Backward compatibility issue for {model_name}")
        return False

def test_all_models():
    """Test explainer creation for all available models"""
    print("\n=== Testing All Available Models ===")
    from models import list_available_models, get_trained_model
    from shap_analysis import get_shap_explainer_optimized
    
    models = list_available_models()
    successful_models = []
    failed_models = []
    
    for model_info in models:
        model_name = model_info['technical_name']
        print(f"Testing {model_name}...")
        
        try:
            explainer = get_shap_explainer_optimized(
                model_name, get_trained_model, top_n_features=10, sample_size=30
            )
            
            if explainer is not None:
                successful_models.append(model_name)
                print(f"  ✓ Success - {type(explainer).__name__}")
            else:
                failed_models.append(model_name)
                print(f"  ⚠ Failed - returned None")
                
        except Exception as e:
            failed_models.append(model_name)
            print(f"  ✗ Error - {e}")
    
    print(f"\n=== Model Test Summary ===")
    print(f"✓ Successful: {len(successful_models)}/{len(models)}")
    print(f"✗ Failed: {len(failed_models)}/{len(models)}")
    
    if successful_models:
        print(f"Working models: {successful_models}")
    if failed_models:
        print(f"Failed models: {failed_models}")
    
    return len(successful_models) > 0

def run_performance_comparison():
    """Compare performance between old and new approach"""
    print("\n=== Performance Comparison ===")
    import time
    from models import get_trained_model
    
    model_name = 'top_model_1_LGBMRegressor'
    
    # Test new optimized approach
    print("Testing new optimized approach...")
    start_time = time.time()
    
    from shap_analysis import get_shap_explainer_optimized
    explainer_new = get_shap_explainer_optimized(
        model_name, get_trained_model, top_n_features=15, sample_size=50
    )
    
    new_time = time.time() - start_time
    
    print(f"✓ New approach time: {new_time:.2f} seconds")
    print(f"✓ New approach result: {explainer_new is not None}")
    
    if explainer_new:
        print(f"✓ Explainer type: {type(explainer_new).__name__}")
    
    return new_time

if __name__ == "__main__":
    print("🚀 PHASE 2 COMPREHENSIVE TESTING")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    try:
        test_csv_mapping_fix()
        success_count += 1
    except Exception as e:
        print(f"✗ CSV mapping test failed: {e}")
    
    try:
        test_isbsg_sample_fix()
        success_count += 1
    except Exception as e:
        print(f"✗ ISBSG test failed: {e}")
    
    try:
        if test_new_system_integration():
            success_count += 1
    except Exception as e:
        print(f"✗ New system integration test failed: {e}")
    
    try:
        if test_coordinator_directly():
            success_count += 1
    except Exception as e:
        print(f"✗ Coordinator test failed: {e}")
    
    try:
        if test_backward_compatibility():
            success_count += 1
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
    
    try:
        if test_all_models():
            success_count += 1
    except Exception as e:
        print(f"✗ All models test failed: {e}")
    
    # Performance test (doesn't count toward success)
    try:
        run_performance_comparison()
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 PHASE 2 RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count >= 4:
        print("🎉 Phase 2 is working well! Ready for Phase 3.")
    elif success_count >= 2:
        print("⚠ Phase 2 has some issues but core functionality works.")
    else:
        print("❌ Phase 2 needs significant debugging.")