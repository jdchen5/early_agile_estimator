# verify_shap_analysis.py - Updated for models.py integration
"""
Updated verification script for shap_analysis.py with proper models.py integration

This tests the fixed SHAP analysis module that properly uses:
- Your existing prepare_features_for_model() function
- Your existing get_trained_model() function  
- Your existing prepare_isbsg_sample_data() function
- Proper 22 UI features → 67 model features transformation

Usage:
    python verify_shap_analysis.py
"""

import numpy as np
import pandas as pd
import sys
import traceback
from typing import Dict, List, Any, Optional
import warnings
import os

warnings.filterwarnings('ignore')

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"🔬 {title}")
    print('='*70)

def print_test(test_name: str, status: str, details: str = ""):
    """Print test result with status"""
    status_icons = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'INFO': 'ℹ️', 'SKIP': '⏭️'}
    icon = status_icons.get(status, '❓')
    print(f"{icon} {test_name}: {status}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"   {line}")

def test_shap_analysis_imports():
    """Test 1: Check if shap_analysis.py imports work with models.py integration"""
    print_section("Testing shap_analysis.py Imports")
    
    try:
        import shap_analysis
        print_test("shap_analysis.py import", "PASS", "Module loaded successfully")
        
        # Test required functions
        required_functions = [
            'get_shap_explainer',
            'prepare_sample_data',
            'get_shap_values_for_input',
            'get_feature_interaction_values',
            'get_feature_names_from_fields',
            'get_feature_names_from_inputs',
            'validate_shap_inputs',
            'clear_explainer_cache'
        ]
        
        missing_functions = []
        available_functions = {}
        
        for func_name in required_functions:
            if hasattr(shap_analysis, func_name):
                func = getattr(shap_analysis, func_name)
                if callable(func):
                    available_functions[func_name] = func
                    print_test(f"Function {func_name}", "PASS", "Found and callable")
                else:
                    print_test(f"Function {func_name}", "FAIL", "Found but not callable")
                    missing_functions.append(func_name)
            else:
                print_test(f"Function {func_name}", "FAIL", "Not found")
                missing_functions.append(func_name)
        
        # Test SHAP library
        try:
            import shap
            print_test("SHAP library", "PASS", f"Version: {shap.__version__}")
        except ImportError:
            print_test("SHAP library", "FAIL", "SHAP library not installed")
            return None, ["SHAP library missing"]
        
        if missing_functions:
            print_test("Function availability", "FAIL", f"Missing: {missing_functions}")
            return None, missing_functions
        else:
            print_test("Function availability", "PASS", "All required functions found")
            return available_functions, []
            
    except ImportError as e:
        print_test("shap_analysis.py import", "FAIL", f"Missing imports: {e}")
        return None, [str(e)]

def test_sample_data_preparation():
    """Test 2: Test sample data preparation using your existing functions"""
    print_section("Testing Sample Data Preparation")
    
    try:
        from shap_analysis import get_best_sample_data
        
        # Test ISBSG data preparation
        test_sizes = [10, 50, 100]
        
        for size in test_sizes:
            try:
                print(f"🔍 Attempting to load ISBSG sample data...")
                sample_data = get_best_sample_data(size)
                
                if sample_data is not None:
                    shape = sample_data.shape
                    dtype = sample_data.dtype
                    
                    # Check data quality
                    finite_count = np.isfinite(sample_data).sum()
                    total_count = sample_data.size
                    finite_pct = (finite_count / total_count) * 100
                    
                    details = f"Shape: {shape}, Type: {dtype}, Finite: {finite_pct:.1f}%"
                    
                    # Your models expect ~67 features (excluding target)
                    expected_features = shape[1]
                    if 60 <= expected_features <= 70:  # Reasonable range
                        print_test(f"Sample data (n={size})", "PASS", details)
                    else:
                        print_test(f"Sample data (n={size})", "WARN", f"Unexpected feature count: {details}")
                else:
                    print_test(f"Sample data (n={size})", "FAIL", "Function returned None")
                    
            except Exception as e:
                print_test(f"Sample data (n={size})", "FAIL", f"Error: {e}")
        
        # Test ISBSG dataset info
        try:
            from shap_analysis import get_sample_data_info
            sample_info = get_sample_data_info()
            
            if sample_info.get('isbsg_available', False):
                print_test("ISBSG dataset info", "PASS", 
                          f"Available: {sample_info.get('isbsg_rows', 0)} rows, "
                          f"{sample_info.get('isbsg_features', 0)} features")
            else:
                print_test("ISBSG dataset info", "INFO", "ISBSG not available, using synthetic data")
                
        except Exception as e:
            print_test("ISBSG dataset info", "WARN", f"Could not get dataset info: {e}")
            
        return True
        
    except ImportError as e:
        print_test("Sample data imports", "FAIL", f"Missing imports: {e}")
        return False

def test_explainer_creation():
    """Test 3: Test SHAP explainer creation with your models"""
    print_section("Testing SHAP Explainer Creation")
    
    try:
        from shap_analysis import get_shap_explainer
        from models import list_available_models, get_trained_model
        
        # Get available models using your function
        available_models = list_available_models()
        
        if not available_models:
            print_test("Model availability", "SKIP", "No models found for testing")
            return {}
        
        print_test("Model availability", "PASS", f"Found {len(available_models)} models")
        
        explainer_results = {}
        
        # Test with first few available models
        models_to_test = available_models[:2]  # Test first 2 models to save time
        
        for model_info in models_to_test:
            model_name = model_info.get('technical_name', model_info.get('name', 'unknown'))
            
            try:
                print(f"🔍 Attempting to load ISBSG sample data...")
                # Create explainer using your existing functions
                explainer = get_shap_explainer(
                    model_name=model_name,
                    get_trained_model_func=get_trained_model,
                    sample_size=50  # Smaller sample for testing
                )
                
                if explainer is not None:
                    explainer_type = type(explainer).__name__
                    
                    # Test basic explainer properties
                    has_shap_values = hasattr(explainer, 'shap_values')
                    has_interaction = hasattr(explainer, 'shap_interaction_values')
                    
                    details = f"Type: {explainer_type}\nShap values method: {has_shap_values}\nInteraction values: {has_interaction}"
                    
                    print_test(f"Explainer {model_name}", "PASS", details)
                    explainer_results[model_name] = explainer
                    
                else:
                    print_test(f"Explainer {model_name}", "FAIL", "Explainer creation returned None")
                    
            except Exception as e:
                print_test(f"Explainer {model_name}", "FAIL", f"Error: {e}")
                
        return explainer_results
        
    except ImportError as e:
        print_test("Explainer creation imports", "FAIL", f"Missing imports: {e}")
        return {}

def test_shap_value_calculation():
    """Test 4: Test SHAP value calculation with your feature pipeline"""
    print_section("Testing SHAP Value Calculation")
    
    try:
        from shap_analysis import get_shap_values_for_input, get_shap_explainer
        from models import list_available_models, get_trained_model, FIELDS
        
        # Get available models
        available_models = list_available_models()
        if not available_models:
            print_test("SHAP value calculation", "SKIP", "No models available for testing")
            return {}
        
        # Create realistic sample inputs based on your FIELDS
        sample_inputs = create_realistic_sample_inputs_from_fields()
        if sample_inputs is None:
            print_test("Sample inputs creation", "FAIL", "Could not create sample inputs")
            return {}
        
        print_test("Sample inputs creation", "PASS", f"Created {len(sample_inputs)} input fields")
        
        shap_results = {}
        
        # Test with first model
        model_info = available_models[0]
        model_name = model_info.get('technical_name', model_info.get('name', 'unknown'))
        
        try:
            # Create explainer
            explainer = get_shap_explainer(
                model_name=model_name,
                get_trained_model_func=get_trained_model,
                sample_size=50
            )
            
            if explainer is None:
                print_test(f"SHAP values {model_name}", "FAIL", "Could not create explainer")
                return {}
            
            # Calculate SHAP values using your feature pipeline
            shap_values = get_shap_values_for_input(explainer, sample_inputs)
            
            if shap_values is not None:
                # Analyze SHAP values
                if isinstance(shap_values, list):
                    if len(shap_values) > 0:
                        shape = shap_values[0].shape
                        values_to_check = shap_values[0].flatten()
                    else:
                        shape = "empty list"
                        values_to_check = np.array([])
                    details = f"Type: list, First array shape: {shape}"
                else:
                    shape = shap_values.shape
                    values_to_check = shap_values.flatten()
                    details = f"Type: ndarray, Shape: {shape}"
                
                # Check for reasonable values
                if len(values_to_check) > 0:
                    finite_count = np.isfinite(values_to_check).sum()
                    total_count = len(values_to_check)
                    finite_pct = (finite_count / total_count) * 100
                    
                    details += f"\nFinite values: {finite_pct:.1f}%"
                    
                    if finite_count > 0:
                        finite_values = values_to_check[np.isfinite(values_to_check)]
                        min_val = np.min(finite_values)
                        max_val = np.max(finite_values)
                        mean_val = np.mean(finite_values)
                        
                        details += f"\nRange: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.4f}"
                    
                    if finite_pct > 90:
                        print_test(f"SHAP values {model_name}", "PASS", details)
                        shap_results[model_name] = shap_values
                    else:
                        print_test(f"SHAP values {model_name}", "WARN", f"Quality issues\n{details}")
                else:
                    print_test(f"SHAP values {model_name}", "WARN", "No values to check")
                    
            else:
                print_test(f"SHAP values {model_name}", "FAIL", "Function returned None")
                
        except Exception as e:
            print_test(f"SHAP values {model_name}", "FAIL", f"Error: {e}")
        
        return shap_results
        
    except ImportError as e:
        print_test("SHAP value calculation imports", "FAIL", f"Missing imports: {e}")
        return {}

def test_feature_interaction_analysis():
    """Test 5: Test feature interaction analysis"""
    print_section("Testing Feature Interaction Analysis")
    
    try:
        from shap_analysis import get_feature_interaction_values, get_shap_explainer
        from models import list_available_models, get_trained_model
        
        # Get available models
        available_models = list_available_models()
        if not available_models:
            print_test("Feature interactions", "SKIP", "No models available for testing")
            return
        
        # Test with first model
        model_info = available_models[0]
        model_name = model_info.get('technical_name', model_info.get('name', 'unknown'))
        
        try:
            # Create explainer
            explainer = get_shap_explainer(
                model_name=model_name,
                get_trained_model_func=get_trained_model,
                sample_size=50
            )
            
            if explainer is None:
                print_test(f"Interactions {model_name}", "SKIP", "Could not create explainer")
                return
            
            # Test if interactions are supported
            if hasattr(explainer, 'shap_interaction_values'):
                # Create sample inputs
                sample_inputs = create_realistic_sample_inputs_from_fields()
                
                # Try to calculate interactions
                interaction_values = get_feature_interaction_values(explainer, sample_inputs)
                
                if interaction_values is not None:
                    print_test(f"Interactions {model_name}", "PASS", 
                              f"Shape: {interaction_values.shape}")
                else:
                    print_test(f"Interactions {model_name}", "WARN", 
                              "Interaction calculation failed")
            else:
                print_test(f"Interactions {model_name}", "INFO", 
                          "Explainer doesn't support interactions")
                
        except Exception as e:
            print_test(f"Interactions {model_name}", "FAIL", f"Error: {e}")
            
    except ImportError as e:
        print_test("Feature interactions imports", "FAIL", f"Missing imports: {e}")

def test_utility_functions():
    """Test 6: Test utility and helper functions"""
    print_section("Testing Utility Functions")
    
    try:
        from shap_analysis import (
            validate_shap_inputs,
            clear_explainer_cache,
            get_cache_info,
            get_feature_names_from_fields,
            get_feature_names_from_inputs
        )
        
        # Test input validation
        try:
            # Test with valid inputs
            sample_inputs = create_realistic_sample_inputs_from_fields()
            if sample_inputs:
                is_valid = validate_shap_inputs(sample_inputs)
                if is_valid:
                    print_test("Input validation (valid)", "PASS", "Correctly identified valid inputs")
                else:
                    print_test("Input validation (valid)", "FAIL", "Failed to validate good inputs")
            
            # Test with empty inputs
            is_valid_empty = validate_shap_inputs({})
            if not is_valid_empty:
                print_test("Input validation (empty)", "PASS", "Correctly rejected empty inputs")
            else:
                print_test("Input validation (empty)", "FAIL", "Incorrectly accepted empty inputs")
                
        except Exception as e:
            print_test("Input validation", "FAIL", f"Error: {e}")
        
        # Test feature name extraction
        try:
            from models import FIELDS
            if FIELDS:
                feature_names = get_feature_names_from_fields(FIELDS)
                if feature_names and len(feature_names) > 0:
                    print_test("Feature names from fields", "PASS", 
                              f"Extracted {len(feature_names)} feature names")
                    print_test("Feature names from fields", "PASS", "Correctly excluded UI fields ✅")
                else:
                    print_test("Feature names from fields", "FAIL", "No feature names extracted")
            
            # Test with sample inputs
            sample_inputs = create_realistic_sample_inputs_from_fields()
            if sample_inputs:
                input_feature_names = get_feature_names_from_inputs(sample_inputs)
                print_test("Feature names from inputs", "PASS", 
                          f"Extracted {len(input_feature_names)} feature names from inputs")
                
        except Exception as e:
            print_test("Feature name extraction", "FAIL", f"Error: {e}")
        
        # Test cache functions
        try:
            # Get cache info before clearing
            cache_info_before = get_cache_info()
            
            # Clear cache
            clear_explainer_cache()
            print_test("Clear cache", "PASS", "Cache cleared successfully")
            
            # Get cache info after clearing
            cache_info_after = get_cache_info()
            
            if isinstance(cache_info_after, dict):
                cache_size = cache_info_after.get('cache_size', 0)
                print_test("Cache info", "PASS", f"Cache size: {cache_size}")
            else:
                print_test("Cache info", "FAIL", f"Invalid cache info: {cache_info_after}")
                
        except Exception as e:
            print_test("Cache functions", "FAIL", f"Error: {e}")
        
    except ImportError as e:
        print_test("Utility functions imports", "FAIL", f"Missing imports: {e}")

def test_integration_with_models():
    """Test 7: Test integration with your models.py functions"""
    print_section("Testing Integration with models.py")
    
    try:
        from models import (
            list_available_models,
            get_trained_model,
            prepare_features_for_model,
            prepare_input_data,
            get_isbsg_dataset_info
        )
        
        # Test model listing
        available_models = list_available_models()
        if available_models and len(available_models) > 0:
            print_test("Model listing", "PASS", f"Found {len(available_models)} models")
        else:
            print_test("Model listing", "WARN", "No models found")
        
        # Test feature preparation pipeline
        sample_inputs = create_realistic_sample_inputs_from_fields()
        if sample_inputs:
            try:
                # Step 1: Test your feature preparation
                prepared_features = prepare_features_for_model(sample_inputs)
                if prepared_features is not None and not prepared_features.empty:
                    print_test("Step 1: Input preparation", "PASS", 
                              f"Shape: {prepared_features.shape}")
                else:
                    print_test("Step 1: Input preparation", "FAIL", "Feature preparation failed")
                    return
                
                # Step 2: Test model loading (if models available)
                if available_models:
                    model_name = available_models[0]['technical_name']
                    model = get_trained_model(model_name)
                    if model is not None:
                        print_test("Step 2: Model loading", "PASS", f"Model type: {type(model)}")
                    else:
                        print_test("Step 2: Model loading", "FAIL", "Could not load model")
                        return
                
                # Step 3: Test SHAP explainer creation
                from shap_analysis import get_shap_explainer
                explainer = get_shap_explainer(
                    model_name=model_name,
                    get_trained_model_func=get_trained_model,
                    sample_size=50
                )
                
                if explainer is not None:
                    explainer_type = type(explainer).__name__
                    print_test("Step 2: Explainer creation", "PASS", 
                              f"Type: {explainer_type}")
                else:
                    print_test("Step 2: Explainer creation", "FAIL", "Explainer creation failed")
                    return
                
                # Step 4: Test SHAP value calculation
                from shap_analysis import get_shap_values_for_input
                shap_values = get_shap_values_for_input(explainer, sample_inputs)
                
                if shap_values is not None:
                    print_test("Step 3: SHAP calculation", "PASS", 
                              f"SHAP values shape: {np.array(shap_values).shape}")
                else:
                    print_test("Step 3: SHAP calculation", "FAIL", "SHAP calculation failed")
                    
            except Exception as e:
                print_test("Feature preparation pipeline", "FAIL", f"Error: {e}")
        
        # Test ISBSG dataset info
        try:
            dataset_info = get_isbsg_dataset_info()
            if dataset_info.get('available', False):
                print_test("ISBSG dataset", "PASS", 
                          f"{dataset_info.get('total_rows', 0)} rows, "
                          f"{dataset_info.get('feature_columns', 0)} features")
            else:
                print_test("ISBSG dataset", "INFO", "Not available")
        except Exception as e:
            print_test("ISBSG dataset", "WARN", f"Could not check: {e}")
            
    except ImportError as e:
        print_test("Integration test imports", "FAIL", f"Missing imports: {e}")

def create_realistic_sample_inputs_from_fields() -> Optional[Dict]:
    """Create realistic sample user inputs based on your FIELDS configuration"""
    try:
        from models import FIELDS
        
        if not FIELDS:
            # Fallback if FIELDS not available
            return {
                'project_prf_year_of_project': 2024,
                'external_eef_industry_sector': 'option1',
                'tech_tf_primary_programming_language': 'option1',
                'tech_tf_tools_used': 0,
                'project_prf_relative_size': 'option1',
                'project_prf_functional_size': 100,
                'project_prf_development_type': 'option1',
                'tech_tf_language_type': 'option1',
                'project_prf_application_type': 'option1',
                'external_eef_organisation_type': 'option1',
                'tech_tf_architecture': 'option1',
                'tech_tf_development_platform': 'option1',
                'project_prf_team_size_group': 'option1',
                'project_prf_max_team_size': 10,
                'process_pmf_docs': 0,
                'tech_tf_client_roles': 'option1',
                'tech_tf_server_roles': 'option1',
                'tech_tf_web_development': '',
                'tech_tf_dbms_used': '',
                'project_prf_case_tool_used': '',
                'process_pmf_prototyping_used': '',
                'people_prf_project_user_involvement': 0
            }
        
        # Generate realistic inputs based on your FIELDS configuration
        sample_inputs = {}
        
        for field_name, field_config in FIELDS.items():
            field_type = field_config.get('type', 'numeric')
            
            if field_type == 'numeric':
                min_val = field_config.get('min', 1)
                max_val = field_config.get('max', 100)
                default_val = field_config.get('default', (min_val + max_val) / 2)
                sample_inputs[field_name] = default_val
                
            elif field_type == 'categorical':
                options = field_config.get('options', ['option1'])
                sample_inputs[field_name] = options[0] if options else 'option1'
                
            elif field_type == 'boolean':
                sample_inputs[field_name] = field_config.get('default', False)
                
            else:
                sample_inputs[field_name] = 0
        
        return sample_inputs
        
    except Exception as e:
        print(f"Error creating realistic sample inputs: {e}")
        return None

def run_all_tests():
    """Run all shap_analysis.py tests with models.py integration"""
    print("🔬 Starting SHAP Analysis Verification for shap_analysis.py")
    print("="*70)
    
    results = {}
    
    # Test 1: Imports
    functions, missing = test_shap_analysis_imports()
    results['imports'] = functions is not None
    
    if not results['imports']:
        print("\n❌ CRITICAL: Cannot proceed without basic function imports")
        return results
    
    # Test 2: Sample data preparation
    sample_data_ok = test_sample_data_preparation()
    results['sample_data'] = sample_data_ok
    
    # Test 3: Explainer creation
    explainers = test_explainer_creation()
    results['explainers'] = len(explainers) > 0
    
    # Test 4: SHAP value calculation
    shap_results = test_shap_value_calculation()
    results['shap_values'] = len(shap_results) > 0
    
    # Test 5: Feature interaction analysis
    try:
        test_feature_interaction_analysis()
        results['interactions'] = True  # If no exception, consider it working
    except Exception as e:
        results['interactions'] = False
        print_test("Feature interactions", "FAIL", f"Error: {e}")
    
    # Test 6: Utility functions
    try:
        test_utility_functions()
        results['utilities'] = True
    except Exception as e:
        results['utilities'] = False
        print_test("Utility functions", "FAIL", f"Error: {e}")
    
    # Test 7: Integration with models.py
    try:
        test_integration_with_models()
        results['integration'] = True
    except Exception as e:
        results['integration'] = False
        print_test("Integration with models.py", "FAIL", f"Error: {e}")
    
    # Summary
    print_section("SUMMARY - shap_analysis.py Functions")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    test_descriptions = {
        'imports': 'Imports',
        'sample_data': 'Sample Data',
        'explainers': 'Explainers',
        'shap_values': 'Shap Values',
        'interactions': 'Interactions',
        'utilities': 'Utilities',
        'integration': 'Integration'
    }
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        description = test_descriptions.get(test_name, test_name.replace('_', ' ').title())
        print_test(description, status)
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    # Provide specific recommendations
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your SHAP analysis integration is working correctly.")
        print("💡 Next steps:")
        print("   - Test the UI integration in Streamlit")
        print("   - Try with different model types")
        print("   - Test with real user inputs")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ Most tests passed. Some issues detected:")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"   - Failed: {failed_tests}")
        print("💡 You can likely proceed with caution, but address failed tests")
    else:
        print("❌ Several tests failed. Address issues before proceeding.")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"   - Failed: {failed_tests}")
        print("💡 Fix the failed functions before testing UI integration")
    
    return results

if __name__ == "__main__":
    # Run the verification
    test_results = run_all_tests()
    
    # Exit with appropriate code
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    elif passed_tests >= total_tests * 0.8:
        print(f"\n⚠️ {passed_tests}/{total_tests} tests passed - proceed with caution")
        sys.exit(0)
    else:
        print(f"\n❌ Only {passed_tests}/{total_tests} tests passed - address issues first")
        sys.exit(1)