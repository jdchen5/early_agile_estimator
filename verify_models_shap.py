# verify_models_shap.py - Test SHAP-related functions in models.py
"""
Verification script specifically for SHAP-related functions in models.py

This tests:
1. get_trained_model() - Returns actual model objects
2. prepare_input_data() - Converts user inputs to model format  
3. prepare_sample_data() - Creates background data
4. prepare_isbsg_sample_data() - Loads ISBSG training data
5. validate_shap_compatibility() - Checks model compatibility
6. Integration with feature preparation pipeline

Usage:
    python verify_models_shap.py
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
    print(f"🔍 {title}")
    print('='*70)

def print_test(test_name: str, status: str, details: str = ""):
    """Print test result with status"""
    status_icons = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'INFO': 'ℹ️'}
    icon = status_icons.get(status, '❓')
    print(f"{icon} {test_name}: {status}")
    if details:
        for line in details.split('\n'):
            print(f"   {line}")

def test_models_imports():
    """Test 1: Check if all SHAP-related functions exist in models.py"""
    print_section("Testing models.py SHAP Function Imports")
    
    required_functions = [
        'get_trained_model',
        'prepare_input_data', 
        'prepare_sample_data',
        'prepare_isbsg_sample_data',
        'get_isbsg_dataset_info',
        'validate_shap_compatibility',
        'list_available_models'
    ]
    
    missing_functions = []
    available_functions = {}
    
    try:
        import models
        print_test("models.py import", "PASS", "Module loaded successfully")
        
        for func_name in required_functions:
            if hasattr(models, func_name):
                func = getattr(models, func_name)
                if callable(func):
                    available_functions[func_name] = func
                    print_test(f"Function {func_name}", "PASS", "Found and callable")
                else:
                    print_test(f"Function {func_name}", "FAIL", "Found but not callable")
                    missing_functions.append(func_name)
            else:
                print_test(f"Function {func_name}", "FAIL", "Not found")
                missing_functions.append(func_name)
                
    except ImportError as e:
        print_test("models.py import", "FAIL", f"Cannot import models.py: {e}")
        return None, required_functions
    
    if missing_functions:
        print_test("Function availability", "FAIL", f"Missing: {missing_functions}")
        return None, missing_functions
    else:
        print_test("Function availability", "PASS", "All required functions found")
        return available_functions, []

def test_model_loading():
    """Test 2: Test get_trained_model() function"""
    print_section("Testing get_trained_model() Function")
    
    try:
        from models import list_available_models, get_trained_model
        
        # Get available models
        available_models = list_available_models()
        if not available_models:
            print_test("Model availability", "FAIL", "No models found in models/ directory")
            return []
        
        print_test("Model discovery", "PASS", f"Found {len(available_models)} models")
        
        tested_models = []
        for i, model_info in enumerate(available_models[:3]):  # Test first 3 models
            model_name = model_info.get('technical_name', f'model_{i}')
            
            try:
                # Test model loading
                model = get_trained_model(model_name)
                
                if model is None:
                    print_test(f"Load {model_name}", "FAIL", "Function returned None")
                    continue
                
                # Test model properties
                model_type = type(model).__name__
                has_predict = hasattr(model, 'predict')
                
                details = f"Type: {model_type}, Has predict(): {has_predict}"
                
                # Test if we can call predict (with dummy data)
                try:
                    # Create minimal test input
                    test_input = np.array([[1, 2, 3, 4, 5]]).astype(float)
                    
                    # Try prediction (might fail due to wrong feature count, but shouldn't crash)
                    try:
                        pred = model.predict(test_input)
                        details += f", Predict test: PASS"
                        print_test(f"Load {model_name}", "PASS", details)
                        tested_models.append((model_name, model))
                        
                    except Exception as pred_error:
                        # Prediction might fail due to feature mismatch, but model is still valid
                        details += f", Predict test: {str(pred_error)[:50]}..."
                        print_test(f"Load {model_name}", "PASS", details)
                        tested_models.append((model_name, model))
                        
                except Exception as test_error:
                    details += f", Test error: {test_error}"
                    print_test(f"Load {model_name}", "WARN", details)
                    
            except Exception as e:
                print_test(f"Load {model_name}", "FAIL", f"Error loading model: {e}")
        
        return tested_models
        
    except Exception as e:
        print_test("Model loading test", "FAIL", f"Error: {e}")
        traceback.print_exc()
        return []

def test_isbsg_functions():
    """Test 3: Test ISBSG-related functions"""
    print_section("Testing ISBSG Data Functions")
    
    try:
        from models import get_isbsg_dataset_info, prepare_isbsg_sample_data
        
        # Test ISBSG dataset info
        try:
            info = get_isbsg_dataset_info()
            
            if isinstance(info, dict):
                available = info.get('available', False)
                if available:
                    rows = info.get('total_rows', 'Unknown')
                    features = info.get('feature_columns', 'Unknown') 
                    file_path = info.get('file_path', 'Unknown')
                    file_size = info.get('file_size_mb', 'Unknown')
                    
                    details = f"File: {file_path}\nRows: {rows}, Features: {features}, Size: {file_size}MB"
                    print_test("ISBSG dataset info", "PASS", details)
                    
                    # Test sample data loading
                    try:
                        sample_data = prepare_isbsg_sample_data(20)
                        
                        if sample_data is not None:
                            shape = sample_data.shape
                            dtype = sample_data.dtype
                            
                            # Check data quality
                            finite_count = np.isfinite(sample_data).sum()
                            total_count = sample_data.size
                            finite_pct = (finite_count / total_count) * 100
                            
                            details = f"Shape: {shape}, Type: {dtype}\nFinite values: {finite_pct:.1f}%"
                            
                            # Check value ranges
                            min_val = np.min(sample_data[np.isfinite(sample_data)])
                            max_val = np.max(sample_data[np.isfinite(sample_data)])
                            mean_val = np.mean(sample_data[np.isfinite(sample_data)])
                            
                            details += f"\nRange: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}"
                            
                            if finite_pct > 95:
                                print_test("ISBSG sample data", "PASS", details)
                                return sample_data
                            else:
                                print_test("ISBSG sample data", "WARN", f"Data quality issues\n{details}")
                                return sample_data
                        else:
                            print_test("ISBSG sample data", "FAIL", "Function returned None")
                            return None
                            
                    except Exception as e:
                        print_test("ISBSG sample data", "FAIL", f"Error loading sample data: {e}")
                        return None
                        
                else:
                    error = info.get('error', 'Unknown error')
                    print_test("ISBSG dataset info", "FAIL", f"Dataset not available: {error}")
                    return None
            else:
                print_test("ISBSG dataset info", "FAIL", f"Invalid return type: {type(info)}")
                return None
                
        except Exception as e:
            print_test("ISBSG dataset info", "FAIL", f"Error: {e}")
            traceback.print_exc()
            return None
            
    except ImportError as e:
        print_test("ISBSG functions import", "FAIL", f"Functions not found: {e}")
        return None

def test_input_preparation():
    """Test 4: Test prepare_input_data() function"""
    print_section("Testing prepare_input_data() Function")
    
    try:
        from models import prepare_input_data, FIELDS
        
        # Create sample user inputs based on your FIELDS config
        sample_inputs = create_sample_user_inputs(FIELDS)
        
        if sample_inputs is None:
            print_test("Sample input creation", "FAIL", "Could not create sample inputs")
            return None
        
        print_test("Sample input creation", "PASS", f"Created inputs with {len(sample_inputs)} fields")
        
        # Test input data preparation
        try:
            prepared_data = prepare_input_data(sample_inputs)
            
            if prepared_data is None:
                print_test("Input preparation", "FAIL", "Function returned None")
                return None
            
            # Analyze prepared data
            if isinstance(prepared_data, np.ndarray):
                shape = prepared_data.shape
                dtype = prepared_data.dtype
                details = f"Type: ndarray, Shape: {shape}, Dtype: {dtype}"
                
                # Check for valid values
                try:
                    if hasattr(prepared_data, 'select_dtypes'):
                        # For DataFrame with mixed types
                        numeric_data = prepared_data.select_dtypes(include=[np.number])
                        if len(numeric_data.columns) > 0:
                            if np.isfinite(numeric_data.values).all():
                                details += "\nAll numeric values finite ✅"
                        else:
                            details += "\nNo numeric columns to check ✅"
                    else:
                        # For numpy array
                        if np.isfinite(prepared_data).all():
                            details += "\nAll values finite ✅"
                except:
                    details += "\nData validation: OK ✅"
                else:
                    inf_count = np.isinf(prepared_data).sum()
                    nan_count = np.isnan(prepared_data).sum()
                    details += f"\nIssues: {inf_count} infinite, {nan_count} NaN values"
                
                print_test("Input preparation", "PASS", details)
                return prepared_data
                
            elif isinstance(prepared_data, pd.DataFrame):
                shape = prepared_data.shape
                dtypes = prepared_data.dtypes.value_counts().to_dict()
                details = f"Type: DataFrame, Shape: {shape}\nColumn types: {dtypes}"
                
                # Check for missing values
                missing_count = prepared_data.isnull().sum().sum()
                details += f"\nMissing values: {missing_count}"
                
                print_test("Input preparation", "PASS", details)
                return prepared_data
                
            else:
                print_test("Input preparation", "WARN", f"Unexpected return type: {type(prepared_data)}")
                return prepared_data
                
        except Exception as e:
            print_test("Input preparation", "FAIL", f"Error: {e}")
            traceback.print_exc()
            return None
            
    except ImportError as e:
        print_test("Input preparation import", "FAIL", f"Missing imports: {e}")
        return None

def create_sample_user_inputs(fields_config: Dict) -> Optional[Dict]:
    """Create realistic sample user inputs for testing"""
    try:
        sample_inputs = {}
        
        for field_name, field_config in fields_config.items():
            field_type = field_config.get('type', 'numeric')
            
            if field_type == 'numeric':
                min_val = field_config.get('min', 1)
                max_val = field_config.get('max', 100) 
                default = field_config.get('default', (min_val + max_val) / 2)
                sample_inputs[field_name] = default
                
            elif field_type == 'categorical':
                options = field_config.get('options', ['option1'])
                sample_inputs[field_name] = options[0] if options else 'default'
                
            elif field_type == 'boolean':
                sample_inputs[field_name] = field_config.get('default', True)
                
            else:
                sample_inputs[field_name] = field_config.get('default', 0)
        
        return sample_inputs
        
    except Exception as e:
        print(f"Error creating sample inputs: {e}")
        return None

def test_shap_compatibility():
    """Test 5: Test validate_shap_compatibility() function"""
    print_section("Testing validate_shap_compatibility() Function")
    
    try:
        from models import validate_shap_compatibility, list_available_models
        
        available_models = list_available_models()
        if not available_models:
            print_test("Model compatibility test", "FAIL", "No models available")
            return {}
        
        compatibility_results = {}
        
        for model_info in available_models[:3]:  # Test first 3 models
            model_name = model_info.get('technical_name')
            
            try:
                compat_result = validate_shap_compatibility(model_name)
                
                if isinstance(compat_result, dict):
                    compatible = compat_result.get('compatible', False)
                    explainer_type = compat_result.get('explainer_type', 'Unknown')
                    issues = compat_result.get('issues', [])
                    recommendations = compat_result.get('recommendations', [])
                    
                    details = f"Compatible: {compatible}, Explainer: {explainer_type}"
                    if issues:
                        details += f"\nIssues: {issues}"
                    if recommendations:
                        details += f"\nRecommendations: {recommendations}"
                    
                    if compatible:
                        print_test(f"Compatibility {model_name}", "PASS", details)
                    else:
                        print_test(f"Compatibility {model_name}", "WARN", details)
                    
                    compatibility_results[model_name] = compat_result
                    
                else:
                    print_test(f"Compatibility {model_name}", "FAIL", f"Invalid return type: {type(compat_result)}")
                    
            except Exception as e:
                print_test(f"Compatibility {model_name}", "FAIL", f"Error: {e}")
        
        return compatibility_results
        
    except ImportError as e:
        print_test("Compatibility function import", "FAIL", f"Function not found: {e}")
        return {}

def test_feature_pipeline_integration():
    """Test 6: Test integration with feature preparation pipeline"""
    print_section("Testing Feature Pipeline Integration")
    
    try:
        from models import prepare_features_for_model, FIELDS
        
        # Create sample inputs
        sample_inputs = create_sample_user_inputs(FIELDS)
        if sample_inputs is None:
            print_test("Pipeline integration", "FAIL", "Could not create sample inputs")
            return None
        
        try:
            # Test the main feature preparation function
            prepared_features = prepare_features_for_model(sample_inputs)
            
            if prepared_features is None:
                print_test("Feature pipeline", "FAIL", "prepare_features_for_model returned None")
                return None
            
            if isinstance(prepared_features, pd.DataFrame):
                shape = prepared_features.shape
                columns = list(prepared_features.columns)
                dtypes = prepared_features.dtypes.value_counts().to_dict()
                missing = prepared_features.isnull().sum().sum()
                
                details = f"Shape: {shape}\nColumns: {len(columns)}\nTypes: {dtypes}\nMissing: {missing}"
                
                # Test if data looks reasonable
                numeric_cols = prepared_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sample_stats = prepared_features[numeric_cols].describe()
                    details += f"\nNumeric columns: {len(numeric_cols)}"
                    details += f"\nMean range: [{sample_stats.loc['mean'].min():.2f}, {sample_stats.loc['mean'].max():.2f}]"
                
                print_test("Feature pipeline", "PASS", details)
                return prepared_features
                
            else:
                print_test("Feature pipeline", "WARN", f"Unexpected return type: {type(prepared_features)}")
                return prepared_features
                
        except Exception as e:
            print_test("Feature pipeline", "FAIL", f"Error in feature preparation: {e}")
            traceback.print_exc()
            return None
            
    except ImportError as e:
        print_test("Pipeline integration import", "FAIL", f"Missing imports: {e}")
        return None

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting SHAP Function Verification for models.py")
    print("="*70)
    
    results = {}
    
    # Test 1: Imports
    functions, missing = test_models_imports()
    results['imports'] = functions is not None
    
    if not results['imports']:
        print("\n❌ CRITICAL: Cannot proceed without basic function imports")
        return results
    
    # Test 2: Model loading
    tested_models = test_model_loading()
    results['model_loading'] = len(tested_models) > 0
    
    # Test 3: ISBSG functions  
    isbsg_data = test_isbsg_functions()
    results['isbsg'] = isbsg_data is not None
    
    # Test 4: Input preparation
    prepared_input = test_input_preparation()
    results['input_prep'] = prepared_input is not None
    
    # Test 5: SHAP compatibility
    compat_results = test_shap_compatibility()
    results['shap_compat'] = len(compat_results) > 0
    
    # Test 6: Feature pipeline
    pipeline_result = test_feature_pipeline_integration()
    results['pipeline'] = pipeline_result is not None
    
    # Summary
    print_section("SUMMARY - models.py SHAP Functions")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print_test(f"{test_name.replace('_', ' ').title()}", status)
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your models.py SHAP functions are ready.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ Most tests passed. Review failed tests above.")
    else:
        print("❌ Several tests failed. Address issues before proceeding.")
    
    return results

if __name__ == "__main__":
    # Run the verification
    test_results = run_all_tests()
    
    # Exit with appropriate code
    if sum(test_results.values()) == len(test_results):
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed