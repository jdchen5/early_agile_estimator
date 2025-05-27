"""
test_models.py - Comprehensive testing for PyCaret models
Run this file to test your models before using them in main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from models import load_model, predict_man_months, get_feature_importance

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models import (
        check_required_models,
        list_available_models,
        load_model,
        load_scaler,
        predict_man_months,
        debug_pycaret_model_files
    )
except ImportError as e:
    print(f"Error importing models.py: {e}")
    print("Make sure models.py is in the same directory as this test file.")
    sys.exit(1)

def print_header(title):
    """Print a formatted header for test sections."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")

def test_1_environment_check():
    """Test 1: Check Python environment and imports."""
    print_header("TEST 1: ENVIRONMENT CHECK")
    
    success = True
    
    try:
        import pycaret
        print(f"✓ PyCaret version: {pycaret.__version__}")
    except ImportError:
        print("✗ PyCaret not installed - Install with: pip install pycaret")
        success = False
    
    try:
        import streamlit
        print(f"✓ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("✗ Streamlit not installed - Install with: pip install streamlit")
        success = False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not installed - Install with: pip install scikit-learn")
        success = False
    
    print(f"✓ Python version: {sys.version}")
    print(f"✓ Current working directory: {os.getcwd()}")
    
    return success

def test_2_model_files():
    """Test 2: Check model files and directory structure."""
    print_header("TEST 2: MODEL FILES CHECK")
    
    # Check if models directory exists
    models_dir = 'models'
    if os.path.exists(models_dir):
        print(f"✓ Models directory exists: {models_dir}")
        
        # List all files
        files = os.listdir(models_dir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        
        print(f"✓ Found {len(pkl_files)} .pkl files:")
        for file in pkl_files:
            file_path = os.path.join(models_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size:,} bytes)")
        
        if len(pkl_files) == 0:
            print("✗ No .pkl files found in models directory!")
            return False
            
    else:
        print(f"✗ Models directory not found: {models_dir}")
        print("Please create the models directory and add your trained models.")
        return False
    
    # Use our function to check models
    try:
        model_status = check_required_models()
        print(f"\nModel Status Summary:")
        for key, value in model_status.items():
            print(f"  {key}: {value}")
        
        return model_status.get('models_available', False)
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False

def test_3_model_loading():
    """Test 3: Try loading each model."""
    print_header("TEST 3: MODEL LOADING")
    
    try:
        available_models = list_available_models()
        print(f"Available models: {available_models}")
        
        if not available_models:
            print("✗ No models found to test")
            return {}
        
        loaded_models = {}
        
        for model_name in available_models:
            print_subheader(f"Loading {model_name}")
            
            try:
                model = load_model(model_name)
                if model is not None:
                    print(f"✓ Successfully loaded")
                    print(f"  Type: {type(model)}")
                    print(f"  Has predict method: {hasattr(model, 'predict')}")
                    
                    # Try to get more info about the model
                    if hasattr(model, 'named_steps'):
                        print(f"  Pipeline steps: {list(model.named_steps.keys())}")
                    elif hasattr(model, '_final_estimator'):
                        print(f"  Final estimator: {type(model._final_estimator)}")
                    
                    loaded_models[model_name] = model
                else:
                    print(f"✗ Failed to load model (returned None)")
            except Exception as e:
                print(f"✗ Error loading model: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(loaded_models)} out of {len(available_models)} models")
        return loaded_models
        
    except Exception as e:
        print(f"✗ Error in model loading test: {e}")
        return {}

def test_4_scaler_loading():
    """Test 4: Try loading scaler."""
    print_header("TEST 4: SCALER LOADING")
    
    try:
        scaler = load_scaler()
        if scaler is not None:
            print(f"✓ Successfully loaded scaler")
            print(f"  Type: {type(scaler)}")
            print(f"  Has transform method: {hasattr(scaler, 'transform')}")
            return scaler
        else:
            print("ℹ No scaler found")
            print("  This is okay for PyCaret models as they often handle scaling internally")
            return None
    except Exception as e:
        print(f"✗ Error loading scaler: {str(e)}")
        return None

def test_5_sample_predictions():
    """Test 5: Make sample predictions with different feature combinations."""
    print_header("TEST 5: SAMPLE PREDICTIONS")
    
    try:
        available_models = list_available_models()
        
        if not available_models:
            print("✗ No models available for prediction testing")
            return False
        
        # Create sample feature arrays of different sizes
        sample_features_sets = [
            {
                'name': '4 features (basic)',
                'features': np.array([1.0, 2.0, 3.0, 4.0])
            },
            {
                'name': '5 features (common)',
                'features': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            },
            {
                'name': '6 features',
                'features': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            },
            {
                'name': '8 features (extended)',
                'features': np.array([10.0, 20.0, 5.0, 15.0, 3.0, 8.0, 12.0, 25.0])
            },
            {
                'name': '10 features (full)',
                'features': np.array([100, 50, 20, 10, 5, 3, 2, 1, 0.5, 0.1])
            }
        ]
        
        success_count = 0
        total_attempts = 0
        
        # Test each model with different feature sets
        for model_name in available_models:
            print_subheader(f"Testing predictions with {model_name}")
            
            model_success = False
            
            for feature_set in sample_features_sets:
                features = feature_set['features']
                feature_name = feature_set['name']
                
                print(f"\n  Testing {feature_name}: {features[:3]}...")
                total_attempts += 1
                
                try:
                    # Test with scaler
                    prediction_with_scaler = predict_man_months(features, model_name, use_scaler=True)
                    
                    if prediction_with_scaler is not None:
                        print(f"  ✓ With scaler: {prediction_with_scaler:.2f} man-months")
                        
                        # Test without scaler
                        prediction_without_scaler = predict_man_months(features, model_name, use_scaler=False)
                        if prediction_without_scaler is not None:
                            print(f"  ✓ Without scaler: {prediction_without_scaler:.2f} man-months")
                        
                        success_count += 1
                        model_success = True
                        break  # Found working feature size for this model
                    else:
                        print(f"  ✗ Prediction returned None")
                        
                except Exception as e:
                    print(f"  ✗ Prediction error: {str(e)}")
            
            if model_success:
                print(f"  🎉 Model {model_name} works!")
            else:
                print(f"  ❌ Model {model_name} failed all feature size tests")
            
            print()  # Empty line between models
        
        print(f"Prediction Success Rate: {success_count}/{total_attempts}")
        return success_count > 0
        
    except Exception as e:
        print(f"✗ Error in prediction testing: {e}")
        return False

def test_6_detailed_debug():
    """Test 6: Run detailed debugging."""
    print_header("TEST 6: DETAILED DEBUGGING")
    
    try:
        debug_pycaret_model_files()
    except Exception as e:
        print(f"Error in detailed debugging: {e}")

def quick_test():
    """Quick test - just check if basic functionality works."""
    print_header("QUICK TEST")
    
    try:
        # Check models exist
        models = list_available_models()
        print(f"Found models: {models}")
        
        if not models:
            print("❌ No models found")
            return False
        
        # Try to load first model
        first_model = models[0]
        model = load_model(first_model)
        
        if model is None:
            print(f"❌ Could not load model: {first_model}")
            return False
        
        print(f"✓ Loaded model: {first_model}")
        
        # Try a simple prediction
        sample_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = predict_man_months(sample_features, first_model)
        
        if prediction is not None:
            print(f"✓ Prediction works: {prediction:.2f} man-months")
            return True
        else:
            print("❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence."""
    print(f"🚀 Starting comprehensive test suite at {datetime.now()}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    test_results = {}
    
    # Run all tests
    try:
        test_results['environment'] = test_1_environment_check()
        test_results['model_files'] = test_2_model_files()
        test_results['model_loading'] = len(test_3_model_loading()) > 0
        test_results['scaler_loading'] = test_4_scaler_loading() is not None
        test_results['predictions'] = test_5_sample_predictions()
        
        # Detailed debug (always run for troubleshooting)
        test_6_detailed_debug()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {e}")
        return
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📈 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your models should work perfectly in the main Streamlit app.")
        print("You can now run: streamlit run main.py")
    elif total_passed >= 3:
        print("\n⚠️ MOST TESTS PASSED")
        print("Your models should mostly work. Check failed tests above.")
    else:
        print("\n❌ MULTIPLE TESTS FAILED")
        print("Please fix the issues above before running the main app.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install pycaret streamlit")
        print("2. Check if models directory exists and has .pkl files")
        print("3. Verify models were saved correctly with PyCaret")
    
    return test_results

def main():
    """Main function with options."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            success = quick_test()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "debug":
            test_6_detailed_debug()
            sys.exit(0)
    
    # Run full test suite
    run_all_tests()

if __name__ == "__main__":
    main()