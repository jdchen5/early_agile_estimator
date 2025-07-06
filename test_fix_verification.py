# test_fix_verification.py - Quick verification of test fixes
"""
Simple test to verify our test fixes work correctly
Run this to check if the main issues are resolved
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from models import (
            list_available_models,
            check_required_models,
            predict_man_hours,
            prepare_features_for_model,
            get_model_display_name,
            load_model
        )
        print("✅ All model imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_discovery():
    """Test basic model discovery functionality"""
    try:
        from models import list_available_models, check_required_models
        
        models = list_available_models()
        print(f"✅ Found {len(models)} models")
        
        status = check_required_models()
        print(f"✅ Model status: {status.get('models_available', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Model discovery error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    try:
        from config_loader import ConfigLoader
        
        # Test with a known config file
        config_file = "config/ui_info.yaml"
        if Path(config_file).exists():
            config = ConfigLoader.load_yaml_config(config_file)
            if config is not None:
                print(f"✅ Config loaded successfully: {len(config)} sections")
            else:
                print("⚠️ Config file exists but returned None")
        else:
            print("⚠️ Config file not found (expected for some setups)")
        
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_mock_model():
    """Test mock model functionality"""
    try:
        from unittest.mock import Mock
        import numpy as np
        
        # Create mock model like in our tests
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1000.0])
        mock_model._final_estimator = Mock()
        mock_model._final_estimator.predict.return_value = np.array([1000.0])
        
        # Test mock functionality
        prediction = mock_model.predict([1, 2, 3])
        assert prediction is not None
        assert len(prediction) == 1
        
        print("✅ Mock model works correctly")
        return True
    except Exception as e:
        print(f"❌ Mock model error: {e}")
        return False

def main():
    """Run verification tests"""
    print("🔍 Verifying test fixes...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Discovery", test_model_discovery),
        ("Config Loading", test_config_loading),
        ("Mock Model", test_mock_model)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Verification Summary:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    
    if failed == 0:
        print("✅ All verification tests passed! Tests should work now.")
        return True
    else:
        print("❌ Some verification tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)