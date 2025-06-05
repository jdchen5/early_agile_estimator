# save this as extract_my_model.py and run it

import pickle
import numpy as np
import sklearn.utils
import os

def fix_and_extract_model():
    """Extract your model by providing the missing sklearn function"""
    
    print("🔧 Applying compatibility fix for sklearn 1.6.1...")
    
    # Create a mock _print_elapsed_time function
    def mock_print_elapsed_time(func):
        """Mock version of the removed sklearn function"""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    # Inject the missing function into sklearn.utils
    sklearn.utils._print_elapsed_time = mock_print_elapsed_time
    print("✅ Compatibility shim installed")
    
    # Navigate to your project directory
    project_dir = r"C:\Users\jdche\Documents\GitHub\early_agile_estimator"
    if os.path.exists(project_dir):
        os.chdir(project_dir)
        print(f"📁 Changed to project directory: {project_dir}")
    else:
        print("⚠️ Using current directory")
    
    model_path = "models/top_model_2_ExtraTreesRegressor.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available files in models folder:")
        if os.path.exists("models"):
            for file in os.listdir("models"):
                print(f"  - {file}")
        return None
    
    try:
        print(f"📦 Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model)}")
        
        # Extract the core sklearn estimator
        core_model = None
        
        if hasattr(model, '_final_estimator'):
            # PyCaret model
            core_model = model._final_estimator
            print("🎯 Extracted core model from PyCaret wrapper")
            print(f"   Core model type: {type(core_model)}")
        elif hasattr(model, 'named_steps'):
            # Pipeline model - find the estimator
            print("🔍 Searching pipeline steps...")
            for step_name, step in model.named_steps.items():
                print(f"   Step '{step_name}': {type(step)}")
                if hasattr(step, 'predict') and hasattr(step, 'fit') and 'Tree' in str(type(step)):
                    core_model = step
                    print(f"🎯 Found estimator in step: {step_name}")
                    break
        else:
            # Direct model
            core_model = model
            print("🎯 Using model directly")
        
        if core_model is None:
            print("❌ Could not extract core model")
            return None
        
        # Test the core model
        print("🧪 Testing extracted model...")
        # Create test input - adjust features based on your model
        n_features = 27  # Common number for your setup
        if hasattr(core_model, 'n_features_in_'):
            n_features = core_model.n_features_in_
            print(f"   Model expects {n_features} features")
        
        test_input = np.random.rand(1, n_features)
        prediction = core_model.predict(test_input)
        print(f"✅ Test prediction successful: {prediction[0]:.2f}")
        
        # Save the extracted core model
        new_model_path = "models/fixed_extra_trees_regressor.pkl"
        with open(new_model_path, 'wb') as f:
            pickle.dump(core_model, f)
        
        print(f"💾 Extracted model saved to: {new_model_path}")
        
        # Verify the new model loads correctly
        print("🔍 Verifying new model...")
        with open(new_model_path, 'rb') as f:
            test_model = pickle.load(f)
        test_pred = test_model.predict(test_input)
        print(f"✅ New model loads and predicts: {test_pred[0]:.2f}")
        
        # Also save with a more user-friendly name
        friendly_path = "models/working_extra_trees_model.pkl"
        with open(friendly_path, 'wb') as f:
            pickle.dump(core_model, f)
        print(f"💾 Also saved as: {friendly_path}")
        
        print("\n🎉 SUCCESS!")
        print("Your model has been extracted and saved with sklearn 1.6.1 compatibility!")
        print("\n📋 Next steps:")
        print("1. Refresh your Streamlit app")
        print("2. Look for these new models in the dropdown:")
        print("   - 'Fixed Extra Trees Regressor'")
        print("   - 'Working Extra Trees Model'")
        print("3. Select one and try making a prediction!")
        
        return core_model
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide specific guidance based on error
        error_str = str(e)
        if "cannot import name" in error_str:
            print("   → Still having import issues. The model may have deeper compatibility problems.")
        elif "pickle" in error_str.lower():
            print("   → Pickle loading issue. File may be corrupted.")
        
        return None

if __name__ == "__main__":
    print("🚀 Starting model extraction process...")
    extracted_model = fix_and_extract_model()
    
    if extracted_model:
        print("\n✨ Model extraction completed successfully!")
    else:
        print("\n❌ Model extraction failed.")
        print("You may need to retrain your model with the current sklearn version.")