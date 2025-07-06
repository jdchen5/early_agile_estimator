# test_model_loading.py
from models import load_model, get_trained_model

model_name = "top_model_1_BayesianRidge"

print("=== TESTING MODEL LOADING ===")

# Test load_model function
model1 = load_model(model_name)
print(f"load_model() result: {type(model1)}")

# Test get_trained_model function  
model2 = get_trained_model(model_name)
print(f"get_trained_model() result: {type(model2)}")

# Try prediction with the working model
if model1 and hasattr(model1, 'predict'):
    print("✅ load_model() works - using this for test")
    working_model = model1
elif model2 and hasattr(model2, 'predict'):
    print("✅ get_trained_model() works - using this for test")
    working_model = model2
else:
    print("❌ Both model loading functions failed")
    working_model = None

# Test model input requirements
if working_model:
    print(f"Model type: {type(working_model)}")
    
    # Check expected features
    if hasattr(working_model, 'feature_names_in_'):
        features = working_model.feature_names_in_
        print(f"Model expects {len(features)} features")
        print(f"First 5: {features[:5]}")
        print(f"Has target: {'project_prf_normalised_work_effort' in features}")