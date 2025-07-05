print("Testing UI import chain...")

try:
    print("1. Testing models import directly...")
    import models
    print("✅ Models import OK")
    
    print("2. Testing shap_analysis import...")
    import shap_analysis
    print("✅ SHAP analysis import OK")
    
    print("3. Testing specific function imports from models...")
    from models import list_available_models
    print("✅ list_available_models import OK")
    
    print("4. Testing the actual UI imports...")
    from models import (
        predict_man_hours,
        list_available_models,
        check_required_models,
        get_feature_importance,
        get_model_display_name,
        get_model_display_name_from_config,
        get_trained_model,
        prepare_input_data,
        prepare_features_for_model,
        load_preprocessing_pipeline
    )
    print("✅ All UI model imports OK")
    
    print("5. Testing SHAP imports...")
    from shap_analysis import (
        get_shap_explainer,
        prepare_sample_data,
        get_shap_values_for_input,
        get_feature_interaction_values,
        get_feature_names_from_fields,
        get_feature_names_from_inputs,
        get_parameter_index
    )
    print("✅ All SHAP imports OK")
    
except RecursionError as e:
    print(f"❌ RECURSION ERROR at step: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()