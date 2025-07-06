# models_patch.py - Quick bug fix for the UnboundLocalError in models.py
"""
This patch fixes the UnboundLocalError in the emergency feature preparation section
Apply this fix to your models.py file around line 804
"""

# ORIGINAL CODE (around line 804 in models.py):
# emergency_df = pd.DataFrame([feature_vector], columns=expected_features)

# FIXED CODE:
"""
Replace the emergency feature preparation section in models.py with this:

    except Exception as e:
        logging.error(f"Complete sequential pipeline failed: {e}")
        
        # Last resort: manual feature preparation
        try:
            logging.warning("Attempting emergency feature preparation...")
            
            # FIX: Define clean_features here since it's not available in this scope
            ui_keys_to_remove = {
                'selected_model', 'selected_models', 'submit', 'clear_results', 
                'show_history', 'save_config', 'config_name', 'comparison_mode'
            }
            clean_features = {k: v for k, v in ui_features.items() if k not in ui_keys_to_remove}
            
            expected_features = get_expected_feature_names_from_config()
            feature_vector = create_feature_vector_from_dict(clean_features, expected_features)
            emergency_df = pd.DataFrame([feature_vector], columns=expected_features)
            logging.warning(f"Emergency preparation successful: {emergency_df.shape}")
            return emergency_df
        except Exception as emergency_e:
            logging.error(f"Emergency feature preparation also failed: {emergency_e}")
            raise Exception(f"All feature preparation methods failed. Sequential error: {e}, Emergency error: {emergency_e}")
"""

def apply_patch_info():
    """Information about applying the patch"""
    print("To fix the bug in models.py:")
    print("1. Open models.py")
    print("2. Find the 'emergency feature preparation' section around line 804")
    print("3. Replace the clean_features reference with the fixed code above")
    print("4. Or add this line before using clean_features:")
    print("   clean_features = {k: v for k, v in ui_features.items() if k not in ui_keys_to_remove}")

if __name__ == "__main__":
    apply_patch_info()