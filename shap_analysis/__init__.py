# shap_analysis/__init__.py
"""
SHAP Analysis Module - Safe integration of modular components
Maintains backward compatibility while introducing new modular system
"""

import logging

# === NEW MODULAR SYSTEM (with import guards) ===
try:
    from shap_analysis.data_preparer import SHAPDataPreparer
    from shap_analysis.explainer_factory import SHAPExplainerFactory  
    from shap_analysis.value_calculator import SHAPValueCalculator
    from shap_analysis.analysis_coordinator import SHAPAnalysisCoordinator
    
    # Create global instances
    _data_preparer = SHAPDataPreparer()
    _explainer_factory = SHAPExplainerFactory()
    _value_calculator = SHAPValueCalculator()
    _coordinator = SHAPAnalysisCoordinator()
    
    MODULAR_SYSTEM_AVAILABLE = True
    print("DEBUG: New modular SHAP system loaded successfully")
    
except ImportError as e:
    print(f"WARNING: New modular system not available: {e}")
    MODULAR_SYSTEM_AVAILABLE = False
    _coordinator = None

# === BACKUP SYSTEM (for safety) ===
try:
    from shap_analysis_backup import *
    BACKUP_SYSTEM_AVAILABLE = True
    print("DEBUG: Backup SHAP system loaded as fallback")
except ImportError as e:
    print(f"ERROR: Backup system also not available: {e}")
    BACKUP_SYSTEM_AVAILABLE = False

# === BACKWARD COMPATIBILITY FUNCTIONS ===

def get_shap_explainer(
    model_name: str,
    get_trained_model_func,
    prepare_sample_data_func=None,
    sample_size: int = 100
):
    """
    Backward compatibility wrapper for get_shap_explainer
    Uses new modular system if available, falls back to backup
    """
    if MODULAR_SYSTEM_AVAILABLE:
        try:
            # Use new modular system
            background_data = _data_preparer.prepare_background_data(sample_size, model_name)
            if background_data is None:
                raise Exception("Background data preparation failed")
                
            explainer = _explainer_factory.create_explainer(
                model_name, get_trained_model_func, background_data, sample_size
            )
            
            if explainer is not None:
                print("DEBUG: Using new modular explainer system")
                return explainer
            else:
                raise Exception("Explainer creation failed")
                
        except Exception as e:
            print(f"WARNING: New system failed, falling back to backup: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_shap_explainer as backup_get_explainer
            print("DEBUG: Using backup explainer system")
            return backup_get_explainer(model_name, get_trained_model_func, prepare_sample_data_func, sample_size)
        except Exception as e:
            print(f"ERROR: Backup system also failed: {e}")
    
    return None

def get_shap_values_for_input(explainer, user_inputs, model=None, feature_names=None):
    """
    Backward compatibility wrapper for get_shap_values_for_input
    """
    if MODULAR_SYSTEM_AVAILABLE and explainer is not None:
        try:
            # Use new modular system
            if isinstance(user_inputs, dict):
                input_data = _data_preparer.prepare_input_data(user_inputs)
            else:
                input_data = user_inputs
            
            if input_data is not None:
                shap_values = _value_calculator.calculate_shap_values(explainer, input_data, feature_names)
                if shap_values is not None:
                    print("DEBUG: Using new modular value calculation")
                    return shap_values
                
        except Exception as e:
            print(f"WARNING: New value calculation failed, falling back: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_shap_values_for_input as backup_get_values
            print("DEBUG: Using backup value calculation")
            return backup_get_values(explainer, user_inputs, model, feature_names)
        except Exception as e:
            print(f"ERROR: Backup value calculation failed: {e}")
    
    return None

def prepare_sample_data(n_samples, fields=None, get_field_options_func=None):
    """
    Backward compatibility wrapper for prepare_sample_data
    """
    if MODULAR_SYSTEM_AVAILABLE:
        try:
            background_data = _data_preparer.prepare_background_data(n_samples)
            if background_data is not None:
                print("DEBUG: Using new modular data preparation")
                return background_data
        except Exception as e:
            print(f"WARNING: New data preparation failed, falling back: {e}")
    
    # Fallback to backup system
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import prepare_sample_data as backup_prepare_data
            print("DEBUG: Using backup data preparation")
            return backup_prepare_data(n_samples, fields, get_field_options_func)
        except Exception as e:
            print(f"ERROR: Backup data preparation failed: {e}")
    
    return None

def clear_explainer_cache():
    """Backward compatibility wrapper for clear_explainer_cache"""
    if MODULAR_SYSTEM_AVAILABLE:
        try:
            _explainer_factory.clear_cache()
            print("DEBUG: Cleared new modular cache")
            return
        except:
            pass
    
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import clear_explainer_cache as backup_clear_cache
            backup_clear_cache()
            print("DEBUG: Cleared backup cache")
        except:
            pass

def get_cache_info():
    """Backward compatibility wrapper for get_cache_info"""
    if MODULAR_SYSTEM_AVAILABLE:
        try:
            return _explainer_factory.get_cache_info()
        except:
            pass
    
    if BACKUP_SYSTEM_AVAILABLE:
        try:
            from shap_analysis_backup import get_cache_info as backup_get_cache_info
            return backup_get_cache_info()
        except:
            pass
    
    return {'cached_models': [], 'cache_size': 0}

# === NEW MODULAR INTERFACE (for advanced users) ===

def get_coordinator():
    """Get the SHAPAnalysisCoordinator instance (new modular interface)"""
    if MODULAR_SYSTEM_AVAILABLE:
        return _coordinator
    return None

def run_complete_analysis(user_inputs, model_name, get_trained_model_func, sample_size=100):
    """
    Run complete SHAP analysis using new modular system
    Returns None if new system not available
    """
    if MODULAR_SYSTEM_AVAILABLE and _coordinator is not None:
        try:
            return _coordinator.run_instance_analysis(user_inputs, model_name, get_trained_model_func, sample_size)
        except Exception as e:
            print(f"ERROR: Complete analysis failed: {e}")
    
    return None

def get_system_status():
    """Get status of both old and new SHAP systems"""
    return {
        'modular_system_available': MODULAR_SYSTEM_AVAILABLE,
        'backup_system_available': BACKUP_SYSTEM_AVAILABLE,
        'recommended_system': 'modular' if MODULAR_SYSTEM_AVAILABLE else 'backup',
        'coordinator_available': _coordinator is not None
    }

# === EXPORTS ===
__all__ = [
    # Backward compatibility interface
    'get_shap_explainer',
    'get_shap_values_for_input', 
    'prepare_sample_data',
    'clear_explainer_cache',
    'get_cache_info',
    
    # New modular interface
    'SHAPAnalysisCoordinator',
    'SHAPDataPreparer',
    'SHAPExplainerFactory', 
    'SHAPValueCalculator',
    'get_coordinator',
    'run_complete_analysis',
    'get_system_status'
]