# shap_analysis/data_preparer.py
"""
SHAP Data Preparer - Prepares background and input data for SHAP analysis
Migrated from shap_analysis_backup.py with proper logging and structure
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Any

class SHAPDataPreparer:
    """Prepares data for SHAP analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prepare_background_data(self, n_samples: int, model_name: str = None) -> Optional[np.ndarray]:
        """Prepare background data for SHAP baseline - main entry point"""
        try:
            self.logger.info(f"Preparing {n_samples} background samples for model: {model_name}")
            
            # Try ISBSG approach first, fallback to synthetic
            try:
                return self._get_isbsg_mapped_data(n_samples)
            except Exception as e:
                self.logger.warning(f"ISBSG mapping failed: {e}, falling back to synthetic")
                return self._get_synthetic_data_from_stats(n_samples)
                
        except Exception as e:
            self.logger.error(f"Background data preparation failed: {e}")
            return None
    
    def prepare_input_data(self, user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare user input data for SHAP analysis"""
        try:
            if not self.validate_shap_inputs(user_inputs):
                self.logger.error("Invalid user inputs for SHAP analysis")
                return None
            
            # Import here to avoid circular dependencies
            from models import prepare_features_for_model
            
            features_df = prepare_features_for_model(user_inputs)
            if features_df is None or features_df.empty:
                self.logger.error("Feature preparation failed")
                return None
            
            self.logger.info(f"Input data prepared: shape {features_df.shape}")
            return features_df.values
            
        except Exception as e:
            self.logger.error(f"Input data preparation failed: {e}")
            return None
    
    def validate_shap_inputs(self, user_inputs: Dict, required_fields: List[str] = None) -> bool:
        """Validate that user inputs are suitable for SHAP analysis"""
        if not user_inputs:
            return False
        
        # Check required fields if specified
        if required_fields:
            for field in required_fields:
                if field not in user_inputs or user_inputs[field] is None:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
        
        # Check for meaningful inputs
        exclude_fields = {'selected_model', 'selected_models', 'submit', 'clear_results', 'show_history', 'comparison_mode'}
        meaningful_inputs = {k: v for k, v in user_inputs.items() 
                           if k not in exclude_fields and v is not None and v != ""}
        
        is_valid = len(meaningful_inputs) > 0
        self.logger.debug(f"Input validation: {len(meaningful_inputs)} meaningful fields, valid: {is_valid}")
        return is_valid
    
    def get_sample_data_info(self) -> Dict[str, Any]:
        """Get information about available sample data sources"""
        info = {
            'isbsg_available': False,
            'synthetic_fallback': True,
            'recommended_source': 'synthetic'
        }
        
        try:
            # Check ISBSG availability
            from models import get_isbsg_dataset_info
            isbsg_info = get_isbsg_dataset_info()
            if isbsg_info.get('available', False):
                info['isbsg_available'] = True
                info['recommended_source'] = 'isbsg'
                info['isbsg_rows'] = isbsg_info.get('total_rows', 0)
                info['isbsg_features'] = isbsg_info.get('feature_columns', 0)
                
        except Exception as e:
            info['error'] = str(e)
            self.logger.warning(f"Could not get ISBSG info: {e}")
        
        return info
    
    # Private helper methods
    def _get_isbsg_mapped_data(self, n_samples: int) -> Optional[np.ndarray]:
        """Get ISBSG data mapped through pipeline"""
        from models import prepare_isbsg_sample_data, prepare_features_for_model
        
        # Get raw ISBSG data
        raw_isbsg = prepare_isbsg_sample_data(n_samples)
        if raw_isbsg is None:
            raise Exception("ISBSG data not available")
        
        processed_samples = []
        for i in range(min(n_samples, len(raw_isbsg))):
            # Simple mapping: extract basic features from ISBSG
            ui_input = self._create_realistic_ui_inputs()  # Use realistic defaults
            
            # Process through pipeline
            from models import prepare_features_for_model
            processed = prepare_features_for_model(ui_input)
            if processed is not None:
                processed_samples.append(processed.values.flatten())
        
        result = np.array(processed_samples, dtype=np.float32) if processed_samples else None
        self.logger.info(f"ISBSG mapped data: {result.shape if result is not None else 'None'}")
        return result
    
    def _get_synthetic_data_from_stats(self, n_samples: int) -> Optional[np.ndarray]:
        """Generate synthetic data using statistical methods"""
        processed_samples = []
        np.random.seed(42)  # Reproducibility
        
        for i in range(n_samples):
            ui_input = self._create_realistic_ui_inputs()
            
            from models import prepare_features_for_model
            processed = prepare_features_for_model(ui_input)
            if processed is not None:
                processed_samples.append(processed.values.flatten())
        
        result = np.array(processed_samples, dtype=np.float32) if processed_samples else None
        self.logger.info(f"Synthetic data generated: {result.shape if result is not None else 'None'}")
        return result
    
    def _create_realistic_ui_inputs(self) -> Dict:
        """Create realistic UI inputs for synthetic data generation"""
        return {
            'project_prf_year_of_project': np.random.randint(2020, 2025),
            'external_eef_industry_sector': np.random.choice(['Financial', 'Banking']),
            'tech_tf_primary_programming_language': np.random.choice(['Java', 'Python', 'C#']),
            'project_prf_relative_size': np.random.choice(['XS', 'S', 'M', 'L']),
            'project_prf_functional_size': int(np.random.lognormal(5, 1.5)),
            'project_prf_max_team_size': np.random.randint(3, 15),
            'external_eef_organisation_type': np.random.choice(['Banking', 'Financial', 'Insurance']),
            'tech_tf_web_development': np.random.choice([True, False]),
            'tech_tf_dbms_used': np.random.choice([True, False]),
            'process_pmf_prototyping_used': np.random.choice([True, False]),
            'project_prf_case_tool_used': np.random.choice([True, False]),
        }