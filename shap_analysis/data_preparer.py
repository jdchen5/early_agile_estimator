# shap_analysis/data_preparer.py
"""
SHAP Data Preparer - Prepares background and input data for SHAP analysis

This module handles all data preparation tasks for SHAP analysis including:
- Background data generation for SHAP baselines
- User input data validation and preparation
- Feature name extraction and management
- Synthetic data generation for testing

Migrated from shap_analysis_backup.py with proper logging and modular structure.
"""

import logging
from typing import Dict, Optional, List, Any, Callable

import numpy as np
import pandas as pd


# Module-level constants
DEFAULT_SAMPLE_SIZE = 100
RANDOM_SEED = 42
EXCLUDE_FIELDS = {
    'selected_model', 'selected_models', 'submit', 
    'clear_results', 'show_history', 'comparison_mode'
}


class SHAPDataPreparer:
    """
    Prepares and validates data for SHAP analysis.
    
    This class handles data preparation for SHAP analysis including background
    data generation, input validation, and feature extraction. It supports both
    ISBSG dataset integration and synthetic data generation.
    """
    
    def __init__(self):
        """Initialize data preparer with logging."""
        self.logger = logging.getLogger(__name__)
    
    def prepare_background_data(
        self, 
        n_samples: int, 
        model_name: str = None
    ) -> Optional[np.ndarray]:
        """
        Prepare background data for SHAP baseline analysis.
        
        This is the main entry point for background data preparation. It attempts
        ISBSG data mapping first, then falls back to synthetic data generation.
        
        Args:
            n_samples: Number of background samples to generate
            model_name: Name of the model (for logging purposes)
            
        Returns:
            numpy array of background data or None if preparation fails
        """
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
        """
        Prepare user input data for SHAP analysis.
        
        Args:
            user_inputs: Dictionary of user input features from UI
            
        Returns:
            numpy array of processed input data or None if preparation fails
        """
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
    
    def validate_shap_inputs(
        self, 
        user_inputs: Dict[str, Any], 
        required_fields: List[str] = None
    ) -> bool:
        """
        Validate that user inputs are suitable for SHAP analysis.
        
        Args:
            user_inputs: Dictionary of user inputs to validate
            required_fields: Optional list of required field names
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not user_inputs:
            return False
        
        # Check required fields if specified
        if required_fields:
            for field in required_fields:
                if field not in user_inputs or user_inputs[field] is None:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
        
        # Check for meaningful inputs
        meaningful_inputs = {
            k: v for k, v in user_inputs.items() 
            if k not in EXCLUDE_FIELDS and v is not None and v != ""
        }
        
        is_valid = len(meaningful_inputs) > 0
        self.logger.debug(f"Input validation: {len(meaningful_inputs)} meaningful fields, valid: {is_valid}")
        return is_valid
    
    def generate_synthetic_data_via_pipeline(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Generate synthetic data through feature preparation pipeline.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            numpy array of synthetic data or None if generation fails
        """
        try:
            from models import prepare_features_for_model
            
            synthetic_samples = []
            np.random.seed(RANDOM_SEED)  # Reproducibility
            
            for i in range(n_samples):
                ui_input = self._create_realistic_ui_inputs()
                processed = prepare_features_for_model(ui_input)
                if processed is not None:
                    synthetic_samples.append(processed.values.flatten())
            
            result = np.array(synthetic_samples, dtype=np.float32) if synthetic_samples else None
            self.logger.info(f"Pipeline synthetic data generated: {result.shape if result is not None else 'None'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline synthetic generation failed: {e}")
            return None
    
    def get_sample_data_info(self) -> Dict[str, Any]:
        """
        Get information about available sample data sources.
        
        Returns:
            Dictionary containing information about data source availability
        """
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
    
    def get_feature_names_from_fields(self, fields: Dict[str, Any]) -> List[str]:
        """
        Extract feature names from fields configuration.
        
        Args:
            fields: Dictionary of field configurations
            
        Returns:
            List of sorted feature names excluding UI control fields
        """
        return [
            name for name in sorted(fields.keys()) 
            if name not in EXCLUDE_FIELDS
        ]
    
    def get_feature_names_from_inputs(self, user_inputs: Dict[str, Any]) -> List[str]:
        """
        Extract feature names from user inputs.
        
        Args:
            user_inputs: Dictionary of user inputs
            
        Returns:
            List of sorted feature names excluding UI control fields
        """
        return [
            name for name in sorted(user_inputs.keys()) 
            if name not in EXCLUDE_FIELDS
        ]
    
    def get_parameter_index(self, param_name: str, feature_names: List[str]) -> Optional[int]:
        """
        Get the index of a parameter in the feature names list.
        
        Args:
            param_name: Name of the parameter to find
            feature_names: List of feature names
            
        Returns:
            Index of the parameter or None if not found
        """
        try:
            return feature_names.index(param_name)
        except (ValueError, AttributeError):
            self.logger.debug(f"Parameter '{param_name}' not found in feature names")
            return None
    
    def prepare_sample_data_wrapper(
        self, 
        n_samples: int, 
        fields: Dict[str, Any] = None, 
        get_field_options_func: Callable = None
    ) -> Optional[np.ndarray]:
        """
        Wrapper function for UI compatibility with legacy interface.
        
        Args:
            n_samples: Number of samples to prepare
            fields: Field configuration (unused, for compatibility)
            get_field_options_func: Field options function (unused, for compatibility)
            
        Returns:
            Background data array or None if preparation fails
        """
        return self.prepare_background_data(n_samples)
    
    def _get_isbsg_mapped_data(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Get ISBSG data mapped through feature preparation pipeline.
        
        Args:
            n_samples: Number of samples to generate from ISBSG data
            
        Returns:
            numpy array of processed ISBSG data or None if unavailable
            
        Raises:
            Exception: If ISBSG data is not available
        """
        from models import prepare_isbsg_sample_data, prepare_features_for_model
        
        # Get raw ISBSG data
        raw_isbsg = prepare_isbsg_sample_data(n_samples)
        if raw_isbsg is None:
            raise Exception("ISBSG data not available")
        
        processed_samples = []
        for i in range(min(n_samples, len(raw_isbsg))):
            # Use realistic defaults for mapping
            ui_input = self._create_realistic_ui_inputs()
            
            # Process through feature preparation pipeline
            processed = prepare_features_for_model(ui_input)
            if processed is not None:
                processed_samples.append(processed.values.flatten())
        
        result = np.array(processed_samples, dtype=np.float32) if processed_samples else None
        self.logger.info(f"ISBSG mapped data: {result.shape if result is not None else 'None'}")
        return result
    
    def _get_synthetic_data_from_stats(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Generate synthetic data using statistical methods.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            numpy array of synthetic data or None if generation fails
        """
        processed_samples = []
        np.random.seed(RANDOM_SEED)  # Reproducibility
        
        for i in range(n_samples):
            ui_input = self._create_realistic_ui_inputs()
            
            try:
                from models import prepare_features_for_model
                processed = prepare_features_for_model(ui_input)
                if processed is not None:
                    processed_samples.append(processed.values.flatten())
            except Exception as e:
                self.logger.debug(f"Failed to process synthetic sample {i}: {e}")
                continue
        
        result = np.array(processed_samples, dtype=np.float32) if processed_samples else None
        self.logger.info(f"Synthetic data generated: {result.shape if result is not None else 'None'}")
        return result
    
    def _create_realistic_ui_inputs(self) -> Dict[str, Any]:
        """
        Create realistic UI inputs for synthetic data generation.
        
        Returns:
            Dictionary of realistic input values for testing and background data
        """
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


# Module-level utility functions
def validate_data_array(data: np.ndarray) -> Dict[str, Any]:
    """
    Validate a data array for SHAP analysis suitability.
    
    Args:
        data: numpy array to validate
        
    Returns:
        Dictionary with validation results and statistics
    """
    if data is None:
        return {'valid': False, 'error': 'Data is None'}
    
    if not isinstance(data, np.ndarray):
        return {'valid': False, 'error': 'Data is not a numpy array'}
    
    if data.size == 0:
        return {'valid': False, 'error': 'Data array is empty'}
    
    # Check for problematic values
    has_nan = np.isnan(data).any()
    has_inf = np.isinf(data).any()
    
    return {
        'valid': not (has_nan or has_inf),
        'shape': data.shape,
        'dtype': str(data.dtype),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'min_value': float(np.min(data)) if not (has_nan or has_inf) else None,
        'max_value': float(np.max(data)) if not (has_nan or has_inf) else None
    }