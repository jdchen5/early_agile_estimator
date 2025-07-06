# shap_analysis/data_preparer.py
"""
SHAP Data Preparer - Prepares background and input data
Single responsibility: Data preparation for SHAP analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

class SHAPDataPreparer:
    """Prepares data for SHAP analysis"""
    
    def prepare_background_data(self, n_samples: int, model_name: str = None) -> Optional[np.ndarray]:
        """Prepare background data for SHAP baseline"""
        
    def prepare_input_data(self, user_inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare user input data for SHAP analysis"""
        
    def validate_shap_inputs(self, user_inputs: Dict, required_fields: List[str] = None) -> bool:
        """Validate that user inputs are suitable for SHAP analysis"""
        
    def create_realistic_ui_inputs(self) -> Dict:
        """Create realistic UI inputs for synthetic data generation"""
        
    def get_isbsg_mapped_data(self, n_samples: int) -> Optional[np.ndarray]:
        """Get ISBSG data mapped through pipeline"""
        
    def get_synthetic_data_from_stats(self, n_samples: int) -> Optional[np.ndarray]:
        """Generate synthetic data using statistical methods"""
        
    def validate_sample_data(self, sample_data: np.ndarray) -> Dict[str, Any]:
        """Validate sample data quality"""
        
    def get_sample_data_info(self) -> Dict[str, Any]:
        """Get information about available sample data sources"""