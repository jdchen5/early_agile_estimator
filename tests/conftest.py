# conftest.py - Pytest configuration and shared fixtures
"""
Central pytest configuration for ML Project Effort Estimator tests
Provides shared fixtures, test data, and environment setup
"""

import pytest
import tempfile
import shutil
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

# Test configuration
pytest_plugins = ["fixtures.mock_streamlit"]

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory fixture"""
    return Path(__file__).parent / "fixtures" / "test_data"

@pytest.fixture(scope="session")
def sample_configs_dir():
    """Sample configuration directory fixture"""
    return Path(__file__).parent / "fixtures" / "sample_configs"

@pytest.fixture
def temp_workspace():
    """Isolated temporary workspace for each test"""
    temp_dir = tempfile.mkdtemp(prefix="ml_estimator_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_ui_inputs():
    """Realistic UI input data for testing"""
    return {
        'project_prf_year_of_project': 2023,
        'external_eef_industry_sector': 'Financial',
        'tech_tf_primary_programming_language': 'Java',
        'project_prf_relative_size': 'M',
        'project_prf_functional_size': 500,
        'project_prf_max_team_size': 8,
        'external_eef_organisation_type': 'Banking',
        'tech_tf_web_development': True,
        'tech_tf_dbms_used': True,
        'process_pmf_prototyping_used': False,
        'project_prf_case_tool_used': True,
        'process_pmf_docs': 5,
        'people_prf_project_user_involvement': 3
    }

@pytest.fixture
def sample_isbsg_data(test_data_dir):
    """Sample ISBSG data for testing (subset of real data)"""
    data_file = test_data_dir / "sample_isbsg_data.csv"
    if data_file.exists():
        return pd.read_csv(data_file)
    
    # Create minimal test data if file doesn't exist
    test_data = pd.DataFrame({
        'project_prf_functional_size': [100, 500, 1000, 200, 800],
        'project_prf_max_team_size': [3, 8, 15, 5, 12],
        'external_eef_industry_sector': ['Financial', 'Banking', 'Insurance', 'Financial', 'Banking'],
        'tech_tf_primary_programming_language': ['Java', 'Python', 'C#', 'Java', 'Python']
    })
    return test_data

@pytest.fixture
def mock_model():
    """Mock ML model for testing without real model dependencies"""
    model = Mock()
    model.predict.return_value = np.array([1000.0])  # Mock prediction
    model._final_estimator = Mock()
    model._final_estimator.predict.return_value = np.array([1000.0])
    return model

@pytest.fixture
def benchmark_config():
    """Performance benchmark configuration"""
    return {
        'targets': {
            'prediction_time_seconds': 5.0,
            'shap_time_seconds': 30.0,
            'memory_usage_mb': 500.0,
            'multi_model_time_seconds': 15.0
        },
        'tolerance': {
            'time_regression_percent': 20.0,
            'memory_regression_percent': 15.0
        }
    }

@pytest.fixture
def expected_feature_count():
    """Expected number of features after pipeline processing"""
    return 67  # Based on your pipeline output

class TestDataValidator:
    """Helper class for validating test data consistency"""
    
    @staticmethod
    def validate_ui_inputs(inputs: Dict[str, Any]) -> bool:
        """Validate UI inputs have required fields"""
        required_fields = [
            'project_prf_functional_size',
            'project_prf_max_team_size',
            'external_eef_industry_sector'
        ]
        return all(field in inputs for field in required_fields)
    
    @staticmethod
    def validate_prediction_output(prediction: float) -> bool:
        """Validate prediction is reasonable"""
        return 1.0 <= prediction <= 100000.0  # Reasonable hour range

@pytest.fixture
def test_validator():
    """Test data validator instance"""
    return TestDataValidator()