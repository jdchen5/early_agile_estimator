# unit/test_config_loader.py - Configuration loading tests
"""
Test configuration loading functionality with error handling
Critical path: Configuration files must load correctly or application fails
"""

import pytest
import yaml
import json
import tempfile
from pathlib import Path
from config_loader import ConfigLoader

class TestConfigLoader:
    """Test configuration loading with various scenarios"""
    
    def test_load_valid_yaml_config(self, temp_workspace):
        """Test loading valid YAML configuration"""
        # Create test YAML file
        config_data = {
            'fields': {'test_field': {'type': 'numeric', 'default': 10}},
            'ui_behavior': {'theme': 'light'}
        }
        
        yaml_file = temp_workspace / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test loading
        result = ConfigLoader.load_yaml_config(str(yaml_file))
        
        assert result is not None
        assert result['fields']['test_field']['type'] == 'numeric'
        assert result['ui_behavior']['theme'] == 'light'
    
    def test_load_valid_json_config(self, temp_workspace):
        """Test loading valid JSON configuration"""
        config_data = {
            'model_display_names': {
                'rf_model': 'Random Forest',
                'xgb_model': 'XGBoost'
            }
        }
        
        json_file = temp_workspace / "test_config.json"
        with open(json_file, 'w') as f:
            json.dump(config_data, f)
        
        result = ConfigLoader.load_json_config(str(json_file))
        
        assert result is not None
        assert result['model_display_names']['rf_model'] == 'Random Forest'
    
    def test_load_missing_file_returns_none(self):
        """Test that missing files return None (don't crash)"""
        result = ConfigLoader.load_yaml_config("nonexistent_file.yaml")
        assert result is None
        
        result = ConfigLoader.load_json_config("nonexistent_file.json")
        assert result is None
    
    def test_load_corrupted_yaml_returns_none(self, temp_workspace):
        """Test corrupted YAML file handling"""
        corrupted_file = temp_workspace / "corrupted.yaml"
        with open(corrupted_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed bracket")
        
        result = ConfigLoader.load_yaml_config(str(corrupted_file))
        assert result is None
    
    def test_load_corrupted_json_returns_none(self, temp_workspace):
        """Test corrupted JSON file handling"""
        corrupted_file = temp_workspace / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"invalid": json content}')
        
        result = ConfigLoader.load_json_config(str(corrupted_file))
        assert result is None
    
    def test_load_empty_file_returns_none(self, temp_workspace):
        """Test empty file handling"""
        empty_file = temp_workspace / "empty.yaml"
        empty_file.touch()
        
        result = ConfigLoader.load_yaml_config(str(empty_file))
        assert result is None
    
    def test_save_yaml_config(self, temp_workspace):
        """Test saving YAML configuration"""
        config_data = {'test': 'data', 'number': 42}
        yaml_file = temp_workspace / "save_test.yaml"
        
        success = ConfigLoader.save_yaml_config(config_data, str(yaml_file))
        
        assert success is True
        assert yaml_file.exists()
        
        # Verify saved content
        loaded = ConfigLoader.load_yaml_config(str(yaml_file))
        assert loaded == config_data
    
    def test_save_json_config(self, temp_workspace):
        """Test saving JSON configuration"""
        config_data = {'test': 'data', 'number': 42}
        json_file = temp_workspace / "save_test.json"
        
        success = ConfigLoader.save_json_config(config_data, str(json_file))
        
        assert success is True
        assert json_file.exists()
        
        # Verify saved content
        loaded = ConfigLoader.load_json_config(str(json_file))
        assert loaded == config_data

class TestConfigIntegration:
    """Test configuration integration scenarios"""
    
    def test_ui_config_structure(self, sample_configs_dir):
        """Test UI configuration has required structure"""
        ui_config_file = sample_configs_dir / "ui_info.yaml"
        
        if ui_config_file.exists():
            config = ConfigLoader.load_yaml_config(str(ui_config_file))
            
            # Validate required sections exist
            assert 'fields' in config
            assert 'tab_organization' in config
            
            # Validate fields structure
            for field_name, field_config in config['fields'].items():
                assert 'type' in field_config
                assert field_config['type'] in ['numeric', 'categorical', 'boolean', 'text']
    
    def test_feature_mapping_structure(self, sample_configs_dir):
        """Test feature mapping configuration structure"""
        mapping_file = sample_configs_dir / "feature_mapping.yaml"
        
        if mapping_file.exists():
            config = ConfigLoader.load_yaml_config(str(mapping_file))
            
            # Validate required sections
            expected_sections = ['categorical_features', 'one_hot_features']
            for section in expected_sections:
                if section in config:
                    assert isinstance(config[section], dict)