# config_loader.py
"""
Centralized configuration loading for ML Project Effort Estimator
Consolidates all YAML and JSON loading logic with consistent error handling
"""

import yaml
import json
import logging
import os
from typing import Dict, Any, Optional
from constants import LoggingConstants

class ConfigLoader:
    """Centralized configuration loader with consistent error handling"""
    
    @staticmethod
    def load_yaml_config(path: str) -> Optional[Dict[str, Any]]:
        """Load YAML config file. Returns None on any failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                if result is None:  # Empty file
                    logging.warning(f"YAML file is empty: {path}")
                    return None
                return result
        except FileNotFoundError:
            logging.warning(f"Configuration file not found: {path}")
            return None
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error loading YAML {path}: {e}")
            return None

    @staticmethod
    def load_json_config(path: str) -> Optional[Dict[str, Any]]:
        """Load JSON config file. Returns None on any failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
                if result is None:
                    logging.warning(f"JSON file is empty: {path}")
                    return None
                return result
        except FileNotFoundError:
            logging.warning(f"JSON configuration file not found: {path}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file {path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error loading JSON {path}: {e}")
            return None    
        
    @staticmethod
    def save_yaml_config(data: Dict[str, Any], path: str) -> bool:
        """Save data to YAML file with error handling"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=True)
            logging.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving YAML to {path}: {e}")
            return False
    
    @staticmethod
    def save_json_config(data: Dict[str, Any], path: str) -> bool:
        """Save data to JSON file with error handling"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"JSON configuration saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving JSON to {path}: {e}")
            return False