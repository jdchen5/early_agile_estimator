# model_display_names.py
"""
Centralized model display name management
Consolidates all display name logic with clear separation of concerns
"""

import os
import re
import logging
from typing import Dict, Optional
import json
import os
from constants import FileConstants, ModelConstants

class ModelDisplayNameManager:
    """Manages model display name mappings and transformations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path"""
        if config_path is None:
            config_path = os.path.join(FileConstants.CONFIG_FOLDER, FileConstants.MODEL_DISPLAY_NAME_FILE)
        
        self.config_path = config_path
        self.display_names = self._load_display_names_config()
        self.type_mappings = ModelConstants.MODEL_TYPE_MAPPINGS
    
    def _load_display_names_config(self) -> Dict[str, str]:
        """Load display names from configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    display_names = json.load(f)
                if display_names:
                    logging.info(f"Loaded {len(display_names)} model display names from config")
                return display_names
            else:
                logging.warning(f"Display names config not found: {self.config_path}")
                return {}
        except Exception as e:
            logging.warning(f"Could not load display names: {e}")
            return {}
    
    def get_display_name(self, technical_name: str) -> str:
        """
        Get display name for a technical model name
        Primary entry point - no recursion risk
        """
        if not technical_name:
            return "Unknown Model"
        
        # Step 1: Try exact match from config
        if technical_name in self.display_names:
            return self.display_names[technical_name]
        
        # Step 2: Try normalized key matching
        normalized_key = self._normalize_model_key(technical_name)
        for config_key, display_name in self.display_names.items():
            if self._normalize_model_key(config_key) == normalized_key:
                return display_name
        
        # Step 3: Try partial matching with model type
        model_type = self._extract_model_type(technical_name)
        if model_type:
            for config_key, display_name in self.display_names.items():
                if model_type.lower() in config_key.lower():
                    return display_name
        
        # Step 4: Generate fallback name
        return self.generate_fallback_name(technical_name)
    
    def generate_fallback_name(self, technical_name: str) -> str:
        """
        Generate a clean fallback display name
        Separated logic with no external dependencies
        """
        try:
            # Handle numbered model pattern: "top_model_1_rf"
            if technical_name.startswith('top_model_'):
                return self._generate_numbered_model_name(technical_name)
            
            # Handle other patterns
            return self._generate_generic_name(technical_name)
            
        except Exception as e:
            logging.error(f"Error generating fallback name for {technical_name}: {e}")
            return f"{technical_name.replace('_', ' ').title()} Model"
    
    def _generate_numbered_model_name(self, technical_name: str) -> str:
        """Generate name for numbered models like 'top_model_1_rf'"""
        parts = technical_name.split('_')
        if len(parts) >= 3:
            # Extract number and model type
            number = None
            model_type = None
            
            for i, part in enumerate(parts):
                if part.isdigit() and number is None:
                    number = part
                elif i > 0 and not part.isdigit():  # Skip 'top', 'model'
                    model_type = part
                    break
            
            if number and model_type:
                # Map model type to display name
                display_type = self.type_mappings.get(model_type.lower(), model_type.upper())
                return f"{display_type} Model #{number}"
        
        # Fallback if pattern doesn't match
        return self._generate_generic_name(technical_name)
    
    def _generate_generic_name(self, technical_name: str) -> str:
        """Generate generic display name"""
        # Clean up the name
        clean_name = technical_name.replace('_', ' ').title()
        
        # Remove common suffixes
        clean_name = clean_name.replace(' Model', '').replace(' Pkl', '')
        
        # Limit length
        if len(clean_name) > 30:
            clean_name = clean_name[:27] + "..."
        
        return f"{clean_name} Model"
    
    def _normalize_model_key(self, key: str) -> str:
        """Normalize model key for consistent matching"""
        key = key.lower()
        key = re.sub(r'^top_model_\d+_', '', key)  # Remove "top_model_X_" prefix
        key = re.sub(r'[^a-z0-9]', '', key)        # Keep only alphanumeric
        return key
    
    def _extract_model_type(self, technical_name: str) -> Optional[str]:
        """Extract model type from technical name"""
        parts = technical_name.split('_')
        if len(parts) > 1:
            return parts[-1]  # Last part is usually the model type
        return None
    
    def _extract_model_number(self, technical_name: str) -> int:
        """Extract number from technical name for sorting"""
        parts = technical_name.split('_')
        if parts and parts[0].startswith('top'):
            m = re.match(r'top(\d+)', parts[0])
            if m:
                return int(m.group(1))
        return 999  # fallback for sorting
    
    def get_all_display_names(self, technical_names: list) -> Dict[str, str]:
        """Get display names for multiple technical names"""
        result = {}
        for technical_name in technical_names:
            result[technical_name] = self.get_display_name(technical_name)
        return result
    
    def save_display_names(self, display_names: Dict[str, str]) -> bool:
        """Save display names to configuration file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(display_names, f, indent=2)
            logging.info(f"Display names saved to {self.config_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving display names: {e}")
            return False
    
    def add_display_name(self, technical_name: str, display_name: str) -> bool:
        """Add a new display name mapping"""
        self.display_names[technical_name] = display_name
        return self.save_display_names(self.display_names)