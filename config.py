"""
Configuration of the MAG
Loads and provides stage-specific settings from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union

class Config:
    """Central configuration handler for the MAG system"""
    
    def __init__(self, stage: int):
        """
        Initialize configuration for a specific training stage
        
        Args:
            stage: Training stage (1, 2, or 3)
        
        Raises:
            ValueError: If stage is not 1, 2, or 3
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3")
            
        self.stage = stage
        self.config_data = self._load_config()
        self.stage_config = self._get_stage_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        try:
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError as e:
            raise RuntimeError("config.yaml not found in the project root") from e
        except yaml.YAMLError as e:
            raise RuntimeError("Error parsing config.yaml") from e
    
    def _get_stage_config(self) -> Dict[str, Any]:
        """Get configuration for the current stage"""
        stage_key = f"stage{self.stage}"
        try:
            return self.config_data['training'][stage_key]
        except KeyError as e:
            raise RuntimeError(f"Missing config section for stage {self.stage}") from e
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Get hyperparameters for the current stage with fallback defaults
        
        Returns:
            Dictionary containing hyperparameter settings
        """
        # Default values per stage
        defaults = {
            1: {"batch_size": 64, "learning_rate": 5e-5},
            2: {"batch_size": 64, "learning_rate": 5e-5, "weight_decay": 0.05, "steps": 500000},
            3: {"batch_size": 64, "learning_rate": 1e-5, "steps": 100000, "freeze_backbone": True}
        }
        
        # Start with defaults, override with config.yaml values
        hyperparams = defaults.get(self.stage, {}).copy()
        config_hp = self.stage_config.get("hyperparameters", {})
        hyperparams.update(config_hp)
        
        return hyperparams
    
    def get_dataset_path(self) -> str:
        """
        Get dataset path for the current stage
        
        Returns:
            String containing the dataset path
            
        Raises:
            RuntimeError: If dataset path is not specified
        """
        dataset_info = self.stage_config.get("dataset", {})
        
        # Prefer explicit path if available
        if "path" in dataset_info:
            return dataset_info["path"]
        
        # Raise error if no path specified
        raise RuntimeError(f"Dataset path not specified for stage {self.stage} in config.yaml")
