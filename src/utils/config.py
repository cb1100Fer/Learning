"""Configuration management for the tree species identifier."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager that loads settings from YAML files."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing config files. Defaults to project config/ dir.
        """
        if config_dir is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._model_config: Optional[Dict[str, Any]] = None
        self._app_config: Optional[Dict[str, Any]] = None

    @property
    def model_config(self) -> Dict[str, Any]:
        """Load and return model configuration."""
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            with open(config_path, 'r') as f:
                self._model_config = yaml.safe_load(f)
        return self._model_config

    @property
    def app_config(self) -> Dict[str, Any]:
        """Load and return application configuration."""
        if self._app_config is None:
            config_path = self.config_dir / "app_config.yaml"
            with open(config_path, 'r') as f:
                self._app_config = yaml.safe_load(f)
        return self._app_config

    def get(self, key: str, default: Any = None, config_type: str = "app") -> Any:
        """
        Get a configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., 'api.port')
            default: Default value if key not found
            config_type: Type of config ('app' or 'model')

        Returns:
            Configuration value or default
        """
        config = self.app_config if config_type == "app" else self.model_config

        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def reload(self) -> None:
        """Reload configuration files."""
        self._model_config = None
        self._app_config = None


# Global config instance
config = Config()
