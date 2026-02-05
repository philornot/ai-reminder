"""Configuration loader and validator."""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate required configuration fields."""
        required_sections = ['discord', 'llm', 'reminder', 'cache', 'logging', 'prompt']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate Discord config
        if 'main_webhook_url' not in self.config['discord']:
            raise ValueError("Missing discord.main_webhook_url")

        # Validate LLM config
        if 'api_key' not in self.config['llm']:
            raise ValueError("Missing llm.api_key")

        # Validate reminder config
        reminder_fields = ['target_name', 'sender_name', 'book_title', 'time_range']
        for field in reminder_fields:
            if field not in self.config['reminder']:
                raise ValueError(f"Missing reminder.{field}")

        # Validate time_range structure
        time_range = self.config['reminder']['time_range']
        if 'start' not in time_range:
            raise ValueError("Missing reminder.time_range.start")
        if self.config['reminder'].get('randomize_time', True) and 'end' not in time_range:
            raise ValueError("Missing reminder.time_range.end (required when randomize_time is true)")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'discord.main_webhook_url')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_prompt(self) -> str:
        """Get formatted prompt with placeholders replaced.

        Returns:
            Formatted prompt string
        """
        prompt_template = self.config['prompt']

        return prompt_template.format(
            sender_name=self.config['reminder']['sender_name'],
            target_name=self.config['reminder']['target_name'],
            book_title=self.config['reminder']['book_title'],
            language=self.config['reminder'].get('language', 'Polish')
        )

    @property
    def discord_main_webhook(self) -> str:
        """Get main Discord webhook URL."""
        return self.config['discord']['main_webhook_url']

    @property
    def discord_debug_webhook(self) -> str:
        """Get debug Discord webhook URL."""
        return self.config['discord'].get('debug_webhook_url', '')

    @property
    def discord_debug_level(self) -> str:
        """Get debug notification level."""
        return self.config['discord'].get('debug_level', 'error')

    @property
    def llm_provider(self) -> str:
        """Get LLM provider."""
        return self.config['llm'].get('provider', 'openai')

    @property
    def llm_api_key(self) -> str:
        """Get LLM API key."""
        return self.config['llm']['api_key']

    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return self.config['llm'].get('model', 'gpt-4')

    @property
    def llm_base_url(self) -> Optional[str]:
        """Get LLM API base URL based on provider."""
        provider = self.llm_provider

        # Check provider-specific settings first
        if provider in self.config['llm'] and isinstance(self.config['llm'][provider], dict):
            if 'base_url' in self.config['llm'][provider]:
                return self.config['llm'][provider]['base_url']

        # Return None to use provider default
        return None

    @property
    def llm_max_tokens(self) -> int:
        """Get LLM max tokens."""
        return self.config['llm'].get('max_tokens', 500)

    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.config['llm'].get('temperature', 0.9)

    @property
    def cache_size(self) -> int:
        """Get cache size."""
        return self.config['cache'].get('cache_size', 10)

    @property
    def cache_dir(self) -> str:
        """Get cache directory."""
        return self.config['cache'].get('cache_dir', 'cache')

    @property
    def log_dir(self) -> str:
        """Get log directory."""
        return self.config['logging'].get('log_dir', 'logs')

    @property
    def log_level(self) -> str:
        """Get log level."""
        return self.config['logging'].get('log_level', 'INFO')

    @property
    def log_max_bytes(self) -> int:
        """Get log file max bytes."""
        return self.config['logging'].get('max_bytes', 10485760)

    @property
    def log_backup_count(self) -> int:
        """Get log backup count."""
        return self.config['logging'].get('backup_count', 5)

    @property
    def log_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return self.config.get('logging', {})

    @property
    def time_randomize(self) -> bool:
        """Get whether time should be randomized."""
        return self.config['reminder'].get('randomize_time', True)

    @property
    def time_range_start(self) -> str:
        """Get reminder time range start."""
        time_range = self.config['reminder'].get('time_range', {})
        return time_range.get('start', '14:00')

    @property
    def time_range_end(self) -> str:
        """Get reminder time range end."""
        time_range = self.config['reminder'].get('time_range', {})
        return time_range.get('end', '17:00')
