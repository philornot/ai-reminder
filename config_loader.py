"""Configuration loader and validator."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_config(self) -> None:
        """Validate required configuration fields.

        Raises:
            ValueError: If a required section or field is missing.
        """
        required_sections = ["discord", "llm", "reminder", "cache", "logging", "prompt"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        if "main_webhook_url" not in self.config["discord"]:
            raise ValueError("Missing discord.main_webhook_url")

        if "api_key" not in self.config["llm"]:
            raise ValueError("Missing llm.api_key")

        reminder_fields = ["target_name", "sender_name", "book_title", "time_range"]
        for field in reminder_fields:
            if field not in self.config["reminder"]:
                raise ValueError(f"Missing reminder.{field}")

        time_range = self.config["reminder"]["time_range"]
        if "start" not in time_range:
            raise ValueError("Missing reminder.time_range.start")
        if self.config["reminder"].get("randomize_time", True):
            if "end" not in time_range:
                raise ValueError(
                    "Missing reminder.time_range.end (required when randomize_time is true)"
                )
            self._validate_time_order(time_range["start"], time_range["end"])

    @staticmethod
    def _validate_time_order(start_str: str, end_str: str) -> None:
        """Ensure the reminder time range does not cross midnight.

        Overnight ranges (e.g. ``22:00``-``02:00``) are not supported by the
        scheduler, which picks a random time by treating both bounds as
        minutes-since-midnight on the same day.

        Args:
            start_str: Range start in ``HH:MM`` format.
            end_str: Range end in ``HH:MM`` format.

        Raises:
            ValueError: If the times are not valid ``HH:MM`` strings, or if
                ``end`` is earlier than ``start``.
        """
        try:
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"reminder.time_range must use HH:MM format, got "
                f"start={start_str!r}, end={end_str!r}"
            ) from exc

        if (end_h * 60 + end_m) < (start_h * 60 + start_m):
            raise ValueError(
                f"reminder.time_range.end ({end_str}) is earlier than "
                f"reminder.time_range.start ({start_str}). Overnight ranges "
                "(e.g. 22:00-02:00) are not supported — pick a range that "
                "stays within the same day."
            )

        prompt_cfg = self.config["prompt"]
        if isinstance(prompt_cfg, dict):
            if "base" not in prompt_cfg:
                raise ValueError("Missing prompt.base")
            if "tasks" not in prompt_cfg or not prompt_cfg["tasks"]:
                raise ValueError(
                    "Missing prompt.tasks (must be a non-empty list of task "
                    "variants when 'prompt' is a dict)"
                )
        elif not isinstance(prompt_cfg, str):
            raise ValueError(
                "prompt must be either a plain string template, or a dict "
                "with 'base' and 'tasks' keys"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value using dot-notation.

        Args:
            key: Dot-separated key path (e.g. ``'discord.main_webhook_url'``).
            default: Value to return when the key is absent.

        Returns:
            Configuration value, or *default* if not found.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_prompt(self, recent_messages: Optional[List[str]] = None) -> str:
        """Return a prompt with all placeholders substituted.

        The ``prompt`` config section supports two formats:

        1. **Plain string** (legacy): used as-is, formatted directly.
        2. **Dict with ``base`` and ``tasks``** (recommended): ``tasks`` is a
           list of single-focus instructions (e.g. "ask how the reading is
           going", "ask what chapter they're on"). One task is chosen at
           random for each call and substituted into ``{task}`` inside
           ``base``. This keeps every generated message focused on ONE
           thing instead of asking the LLM to juggle several instructions
           at once ("mix it up: do A, or B, or C, ..."), which is what was
           causing incoherent/nonsensical messages with smaller models —
           the model would try to satisfy multiple instructions in one
           short reply and produce garbled text.

        Supported placeholders in both ``base`` and ``tasks``:
            ``{sender_name}``, ``{target_name}``, ``{book_title}``,
            ``{language}``, ``{target_gender}``, ``{recent_messages}``.
            ``base`` additionally supports ``{task}`` (the chosen task text).

        Args:
            recent_messages: Recent sent messages used as LLM context.
                When *None* or empty a placeholder text is inserted instead.

        Returns:
            Fully formatted prompt string.

        Raises:
            ValueError: If the template contains an unknown placeholder, or
                if ``prompt.tasks`` is empty.
        """
        prompt_cfg = self.config["prompt"]
        reminder_cfg = self.config["reminder"]

        if recent_messages:
            messages_text = "\n".join(f"- {msg}" for msg in recent_messages)
        else:
            messages_text = "(No previous messages yet)"

        base_kwargs = dict(
            sender_name=reminder_cfg["sender_name"],
            target_name=reminder_cfg["target_name"],
            book_title=reminder_cfg["book_title"],
            language=reminder_cfg.get("language", "Polish"),
            target_gender=reminder_cfg.get("target_gender", "female"),
            recent_messages=messages_text,
        )

        # Legacy plain-string prompt — used verbatim, no task selection.
        if isinstance(prompt_cfg, str):
            try:
                return prompt_cfg.format(**base_kwargs)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown placeholder in prompt template: {exc}. "
                    "Supported placeholders: sender_name, target_name, "
                    "book_title, language, target_gender, recent_messages."
                ) from exc

        # New base+tasks format.
        tasks = prompt_cfg["tasks"]
        if not tasks:
            raise ValueError("prompt.tasks must contain at least one task")

        task_template = random.choice(tasks)
        try:
            task_text = task_template.format(**base_kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder in prompt task variant: {exc}. "
                "Supported placeholders: sender_name, target_name, "
                "book_title, language, target_gender, recent_messages."
            ) from exc

        try:
            return prompt_cfg["base"].format(task=task_text, **base_kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder in prompt.base: {exc}. Supported "
                "placeholders: task, sender_name, target_name, book_title, "
                "language, target_gender, recent_messages."
            ) from exc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def discord_main_webhook(self) -> str:
        """Main Discord webhook URL."""
        return self.config["discord"]["main_webhook_url"]

    @property
    def discord_debug_webhook(self) -> str:
        """Debug Discord webhook URL (empty string when not configured)."""
        return self.config["discord"].get("debug_webhook_url", "")

    @property
    def discord_debug_level(self) -> str:
        """Minimum level for debug webhook notifications."""
        return self.config["discord"].get("debug_level", "error")

    @property
    def llm_provider(self) -> str:
        """LLM provider name."""
        return self.config["llm"].get("provider", "openai")

    @property
    def llm_api_key(self) -> str:
        """LLM API key."""
        return self.config["llm"]["api_key"]

    @property
    def llm_model(self) -> str:
        """LLM model name."""
        return self.config["llm"].get("model", "gpt-4")

    @property
    def llm_base_url(self) -> Optional[str]:
        """LLM API base URL derived from the active provider config.

        Returns *None* to let the client use its built-in default.
        """
        provider = self.llm_provider
        provider_cfg = self.config["llm"].get(provider)
        if isinstance(provider_cfg, dict):
            return provider_cfg.get("base_url")
        return None

    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens allowed in a single LLM response."""
        return self.config["llm"].get("max_tokens", 500)

    @property
    def llm_temperature(self) -> float:
        """LLM sampling temperature."""
        return self.config["llm"].get("temperature", 0.9)

    @property
    def cache_size(self) -> int:
        """Target number of messages to keep in the pending cache."""
        return self.config["cache"].get("cache_size", 10)

    @property
    def cache_dir(self) -> str:
        """Directory used for cache files."""
        return self.config["cache"].get("cache_dir", "cache")

    @property
    def log_dir(self) -> str:
        """Directory used for log files."""
        return self.config["logging"].get("log_dir", "logs")

    @property
    def log_level(self) -> str:
        """Logging level string (DEBUG / INFO / WARNING / ERROR)."""
        return self.config["logging"].get("log_level", "INFO")

    @property
    def log_max_bytes(self) -> int:
        """Maximum size of a single log file before rotation."""
        return self.config["logging"].get("max_bytes", 10_485_760)

    @property
    def log_backup_count(self) -> int:
        """Number of rotated log files to keep."""
        return self.config["logging"].get("backup_count", 5)

    @property
    def log_config(self) -> Dict[str, Any]:
        """Full logging configuration sub-dictionary."""
        return self.config.get("logging", {})

    @property
    def time_randomize(self) -> bool:
        """Whether the reminder time should be randomised within the range."""
        return self.config["reminder"].get("randomize_time", True)

    @property
    def time_range_start(self) -> str:
        """Reminder window start time in ``HH:MM`` format."""
        return self.config["reminder"].get("time_range", {}).get("start", "14:00")

    @property
    def time_range_end(self) -> str:
        """Reminder window end time in ``HH:MM`` format."""
        return self.config["reminder"].get("time_range", {}).get("end", "17:00")

    @property
    def target_gender(self) -> str:
        """Grammatical gender of the reminder target (``female`` / ``male``)."""
        return self.config["reminder"].get("target_gender", "female")

    @property
    def response_window_hours(self) -> float:
        """Hours the response window stays open after each sent reminder.

        Configurable via ``reminder.response_window_hours`` in config.yaml.
        Defaults to 3.0 hours.
        """
        return float(self.config["reminder"].get("response_window_hours", 3.0))