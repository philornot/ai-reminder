"""Custom logging module with colorama support and file rotation."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
import re

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ANSIStripFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes for file logging."""

    ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def format(self, record):
        """Format the record and strip ANSI codes.

        Args:
            record: Log record to format

        Returns:
            Formatted string without ANSI codes
        """
        formatted = super().format(record)
        return self.ANSI_ESCAPE_PATTERN.sub('', formatted)


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    COLORS = {
        'DEBUG': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'INFO': Fore.GREEN if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'ERROR': Fore.RED if COLORAMA_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else '',
    }

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        """Initialize colored formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            use_colors: Whether to use colors
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and COLORAMA_AVAILABLE

    def format(self, record):
        """Format the record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted colored string
        """
        if self.use_colors:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"

        return super().format(record)


def setup_logger(
        name: str,
        config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Set up a logger with configuration from dict.

    Args:
        name: Logger name
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger instance
    """
    # Default configuration
    default_config = {
        'log_dir': 'logs',
        'log_level': 'INFO',
        'max_bytes': 10485760,
        'backup_count': 5,
        'console': {
            'enabled': True,
            'colored': True
        },
        'file': {
            'enabled': True,
            'include_timestamp': True,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S'
        }
    }

    # Merge with provided config
    if config:
        default_config.update(config)
        if 'console' in config:
            default_config['console'].update(config.get('console', {}))
        if 'file' in config:
            default_config['file'].update(config.get('file', {}))

    cfg = default_config

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, cfg['log_level'].upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler if enabled
    if cfg['file']['enabled']:
        log_path = Path(cfg['log_dir'])
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=cfg['max_bytes'],
            backupCount=cfg['backup_count']
        )

        file_format = cfg['file']['format']
        file_datefmt = cfg['file'].get('date_format', '%Y-%m-%d %H:%M:%S')

        file_formatter = ANSIStripFormatter(
            file_format,
            datefmt=file_datefmt
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler if enabled
    if cfg['console']['enabled']:
        console_handler = logging.StreamHandler(sys.stdout)

        console_format = cfg['file']['format']
        console_datefmt = cfg['file'].get('date_format', '%Y-%m-%d %H:%M:%S')
        use_colors = cfg['console'].get('colored', True)

        console_formatter = ColoredConsoleFormatter(
            console_format,
            datefmt=console_datefmt,
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger