"""Discord webhook integration module."""

import requests
import json
from typing import Optional
import logging
from datetime import datetime


class DiscordWebhook:
    """Handler for sending messages to Discord webhooks."""

    LEVEL_MAP = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }

    def __init__(
            self,
            main_webhook_url: str,
            debug_webhook_url: Optional[str] = None,
            debug_level: str = "error",
            logger: Optional[logging.Logger] = None
    ):
        """Initialize Discord webhook handler.

        Args:
            main_webhook_url: URL for main reminder messages
            debug_webhook_url: URL for debug/error notifications
            debug_level: Minimum level for debug notifications (debug, info, warning, error)
            logger: Logger instance for logging
        """
        self.main_webhook_url = main_webhook_url
        self.debug_webhook_url = debug_webhook_url
        self.debug_level = self.LEVEL_MAP.get(debug_level.lower(), logging.ERROR)
        self.logger = logger or logging.getLogger(__name__)

    def send_message(self, content: str, webhook_url: Optional[str] = None) -> bool:
        """Send a message to Discord webhook.

        Args:
            content: Message content to send
            webhook_url: Webhook URL (defaults to main webhook)

        Returns:
            True if successful, False otherwise
        """
        url = webhook_url or self.main_webhook_url

        if not url or url == "YOUR_MAIN_WEBHOOK_URL_HERE" or url == "YOUR_DEBUG_WEBHOOK_URL_HERE":
            self.logger.error("Webhook URL not configured")
            return False

        payload = {
            "content": content
        }

        try:
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            self.logger.info(f"Message sent successfully to Discord")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send message to Discord: {e}")
            return False

    def send_reminder(self, message: str) -> bool:
        """Send reminder message to main webhook.

        Args:
            message: Reminder message to send

        Returns:
            True if successful, False otherwise
        """
        return self.send_message(message, self.main_webhook_url)

    def send_debug(self, message: str, level: int = logging.ERROR) -> bool:
        """Send debug/error message to debug webhook if configured.

        Args:
            message: Debug message to send
            level: Log level of the message

        Returns:
            True if successful, False otherwise
        """
        if not self.debug_webhook_url:
            self.logger.debug("Debug webhook not configured, skipping debug message")
            return False

        if level < self.debug_level:
            self.logger.debug(f"Message level {level} below threshold {self.debug_level}, skipping")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = logging.getLevelName(level)
        formatted_message = f"[{timestamp}] **{level_name}**: {message}"

        return self.send_message(formatted_message, self.debug_webhook_url)

    def send_error(self, error_message: str, exception: Optional[Exception] = None) -> bool:
        """Send error notification to debug webhook.

        Args:
            error_message: Error description
            exception: Optional exception object

        Returns:
            True if successful, False otherwise
        """
        if exception:
            full_message = f"{error_message}\n```\n{type(exception).__name__}: {str(exception)}\n```"
        else:
            full_message = error_message

        return self.send_debug(full_message, logging.ERROR)