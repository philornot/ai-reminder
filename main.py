"""Main reminder application."""

import time
import sys
from pathlib import Path
from typing import Optional
import logging

from config_loader import Config
from logger import setup_logger
from discord_webhook import DiscordWebhook
from llm_client import LLMClient
from cache_manager import CacheManager
from scheduler import ReminderScheduler


class ReminderApp:
    """Main application for AI-powered reminders."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the reminder application.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)

        # Setup logger with config
        self.logger = setup_logger(
            name="ai-reminder",
            config=self.config.log_config
        )

        self.logger.info("=" * 60)
        self.logger.info("AI Reminder Application Starting")
        self.logger.info("=" * 60)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Discord webhook
            self.webhook = DiscordWebhook(
                main_webhook_url=self.config.discord_main_webhook,
                debug_webhook_url=self.config.discord_debug_webhook,
                debug_level=self.config.discord_debug_level,
                logger=self.logger
            )

            # LLM client
            self.llm = LLMClient(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                logger=self.logger
            )

            # Cache manager
            self.cache = CacheManager(
                cache_dir=self.config.cache_dir,
                cache_size=self.config.cache_size,
                logger=self.logger
            )

            # Scheduler
            self.scheduler = ReminderScheduler(
                time_range_start=self.config.time_range_start,
                time_range_end=self.config.time_range_end,
                randomize=self.config.time_randomize,
                logger=self.logger
            )

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def _generate_and_cache_message(self) -> Optional[str]:
        """Generate a message from LLM and add to cache.

        Returns:
            Generated message or None if failed
        """
        try:
            prompt = self.config.get_prompt()
            message = self.llm.generate_message(prompt)

            if message:
                self.cache.add_message(message)
                return message

            return None

        except Exception as e:
            self.logger.error(f"Error generating message: {e}")
            self.webhook.send_error("Failed to generate message from LLM", e)
            return None

    def _initialize_cache(self):
        """Fill cache with initial messages."""
        needed = self.cache.needs_refill()

        if needed == 0:
            self.logger.info(f"Cache already full ({self.cache.get_cache_count()} messages)")
            return

        self.logger.info(f"Initializing cache with {needed} messages...")

        success_count = 0
        for i in range(needed):
            self.logger.info(f"Generating message {i + 1}/{needed}...")

            message = self._generate_and_cache_message()
            if message:
                success_count += 1
            else:
                self.logger.warning(f"Failed to generate message {i + 1}")

            # Small delay between requests
            if i < needed - 1:
                time.sleep(1)

        self.logger.info(f"Cache initialization complete: {success_count}/{needed} messages generated")

        if success_count == 0:
            raise RuntimeError("Failed to generate any cache messages")

    def _send_reminder(self) -> bool:
        """Send a reminder message.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get message from cache
            message = self.cache.get_oldest_message()

            if not message:
                self.logger.error("No cached message available")
                self.webhook.send_error("Cache is empty, cannot send reminder")
                return False

            # Send to Discord
            success = self.webhook.send_reminder(message)

            if success:
                self.logger.info("Reminder sent successfully")
                # Refill cache asynchronously
                self._refill_cache()
            else:
                self.logger.error("Failed to send reminder to Discord")
                # Put message back in cache
                self.cache.add_message(message)

            return success

        except Exception as e:
            self.logger.error(f"Error sending reminder: {e}")
            self.webhook.send_error("Error sending reminder", e)
            return False

    def _refill_cache(self):
        """Refill cache with one new message if needed."""
        needed = self.cache.needs_refill()

        if needed > 0:
            self.logger.info(f"Cache needs refill ({needed} messages needed)")
            message = self._generate_and_cache_message()

            if message:
                self.logger.info("Successfully refilled cache")
            else:
                self.logger.warning("Failed to refill cache")

    def run(self):
        """Run the main application loop."""
        try:
            # Initialize cache
            self._initialize_cache()

            # Schedule first reminder
            self.scheduler.schedule_next_reminder()

            self.logger.info("Application running. Press Ctrl+C to stop.")

            # Main loop
            while True:
                if self.scheduler.should_send_reminder():
                    self._send_reminder()
                    self.scheduler.schedule_next_reminder()

                # Wait before next check
                interval = self.scheduler.get_next_check_interval()
                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("Application stopped by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            self.webhook.send_error("Application crashed", e)
            raise


def main():
    """Main entry point."""
    try:
        app = ReminderApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()