"""Main reminder application."""

import sys
import time
from pathlib import Path
from typing import Optional

from cache_manager import CacheManager
from config_loader import Config
from discord_webhook import DiscordWebhook
from llm_client import LLMClient
from logger import setup_logger
from scheduler import ReminderScheduler

# Default response-window duration used when the key is absent from config.
_DEFAULT_RESPONSE_WINDOW_HOURS: float = 3.0

# How often (seconds) to check for a manual trigger file while sleeping
# between scheduler checks. Keeps manual triggers responsive even when the
# scheduler's own next check is far away (e.g. hours before the next
# scheduled reminder).
_MANUAL_TRIGGER_POLL_SECONDS: int = 5

# Filename (inside the configured cache dir) that, when it exists, causes the
# app to send a reminder immediately. Create it with e.g.:
#   touch cache/manual_trigger
# The file is deleted as soon as it's picked up. Sending this way does NOT
# touch the scheduler's already-randomized next reminder time — the regular
# scheduled reminder still fires later as usual.
_MANUAL_TRIGGER_FILENAME: str = "manual_trigger"


class ReminderApp:
    """Main application for AI-powered reminders."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the reminder application.

        Args:
            config_path: Path to configuration file.
        """
        self.config = Config(config_path)

        self.logger = setup_logger(
            name="ai-reminder",
            config=self.config.log_config,
        )

        self.logger.info("=" * 60)
        self.logger.info("AI Reminder Application Starting")
        self.logger.info("=" * 60)

        # Flag to prevent multiple simultaneous sends.
        self._is_sending = False

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all application components."""
        try:
            self.webhook = DiscordWebhook(
                main_webhook_url=self.config.discord_main_webhook,
                debug_webhook_url=self.config.discord_debug_webhook,
                debug_level=self.config.discord_debug_level,
                logger=self.logger,
            )

            self.llm = LLMClient(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                logger=self.logger,
            )

            self.cache = CacheManager(
                cache_dir=self.config.cache_dir,
                cache_size=self.config.cache_size,
                logger=self.logger,
            )

            self.scheduler = ReminderScheduler(
                time_range_start=self.config.time_range_start,
                time_range_end=self.config.time_range_end,
                randomize=self.config.time_randomize,
                logger=self.logger,
            )

            # How long the response window stays open after a sent reminder.
            self._response_window_hours: float = float(
                self.config.get(
                    "reminder.response_window_hours",
                    _DEFAULT_RESPONSE_WINDOW_HOURS,
                )
            )
            self.logger.info(
                "Response window: %.1f h after each sent reminder",
                self._response_window_hours,
            )

            # Path to the manual-trigger marker file (see module docstring
            # constant _MANUAL_TRIGGER_FILENAME for how it's used).
            self._manual_trigger_path: Path = (
                    self.cache.cache_dir / _MANUAL_TRIGGER_FILENAME
            )

            self.logger.info("All components initialized successfully")

        except Exception as exc:
            self.logger.error("Failed to initialize components: %s", exc)
            raise

    def _generate_and_cache_message(self) -> Optional[str]:
        """Generate a message from LLM and add to cache.

        Returns:
            Generated message or None if failed.
        """
        try:
            recent_messages = self.cache.get_recent_sent_messages(count=5)
            prompt = self.config.get_prompt(recent_messages=recent_messages)
            message = self.llm.generate_message(prompt)

            if message:
                self.cache.add_message(message)
                return message

            return None

        except Exception as exc:
            self.logger.error("Error generating message: %s", exc)
            self.webhook.send_error("Failed to generate message from LLM", exc)
            return None

    def _initialize_cache(self) -> None:
        """Fill cache with initial messages."""
        self.logger.info("Validating existing cache...")
        self.cache.validate_and_repair_cache()

        needed = self.cache.needs_refill()

        if needed == 0:
            self.logger.info(
                "Cache already full (%d messages)", self.cache.get_cache_count()
            )
            return

        self.logger.info("Initializing cache with %d messages...", needed)

        success_count = 0
        for i in range(needed):
            self.logger.info("Generating message %d/%d...", i + 1, needed)
            message = self._generate_and_cache_message()
            if message:
                success_count += 1
            else:
                self.logger.warning("Failed to generate message %d", i + 1)

            if i < needed - 1:
                time.sleep(1)

        self.logger.info(
            "Cache initialization complete: %d/%d messages generated",
            success_count, needed,
        )

        if success_count == 0:
            raise RuntimeError("Failed to generate any cache messages")

    def _send_reminder(self) -> tuple[bool, bool]:
        """Send a reminder message and open a response window on success.

        After a reminder is delivered to Discord, ``CacheManager.open_response_window()``
        is called so that ``ContextualReminder`` switches to response-window mode
        and can reply to whatever the target writes back within the configured window.

        Returns:
            A ``(success, skipped)`` tuple.
                success: True if the reminder was delivered successfully.
                skipped: True if sending was deliberately skipped on purpose
                    (contextual reminder already sent today, or a duplicate
                    attempt while one was already in progress) rather than
                    attempted and failed. The caller uses this to decide
                    whether today should retry shortly (real failure) or
                    wait until tomorrow (success or deliberate skip).
        """
        if self._is_sending:
            self.logger.warning(
                "Already sending a reminder, skipping duplicate send attempt"
            )
            return False, True

        try:
            self._is_sending = True

            # Skip the regular reminder when the contextual mechanism already
            # delivered a message today.
            if self.cache.was_contextual_sent_today():
                self.logger.info(
                    "Skipping regular reminder — contextual reminder already sent today"
                )
                return False, True

            self.logger.info("=" * 60)
            self.logger.info("Starting reminder send process")

            message = self.cache.get_oldest_message()

            if not message:
                self.logger.error("No cached message available")
                self.webhook.send_error("Cache is empty, cannot send reminder")
                return False, False

            if not isinstance(message, str) or not message.strip():
                self.logger.error("Invalid message retrieved: %s", type(message))
                self.webhook.send_error("Invalid message format in cache")
                return False, False

            message = message.strip()
            self.logger.info(
                "Sending message: %s",
                (message[:100] + "...") if len(message) > 100 else message,
            )

            success = self.webhook.send_reminder(message)

            if success:
                self.logger.info("✓ Reminder sent successfully")
                self.cache.mark_as_sent(message)

                # Open the response window so ContextualReminder can reply
                # to whatever the target writes back.
                self.cache.open_response_window(self._response_window_hours)

                self._refill_cache()
            else:
                self.logger.error("✗ Failed to send reminder to Discord")
                self.logger.info("Re-adding message to cache")
                self.cache.add_message(message)

            self.logger.info("=" * 60)
            return success, False

        except Exception as exc:
            self.logger.error("Error sending reminder: %s", exc)
            self.webhook.send_error("Error sending reminder", exc)
            return False, False
        finally:
            self._is_sending = False

    def _refill_cache(self) -> None:
        """Refill cache with one new message if needed."""
        needed = self.cache.needs_refill()

        if needed > 0:
            self.logger.info("Cache needs refill (%d messages needed)", needed)
            message = self._generate_and_cache_message()

            if message:
                self.logger.info("Successfully refilled cache")
            else:
                self.logger.warning("Failed to refill cache")

    def _check_and_clear_manual_trigger(self) -> bool:
        """Check for a manual-trigger marker file and consume it if present.

        The marker is deleted immediately upon detection so a single
        ``touch`` only fires one reminder, not one per poll.

        Returns:
            True if a manual trigger was detected (and has now been cleared).
        """
        if not self._manual_trigger_path.exists():
            return False

        try:
            self._manual_trigger_path.unlink()
        except OSError as exc:
            self.logger.warning(
                "Manual trigger detected but could not remove marker file "
                "'%s': %s — proceeding anyway, but it may fire again",
                self._manual_trigger_path, exc,
            )
        return True

    def _send_manual_reminder(self) -> None:
        """Send a reminder immediately in response to a manual trigger.

        This deliberately does NOT touch ``self.scheduler`` in any way: the
        already-randomized next scheduled reminder time is left completely
        untouched, and the "sent today" flag used by the scheduler is not
        set either. In other words, this is purely an extra, on-demand send
        — the regular scheduled reminder still fires later exactly as
        planned.

        Mirrors the success/failure handling of ``_send_reminder()`` (cache
        bookkeeping, response window, refill) minus anything scheduler-related.
        """
        if self._is_sending:
            self.logger.warning(
                "Manual trigger received while already sending a reminder — ignoring"
            )
            return

        try:
            self._is_sending = True

            self.logger.info("=" * 60)
            self.logger.info("Manual trigger detected — sending reminder now")

            message = self.cache.get_oldest_message()

            if not message or not isinstance(message, str) or not message.strip():
                self.logger.error("No valid cached message available for manual trigger")
                self.webhook.send_error(
                    "Manual trigger: cache is empty or invalid, cannot send reminder"
                )
                return

            message = message.strip()
            self.logger.info(
                "Sending manual reminder: %s",
                (message[:100] + "...") if len(message) > 100 else message,
            )

            success = self.webhook.send_reminder(message)

            if success:
                self.logger.info("✓ Manual reminder sent successfully")
                self.cache.mark_as_sent(message)
                self.cache.open_response_window(self._response_window_hours)
                self._refill_cache()
            else:
                self.logger.error("✗ Failed to send manual reminder")
                self.logger.info("Re-adding message to cache")
                self.cache.add_message(message)

        except Exception as exc:
            self.logger.error("Error sending manual reminder: %s", exc)
            self.webhook.send_error("Error sending manual reminder", exc)
        finally:
            self._is_sending = False
            self.logger.info("=" * 60)

    def run(self) -> None:
        """Run the main application loop."""
        try:
            self._initialize_cache()
            self.scheduler.schedule_next_reminder()

            self.logger.info("Application running. Press Ctrl+C to stop.")
            self.logger.info(
                "Manual trigger: run `touch %s` to send a reminder "
                "immediately — this does not affect the next scheduled time.",
                self._manual_trigger_path,
            )

            while True:
                if self.scheduler.should_send_reminder():
                    self.logger.debug("Scheduler indicates it's time to attempt a send")
                    success, skipped = self._send_reminder()
                    self.scheduler.report_send_result(success, skipped)

                if self._check_and_clear_manual_trigger():
                    self._send_manual_reminder()

                # Sleep in short chunks so a manual trigger is picked up
                # quickly even when the scheduler's own next check is far
                # away (e.g. hours before the next scheduled reminder).
                remaining = self.scheduler.get_next_check_interval()
                while remaining > 0:
                    chunk = min(remaining, _MANUAL_TRIGGER_POLL_SECONDS)
                    time.sleep(chunk)
                    remaining -= chunk

                    if self._check_and_clear_manual_trigger():
                        self._send_manual_reminder()

        except KeyboardInterrupt:
            self.logger.info("\nApplication stopped by user")
            sys.exit(0)
        except Exception as exc:
            self.logger.error("Application error: %s", exc, exc_info=True)
            self.webhook.send_error("Application crashed", exc)
            raise


def main() -> None:
    """Main entry point."""
    try:
        app = ReminderApp()
        app.run()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()