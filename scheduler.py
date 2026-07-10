"""Scheduler for managing reminder timing."""

import logging
import random
from datetime import datetime, time, timedelta
from typing import Optional


class ReminderScheduler:
    """Scheduler for managing when reminders should be sent."""

    # How long to wait before retrying after a genuine send failure
    # (e.g. webhook/network error), instead of waiting until tomorrow.
    _RETRY_COOLDOWN_SECONDS: int = 120

    def __init__(
            self,
            time_range_start: str,
            time_range_end: str,
            randomize: bool = True,
            logger: Optional[logging.Logger] = None
    ):
        """Initialize reminder scheduler.

        Args:
            time_range_start: Start time in HH:MM format (or exact time if not randomizing)
            time_range_end: End time in HH:MM format (ignored if not randomizing)
            randomize: Whether to randomize time within range
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.randomize = randomize

        # Parse time range
        self.range_start = self._parse_time(time_range_start)
        self.range_end = self._parse_time(time_range_end) if randomize else self.range_start

        if randomize and self._to_minutes(self.range_end) < self._to_minutes(self.range_start):
            raise ValueError(
                f"reminder.time_range.end ({time_range_end}) is earlier than "
                f"reminder.time_range.start ({time_range_start}). Overnight "
                "ranges (e.g. 22:00-02:00) are not supported — pick a range "
                "that stays within the same day."
            )

        self.next_reminder_time: Optional[datetime] = None
        self._reminder_sent_today = False  # Flag to prevent double sending

        if randomize:
            self.logger.info(f"Scheduler initialized with random time between {time_range_start} and {time_range_end}")
        else:
            self.logger.info(f"Scheduler initialized with fixed time at {time_range_start}")

    @staticmethod
    def _to_minutes(t: time) -> int:
        """Convert a time object to minutes since midnight.

        Args:
            t: Time object to convert.

        Returns:
            Number of minutes since 00:00.
        """
        return t.hour * 60 + t.minute

    def _parse_time(self, time_str: str) -> time:
        """Parse time string in HH:MM format.

        Args:
            time_str: Time string in HH:MM format

        Returns:
            Time object
        """
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour=hour, minute=minute)
        except (ValueError, AttributeError) as e:
            self.logger.error(f"Invalid time format '{time_str}': {e}")
            raise ValueError(f"Time must be in HH:MM format, got: {time_str}")

    def _generate_random_time(self, date: datetime.date) -> datetime:
        """Generate random time within configured range for given date.

        Args:
            date: Date for which to generate time

        Returns:
            Random datetime within time range
        """
        # Convert time objects to minutes since midnight
        start_minutes = self._to_minutes(self.range_start)
        end_minutes = self._to_minutes(self.range_end)

        # Generate random minute within range
        random_minutes = random.randint(start_minutes, end_minutes)

        hour = random_minutes // 60
        minute = random_minutes % 60

        return datetime.combine(date, time(hour=hour, minute=minute))

    def _generate_fixed_time(self, date: datetime.date) -> datetime:
        """Generate fixed time for given date.

        Args:
            date: Date for which to generate time

        Returns:
            Fixed datetime
        """
        return datetime.combine(date, self.range_start)

    def schedule_next_reminder(self) -> datetime:
        """Schedule the next reminder time.

        If today's slot has already passed but no reminder was actually sent
        today (e.g. the process was down, crashed, or was only just started
        after the scheduled time), this catches up by scheduling for "now"
        instead of silently jumping to tomorrow — otherwise a restart shortly
        after the target time would cause that day's reminder to be skipped
        entirely.

        Returns:
            Datetime when next reminder should be sent
        """
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        now = datetime.now()

        # Generate today's slot based on mode
        if self.randomize:
            today_reminder = self._generate_random_time(today)
        else:
            today_reminder = self._generate_fixed_time(today)

        if self._reminder_sent_today:
            # Today is already handled (delivered or deliberately skipped) —
            # move on to tomorrow's slot.
            if self.randomize:
                self.next_reminder_time = self._generate_random_time(tomorrow)
            else:
                self.next_reminder_time = self._generate_fixed_time(tomorrow)
            self._reminder_sent_today = False
        elif now < today_reminder:
            # Normal case: today's slot hasn't arrived yet.
            self.next_reminder_time = today_reminder
        else:
            # today's slot already passed, but nothing was actually sent
            # today — most likely the process was down or just (re)started
            # after the scheduled time. Catch up now rather than skipping
            # today's reminder entirely.
            self.next_reminder_time = now
            self.logger.warning(
                "Missed today's scheduled reminder time (%s) — the process "
                "was probably down or just started. Catching up now instead "
                "of waiting until tomorrow.",
                today_reminder.strftime("%H:%M:%S"),
            )

        self.logger.info(f"Next reminder scheduled for: {self.next_reminder_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.next_reminder_time

    def should_send_reminder(self) -> bool:
        """Check if it's time to attempt sending a reminder.

        This only looks at timing and the "already sent today" flag. It does
        NOT mark the reminder as sent — that happens later, in
        ``report_send_result()``, once the caller actually knows whether the
        send succeeded. Keeping those two steps separate avoids the previous
        bug where a skipped or failed send was incorrectly treated as if it
        had gone out, silently blocking the rest of the day.

        To avoid re-triggering on every check while a send attempt is still
        being processed by the caller, the scheduled time is pushed forward
        by a short cooldown. ``report_send_result()`` either confirms that
        push (success, or deliberately skipped) or cancels it (transient
        failure → retry shortly).

        Returns:
            True if a send attempt should be made now.
        """
        if self.next_reminder_time is None:
            self.logger.warning("No reminder scheduled, scheduling now")
            self.schedule_next_reminder()
            return False

        now = datetime.now()

        if now < self.next_reminder_time:
            return False

        if self._reminder_sent_today:
            self.logger.debug("Reminder already sent today, skipping")
            return False

        self.logger.info("Time to attempt sending reminder")

        # Push the trigger time forward briefly so should_send_reminder()
        # doesn't fire again on the next check while the caller is still
        # handling this attempt. report_send_result() will adjust this
        # properly once the actual outcome is known.
        self.next_reminder_time = now + timedelta(seconds=self._RETRY_COOLDOWN_SECONDS)

        return True

    def report_send_result(self, success: bool, skipped: bool = False) -> None:
        """Record the outcome of a send attempt triggered by should_send_reminder().

        Args:
            success: True if the reminder was actually delivered.
            skipped: True if sending was deliberately skipped on purpose
                (e.g. a contextual reminder already went out today) rather
                than attempted and failed. A skipped send counts as "handled
                for today", same as a successful one — the regular reminder
                should not retry or fire again until tomorrow.

        Behavior:
            * success or skipped → today is considered handled; the next
              attempt is scheduled for tomorrow.
            * neither (a real failure, e.g. webhook/network error) → today
              is NOT marked as handled, and the next attempt is scheduled
              shortly (see ``_RETRY_COOLDOWN_SECONDS``) so the bot retries
              the same day instead of waiting until tomorrow.
        """
        if success or skipped:
            self._reminder_sent_today = True
            reason = "delivered" if success else "skipped on purpose"
            self.logger.info("Reminder attempt %s — done for today", reason)
            # next_reminder_time currently holds the short cooldown value set
            # by should_send_reminder(); replace it with tomorrow's slot now
            # that today is settled.
            self.schedule_next_reminder()
        else:
            self._reminder_sent_today = False
            retry_at = datetime.now() + timedelta(seconds=self._RETRY_COOLDOWN_SECONDS)
            self.next_reminder_time = retry_at
            self.logger.warning(
                "Reminder send failed — will retry at %s",
                retry_at.strftime("%Y-%m-%d %H:%M:%S"),
            )

    def get_seconds_until_next(self) -> float:
        """Get seconds until next scheduled reminder.

        Returns:
            Seconds until next reminder
        """
        if self.next_reminder_time is None:
            return 0

        now = datetime.now()
        delta = self.next_reminder_time - now
        return max(0, delta.total_seconds())

    def get_next_check_interval(self) -> int:
        """Get recommended interval for next check in seconds.

        Returns number of seconds to wait before checking again.
        Checks more frequently as we get closer to reminder time.

        Returns:
            Seconds to wait before next check
        """
        seconds_until = self.get_seconds_until_next()

        # If more than 1 hour away, check every 5 minutes
        if seconds_until > 3600:
            return 300
        # If more than 10 minutes away, check every minute
        elif seconds_until > 600:
            return 60
        # If close, check every 30 seconds (changed from 10 to reduce double-send risk)
        else:
            return 30	