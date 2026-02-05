"""Scheduler for managing reminder timing."""

import random
from datetime import datetime, time, timedelta
import time as time_module
from typing import Optional
import logging


class ReminderScheduler:
    """Scheduler for managing when reminders should be sent."""

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

        self.next_reminder_time: Optional[datetime] = None

        if randomize:
            self.logger.info(f"Scheduler initialized with random time between {time_range_start} and {time_range_end}")
        else:
            self.logger.info(f"Scheduler initialized with fixed time at {time_range_start}")

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
        start_minutes = self.range_start.hour * 60 + self.range_start.minute
        end_minutes = self.range_end.hour * 60 + self.range_end.minute

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

        Returns:
            Datetime when next reminder should be sent
        """
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)

        # Generate time based on mode
        if self.randomize:
            today_reminder = self._generate_random_time(today)
        else:
            today_reminder = self._generate_fixed_time(today)

        # Check if we can schedule for today
        now = datetime.now()

        if now < today_reminder:
            self.next_reminder_time = today_reminder
        else:
            # Schedule for tomorrow
            if self.randomize:
                self.next_reminder_time = self._generate_random_time(tomorrow)
            else:
                self.next_reminder_time = self._generate_fixed_time(tomorrow)

        self.logger.info(f"Next reminder scheduled for: {self.next_reminder_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.next_reminder_time

    def should_send_reminder(self) -> bool:
        """Check if it's time to send a reminder.

        Returns:
            True if reminder should be sent now
        """
        if self.next_reminder_time is None:
            self.logger.warning("No reminder scheduled, scheduling now")
            self.schedule_next_reminder()
            return False

        now = datetime.now()

        if now >= self.next_reminder_time:
            self.logger.info("Time to send reminder")
            return True

        return False

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
        # If close, check every 10 seconds
        else:
            return 10