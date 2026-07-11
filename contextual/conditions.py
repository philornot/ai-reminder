"""Send-condition checks: daily contextual-sent state, time window, and response cooldown.

``_ConditionsMixin`` is mixed into
``ContextualReminder`` (see ``contextual.reminder``) and assumes the host
class provides ``self._cache``, ``self._ai_config``, ``self.logger``,
``self._contextual_sent_path``, ``self._response_cooldown_path``, and
``self.response_cooldown_minutes``.
"""

import json
from datetime import date, datetime, time as dtime, timedelta, timezone


class _ConditionsMixin:
    """Evaluates whether contextual-mode conditions are met and tracks cooldowns."""

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _should_send(self) -> bool:
        """Evaluate whether contextual-mode conditions are satisfied.

        Checks whether a contextual reminder was already sent today and,
        when ``randomize_time`` is enabled, whether the current local time
        falls inside the configured window.

        Returns:
            True if a contextual-mode reminder should be sent right now.
        """
        if self._was_sent_today():
            return False

        if self._ai_config.time_randomize and not self._is_within_time_range():
            self.logger.debug(
                "Contextual reminder: outside time window [%s – %s] — skipping",
                self._ai_config.time_range_start,
                self._ai_config.time_range_end,
            )
            return False

        return True

    def _was_sent_today(self) -> bool:
        """Check whether a contextual reminder was already sent today.

        Returns:
            True if ``contextual_sent.json`` records today's ISO date.
        """
        today = date.today().isoformat()
        # contextual_sent.json is also written/read by main.py's CacheManager
        # (was_contextual_sent_today) in a separate process — share the same
        # cross-process lock so the two never race on this file.
        with self._cache.lock():
            try:
                if not self._contextual_sent_path.exists():
                    return False
                data = json.loads(
                    self._contextual_sent_path.read_text(encoding="utf-8")
                )
                return data.get("last_sent_date") == today
            except Exception as exc:
                self.logger.warning(
                    "Could not read contextual_sent.json: %s — assuming not sent today",
                    exc,
                )
                return False

    def _is_within_time_range(self) -> bool:
        """Return True if the current local time is inside the reminder window.

        Returns:
            True when ``time_range.start <= now <= time_range.end``.
        """

        def _parse(s: str) -> dtime:
            h, m = map(int, s.split(":"))
            return dtime(h, m)

        now = datetime.now().time()
        start = _parse(self._ai_config.time_range_start)
        end = _parse(self._ai_config.time_range_end)
        return start <= now <= end

    def _mark_sent_today(self) -> None:
        """Persist today's date to ``contextual_sent.json``."""
        today = date.today().isoformat()
        with self._cache.lock():
            try:
                self._contextual_sent_path.parent.mkdir(parents=True, exist_ok=True)
                self._contextual_sent_path.write_text(
                    json.dumps(
                        {
                            "last_sent_date": today,
                            "sent_at": datetime.now().isoformat(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("Could not write contextual_sent.json: %s", exc)

    # ------------------------------------------------------------------
    # Response-window cooldown helpers
    # ------------------------------------------------------------------

    def _response_cooldown_elapsed(self) -> bool:
        """Return True if enough time has passed since the last response-mode reply.

        Reads ``response_cooldown.json`` which stores the timestamp of the last
        successful reply.  If the file is absent (no reply sent yet in this
        window) the cooldown is considered elapsed.

        Returns:
            True if the bot is allowed to reply again.
        """
        with self._cache.lock():
            try:
                if not self._response_cooldown_path.exists():
                    return True
                data = json.loads(
                    self._response_cooldown_path.read_text(encoding="utf-8")
                )
                last_reply_at = datetime.fromisoformat(data["last_reply_at"])
                if last_reply_at.tzinfo is None:
                    last_reply_at = last_reply_at.replace(tzinfo=timezone.utc)
                elapsed = datetime.now(timezone.utc) - last_reply_at
                cooldown = timedelta(minutes=self.response_cooldown_minutes)
                if elapsed >= cooldown:
                    return True
                remaining = (cooldown - elapsed).total_seconds() / 60
                self.logger.debug(
                    "Response cooldown: %.1f min remaining", remaining
                )
                return False
            except Exception as exc:
                self.logger.warning(
                    "Could not read response_cooldown.json: %s — assuming cooldown elapsed",
                    exc,
                )
                return True

    def _reset_response_cooldown(self) -> None:
        """Write the current UTC timestamp to ``response_cooldown.json``.

        Called immediately after a successful response-mode reply so that
        the next incoming message is held until the cooldown expires.
        """
        now = datetime.now(timezone.utc)
        with self._cache.lock():
            try:
                self._response_cooldown_path.parent.mkdir(parents=True, exist_ok=True)
                self._response_cooldown_path.write_text(
                    json.dumps(
                        {"last_reply_at": now.isoformat()},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                self.logger.debug(
                    "Response cooldown reset — next reply allowed in %d min",
                    self.response_cooldown_minutes,
                )
            except Exception as exc:
                self.logger.warning(
                    "Could not write response_cooldown.json: %s", exc
                )