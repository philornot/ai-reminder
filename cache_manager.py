"""Cache manager for AI-generated messages."""

import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:
    # fcntl is POSIX-only. On platforms without it, cross-process locking is
    # silently skipped and only same-process (threading.RLock) safety holds.
    _FCNTL_AVAILABLE = False


class _InterProcessLock:
    """Reentrant lock that is also safe across separate OS processes.

    ``main.py`` (the scheduler loop) and ``bot_listener.py`` (via
    ``ContextualReminder``) run as two independent processes that both read
    and write the same JSON files in the cache directory. A plain
    ``threading.RLock`` only protects against concurrent *threads* within a
    single process — it does nothing to stop the other process from reading
    or writing the same file at the same moment, which can silently drop an
    update (e.g. two near-simultaneous ``mark_as_sent()`` calls, one from
    each process, where the last writer wins and the other's entry is lost).

    This class combines:
        * a ``threading.RLock`` for cheap, reentrant same-process safety
          (public methods calling other public methods, e.g.
          ``get_oldest_message()`` → ``get_cache_count()``), and
        * an ``fcntl.flock()`` on a dedicated lock file for cross-process
          mutual exclusion.

    The OS-level flock is only acquired/released on the outermost
    lock/unlock (depth 0 → 1 and 1 → 0); nested/reentrant acquisitions within
    the same process just bump a counter. This avoids a self-deadlock that
    would otherwise happen if the same process tried to flock() the same
    file twice via two different file descriptors.

    Falls back to thread-only safety (with a one-time warning) on platforms
    without ``fcntl`` (i.e. non-POSIX systems).
    """

    def __init__(self, lock_path: Path, logger: logging.Logger):
        """Initialize the lock.

        Args:
            lock_path: Path to the dedicated lock file (created if missing).
            logger: Logger instance used for one-time warnings.
        """
        self._lock_path = lock_path
        self._logger = logger
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fd = None

        if _FCNTL_AVAILABLE:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            # Opened once and kept open for the lifetime of this object so
            # repeated flock() calls always target the same open file
            # description (required for correct reentrancy bookkeeping).
            self._fd = open(self._lock_path, "a+")
        else:
            self._logger.warning(
                "fcntl not available on this platform — cache file locking "
                "is limited to within-process only. Running main.py and "
                "bot_listener.py against the same cache directory "
                "concurrently may lose updates."
            )

    def acquire(self) -> None:
        """Acquire the lock, blocking until available."""
        self._thread_lock.acquire()
        if self._fd is not None and self._depth == 0:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._depth += 1

    def release(self) -> None:
        """Release the lock."""
        self._depth -= 1
        if self._fd is not None and self._depth == 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._thread_lock.release()

    def __enter__(self) -> "_InterProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class CacheManager:
    """Manager for caching AI-generated messages.

    All public methods are safe to call concurrently — both from multiple
    threads within one process, and from ``main.py`` and ``bot_listener.py``
    running as separate OS processes against the same cache directory. A
    single lock (``_InterProcessLock``, combining a reentrant thread lock
    with a cross-process ``flock``) guards every read-modify-write operation
    so concurrent callers cannot corrupt the JSON files or silently drop
    each other's updates.
    """

    _CONTEXTUAL_SENT_FILENAME = "contextual_sent.json"
    _RESPONSE_WINDOW_FILENAME = "response_window.json"
    _LOCK_FILENAME = ".cache.lock"

    def __init__(
            self,
            cache_dir: str,
            cache_size: int = 10,
            logger: Optional[logging.Logger] = None,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files.
            cache_size: Number of messages to keep in cache.
            logger: Logger instance.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size
        self.logger = logger or logging.getLogger(__name__)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Combined reentrant + cross-process lock so that main.py and
        # bot_listener.py (via ContextualReminder), which run as separate
        # processes sharing this same cache directory, cannot clobber each
        # other's writes to messages.json / sent_messages.json / etc.
        self._lock = _InterProcessLock(
            self.cache_dir / self._LOCK_FILENAME, self.logger
        )

        self.cache_file = self.cache_dir / "messages.json"
        self.sent_file = self.cache_dir / "sent_messages.json"
        self._ensure_cache_file()
        self._ensure_sent_file()

    # ------------------------------------------------------------------
    # Internal helpers – callers must already hold self._lock
    # ------------------------------------------------------------------

    def _ensure_cache_file(self) -> None:
        """Ensure cache file exists with proper structure."""
        if not self.cache_file.exists():
            self._write_cache([])
            self.logger.info("Created new cache file")

    def _ensure_sent_file(self) -> None:
        """Ensure sent messages file exists with proper structure."""
        if not self.sent_file.exists():
            self._write_sent([])
            self.logger.info("Created new sent messages file")

    def _read_cache(self) -> List[dict]:
        """Read and validate the cache file.

        Returns:
            List of valid cached message entries.
        """
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)

            if not isinstance(cache, list):
                self.logger.error(
                    "Invalid cache structure: expected list, got %s", type(cache)
                )
                return []

            valid_cache: List[dict] = []
            for i, entry in enumerate(cache):
                if not isinstance(entry, dict):
                    self.logger.warning("Skipping entry %d: not a dict", i)
                    continue
                if "message" not in entry:
                    self.logger.warning("Skipping entry %d: missing 'message' key", i)
                    continue
                if not isinstance(entry["message"], str):
                    self.logger.warning("Skipping entry %d: 'message' is not a string", i)
                    continue
                if not entry["message"].strip():
                    self.logger.warning("Skipping entry %d: empty message", i)
                    continue
                valid_cache.append(entry)

            removed = len(cache) - len(valid_cache)
            if removed:
                self.logger.warning("Removed %d invalid entries from cache", removed)
                self._write_cache(valid_cache)

            return valid_cache

        except (json.JSONDecodeError, FileNotFoundError) as exc:
            self.logger.error("Error reading cache: %s", exc)
            return []

    def _read_sent(self) -> List[dict]:
        """Read sent messages file.

        Returns:
            List of sent message entries.
        """
        try:
            with open(self.sent_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_cache(self, cache: List[dict]) -> None:
        """Write cache list to file.

        Args:
            cache: List of message entries to persist.

        Raises:
            OSError: If the file cannot be written.
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.error("Error writing cache: %s", exc)
            raise

    def _write_sent(self, sent: List[dict]) -> None:
        """Write sent messages list to file.

        Args:
            sent: List of sent message entries to persist.

        Raises:
            OSError: If the file cannot be written.
        """
        try:
            with open(self.sent_file, "w", encoding="utf-8") as f:
                json.dump(sent, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.error("Error writing sent messages: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API – every method acquires the lock for its entire operation
    # ------------------------------------------------------------------

    def lock(self) -> _InterProcessLock:
        """Return the shared cross-process/cross-thread lock for this cache dir.

        Intended for callers outside ``CacheManager`` (currently
        ``ContextualReminder``) that read or write other files in the same
        cache directory (``contextual_sent.json``, ``response_cooldown.json``,
        ``reactable_messages.json``) and need to serialise those accesses
        against ``main.py``'s ``CacheManager`` operations too, since both
        processes share the same directory. Use as a context manager::

            with cache.lock():
                ...read/write a file in cache.cache_dir...

        Returns:
            The ``_InterProcessLock`` instance guarding this cache directory.
        """
        return self._lock

    def get_recent_sent_messages(self, count: int = 5) -> List[str]:
        """Return the most recently sent messages for LLM context.

        Falls back to the pending cache when no messages have been sent yet.

        Args:
            count: Number of recent messages to retrieve.

        Returns:
            List of message strings, oldest first.
        """
        with self._lock:
            sent = self._read_sent()

            if not sent:
                self.logger.debug(
                    "No sent messages yet – using cached messages for context"
                )
                cache = self._read_cache()
                return [entry["message"] for entry in cache[-count:]]

            recent = sent[-count:]
            messages = [entry["message"] for entry in recent]
            self.logger.debug(
                "Retrieved %d recent sent messages for context", len(messages)
            )
            return messages

    def mark_as_sent(self, message: str) -> None:
        """Record a message as sent.

        Appends the message to the sent-messages file and trims the history
        to the last 20 entries.

        Args:
            message: The message that was delivered to Discord.
        """
        with self._lock:
            try:
                sent = self._read_sent()
                sent.append(
                    {"message": message, "timestamp": datetime.now().isoformat()}
                )
                if len(sent) > 20:
                    sent = sent[-20:]
                self._write_sent(sent)
                self.logger.info("Marked message as sent (total sent: %d)", len(sent))
            except Exception as exc:
                self.logger.error("Failed to mark message as sent: %s", exc)

    def add_message(self, message: str) -> bool:
        """Append a message to the pending cache.

        Args:
            message: Message text to cache.

        Returns:
            True if the message was added successfully, False otherwise.
        """
        with self._lock:
            try:
                if not message or not isinstance(message, str):
                    self.logger.error("Invalid message: must be a non-empty string")
                    return False

                message = message.strip()
                if not message:
                    self.logger.error("Invalid message: empty after stripping whitespace")
                    return False

                cache = self._read_cache()
                cache.append(
                    {"message": message, "timestamp": datetime.now().isoformat()}
                )
                self._write_cache(cache)
                self.logger.info("Added message to cache (total: %d)", len(cache))
                return True

            except Exception as exc:
                self.logger.error("Failed to add message to cache: %s", exc)
                return False

    def get_oldest_message(self) -> Optional[str]:
        """Remove and return the oldest message from the cache.

        Returns:
            The oldest cached message string, or None if the cache is empty.
        """
        with self._lock:
            try:
                cache = self._read_cache()

                if not cache:
                    self.logger.warning("Cache is empty")
                    return None

                oldest = cache.pop(0)
                message = oldest.get("message", "")

                if not isinstance(message, str) or not message.strip():
                    self.logger.error(
                        "Retrieved invalid message from cache; discarding and retrying"
                    )
                    self._write_cache(cache)
                    # Recurse – lock is reentrant so this is safe.
                    return self.get_oldest_message()

                message = message.strip()
                self._write_cache(cache)

                self.logger.info(
                    "Retrieved oldest message from cache (remaining: %d)", len(cache)
                )
                self.logger.debug(
                    "Message content: %s",
                    (message[:50] + "...") if len(message) > 50 else message,
                )
                return message

            except Exception as exc:
                self.logger.error("Failed to get message from cache: %s", exc)
                return None

    def get_cache_count(self) -> int:
        """Return the number of messages currently in the cache.

        Returns:
            Count of cached messages.
        """
        with self._lock:
            return len(self._read_cache())

    def is_cache_full(self) -> bool:
        """Check whether the cache has reached its target size.

        Returns:
            True if the cache is at or above the configured size limit.
        """
        return self.get_cache_count() >= self.cache_size

    def needs_refill(self) -> int:
        """Calculate how many messages are needed to fill the cache.

        Returns:
            Number of messages that should be generated to top up the cache.
        """
        return max(0, self.cache_size - self.get_cache_count())

    def clear_cache(self) -> None:
        """Remove all pending messages from the cache."""
        with self._lock:
            self._write_cache([])
            self.logger.info("Cache cleared")

    def validate_and_repair_cache(self) -> bool:
        """Validate the cache file and remove duplicates.

        Returns:
            True if the cache is valid or was repaired successfully.
        """
        with self._lock:
            try:
                cache = self._read_cache()

                seen: set = set()
                unique_cache: List[dict] = []
                duplicates = 0

                for entry in cache:
                    msg = entry.get("message", "").strip()
                    if msg and msg not in seen:
                        seen.add(msg)
                        unique_cache.append(entry)
                    else:
                        duplicates += 1

                if duplicates:
                    self.logger.warning(
                        "Removed %d duplicate messages from cache", duplicates
                    )
                    self._write_cache(unique_cache)

                self.logger.info(
                    "Cache validation complete: %d valid unique messages",
                    len(unique_cache),
                )
                return True

            except Exception as exc:
                self.logger.error("Cache validation failed: %s", exc)
                return False

    def was_contextual_sent_today(self) -> bool:
        """Check whether a contextual reminder was already sent today.

        Reads the ``contextual_sent.json`` file written by ``ContextualReminder``
        in the same cache directory and compares its ``last_sent_date`` field
        against today's ISO date.

        Returns:
            True if a contextual reminder was delivered today, False otherwise
            (including when the file is absent or unreadable).
        """
        contextual_sent_path = self.cache_dir / self._CONTEXTUAL_SENT_FILENAME
        today = date.today().isoformat()

        with self._lock:
            try:
                if not contextual_sent_path.exists():
                    return False
                with open(contextual_sent_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("last_sent_date") == today
            except Exception as exc:
                self.logger.warning(
                    "Could not read %s: %s — assuming no contextual reminder sent today",
                    contextual_sent_path,
                    exc,
                )
                return False

    # ------------------------------------------------------------------
    # Response-window API
    # ------------------------------------------------------------------

    def open_response_window(self, duration_hours: float) -> None:
        """Record that a reminder was just sent and open a reply window.

        While the window is open, ``ContextualReminder`` switches to
        "response mode": instead of generating a fresh daily reminder it
        replies directly to whatever the target writes next — "tak,
        przeczytałam", "stfu", anything — with a witty comeback that still
        weaves in the book.

        The window state is stored in ``response_window.json`` in the cache
        directory so it survives process restarts.

        Args:
            duration_hours: How many hours the window should remain open.
                Configured via ``reminder.response_window_hours`` in
                ``config.yaml``; defaults to 3 when omitted.
        """
        from datetime import timedelta, timezone

        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(hours=duration_hours)).isoformat()

        path = self.cache_dir / self._RESPONSE_WINDOW_FILENAME
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "opened_at": now.isoformat(),
                            "expires_at": expires_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                self.logger.info(
                    "Response window opened — expires at %s (%.1f h)",
                    expires_at,
                    duration_hours,
                )
            except Exception as exc:
                self.logger.warning("Could not write response_window.json: %s", exc)

    def is_response_window_open(self) -> bool:
        """Return True when a response window is currently active.

        Reads ``response_window.json`` and checks whether the ``expires_at``
        timestamp is still in the future.  Naive timestamps written by older
        versions of the code are treated as UTC.

        Returns:
            True if the window exists and has not yet expired.
        """
        from datetime import timezone

        path = self.cache_dir / self._RESPONSE_WINDOW_FILENAME
        with self._lock:
            try:
                if not path.exists():
                    return False
                data = json.loads(path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(data["expires_at"])
                now = datetime.now(timezone.utc)
                # Normalise naive datetimes (e.g. written without tz by old code) to UTC.
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                return now < expires_at
            except Exception as exc:
                self.logger.warning(
                    "Could not read response_window.json: %s — treating window as closed",
                    exc,
                )
                return False

    def close_response_window(self) -> None:
        """Explicitly close the response window before it naturally expires.

        Called by ``ContextualReminder`` after the per-response cooldown has
        elapsed and it decides not to reply again within the same window.
        """
        path = self.cache_dir / self._RESPONSE_WINDOW_FILENAME
        with self._lock:
            try:
                if path.exists():
                    path.unlink()
                    self.logger.info("Response window closed")
            except Exception as exc:
                self.logger.warning("Could not remove response_window.json: %s", exc)