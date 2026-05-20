"""Cache manager for AI-generated messages."""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class CacheManager:
    """Manager for caching AI-generated messages.

    All public methods are thread-safe. A single reentrant lock guards every
    read-modify-write operation so concurrent callers (e.g. a background
    refill thread and the main send loop) cannot corrupt the JSON files.
    """

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

        # RLock so that public methods can safely call other public methods
        # without deadlocking (e.g. get_oldest_message → get_cache_count).
        self._lock = threading.RLock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
