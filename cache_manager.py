"""Cache manager for AI-generated messages."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List


class CacheManager:
    """Manager for caching AI-generated messages."""

    def __init__(
            self,
            cache_dir: str,
            cache_size: int = 10,
            logger: Optional[logging.Logger] = None
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            cache_size: Number of messages to keep in cache
            logger: Logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size
        self.logger = logger or logging.getLogger(__name__)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "messages.json"
        self.sent_file = self.cache_dir / "sent_messages.json"
        self._ensure_cache_file()
        self._ensure_sent_file()

    def _ensure_cache_file(self):
        """Ensure cache file exists with proper structure."""
        if not self.cache_file.exists():
            self._write_cache([])
            self.logger.info("Created new cache file")

    def _ensure_sent_file(self):
        """Ensure sent messages file exists with proper structure."""
        if not self.sent_file.exists():
            self._write_sent([])
            self.logger.info("Created new sent messages file")

    def _read_cache(self) -> List[dict]:
        """Read cache from file with validation.

        Returns:
            List of cached message entries
        """
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)

                # Validate cache structure
                if not isinstance(cache, list):
                    self.logger.error(f"Invalid cache structure: expected list, got {type(cache)}")
                    return []

                # Validate each entry
                valid_cache = []
                for i, entry in enumerate(cache):
                    if not isinstance(entry, dict):
                        self.logger.warning(f"Skipping invalid entry at index {i}: not a dict")
                        continue

                    if "message" not in entry:
                        self.logger.warning(f"Skipping invalid entry at index {i}: missing 'message' key")
                        continue

                    if not isinstance(entry["message"], str):
                        self.logger.warning(f"Skipping invalid entry at index {i}: 'message' is not a string")
                        continue

                    # Additional check: message should not be empty
                    if not entry["message"].strip():
                        self.logger.warning(f"Skipping invalid entry at index {i}: empty message")
                        continue

                    valid_cache.append(entry)

                if len(valid_cache) < len(cache):
                    self.logger.warning(f"Removed {len(cache) - len(valid_cache)} invalid entries from cache")
                    # Save cleaned cache
                    self._write_cache(valid_cache)

                return valid_cache

        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error reading cache: {e}")
            return []

    def _read_sent(self) -> List[dict]:
        """Read sent messages from file.

        Returns:
            List of sent message entries
        """
        try:
            with open(self.sent_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_cache(self, cache: List[dict]):
        """Write cache to file.

        Args:
            cache: List of message entries to write
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error writing cache: {e}")
            raise

    def _write_sent(self, sent: List[dict]):
        """Write sent messages to file.

        Args:
            sent: List of sent message entries to write
        """
        try:
            with open(self.sent_file, 'w', encoding='utf-8') as f:
                json.dump(sent, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error writing sent messages: {e}")
            raise

    def get_recent_sent_messages(self, count: int = 5) -> List[str]:
        """Get recent sent messages for context.

        Args:
            count: Number of recent messages to retrieve

        Returns:
            List of recent message strings
        """
        sent = self._read_sent()

        # If sent file is empty, use messages from cache instead
        if not sent:
            self.logger.debug("No sent messages yet, using cached messages for context")
            cache = self._read_cache()
            messages = [entry["message"] for entry in cache[-count:]]
            return messages

        # Get last N messages
        recent = sent[-count:]
        messages = [entry["message"] for entry in recent]

        self.logger.debug(f"Retrieved {len(messages)} recent sent messages for context")
        return messages

    def mark_as_sent(self, message: str):
        """Mark a message as sent by moving it to sent_messages.

        Args:
            message: The message that was sent
        """
        try:
            sent = self._read_sent()

            entry = {
                "message": message,
                "timestamp": datetime.now().isoformat()
            }

            sent.append(entry)

            # Keep only last 20 sent messages
            if len(sent) > 20:
                sent = sent[-20:]

            self._write_sent(sent)
            self.logger.info(f"Marked message as sent (total sent: {len(sent)})")

        except Exception as e:
            self.logger.error(f"Failed to mark message as sent: {e}")

    def add_message(self, message: str) -> bool:
        """Add a message to the cache.

        Args:
            message: Message to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate message
            if not message or not isinstance(message, str):
                self.logger.error(f"Invalid message: must be non-empty string")
                return False

            message = message.strip()
            if not message:
                self.logger.error(f"Invalid message: empty after stripping")
                return False

            cache = self._read_cache()

            entry = {
                "message": message,
                "timestamp": datetime.now().isoformat()
            }

            cache.append(entry)
            self._write_cache(cache)

            self.logger.info(f"Added message to cache (total: {len(cache)})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add message to cache: {e}")
            return False

    def get_oldest_message(self) -> Optional[str]:
        """Get and remove the oldest message from cache.

        Returns:
            Oldest cached message or None if cache is empty
        """
        try:
            cache = self._read_cache()

            if not cache:
                self.logger.warning("Cache is empty")
                return None

            # Get oldest message (first in list)
            oldest = cache.pop(0)
            message = oldest["message"]

            # Validate message before returning
            if not message or not isinstance(message, str):
                self.logger.error(f"Retrieved invalid message from cache: {type(message)}")
                # Try to get next message
                self._write_cache(cache)
                if cache:
                    return self.get_oldest_message()
                return None

            message = message.strip()
            if not message:
                self.logger.error(f"Retrieved empty message from cache")
                # Try to get next message
                self._write_cache(cache)
                if cache:
                    return self.get_oldest_message()
                return None

            # Write updated cache
            self._write_cache(cache)

            self.logger.info(f"Retrieved oldest message from cache (remaining: {len(cache)})")
            self.logger.debug(
                f"Message content: {message[:50]}..." if len(message) > 50 else f"Message content: {message}")

            return message

        except Exception as e:
            self.logger.error(f"Failed to get message from cache: {e}")
            return None

    def get_cache_count(self) -> int:
        """Get number of messages currently in cache.

        Returns:
            Number of cached messages
        """
        cache = self._read_cache()
        return len(cache)

    def is_cache_full(self) -> bool:
        """Check if cache has reached target size.

        Returns:
            True if cache is full, False otherwise
        """
        return self.get_cache_count() >= self.cache_size

    def needs_refill(self) -> int:
        """Calculate how many messages needed to fill cache.

        Returns:
            Number of messages needed
        """
        current = self.get_cache_count()
        needed = self.cache_size - current
        return max(0, needed)

    def clear_cache(self):
        """Clear all cached messages."""
        self._write_cache([])
        self.logger.info("Cache cleared")

    def validate_and_repair_cache(self) -> bool:
        """Validate cache file and repair if needed.

        Returns:
            True if cache is valid or was repaired successfully
        """
        try:
            cache = self._read_cache()

            # Check for duplicates
            seen_messages = set()
            unique_cache = []
            duplicates = 0

            for entry in cache:
                msg = entry.get("message", "").strip()
                if msg and msg not in seen_messages:
                    seen_messages.add(msg)
                    unique_cache.append(entry)
                else:
                    duplicates += 1

            if duplicates > 0:
                self.logger.warning(f"Removed {duplicates} duplicate messages from cache")
                self._write_cache(unique_cache)

            self.logger.info(f"Cache validation complete: {len(unique_cache)} valid unique messages")
            return True

        except Exception as e:
            self.logger.error(f"Cache validation failed: {e}")
            return False
