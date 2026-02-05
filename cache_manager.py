"""Cache manager for AI-generated messages."""

import json
from pathlib import Path
from typing import Optional, List
import logging
from datetime import datetime


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
        self._ensure_cache_file()

    def _ensure_cache_file(self):
        """Ensure cache file exists with proper structure."""
        if not self.cache_file.exists():
            self._write_cache([])
            self.logger.info("Created new cache file")

    def _read_cache(self) -> List[dict]:
        """Read cache from file.

        Returns:
            List of cached message entries
        """
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error reading cache: {e}")
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

    def add_message(self, message: str) -> bool:
        """Add a message to the cache.

        Args:
            message: Message to cache

        Returns:
            True if successful, False otherwise
        """
        try:
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

            # Write updated cache
            self._write_cache(cache)

            self.logger.info(f"Retrieved oldest message from cache (remaining: {len(cache)})")
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