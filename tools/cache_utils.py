#!/usr/bin/env python3
"""Utility script to inspect and repair the message cache.

Usage:
    python tools/cache_utils.py [inspect|repair|clear]
        [--config config/config.yaml | --cache-dir cache]

By default the cache directory is resolved from config/config.yaml (same as
the running app), so it works out of the box for both a single-target setup
and multi-target setups where each target has its own config file and
cache_dir. Pass --config to point at a specific target's config file, or
--cache-dir to bypass config loading entirely and point directly at a cache
directory, e.g.:

    python tools/cache_utils.py inspect --config config/config-marek.yaml
    python tools/cache_utils.py repair --cache-dir cache/agnieszka
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running this script directly from the tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_cache(cache_path: Path):
    """Load cache file.

    Args:
        cache_path: Path to cache file

    Returns:
        Cache data or None if failed
    """
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Cache file not found: {cache_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in cache file: {e}")
        return None


def save_cache(cache_path: Path, cache_data):
    """Save cache file.

    Args:
        cache_path: Path to cache file
        cache_data: Data to save
    """
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Cache saved successfully")
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")


def inspect_cache(cache_path: Path):
    """Inspect cache and show statistics.

    Args:
        cache_path: Path to cache file
    """
    print("=" * 60)
    print("CACHE INSPECTION")
    print("=" * 60)

    cache = load_cache(cache_path)
    if cache is None:
        return

    print(f"\n📊 Cache Statistics:")
    print(f"   Total entries: {len(cache)}")

    if not cache:
        print("   Cache is empty")
        return

    # Validate structure
    valid_entries = 0
    invalid_entries = 0
    empty_messages = 0

    for i, entry in enumerate(cache):
        if not isinstance(entry, dict):
            invalid_entries += 1
            print(f"   ⚠️  Entry {i}: Not a dict")
            continue

        if "message" not in entry:
            invalid_entries += 1
            print(f"   ⚠️  Entry {i}: Missing 'message' key")
            continue

        if not isinstance(entry["message"], str):
            invalid_entries += 1
            print(f"   ⚠️  Entry {i}: 'message' is not a string")
            continue

        if not entry["message"].strip():
            empty_messages += 1
            print(f"   ⚠️  Entry {i}: Empty message")
            continue

        valid_entries += 1

    print(f"\n   Valid entries: {valid_entries}")
    print(f"   Invalid entries: {invalid_entries}")
    print(f"   Empty messages: {empty_messages}")

    # Show messages
    print(f"\n📝 Messages in cache:")
    for i, entry in enumerate(cache):
        if isinstance(entry, dict) and "message" in entry:
            msg = entry["message"]
            timestamp = entry.get("timestamp", "Unknown")

            print(f"\n   [{i}] {timestamp}")

            # Show first 100 chars of message
            if len(msg) > 100:
                print(f"   {msg[:100]}...")
            else:
                print(f"   {msg}")


def repair_cache(cache_path: Path):
    """Repair cache by removing invalid entries.

    Args:
        cache_path: Path to cache file
    """
    print("=" * 60)
    print("CACHE REPAIR")
    print("=" * 60)

    cache = load_cache(cache_path)
    if cache is None:
        return

    original_count = len(cache)
    print(f"\n📊 Original cache size: {original_count}")

    # Filter valid entries
    valid_cache = []
    seen_messages = set()

    for i, entry in enumerate(cache):
        # Check if entry is valid dict
        if not isinstance(entry, dict):
            print(f"   ⚠️  Removing entry {i}: Not a dict")
            continue

        # Check if has message key
        if "message" not in entry:
            print(f"   ⚠️  Removing entry {i}: Missing 'message' key")
            continue

        # Check if message is string
        if not isinstance(entry["message"], str):
            print(f"   ⚠️  Removing entry {i}: 'message' is not a string")
            continue

        # Check if message is not empty
        msg = entry["message"].strip()
        if not msg:
            print(f"   ⚠️  Removing entry {i}: Empty message")
            continue

        # Check for duplicates
        if msg in seen_messages:
            print(f"   ⚠️  Removing entry {i}: Duplicate message")
            continue

        seen_messages.add(msg)

        # Add timestamp if missing
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
            print(f"   ℹ️  Added timestamp to entry {i}")

        valid_cache.append(entry)

    repaired_count = len(valid_cache)
    removed_count = original_count - repaired_count

    print(f"\n📊 Repair complete:")
    print(f"   Valid entries: {repaired_count}")
    print(f"   Removed entries: {removed_count}")

    if removed_count > 0:
        # Backup original
        backup_path = cache_path.with_suffix('.json.backup')
        print(f"\n💾 Creating backup: {backup_path}")
        save_cache(backup_path, cache)

        # Save repaired cache
        print(f"💾 Saving repaired cache: {cache_path}")
        save_cache(cache_path, valid_cache)
    else:
        print("\n✓ No repairs needed, cache is valid!")


def clear_cache(cache_path: Path):
    """Clear all messages from cache.

    Args:
        cache_path: Path to cache file
    """
    print("=" * 60)
    print("CACHE CLEAR")
    print("=" * 60)

    cache = load_cache(cache_path)
    if cache is None:
        return

    count = len(cache)

    if count == 0:
        print("\n✓ Cache is already empty")
        return

    # Confirm
    response = input(f"\n⚠️  This will remove {count} messages. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return

    # Backup original
    backup_path = cache_path.with_suffix('.json.backup')
    print(f"\n💾 Creating backup: {backup_path}")
    save_cache(backup_path, cache)

    # Clear cache
    print(f"🗑️  Clearing cache: {cache_path}")
    save_cache(cache_path, [])
    print(f"\n✓ Removed {count} messages")


def _resolve_cache_dir(args: argparse.Namespace) -> Path:
    """Resolve the cache directory from --cache-dir or --config.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Resolved cache directory path.
    """
    if args.cache_dir:
        return Path(args.cache_dir)

    from config_loader import Config  # noqa: E402

    try:
        config = Config(args.config)
    except Exception as exc:
        print(f"❌ Could not load config '{args.config}': {exc}")
        sys.exit(1)

    return Path(config.cache_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["inspect", "repair", "clear"],
        help="Command to run",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml to resolve the cache dir from "
             "(default: config/config.yaml). Ignored if --cache-dir is given.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory to operate on directly (skips config loading). "
             "E.g. --cache-dir cache/agnieszka",
    )
    args = parser.parse_args()

    cache_dir = _resolve_cache_dir(args)
    cache_path = cache_dir / "messages.json"

    command = args.command.lower()

    if command == "inspect":
        inspect_cache(cache_path)
    elif command == "repair":
        repair_cache(cache_path)
    elif command == "clear":
        clear_cache(cache_path)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: inspect, repair, clear")
        sys.exit(1)


if __name__ == "__main__":
    main()