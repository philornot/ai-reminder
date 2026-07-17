#!/usr/bin/env python3
"""Script to clean up malformed messages in cache.

Usage:
    python tools/fix_cache.py [--config config/config.yaml | --cache-dir cache]

By default this resolves the cache directory from config/config.yaml (same
as the running app), so it works out of the box for both a single-target
setup and multi-target setups where each target has its own config file and
cache_dir. Pass --config to point at a specific target's config file, or
--cache-dir to bypass config loading entirely and point directly at a cache
directory.
"""

import argparse
import json
import sys
from pathlib import Path
import re

# Allow running this script directly from the tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def clean_message(message: str) -> str:
    """Clean a malformed message by extracting only the first variant.

    Args:
        message: Potentially malformed message

    Returns:
        Cleaned message
    """
    # Patterns that indicate multiple examples
    separators = [
        '\n\nLub:',
        '\n\nAlbo:',
        '\n\nMoże:',
        '\n\nPrzykład:',
        '\nLub:',
        '\nAlbo:',
        '\nMoże:',
        '\nNastępnie:',
        '\nI jeszcze:',
        '\nLub tak:',
        '\nLub też:',
        '\nEwentualnie:',
        '\nPotem:',
        '\nI znów:',
        '\nI tak dalej:',
        '\nPrzykładowe',
    ]

    # Find the earliest separator
    first_sep_pos = len(message)
    for sep in separators:
        pos = message.find(sep)
        if pos != -1 and pos < first_sep_pos:
            first_sep_pos = pos

    # Extract first part
    if first_sep_pos < len(message):
        cleaned = message[:first_sep_pos].strip()
    else:
        cleaned = message.strip()

    # Remove markdown formatting
    cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)
    cleaned = re.sub(r'`(.+?)`', r'\1', cleaned)

    # If still too long, take first 2 sentences
    if len(cleaned) > 300:
        sentences = re.split(r'[.!?]+\s+', cleaned)
        if len(sentences) > 1:
            cleaned = sentences[0] + '. ' + sentences[1]
            if not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
        else:
            cleaned = sentences[0]

    return cleaned.strip()


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

    if not cache_path.exists():
        print(f"❌ Cache file not found: {cache_path}")
        sys.exit(1)

    print("=" * 60)
    print("CACHE CLEANUP - Fixing Malformed Messages")
    print("=" * 60)

    # Load cache
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        sys.exit(1)

    print(f"\n📊 Original cache size: {len(cache)} messages")

    # Analyze messages
    malformed_count = 0
    fixed_messages = []

    for i, entry in enumerate(cache):
        if not isinstance(entry, dict) or 'message' not in entry:
            print(f"⚠️  Skipping invalid entry {i}")
            continue

        original = entry['message']

        # Check if malformed (contains example indicators)
        is_malformed = any(sep in original for sep in [
            '\n\nLub:', '\nLub:', '\n\nAlbo:', '\nAlbo:',
            '\n\nPrzykład:', '\nPrzykład:', 'Przykładowe'
        ])

        if is_malformed or len(original) > 300:
            malformed_count += 1
            cleaned = clean_message(original)

            print(f"\n📝 Message {i}:")
            print(f"   Original length: {len(original)} chars")
            print(f"   Cleaned length: {len(cleaned)} chars")
            print(f"   Original: {original[:100]}...")
            print(f"   Cleaned: {cleaned[:100]}..." if len(cleaned) > 100 else f"   Cleaned: {cleaned}")

            entry['message'] = cleaned

        fixed_messages.append(entry)

    print(f"\n📊 Results:")
    print(f"   Total messages: {len(cache)}")
    print(f"   Malformed messages: {malformed_count}")
    print(f"   Fixed messages: {len(fixed_messages)}")

    if malformed_count == 0:
        print("\n✓ No malformed messages found!")
        return

    # Confirm save
    response = input(f"\n💾 Save cleaned cache? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return

    # Backup original
    backup_path = cache_path.with_suffix('.json.backup-cleanup')
    print(f"\n💾 Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    # Save cleaned cache
    print(f"💾 Saving cleaned cache: {cache_path}")
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_messages, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done! Fixed {malformed_count} messages")


if __name__ == "__main__":
    main()