#!/usr/bin/env python3
"""Manually trigger an immediate reminder send from the terminal.

Creates the marker file that ``main.py``'s running instance polls for. The
running app picks it up within a few seconds, sends the oldest cached
message right away, and deletes the marker. This does NOT affect the
already-randomized next scheduled reminder time — the regular scheduled
reminder still fires later as usual.

Usage:
    python tools/trigger_now.py [--config config/config.yaml]

Equivalent to:
    touch cache/manual_trigger
but resolves the correct cache directory from config.yaml instead of
assuming it's "cache/".
"""

import argparse
import sys
from pathlib import Path

# Allow running this script directly from the tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import Config  # noqa: E402

_MANUAL_TRIGGER_FILENAME = "manual_trigger"


def main() -> None:
    """Parse arguments, resolve the cache dir, and create the trigger marker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()

    try:
        config = Config(args.config)
    except Exception as exc:
        print(f"❌ Could not load config '{args.config}': {exc}")
        sys.exit(1)

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trigger_path = cache_dir / _MANUAL_TRIGGER_FILENAME

    trigger_path.touch()
    print(f"✓ Manual trigger created: {trigger_path}")
    print(
        "  The running ai-reminder service will send a reminder within a "
        "few seconds. The next scheduled reminder time is unaffected."
    )
    print(
        "  (If nothing happens, make sure ai-reminder.service is actually "
        "running and using this same config/cache directory.)"
    )


if __name__ == "__main__":
    main()