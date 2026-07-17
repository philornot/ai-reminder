"""Discord Listener Bot.

Lightweight Discord bot that acts as the listener layer for ai-reminder.
Its only job is to watch for messages from one or more configured target
users and hand each off to that target's own ``ContextualReminder``.

What it does:
    * Connects to Discord with the Message Content privileged intent.
    * On every guild message: delegates to ``ContextualReminder.handle_message()``
      for every configured target (each target's own ``handle_message()``
      already ignores messages that aren't from its ``target_discord_id``,
      so routing "who should handle this message" needs no extra code here).
    * Registers the bot's own user with every ``ContextualReminder`` on
      login so that direct mention detection works correctly.
    * Stays invisible at all times (status refreshed every 60 s) while still
      listening and processing messages normally in the background.
    * Logs to both console and a rotating file (bot_listener.log).

Configuration:
    Reads config_listener.yaml (copy config_listener.example.yaml and fill in):
        discord_token — bot token (needs Message Content intent enabled)
        targets       — list of targets to watch. Each entry has:
            name                — free-text label used only for log lines
            contextual_reminder — same block that used to be top-level
                                   (target_discord_id, ai_reminder_config,
                                   channel_id, response_cooldown_minutes,
                                   etc.) — see that example for details.

    For backward compatibility, a config file that still has a top-level
    ``contextual_reminder`` block (the old single-target format) is
    automatically treated as a single-entry ``targets`` list named
    "default" — no changes needed for existing single-target installs.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import discord
import yaml
from discord.ext import commands, tasks

from contextual import ContextualReminder


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure console + rotating-file logging for the listener process."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    fh = RotatingFileHandler(
        "bot_listener.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    root.addHandler(ch)
    root.addHandler(fh)

    logging.getLogger("discord").setLevel(logging.INFO)
    logging.getLogger("discord.http").setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("bot_listener")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_FILE = "config/config_listener.yaml"


def _load_config() -> dict:
    """Load and return bot configuration from config_listener.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: If the file is missing or contains invalid YAML.
    """
    p = Path(CONFIG_FILE)
    if not p.exists():
        logger.critical(
            "Config file '%s' not found — copy config_listener.example.yaml "
            "to %s and fill in your values.",
            CONFIG_FILE, CONFIG_FILE,
        )
        sys.exit(1)
    try:
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        logger.critical("Invalid YAML in %s: %s", CONFIG_FILE, exc)
        sys.exit(1)


def _resolve_target_entries(cfg: dict) -> list[dict]:
    """Normalize config into a list of ``{"name": ..., "contextual_reminder": ...}``.

    Supports the current multi-target ``targets:`` list format as well as
    the legacy single-target format where ``contextual_reminder`` sits
    directly at the top level of the config file.

    Args:
        cfg: Full parsed config_listener.yaml.

    Returns:
        A list of target entries, each with "name" and "contextual_reminder"
        keys. Empty list if none are configured.
    """
    targets = cfg.get("targets")
    if targets:
        normalized = []
        for i, entry in enumerate(targets):
            name = entry.get("name") or f"target-{i + 1}"
            normalized.append({
                "name": name,
                "contextual_reminder": entry.get("contextual_reminder", {}),
            })
        return normalized

    # Legacy single-target format.
    legacy_ctx = cfg.get("contextual_reminder")
    if legacy_ctx:
        return [{"name": "default", "contextual_reminder": legacy_ctx}]

    return []


cfg = _load_config()

TOKEN: str = cfg.get("discord_token", "")
if not TOKEN:
    logger.critical("'discord_token' missing from %s", CONFIG_FILE)
    sys.exit(1)

# ---------------------------------------------------------------------------
# ContextualReminder initialisation — one instance per target
# ---------------------------------------------------------------------------

_target_entries = _resolve_target_entries(cfg)

if not _target_entries:
    logger.critical(
        "No targets configured in %s — add a 'targets:' list (or the "
        "legacy top-level 'contextual_reminder:' block) with at least one "
        "enabled entry.",
        CONFIG_FILE,
    )
    sys.exit(1)

# List of (name, ContextualReminder) pairs for every enabled target. A
# target whose contextual_reminder.enabled is false is skipped entirely
# (mirrors the previous single-target behaviour of exiting on disabled,
# except now it just means "not watching this particular target").
contextual_reminders: list[tuple[str, ContextualReminder]] = []

for _entry in _target_entries:
    _name = _entry["name"]
    _ctx_cfg = _entry["contextual_reminder"]

    if not _ctx_cfg.get("enabled", False):
        logger.warning(
            "Target '%s': contextual_reminder.enabled is false — skipping",
            _name,
        )
        continue

    try:
        _reminder = ContextualReminder({"contextual_reminder": _ctx_cfg}, logger)
        contextual_reminders.append((_name, _reminder))
        logger.info("Target '%s': ContextualReminder initialised successfully", _name)
    except FileNotFoundError as exc:
        logger.critical("Target '%s': ai-reminder config not found: %s", _name, exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical(
            "Target '%s': failed to initialise ContextualReminder: %s",
            _name, exc, exc_info=True,
        )
        sys.exit(1)

if not contextual_reminders:
    logger.critical(
        "All configured targets have contextual_reminder.enabled = false — "
        "nothing to do. Enable at least one target in %s.",
        CONFIG_FILE,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class ListenerBot(commands.Bot):
    """Minimal Discord bot whose sole purpose is message listening."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged — must be enabled in the Developer Portal
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        """Start background tasks after the bot is ready."""
        presence_updater.start()
        logger.info("Background tasks started")

    async def on_ready(self) -> None:
        """Log successful connection, register bot user, and switch to invisible.

        The bot keeps functioning normally in the background (still receiving
        and processing every message event); it just won't show any status to
        other Discord users.

        Registering ``self.user`` with every ``ContextualReminder`` is
        required so that mention detection (``message.mentions`` check) can
        identify pings directed at this specific bot account.
        """
        logger.info(
            "Listener bot connected as %s (id=%d) — watching %d target(s): %s",
            self.user, self.user.id,
            len(contextual_reminders),
            ", ".join(name for name, _ in contextual_reminders),
        )
        await self.change_presence(status=discord.Status.invisible)

        for _name, reminder in contextual_reminders:
            reminder.set_bot_user(self.user)

    async def on_message(self, message: discord.Message) -> None:
        """Forward every guild message to every configured target's handler.

        Each ``ContextualReminder.handle_message()`` already checks
        ``message.author.id`` against its own ``target_discord_id`` and
        returns immediately for messages from anyone else, so it's safe to
        offer every message to every target here.

        Args:
            message: Incoming Discord message event.
        """
        for name, reminder in contextual_reminders:
            try:
                await reminder.handle_message(message)
            except Exception:
                logger.exception(
                    "Target '%s': unhandled error in handle_message", name,
                )
        await self.process_commands(message)

    async def on_error(self, event: str, *args, **kwargs) -> None:
        """Log unhandled errors in event handlers.

        Args:
            event: Name of the event that raised the error.
        """
        logger.exception("Unhandled error in event '%s'", event)


bot = ListenerBot()


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

@tasks.loop(seconds=60)
async def presence_updater() -> None:
    """Re-apply an invisible, activity-less presence every 60 seconds.

    Discord can reset presence after reconnects/resumes, so this loop keeps
    re-asserting invisible to make sure the bot never shows up as online.
    """
    try:
        await bot.change_presence(status=discord.Status.invisible, activity=None)
    except Exception:
        logger.exception("presence_updater error")


@presence_updater.before_loop
async def _before_presence() -> None:
    await bot.wait_until_ready()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Discord Listener Bot starting")
    logger.info("Config file : %s", CONFIG_FILE)
    logger.info("Targets     : %s", ", ".join(name for name, _ in contextual_reminders))
    logger.info("Log rotation: 10 MB max, 5 backups")
    logger.info("=" * 60)

    try:
        bot.run(TOKEN, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    except discord.LoginFailure:
        logger.critical("Invalid Discord token — check '%s'", CONFIG_FILE)
        sys.exit(1)
    except Exception:
        logger.exception("Bot crashed")
        raise
