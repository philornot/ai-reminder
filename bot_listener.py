"""
Discord Listener Bot
====================
Lightweight Discord bot that acts as the listener layer for ai-reminder.
No Minecraft server management — its only job is to watch for messages from
the configured target user and hand them off to ContextualReminder.

What it does
------------
* Connects to Discord with the Message Content privileged intent.
* On every guild message: delegates to ContextualReminder.handle_message().
* Refreshes presence every 60 s (just a static "Watching for messages" label).
* Logs to both console and a rotating file (bot_listener.log).

Configuration
-------------
Reads config_listener.yaml (copy config_listener.example.yaml and fill in):
    discord_token       — bot token (needs Message Content intent enabled)
    contextual_reminder — same block as in config_rpi.yaml; see that example
"""

import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import discord
import yaml
from discord.ext import commands, tasks

from contextual_reminder import ContextualReminder


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure console + rotating-file logging."""
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

    # Suppress noisy libraries
    logging.getLogger("discord").setLevel(logging.INFO)
    logging.getLogger("discord.http").setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("bot_listener")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_FILE = "config_listener.yaml"


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


cfg = _load_config()

TOKEN: str = cfg.get("discord_token", "")
if not TOKEN:
    logger.critical("'discord_token' missing from %s", CONFIG_FILE)
    sys.exit(1)


# ---------------------------------------------------------------------------
# ContextualReminder initialisation
# ---------------------------------------------------------------------------

contextual_reminder: Optional[ContextualReminder] = None

_ctx_cfg = cfg.get("contextual_reminder", {})
if not _ctx_cfg.get("enabled", False):
    logger.critical(
        "contextual_reminder.enabled is false in %s — nothing to do. "
        "Set it to true and configure the block.",
        CONFIG_FILE,
    )
    sys.exit(1)

try:
    contextual_reminder = ContextualReminder(cfg, logger)
    logger.info("ContextualReminder initialised successfully")
except FileNotFoundError as exc:
    logger.critical("ai-reminder config not found: %s", exc)
    sys.exit(1)
except Exception as exc:
    logger.critical("Failed to initialise ContextualReminder: %s", exc, exc_info=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class ListenerBot(commands.Bot):
    """Minimal Discord bot whose sole purpose is message listening."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True   # privileged — must be enabled in the Developer Portal
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        """Start background tasks after the bot is ready."""
        presence_updater.start()
        logger.info("Background tasks started")

    async def on_ready(self) -> None:
        """Log successful connection."""
        logger.info(
            "Listener bot connected as %s (id=%d)",
            self.user, self.user.id,
        )

    async def on_message(self, message: discord.Message) -> None:
        """Forward every guild message to ContextualReminder.

        Args:
            message: Incoming Discord message event.
        """
        if contextual_reminder is not None:
            await contextual_reminder.handle_message(message)
        # Still process prefix commands in case you add any later.
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
    """Refresh Discord presence every 60 seconds."""
    try:
        await bot.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for messages 📖",
            ),
        )
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
