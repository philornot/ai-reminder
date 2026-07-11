"""Contextual book reminder triggered by the target user's Discord messages.

Three operating modes
---------------------
1. **Contextual mode** (default, existing behaviour):
   When the target user sends a message and the daily reminder has not yet
   been sent, the module asks the LLM to craft a reply that playfully
   references what they said while sneaking in a book reminder.

2. **Response-window mode**:
   After ``main.py`` sends a regular scheduled reminder it calls
   ``CacheManager.open_response_window()``.  While that window is open
   (duration configured via ``reminder.response_window_hours``, default 3 h)
   this module switches to "response mode": every message the target sends
   triggers a witty comeback that acknowledges what they wrote and still
   weaves in the book. A per-reply cooldown (``contextual_reminder.
   response_cooldown_minutes``, default 30 min) prevents flooding if the
   target keeps chatting.

3. **Mention mode** (new):
   When the target directly pings the bot (``@bot``) or pings a role the
   bot has (``@RoleName``), a reply is sent immediately regardless of
   time-of-day window or daily-sent state. The book is woven in where it
   fits naturally, but the primary goal is a genuine response to the ping.
   No cooldown applies.

Reply reactions
----------------
Independently of the three modes above: whenever the target sends a message
that's plausibly answering one of the bot's own sent messages (mention,
contextual, or response-window) — either an explicit Discord "reply", or
simply the next thing they write shortly after — the module asks the LLM a
small yes/no question: does adding an emoji reaction actually fit here, or
would it look tone-deaf (e.g. the bot's message landed in the middle of an
unrelated conversation the target is really continuing)? Only when the LLM
says yes does the bot add a reaction, picking one emoji from a small
hardcoded pool (see ``contextual.constants._REACTION_EMOJI_CHOICES``). This
runs as a best-effort background task and never blocks or affects the normal
reply-generation flow. See ``contextual.reactions`` for the implementation.

Conversation context
--------------------
Before generating a reply, the module fetches the recent message history
from the Discord channel.  It walks backwards from the trigger message until
it finds the last message sent by the bot (identified by ``client_user.id``),
then passes every message in that window to the LLM prompt so the reply feels
aware of the ongoing conversation.

If the bot's last message cannot be found the window is capped at
``_CONTEXT_FALLBACK_LIMIT`` most-recent messages so the prompt never arrives
empty and the feature degrades gracefully.

Integration
-----------
Loaded by ``bot_listener.py`` at startup.  Requires an ai-reminder project on
the same host whose config path is stored under
``contextual_reminder.ai_reminder_config`` in the listener's YAML config.

After the Discord client logs in, ``bot_listener.py`` must call
``set_bot_user(client.user)`` so that mention detection works correctly.

Required config keys (under ``contextual_reminder``):
    enabled (bool): Master switch.
    ai_reminder_config (str): Path to ai-reminder's ``config/config.yaml``.
    target_discord_id (int): Discord user ID to watch.

Optional config keys (under ``contextual_reminder``):
    channel_id (int | null): Restrict monitoring to a single channel.
        ``null`` (default) watches all guild channels.
    response_cooldown_minutes (int): Minimum gap between two response-mode
        replies within the same window.  Default: 30.
    context_message_limit (int): Maximum number of messages to include as
        conversation context (hard cap regardless of bot-message search).
        Default: 20.

IMPORTANT — Discord privileged intent:
    Reading message content requires the **Message Content** privileged intent.
    In the Discord Developer Portal for your bot go to:
        Bot -> Privileged Gateway Intents -> Message Content Intent -> Enable
    Then set ``intents.message_content = True`` in the bot class.

Package layout
--------------
This module only holds the ``ContextualReminder`` orchestrator: config
bootstrap, the top-level ``handle_message()`` dispatch, mention detection,
and conversation-context fetching. Everything else lives in sibling modules
and is mixed in:

- ``contextual.constants`` — tunables, default task lists, emoji tables.
- ``contextual.context`` — the ``ConversationContext`` dataclass.
- ``contextual.prompts`` — ``_PromptBuilderMixin`` (prompt construction).
- ``contextual.reactions`` — ``_ReactionMixin`` (reply-reaction tracking + decision).
- ``contextual.conditions`` — ``_ConditionsMixin`` (daily-sent / time-window / cooldown).
- ``contextual.modes`` — ``_ModeHandlerMixin`` (per-mode generation + send).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import discord

from .conditions import _ConditionsMixin
from .constants import (
    _CONTEXT_FALLBACK_LIMIT,
    _DEFAULT_CONTEXT_MESSAGE_LIMIT,
    _DEFAULT_RESPONSE_COOLDOWN_MINUTES,
    _GENERATION_LOCK_TIMEOUT_S,
    _HISTORY_FETCH_LIMIT,
)
from .context import ConversationContext as _ConversationContext
from .modes import _ModeHandlerMixin
from .prompts import _PromptBuilderMixin
from .reactions import _ReactionMixin


class ContextualReminder(
    _PromptBuilderMixin,
    _ReactionMixin,
    _ConditionsMixin,
    _ModeHandlerMixin,
):
    """Generates and sends context-aware book reminders triggered by Discord messages.

    Watches for messages from a specific Discord user and, when conditions are
    met, asks the configured LLM to craft a reply.  The reply style depends on
    the active mode:

    - **Mention mode**: fires when the bot is directly pinged (``@bot``) or
      when one of its roles is pinged (``@RoleName``). Replies immediately
      regardless of time window or daily-sent state.
    - **Response-window mode**: fires while a response window is open (i.e.
      shortly after a scheduled reminder was sent). Acknowledges the target's
      reply with a witty comeback that keeps the book in the conversation.
    - **Contextual mode**: weaves a reminder into a reaction to what the target
      just said (fires when no window is open and the daily reminder has not
      been sent yet).

    In all modes the LLM receives a window of recent channel messages so its
    reply is aware of the conversation that led up to the trigger.

    All blocking I/O (LLM API calls, webhook HTTP requests) runs in a
    thread-pool executor so the Discord event loop is never stalled.

    A per-instance asyncio lock with a timeout prevents a burst of rapid
    messages from the target triggering multiple simultaneous generations.

    Attributes:
        enabled: Whether this feature is active (mirrors config).
        target_discord_id: Discord snowflake of the monitored user.
        channel_id: Optional channel snowflake to restrict monitoring.
        response_cooldown_minutes: Minimum gap (minutes) between two
            response-mode replies within the same window.
        context_message_limit: Hard cap on context messages passed to the LLM.
    """

    _CONTEXTUAL_SENT_FILENAME = "contextual_sent.json"
    _RESPONSE_COOLDOWN_FILENAME = "response_cooldown.json"
    _REACTABLE_MESSAGES_FILENAME = "reactable_messages.json"

    def __init__(self, mc_config: dict, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the contextual reminder handler.

        Reads the ``contextual_reminder`` block from the MC bot config, then
        bootstraps the ai-reminder project (Config, LLMClient, DiscordWebhook,
        CacheManager) by temporarily extending ``sys.path``.

        Args:
            mc_config: Full MC bot configuration dictionary (already parsed YAML).
            logger: Logger instance; a module-level logger is used when omitted.

        Raises:
            FileNotFoundError: If the ai-reminder config path does not exist.
            KeyError: If ``target_discord_id`` is missing from the config block.
            ImportError: If ai-reminder modules cannot be imported from the
                resolved directory.
        """
        self.logger = logger or logging.getLogger(__name__)
        cfg = mc_config.get("contextual_reminder", {})

        if not cfg.get("enabled", False):
            self.enabled = False
            self.logger.info("Contextual reminder: disabled in config")
            return

        self.enabled = True
        self.target_discord_id: int = int(cfg["target_discord_id"])
        self.channel_id: Optional[int] = (
            int(cfg["channel_id"]) if cfg.get("channel_id") else None
        )
        self.response_cooldown_minutes: int = int(
            cfg.get("response_cooldown_minutes", _DEFAULT_RESPONSE_COOLDOWN_MINUTES)
        )
        self.context_message_limit: int = int(
            cfg.get("context_message_limit", _DEFAULT_CONTEXT_MESSAGE_LIMIT)
        )

        # One generation at a time — prevents double-send when the target
        # sends a burst of messages before the first reply is marked as sent.
        self._generation_lock = asyncio.Lock()

        # Reply-reaction decisions run as fire-and-forget background tasks
        # (see handle_message / _maybe_react_to_reply). asyncio only holds a
        # weak reference to a task, so we keep a strong one here until it
        # finishes to make sure it isn't garbage-collected mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

        # Set by set_bot_user() after the Discord client logs in so that
        # mention detection can compare message.mentions against the real
        # bot account.
        self._bot_user: Optional[discord.ClientUser] = None

        # ------------------------------------------------------------------
        # Bootstrap ai-reminder imports
        # ------------------------------------------------------------------
        ai_config_path = Path(cfg["ai_reminder_config"]).resolve()
        if not ai_config_path.exists():
            raise FileNotFoundError(
                f"ai-reminder config not found: {ai_config_path}\n"
                "Check 'contextual_reminder.ai_reminder_config' in your MC bot config."
            )

        ai_reminder_dir = str(ai_config_path.parent.parent)
        if ai_reminder_dir not in sys.path:
            sys.path.insert(0, ai_reminder_dir)
            self.logger.debug(
                "Contextual reminder: added ai-reminder to sys.path: %s", ai_reminder_dir
            )

        try:
            from config_loader import Config  # noqa: PLC0415
            from llm_client import LLMClient  # noqa: PLC0415
            from discord_webhook import DiscordWebhook  # noqa: PLC0415
            from cache_manager import CacheManager  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"Could not import ai-reminder modules from '{ai_reminder_dir}': {exc}\n"
                "Make sure 'ai_reminder_config' points to the correct directory."
            ) from exc

        self._ai_config = Config(str(ai_config_path))

        self._llm = LLMClient(
            provider=self._ai_config.llm_provider,
            api_key=self._ai_config.llm_api_key,
            model=self._ai_config.llm_model,
            base_url=self._ai_config.llm_base_url,
            max_tokens=self._ai_config.llm_max_tokens,
            temperature=self._ai_config.llm_temperature,
            logger=self.logger,
        )

        self._webhook = DiscordWebhook(
            main_webhook_url=self._ai_config.discord_main_webhook,
            debug_webhook_url=self._ai_config.discord_debug_webhook,
            debug_level=self._ai_config.discord_debug_level,
            logger=self.logger,
        )

        self._cache = CacheManager(
            cache_dir=self._ai_config.cache_dir,
            cache_size=self._ai_config.cache_size,
            logger=self.logger,
        )

        self._contextual_sent_path = (
                Path(self._ai_config.cache_dir) / self._CONTEXTUAL_SENT_FILENAME
        )
        self._response_cooldown_path = (
                Path(self._ai_config.cache_dir) / self._RESPONSE_COOLDOWN_FILENAME
        )
        self._reactable_messages_path = (
                Path(self._ai_config.cache_dir) / self._REACTABLE_MESSAGES_FILENAME
        )

        self.logger.info(
            "Contextual reminder: enabled (watching user_id=%d%s, "
            "response_cooldown=%d min, context_limit=%d messages)",
            self.target_discord_id,
            f", channel_id={self.channel_id}" if self.channel_id else " in all channels",
            self.response_cooldown_minutes,
            self.context_message_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_bot_user(self, bot_user: discord.ClientUser) -> None:
        """Register the bot's own Discord user so mention detection works.

        Must be called from ``on_ready`` after the bot has successfully
        connected and ``client.user`` is populated.

        Args:
            bot_user: The logged-in bot's ClientUser object.
        """
        self._bot_user = bot_user
        self.logger.debug(
            "Contextual reminder: bot user registered (id=%d, name=%s)",
            bot_user.id,
            bot_user.name,
        )

    def _is_bot_mentioned(self, message: discord.Message) -> bool:
        """Check whether the bot was pinged directly or via one of its roles.

        Discord keeps direct user mentions and role mentions in separate
        lists: ``message.mentions`` only contains users/members explicitly
        @-mentioned, while pinging a role (e.g. ``@Reminders``) shows up in
        ``message.role_mentions`` instead. A bot is never added to
        ``message.mentions`` just because someone pinged a role it has, so
        without this check those pings would silently fall through to
        contextual or response-window mode instead of triggering a direct
        reply.

        Args:
            message: Incoming Discord message event.

        Returns:
            True if the bot was mentioned directly, or if the message pings
            a role that the bot currently has in that guild.
        """
        if self._bot_user is None:
            return False

        if self._bot_user in message.mentions:
            return True

        if message.role_mentions and message.guild is not None:
            bot_member = message.guild.me
            if bot_member is not None:
                return any(role in bot_member.roles for role in message.role_mentions)

        return False

    async def handle_message(self, message: discord.Message) -> bool:
        """Process an incoming Discord message and act if appropriate.

        Decides which mode to run based on the current state:

        - **Mention mode**: fires when the bot is mentioned directly or
          through one of its roles (see ``_is_bot_mentioned``), bypassing
          time-window and daily-sent checks.
        - **Response-window mode**: fires when ``CacheManager.is_response_window_open()``
          is True and the per-reply cooldown has elapsed.
        - **Contextual mode**: fires when no window is open, the daily
          contextual reminder has not been sent yet, and the time-window
          condition passes.

        Both non-mention modes share the same pre-checks (bot filter, user
        filter, channel filter, non-empty content) and the same generation
        lock.  All modes receive the conversation context window fetched from
        the channel history.

        Args:
            message: Incoming Discord message event.

        Returns:
            True if a reply was sent successfully.
        """
        if not self.enabled:
            return False

        # ------------------------------------------------------------------
        # Fast pre-checks — no lock, no blocking I/O
        # ------------------------------------------------------------------
        if message.author.bot:
            return False

        if message.author.id != self.target_discord_id:
            return False

        if self.channel_id is not None and message.channel.id != self.channel_id:
            return False

        trigger_content = (message.content or "").strip()
        if not trigger_content:
            return False

        # ------------------------------------------------------------------
        # Reply-reaction check — independent of the mode logic below.
        # Fires on any message from the target as long as there's a recently
        # sent bot message to react to; an explicit Discord "reply" link is
        # used when present (more precise about *what* is being answered),
        # but is no longer required — the target answering in plain text
        # right after the bot's message counts too.
        # Runs in the background so it can never delay or break the normal
        # reply-generation flow (see _maybe_react_to_reply's own try/except).
        # ------------------------------------------------------------------
        reactable_match: Optional[tuple[int, str]] = None
        if message.reference is not None and message.reference.message_id is not None:
            reactable_match = self._find_reactable_content(message.reference.message_id)
        if reactable_match is None:
            reactable_match = self._get_last_reactable_content()

        if reactable_match is not None:
            reactable_message_id, replied_to_content = reactable_match
            # Consume immediately (synchronously, before the background task
            # even starts) so a burst of several messages from the target
            # can only ever match this bot message once — the reaction (or
            # no-react decision) attaches to this single message only, never
            # to subsequent ones.
            self._mark_reactable_consumed(reactable_message_id)
            task = asyncio.create_task(
                self._maybe_react_to_reply(message, trigger_content, replied_to_content)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # ------------------------------------------------------------------
        # Determine mode without the lock (full re-check happens inside)
        # ------------------------------------------------------------------
        is_mention = self._is_bot_mentioned(message)
        in_response_window = self._cache.is_response_window_open()

        if is_mention:
            # Mention mode bypasses all cooldown/time-window guards.
            pass
        elif in_response_window:
            if not self._response_cooldown_elapsed():
                self.logger.debug(
                    "Response-window mode: still on cooldown — skipping"
                )
                return False
        else:
            if not self._should_send():
                return False

        # ------------------------------------------------------------------
        # Fetch conversation context from channel history.
        # Done before acquiring the generation lock — it's a read-only
        # Discord API call and doesn't need to be serialised.
        # ------------------------------------------------------------------
        context = await self._fetch_conversation_context(message)
        self.logger.debug(
            "Context window: %d message(s), bot_message_found=%s",
            len(context.messages),
            context.bot_message_found,
        )

        # ------------------------------------------------------------------
        # Serialised generation — one in-flight at a time.
        # ------------------------------------------------------------------
        try:
            acquired = await asyncio.wait_for(
                self._generation_lock.acquire(),
                timeout=_GENERATION_LOCK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Contextual reminder: could not acquire generation lock within %.1fs — skipping",
                _GENERATION_LOCK_TIMEOUT_S,
            )
            return False

        try:
            # Re-check conditions now that we hold the lock.
            is_mention = self._is_bot_mentioned(message)
            in_response_window = self._cache.is_response_window_open()

            if is_mention:
                return await self._handle_mention_mode(trigger_content, context)
            elif in_response_window:
                if not self._response_cooldown_elapsed():
                    self.logger.debug(
                        "Response-window mode: cooldown not elapsed (re-check) — skipping"
                    )
                    return False
                return await self._handle_response_mode(trigger_content, context)
            else:
                if not self._should_send():
                    return False
                return await self._handle_contextual_mode(trigger_content, context)

        finally:
            self._generation_lock.release()

    # ------------------------------------------------------------------
    # Conversation context fetcher
    # ------------------------------------------------------------------

    async def _fetch_conversation_context(
            self, trigger_message: discord.Message
    ) -> _ConversationContext:
        """Fetch the channel history window between the last bot message and the trigger.

        Walks backwards through the channel history (up to
        ``_HISTORY_FETCH_LIMIT`` messages) looking for the most recent message
        sent by the bot itself.  Everything between that anchor point and the
        trigger message (exclusive on both ends) is collected as context.

        Falls back to the ``_CONTEXT_FALLBACK_LIMIT`` most-recent messages
        when the bot anchor cannot be found (bot was offline or message was
        deleted), so the LLM still receives some context.

        The result is always capped at ``self.context_message_limit`` entries
        to keep prompts from growing unbounded.

        Args:
            trigger_message: The Discord message that triggered this invocation.

        Returns:
            A ``_ConversationContext`` instance ready to be formatted for the prompt.
        """
        channel = trigger_message.channel
        bot_user = channel.guild.me if hasattr(channel, "guild") else None
        bot_id: Optional[int] = bot_user.id if bot_user is not None else None

        collected: list[tuple[str, str]] = []
        bot_message_found = False

        try:
            async for msg in channel.history(
                    limit=_HISTORY_FETCH_LIMIT,
                    before=trigger_message,
            ):
                if bot_id is not None and msg.author.id == bot_id:
                    bot_message_found = True
                    break

                content = (msg.content or "").strip()
                if not content:
                    continue

                author_name = (
                        msg.author.display_name or msg.author.name or str(msg.author.id)
                )
                collected.append((author_name, content))

        except discord.Forbidden:
            self.logger.warning(
                "Cannot read history in channel %s — missing Read Message History permission",
                getattr(channel, "name", channel.id),
            )
        except discord.HTTPException as exc:
            self.logger.warning(
                "Failed to fetch channel history for context: %s", exc
            )

        if not bot_message_found and not collected:
            return _ConversationContext(messages=[], bot_message_found=False)

        if not bot_message_found:
            self.logger.debug(
                "Bot anchor message not found in last %d messages "
                "(offline or deleted) — using fallback context (%d most-recent messages)",
                _HISTORY_FETCH_LIMIT,
                _CONTEXT_FALLBACK_LIMIT,
            )
            collected = collected[:_CONTEXT_FALLBACK_LIMIT]

        collected.reverse()

        if len(collected) > self.context_message_limit:
            self.logger.debug(
                "Context window trimmed from %d to %d messages (limit reached)",
                len(collected),
                self.context_message_limit,
            )
            collected = collected[-self.context_message_limit:]

        return _ConversationContext(messages=collected, bot_message_found=bot_message_found)