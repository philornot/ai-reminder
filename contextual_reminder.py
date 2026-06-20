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
   When the target directly pings the bot (``@bot``), a reply is sent
   immediately regardless of time-of-day window or daily-sent state.
   The book is woven in where it fits naturally, but the primary goal is
   a genuine response to the ping.  No cooldown applies.

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
"""

from pathlib import Path

import asyncio
import discord
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional

# How long to wait for the generation lock before giving up.
_GENERATION_LOCK_TIMEOUT_S: float = 5.0

# Default cooldown between response-mode replies (minutes).
_DEFAULT_RESPONSE_COOLDOWN_MINUTES: int = 30

# Default hard cap on how many messages we pull for context, regardless of
# how far back the last bot message is.
_DEFAULT_CONTEXT_MESSAGE_LIMIT: int = 20

# When the bot's last message cannot be found in history, fall back to this
# many most-recent messages so the LLM still has *some* context.
_CONTEXT_FALLBACK_LIMIT: int = 10

# discord.py history() fetch limit — fetched in one batch; large enough to
# find the bot message in most normal channels without being wasteful.
_HISTORY_FETCH_LIMIT: int = 50


@dataclass
class _ConversationContext:
    """A slice of channel history between the last bot message and the trigger.

    Attributes:
        messages: Ordered list of (author_display_name, content) tuples,
            oldest first, NOT including the trigger message itself.
        bot_message_found: True when the window was anchored to an actual bot
            message; False when it fell back to the most-recent N messages.
    """

    messages: list[tuple[str, str]] = field(default_factory=list)
    bot_message_found: bool = False

    def is_empty(self) -> bool:
        """Return True when there are no messages in the context window.

        Returns:
            True if the message list is empty.
        """
        return len(self.messages) == 0

    def format_for_prompt(self) -> str:
        """Render the context as a human-readable block for LLM prompts.

        Returns:
            Formatted string, one line per message, or a placeholder when empty.
        """
        if self.is_empty():
            return "(no recent conversation context available)"
        return "\n".join(f"{author}: {content}" for author, content in self.messages)


class ContextualReminder:
    """Generates and sends context-aware book reminders triggered by Discord messages.

    Watches for messages from a specific Discord user and, when conditions are
    met, asks the configured LLM to craft a reply.  The reply style depends on
    the active mode:

    - **Mention mode**: fires when the bot is directly pinged (``@bot``).
      Replies immediately regardless of time window or daily-sent state.
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

    async def handle_message(self, message: discord.Message) -> bool:
        """Process an incoming Discord message and act if appropriate.

        Decides which mode to run based on the current state:

        - **Mention mode**: fires when ``self._bot_user`` is in
          ``message.mentions``, bypassing time-window and daily-sent checks.
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
        # Determine mode without the lock (full re-check happens inside)
        # ------------------------------------------------------------------
        is_mention = (
            self._bot_user is not None
            and self._bot_user in message.mentions
        )
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
            is_mention = (
                self._bot_user is not None
                and self._bot_user in message.mentions
            )
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

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    async def _handle_mention_mode(
            self, trigger_content: str, context: _ConversationContext
    ) -> bool:
        """Generate and send a reply when the bot is directly mentioned.

        Fires regardless of time-of-day window or daily-sent state — if the
        target pings the bot they deserve a response.  The book reminder is
        woven in naturally but the primary goal is to feel like a genuine
        reply to the mention, not a canned reminder.

        Does NOT mark ``contextual_sent_today`` so the regular daily reminder
        can still fire later in the day.

        Args:
            trigger_content: Raw text of the message that mentioned the bot.
            context: Conversation history window fetched from the channel.

        Returns:
            True if a reply was delivered successfully.
        """
        self.logger.info(
            "Mention mode — generating reply for: %.80s%s",
            trigger_content,
            "..." if len(trigger_content) > 80 else "",
        )

        loop = asyncio.get_running_loop()

        try:
            reply_msg: Optional[str] = await loop.run_in_executor(
                None, self._generate_mention_message, trigger_content, context
            )
        except Exception as exc:
            self.logger.error("Mention-mode generation failed: %s", exc)
            return False

        if not reply_msg:
            self.logger.warning("LLM returned empty message for mention reply")
            return False

        success: bool = await loop.run_in_executor(
            None, self._webhook.send_reminder, reply_msg
        )

        if success:
            self._cache.mark_as_sent(reply_msg)
            self.logger.info(
                "Mention-mode reply sent: %.80s%s",
                reply_msg,
                "..." if len(reply_msg) > 80 else "",
            )
        else:
            self.logger.error("Mention-mode reply: webhook delivery failed")

        return success

    async def _handle_contextual_mode(
            self, trigger_content: str, context: _ConversationContext
    ) -> bool:
        """Generate and send a contextual reminder reacting to trigger_content.

        Crafts a reply that feels like a genuine reaction to the target's
        message while weaving in the book reminder. The conversation context
        window is included in the prompt so the LLM can reference the broader
        conversation, not just the single trigger line.

        Args:
            trigger_content: Raw text of the target's Discord message.
            context: Conversation history window fetched from the channel.

        Returns:
            True if the reminder was delivered successfully.
        """
        self.logger.info(
            "Contextual mode — generating reminder for trigger: %.80s%s",
            trigger_content,
            "..." if len(trigger_content) > 80 else "",
        )

        loop = asyncio.get_running_loop()

        try:
            reminder_msg: Optional[str] = await loop.run_in_executor(
                None, self._generate_contextual_message, trigger_content, context
            )
        except Exception as exc:
            self.logger.error("Contextual reminder generation failed: %s", exc)
            return False

        if not reminder_msg:
            self.logger.warning("LLM returned empty message for contextual reminder")
            return False

        success: bool = await loop.run_in_executor(
            None, self._webhook.send_reminder, reminder_msg
        )

        if success:
            self._mark_sent_today()
            self._cache.mark_as_sent(reminder_msg)
            self.logger.info(
                "Contextual reminder sent: %.80s%s",
                reminder_msg,
                "..." if len(reminder_msg) > 80 else "",
            )
        else:
            self.logger.error("Contextual reminder: webhook delivery failed")

        return success

    async def _handle_response_mode(
            self, trigger_content: str, context: _ConversationContext
    ) -> bool:
        """Generate and send a witty comeback to the target's reply.

        Fires while the response window is open (i.e. shortly after a
        scheduled reminder was sent).  The LLM is prompted to acknowledge
        what the target wrote and keep the book in the conversation.
        After a successful send the per-reply cooldown is reset.

        Args:
            trigger_content: Raw text of the target's Discord message.
            context: Conversation history window fetched from the channel.

        Returns:
            True if the comeback was delivered successfully.
        """
        self.logger.info(
            "Response-window mode — generating comeback for: %.80s%s",
            trigger_content,
            "..." if len(trigger_content) > 80 else "",
        )

        loop = asyncio.get_running_loop()

        try:
            reply_msg: Optional[str] = await loop.run_in_executor(
                None, self._generate_response_message, trigger_content, context
            )
        except Exception as exc:
            self.logger.error("Response-mode generation failed: %s", exc)
            return False

        if not reply_msg:
            self.logger.warning("LLM returned empty message for response-mode reply")
            return False

        success: bool = await loop.run_in_executor(
            None, self._webhook.send_reminder, reply_msg
        )

        if success:
            self._reset_response_cooldown()
            self._cache.mark_as_sent(reply_msg)
            self.logger.info(
                "Response-mode reply sent: %.80s%s",
                reply_msg,
                "..." if len(reply_msg) > 80 else "",
            )
        else:
            self.logger.error("Response-mode reply: webhook delivery failed")

        return success

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _should_send(self) -> bool:
        """Evaluate whether contextual-mode conditions are satisfied.

        Checks whether a contextual reminder was already sent today and,
        when ``randomize_time`` is enabled, whether the current local time
        falls inside the configured window.

        Returns:
            True if a contextual-mode reminder should be sent right now.
        """
        if self._was_sent_today():
            return False

        if self._ai_config.time_randomize and not self._is_within_time_range():
            self.logger.debug(
                "Contextual reminder: outside time window [%s – %s] — skipping",
                self._ai_config.time_range_start,
                self._ai_config.time_range_end,
            )
            return False

        return True

    def _was_sent_today(self) -> bool:
        """Check whether a contextual reminder was already sent today.

        Returns:
            True if ``contextual_sent.json`` records today's ISO date.
        """
        today = date.today().isoformat()
        try:
            if not self._contextual_sent_path.exists():
                return False
            data = json.loads(
                self._contextual_sent_path.read_text(encoding="utf-8")
            )
            return data.get("last_sent_date") == today
        except Exception as exc:
            self.logger.warning(
                "Could not read contextual_sent.json: %s — assuming not sent today",
                exc,
            )
            return False

    def _is_within_time_range(self) -> bool:
        """Return True if the current local time is inside the reminder window.

        Returns:
            True when ``time_range.start <= now <= time_range.end``.
        """

        def _parse(s: str) -> dtime:
            h, m = map(int, s.split(":"))
            return dtime(h, m)

        now = datetime.now().time()
        start = _parse(self._ai_config.time_range_start)
        end = _parse(self._ai_config.time_range_end)
        return start <= now <= end

    def _mark_sent_today(self) -> None:
        """Persist today's date to ``contextual_sent.json``."""
        today = date.today().isoformat()
        try:
            self._contextual_sent_path.parent.mkdir(parents=True, exist_ok=True)
            self._contextual_sent_path.write_text(
                json.dumps(
                    {
                        "last_sent_date": today,
                        "sent_at": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("Could not write contextual_sent.json: %s", exc)

    # ------------------------------------------------------------------
    # Response-window cooldown helpers
    # ------------------------------------------------------------------

    def _response_cooldown_elapsed(self) -> bool:
        """Return True if enough time has passed since the last response-mode reply.

        Reads ``response_cooldown.json`` which stores the timestamp of the last
        successful reply.  If the file is absent (no reply sent yet in this
        window) the cooldown is considered elapsed.

        Returns:
            True if the bot is allowed to reply again.
        """
        try:
            if not self._response_cooldown_path.exists():
                return True
            data = json.loads(
                self._response_cooldown_path.read_text(encoding="utf-8")
            )
            last_reply_at = datetime.fromisoformat(data["last_reply_at"])
            if last_reply_at.tzinfo is None:
                last_reply_at = last_reply_at.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - last_reply_at
            cooldown = timedelta(minutes=self.response_cooldown_minutes)
            if elapsed >= cooldown:
                return True
            remaining = (cooldown - elapsed).total_seconds() / 60
            self.logger.debug(
                "Response cooldown: %.1f min remaining", remaining
            )
            return False
        except Exception as exc:
            self.logger.warning(
                "Could not read response_cooldown.json: %s — assuming cooldown elapsed",
                exc,
            )
            return True

    def _reset_response_cooldown(self) -> None:
        """Write the current UTC timestamp to ``response_cooldown.json``.

        Called immediately after a successful response-mode reply so that
        the next incoming message is held until the cooldown expires.
        """
        now = datetime.now(timezone.utc)
        try:
            self._response_cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            self._response_cooldown_path.write_text(
                json.dumps(
                    {"last_reply_at": now.isoformat()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.logger.debug(
                "Response cooldown reset — next reply allowed in %d min",
                self.response_cooldown_minutes,
            )
        except Exception as exc:
            self.logger.warning(
                "Could not write response_cooldown.json: %s", exc
            )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_mention_prompt(
            self, trigger_message: str, context: _ConversationContext
    ) -> str:
        """Build an LLM prompt for a direct-mention reply.

        The target explicitly pinged the bot, so the reply should feel like a
        direct, personal response rather than a canned reminder. The book is
        woven in only where it fits naturally.

        Args:
            trigger_message: Raw text of the message that mentioned the bot.
                Mention tokens like ``<@123>`` are left in; the LLM ignores them.
            context: Conversation history window between the last bot message
                and the trigger.

        Returns:
            Fully formatted prompt string ready to be sent to the LLM.
        """
        recent = self._cache.get_recent_sent_messages(count=3)
        recent_text = (
            "\n".join(f"- {m}" for m in recent) if recent else "(no previous messages)"
        )

        target = self._ai_config.get("reminder.target_name", "target")
        book = self._ai_config.get("reminder.book_title", "the book")
        language = self._ai_config.get("reminder.language", "Polish")
        gender = self._ai_config.get("reminder.target_gender", "female")

        context_section = self._format_context_section(context)

        return (
            f'You are a reminder bot whose job is to nudge {target} to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f"{target} just directly mentioned (pinged) you and wrote:\n"
            f'"{trigger_message}"\n\n'
            f"Write ONE SHORT (1-2 sentences) reply that:\n"
            f"1. Directly and naturally responds to what they said or asked — "
            f"this is a direct ping so they expect a real answer\n"
            f'2. If there\'s a natural opportunity, weave in a mention of "{book}", '
            f"but don't force it — a genuine reply to the ping comes first\n"
            f"3. Sounds like a real person texting, not a bot\n"
            f"4. Takes the conversation context shown above into account\n\n"
            f"Recent reminders already sent (do not repeat these patterns):\n"
            f"{recent_text}\n\n"
            f"Respond in {language}. Output only the message, nothing else."
        )

    def _build_contextual_prompt(
            self, trigger_message: str, context: _ConversationContext
    ) -> str:
        """Build an LLM prompt for contextual-mode reply generation.

        Instructs the model to craft a short reply that feels like a genuine
        reaction to the trigger message (and the conversation leading up to it)
        while weaving in the book reminder.

        Args:
            trigger_message: Raw text content of the target's Discord message.
            context: Conversation history window between the last bot message
                and the trigger.

        Returns:
            Fully formatted prompt string ready to be sent to the LLM.
        """
        recent = self._cache.get_recent_sent_messages(count=3)
        recent_text = (
            "\n".join(f"- {m}" for m in recent) if recent else "(no previous messages)"
        )

        target = self._ai_config.get("reminder.target_name", "target")
        book = self._ai_config.get("reminder.book_title", "the book")
        language = self._ai_config.get("reminder.language", "Polish")
        gender = self._ai_config.get("reminder.target_gender", "female")

        context_section = self._format_context_section(context)

        return (
            f'You remind {target} to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f'{target} just wrote on Discord:\n'
            f'"{trigger_message}"\n\n'
            f"Write ONE SHORT (1-2 sentences) casual reply that:\n"
            f"1. Playfully references or riffs on what they just said "
            f"(taking the recent conversation into account if relevant)\n"
            f'2. Sneaks in a reminder about "{book}" — the connection should '
            f"feel witty, not forced\n"
            f"3. Sounds like a real person texting, not a bot\n\n"
            f"Tone examples (adapt, don't copy):\n"
            f'- They said "Anyone want to play Valorant?" → '
            f'"Maybe you\'d like to play reading {book}?"\n'
            f'- They said "I\'m so tired today" → '
            f'"Lie in bed with {book} then, perfect excuse"\n'
            f'- They said "What should I cook for dinner?" → '
            f'"No idea, but I know what you should do while it\'s on the stove"\n\n'
            f"Recent reminders already sent (do not repeat these patterns):\n"
            f"{recent_text}\n\n"
            f"Respond in {language}. Output only the message, nothing else."
        )

    def _build_response_prompt(
            self, trigger_message: str, context: _ConversationContext
    ) -> str:
        """Build an LLM prompt for response-window-mode reply generation.

        The target has just replied to a scheduled reminder. The model should
        acknowledge what they wrote with a witty comeback and keep the book in
        the conversation without being annoying. The conversation context window
        gives the LLM awareness of the tone of the exchange so far.

        Args:
            trigger_message: Raw text of the target's reply to the reminder.
            context: Conversation history window between the last bot message
                (the reminder itself) and the trigger.

        Returns:
            Fully formatted prompt string ready to be sent to the LLM.
        """
        recent = self._cache.get_recent_sent_messages(count=3)
        recent_text = (
            "\n".join(f"- {m}" for m in recent) if recent else "(no previous messages)"
        )

        target = self._ai_config.get("reminder.target_name", "target")
        book = self._ai_config.get("reminder.book_title", "the book")
        language = self._ai_config.get("reminder.language", "Polish")
        gender = self._ai_config.get("reminder.target_gender", "female")

        context_section = self._format_context_section(context)

        return (
            f'You just sent {target} a reminder to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f"They replied with:\n"
            f'"{trigger_message}"\n\n'
            f"Write ONE SHORT (1-2 sentences) witty comeback that:\n"
            f"1. Directly acknowledges what they said — if they're dismissive "
            f'(e.g. "stfu", "nie", "zostaw mnie"), lean into it with playful '
            f"sass; if they're positive (e.g. \"ok\", \"przeczytałam\"), "
            f"celebrate a little but stay in character\n"
            f'2. Keeps "{book}" in the conversation naturally — '
            f"don't force it if the reply is very short\n"
            f"3. Sounds like a real person texting back, not a bot\n"
            f"4. Takes into account the tone of the conversation so far "
            f"(visible above) — don't be warmer or colder than the exchange warrants\n\n"
            f"Tone examples (adapt, don't copy):\n"
            f'- They said "stfu" → "nie, nie i jeszcze raz nie"\n'
            f'- They said "ok ok" → "ok ok to za mało, strona minimum"\n'
            f'- They said "przeczytałam już" → '
            f'"serio?? ile stron? i co sądzisz?"\n'
            f'- They said "zostaw mnie" → '
            f'"nie mogę, {book} by mi tego nie wybaczyła"\n\n'
            f"Recent messages you already sent (do not repeat these patterns):\n"
            f"{recent_text}\n\n"
            f"Respond in {language}. Output only the message, nothing else."
        )

    def _format_context_section(self, context: _ConversationContext) -> str:
        """Render the context window as a labelled prompt section.

        When the context is empty a short note is emitted instead so the LLM
        understands there simply was no prior conversation to reference.

        Args:
            context: Fetched conversation context.

        Returns:
            A ready-to-embed string block (always ends with two newlines).
        """
        if context.is_empty():
            return "Recent conversation context: (none — this appears to be the start of the conversation)\n\n"

        anchor_note = (
            "since your last message"
            if context.bot_message_found
            else "most recent messages (your last message was not found in history)"
        )
        formatted = context.format_for_prompt()
        return (
            f"Recent conversation context ({anchor_note}):\n"
            f"{formatted}\n\n"
        )

    # ------------------------------------------------------------------
    # Generation (blocking — must be called via run_in_executor)
    # ------------------------------------------------------------------

    def _generate_mention_message(
            self, trigger_content: str, context: _ConversationContext
    ) -> Optional[str]:
        """Synchronous mention-mode generation step.

        Args:
            trigger_content: Text content of the message that mentioned the bot.
            context: Conversation history window.

        Returns:
            Generated reply text, or ``None`` on failure.
        """
        prompt = self._build_mention_prompt(trigger_content, context)
        return self._llm.generate_message(prompt)

    def _generate_contextual_message(
            self, trigger_content: str, context: _ConversationContext
    ) -> Optional[str]:
        """Synchronous contextual-mode generation step.

        Args:
            trigger_content: Text content of the trigger message.
            context: Conversation history window.

        Returns:
            Generated reminder text, or ``None`` on failure.
        """
        prompt = self._build_contextual_prompt(trigger_content, context)
        return self._llm.generate_message(prompt)

    def _generate_response_message(
            self, trigger_content: str, context: _ConversationContext
    ) -> Optional[str]:
        """Synchronous response-window-mode generation step.

        Args:
            trigger_content: Text content of the target's reply to the reminder.
            context: Conversation history window.

        Returns:
            Generated comeback text, or ``None`` on failure.
        """
        prompt = self._build_response_prompt(trigger_content, context)
        return self._llm.generate_message(prompt)