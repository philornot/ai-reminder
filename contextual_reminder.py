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
hardcoded pool (see ``_REACTION_EMOJI_CHOICES``). This runs as a best-effort
background task and never blocks or affects the normal reply-generation flow.

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

import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

import discord

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

# ---------------------------------------------------------------------------
# Single-task prompt variants
# ---------------------------------------------------------------------------
# Each mode used to ask the LLM to satisfy several instructions at once
# ("react to what they said" + "sneak in the book" + "match the tone" + ...).
# Small/quick models (like the default gpt-oss-120b) tend to produce
# nonsensical or ungrammatical output when asked to juggle that many
# constraints in a single short reply.
#
# Instead, Python picks ONE task per generation (random.choice) and the LLM
# only has to do that one thing. This mirrors the approach used for the
# scheduled reminder prompt in config.yaml (see Config.get_prompt()).
#
# These lists can be overridden per-deployment via the ai-reminder config.yaml
# under reminder.contextual_tasks / reminder.response_tasks / reminder.mention_tasks
# (each a list of strings). Placeholders {target_name}, {book_title} and
# {sender_name} are substituted before the task is inserted into the prompt.
_DEFAULT_CONTEXTUAL_TASKS: list[str] = [
    'Playfully riff on what they just said in one short phrase, then '
    'casually mention "{book_title}" — keep the link light, it does not '
    'need to make perfect logical sense.',
    'React to their message with a brief joke, then ask a quick question '
    'about how "{book_title}" is going.',
    'Give a short, deadpan reaction to what they said, then bring up '
    '"{book_title}" almost as an afterthought.',
    'Acknowledge their message in a few words, then pivot straight to '
    'asking whether they read "{book_title}" today.',
]

_DEFAULT_RESPONSE_TASKS: list[str] = [
    'If their reply is dismissive, tease them lightly about it. If it is '
    'positive, react with a bit of enthusiasm. Either way, keep '
    '"{book_title}" in the reply.',
    'Give a short witty comeback to what they said, and mention '
    '"{book_title}" only if it fits naturally — do not force it into a '
    'very short reply.',
    'React to their reply in one short line, then ask a quick follow-up '
    'question about "{book_title}".',
]

_DEFAULT_MENTION_TASKS: list[str] = [
    'Give a direct, genuine answer to what they asked or said — that comes '
    'first since this is a direct ping — and only weave in "{book_title}" '
    'if there is a natural opening.',
    'Answer them properly like a real reply to a ping, then add one short '
    'closing line nudging them toward "{book_title}".',
]

# ---------------------------------------------------------------------------
# Reply-reaction feature
# ---------------------------------------------------------------------------
# Hardcoded pool the LLM must pick from when it decides a reaction fits the
# target's reply. Kept small and hardcoded on purpose — this is a light
# flourish, not something that needs to be configurable per deployment.
# Maps a short name (what the LLM answers with) to the actual Discord/unicode
# emoji character (what gets passed to Message.add_reaction()).
_REACTION_EMOJI_CHOICES: dict[str, str] = {
    "flushed": "😳",
    "eyes": "👀",
    "joy": "😂",
    "skull": "💀",
    "thinking": "🤔",
    "salute": "🫡",
    "melting_face": "🫠",
    "sob": "😭",
    "fire": "🔥",
    "heart": "❤️",
}

# Short human-readable meaning for each emoji, shown to the LLM alongside the
# character itself. Small/quick models otherwise tend to pick an emoji based
# on a vague vibe rather than its actual meaning — most notably reaching for
# ``joy`` (😂, "laughing/crying with laughter", i.e. LOL) any time something
# is merely pleasant or nice, when that emoji specifically signals something
# was FUNNY, not that it made someone happy or proud. Spelling these out
# keeps the picked emoji aligned with what it actually communicates.
_REACTION_EMOJI_MEANINGS: dict[str, str] = {
    "flushed": "something unexpected, absurd or embarrassing",
    "eyes": "suspicious or intrigued side-eye, \"I see you\" / \"spill the tea\"",
    "joy": "laughing-crying at something FUNNY (LOL). NOT the same as being "
           "happy, glad, proud, or pleased — never use it for good news or "
           "compliments, only for genuine humor",
    "skull": "\"I'm dead\", something so funny or unhinged it killed you — dark/exaggerated humor",
    "thinking": "considering, skeptical, or mildly doubtful — a raised-eyebrow \"hmm\"",
    "salute": "respect, acknowledgement, \"got it\" / \"o7\" — earnest, not sarcastic",
    "melting_face": "cringing or dying of secondhand embarrassment (or heat) — awkward, not sad",
    "sob": "genuinely moved, touched, or sad — crying for real, not laughing",
    "fire": "impressed, \"that's awesome\" / \"this slaps\"",
    "heart": "genuine warmth or affection",
}

# Hard cap on how many sent messages we keep track of as "reactable" (i.e.
# eligible for a reply-reaction). Old entries are dropped first.
_MAX_REACTABLE_MESSAGES: int = 25

# When the target's message isn't an explicit Discord reply, we fall back to
# "the last message the bot sent" as the thing they're presumably answering.
# This window bounds how stale that last message may be before we stop
# treating an unrelated later message as a response to it.
_REACTABLE_LOOKBACK_MINUTES: int = 30

# How many messages of surrounding channel history to show the LLM when
# deciding whether a reaction fits. Two isolated sentences (the bot's message
# + the target's reply) can't reveal that the bot's message actually
# interrupted an unrelated conversation — this gives it enough to notice.
_REACTION_CONTEXT_LIMIT: int = 10


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

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    async def _handle_mention_mode(
            self, trigger_content: str, context: _ConversationContext
    ) -> bool:
        """Generate and send a reply when the bot is directly mentioned.

        Fires regardless of time-of-day window or daily-sent state — if the
        target pings the bot (or a role it has) they deserve a response.
        The book reminder is woven in naturally but the primary goal is a
        genuine reply to the mention, not a canned reminder.

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

        sent_message_id: Optional[int] = await loop.run_in_executor(
            None, self._webhook.send_reminder_get_id, reply_msg
        )
        success = sent_message_id is not None

        if success:
            self._cache.mark_as_sent(reply_msg)
            self._remember_reactable_message(sent_message_id, reply_msg)
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

        sent_message_id: Optional[int] = await loop.run_in_executor(
            None, self._webhook.send_reminder_get_id, reminder_msg
        )
        success = sent_message_id is not None

        if success:
            self._mark_sent_today()
            self._cache.mark_as_sent(reminder_msg)
            self._remember_reactable_message(sent_message_id, reminder_msg)
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

        sent_message_id: Optional[int] = await loop.run_in_executor(
            None, self._webhook.send_reminder_get_id, reply_msg
        )
        success = sent_message_id is not None

        if success:
            self._reset_response_cooldown()
            self._cache.mark_as_sent(reply_msg)
            self._remember_reactable_message(sent_message_id, reply_msg)
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
        # contextual_sent.json is also written/read by main.py's CacheManager
        # (was_contextual_sent_today) in a separate process — share the same
        # cross-process lock so the two never race on this file.
        with self._cache.lock():
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
        with self._cache.lock():
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
        with self._cache.lock():
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
        with self._cache.lock():
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
    # Reply-reaction helpers
    # ------------------------------------------------------------------

    def _remember_reactable_message(self, message_id: int, content: str) -> None:
        """Record a just-sent message as eligible for a future reply-reaction.

        Persisted to ``reactable_messages.json`` in the ai-reminder cache
        directory so that when the target later responds to it — via an
        explicit Discord reply, or just by writing again shortly after —
        ``handle_message()`` can recognise which message it was and hand it
        off to the LLM for a react/don't-react decision. The content is
        stored alongside the ID so that decision doesn't need an extra
        Discord API round trip later.

        The list is capped at ``_MAX_REACTABLE_MESSAGES`` entries (oldest
        dropped first) so the file cannot grow without bound.

        Args:
            message_id: Discord snowflake ID of the message that was just sent.
            content: Text content of that message.
        """
        with self._cache.lock():
            try:
                entries = self._read_reactable_messages()
                entries.append(
                    {
                        "message_id": message_id,
                        "content": content,
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                        # Set once this message has been picked as "the one
                        # being answered" for a reply-reaction decision, so
                        # it can't be picked again for a later message from
                        # the target (see _mark_reactable_consumed).
                        "consumed": False,
                    }
                )
                if len(entries) > _MAX_REACTABLE_MESSAGES:
                    entries = entries[-_MAX_REACTABLE_MESSAGES:]

                self._reactable_messages_path.parent.mkdir(parents=True, exist_ok=True)
                self._reactable_messages_path.write_text(
                    json.dumps(entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("Could not persist reactable message id: %s", exc)

    def _read_reactable_messages(self) -> list[dict]:
        """Load the list of bot-sent messages eligible for a reply-reaction.

        Returns:
            Parsed list of ``{"message_id", "content", "sent_at"}`` entries,
            or an empty list if the file is missing or unreadable.
        """
        with self._cache.lock():
            try:
                if not self._reactable_messages_path.exists():
                    return []
                return json.loads(
                    self._reactable_messages_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                self.logger.warning("Could not read reactable_messages.json: %s", exc)
                return []

    def _get_last_reactable_content(self) -> Optional[tuple[int, str]]:
        """Get the most recently sent, not-yet-consumed trackable bot message.

        Used as a fallback when the target's message isn't an explicit
        Discord reply: we still treat it as a plausible response to "whatever
        the bot last said", as long as that message is recent enough (see
        ``_REACTABLE_LOOKBACK_MINUTES``) to make that a reasonable guess.

        Entries already marked ``consumed`` are skipped — a single bot
        message may only be treated as "being answered" once, so that a
        reaction (or a no-react decision) only ever attaches to the first
        target message that follows it, not to every message the target
        sends afterwards. See ``_mark_reactable_consumed``.

        Returns:
            A ``(message_id, content)`` tuple for the most recent eligible
            tracked message, or None if there isn't one, it's already been
            consumed, or it's older than the lookback window.
        """
        entries = self._read_reactable_messages()
        if not entries:
            return None

        last_entry = entries[-1]
        if last_entry.get("consumed"):
            return None

        sent_at_raw = last_entry.get("sent_at")
        try:
            sent_at = datetime.fromisoformat(sent_at_raw)
        except (TypeError, ValueError):
            return None

        age = datetime.now(timezone.utc) - sent_at
        if age > timedelta(minutes=_REACTABLE_LOOKBACK_MINUTES):
            return None

        message_id = last_entry.get("message_id")
        content = last_entry.get("content")
        if message_id is None or content is None:
            return None
        return message_id, content

    def _find_reactable_content(self, message_id: int) -> Optional[tuple[int, str]]:
        """Look up the text of a tracked, not-yet-consumed bot-sent message.

        Args:
            message_id: Discord snowflake ID, typically taken from
                ``message.reference.message_id`` on an incoming reply.

        Returns:
            A ``(message_id, content)`` tuple if it is one the bot sent, is
            still tracked, and hasn't already been consumed by an earlier
            reply-reaction decision; otherwise None.
        """
        for entry in self._read_reactable_messages():
            if entry.get("message_id") == message_id:
                if entry.get("consumed"):
                    return None
                content = entry.get("content")
                if content is None:
                    return None
                return message_id, content
        return None

    def _mark_reactable_consumed(self, message_id: int) -> None:
        """Mark a tracked bot message as already used for a reaction decision.

        Called as soon as a target message is matched to a reactable bot
        message — *before* the (async, best-effort) react/don't-react
        decision even runs — so that a burst of several messages from the
        target right after a single bot message can only ever trigger one
        reply-reaction decision, tied to the first of those messages, rather
        than one decision (and potentially one reaction) per message.

        Args:
            message_id: Discord snowflake ID of the bot message that was
                matched as "the one being answered".
        """
        with self._cache.lock():
            try:
                entries = self._read_reactable_messages()
                changed = False
                for entry in entries:
                    if entry.get("message_id") == message_id and not entry.get("consumed"):
                        entry["consumed"] = True
                        changed = True
                        break
                if not changed:
                    return
                self._reactable_messages_path.parent.mkdir(parents=True, exist_ok=True)
                self._reactable_messages_path.write_text(
                    json.dumps(entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("Could not mark reactable message consumed: %s", exc)

    async def _fetch_recent_channel_messages(
            self, trigger_message: discord.Message, limit: int = _REACTION_CONTEXT_LIMIT
    ) -> _ConversationContext:
        """Fetch the last few channel messages before trigger_message.

        A flat, simple history read — unlike ``_fetch_conversation_context()``
        this doesn't try to anchor on the bot's last message, it just grabs
        "whatever was said recently". Used to give the reply-reaction
        decision enough surrounding context to notice when the bot's message
        actually interrupted an unrelated conversation, which the bot's
        message and the target's reply alone can't show.

        Args:
            trigger_message: The message to fetch history before.
            limit: Maximum number of prior messages to fetch.

        Returns:
            A ``_ConversationContext`` with up to ``limit`` messages, oldest
            first, or an empty one if history couldn't be read.
        """
        channel = trigger_message.channel
        collected: list[tuple[str, str]] = []

        try:
            async for msg in channel.history(limit=limit, before=trigger_message):
                content = (msg.content or "").strip()
                if not content:
                    continue
                author_name = (
                        msg.author.display_name or msg.author.name or str(msg.author.id)
                )
                collected.append((author_name, content))
        except discord.Forbidden:
            self.logger.warning(
                "Cannot read history in channel %s for reaction context — "
                "missing Read Message History permission",
                getattr(channel, "name", channel.id),
            )
        except discord.HTTPException as exc:
            self.logger.warning(
                "Failed to fetch channel history for reaction context: %s", exc
            )

        collected.reverse()
        return _ConversationContext(messages=collected, bot_message_found=False)

    def _build_reaction_decision_prompt(
            self,
            reply_content: str,
            replied_to_content: str,
            context: _ConversationContext,
    ) -> str:
        """Build a prompt asking the LLM whether a reaction fits this reply.

        Deliberately framed as a judgement call rather than a default-yes:
        the target's "reply" might just be them continuing an unrelated
        conversation the bot's message happened to land in the middle of,
        in which case a reaction would look like the bot wasn't paying
        attention.

        Args:
            reply_content: Stripped text of the target's Discord reply.
            replied_to_content: Text of the bot's message being replied to.
            context: Recent surrounding channel history, so the LLM can spot
                whether the bot's message actually interrupted an unrelated
                conversation.

        Returns:
            Fully formatted prompt string ready to be sent to the LLM.
        """
        target = self._ai_config.get("reminder.target_name", "target")
        emoji_list = "\n".join(
            f"- {name}: {char} — {_REACTION_EMOJI_MEANINGS.get(name, '')}"
            for name, char in _REACTION_EMOJI_CHOICES.items()
        )

        return (
            f"Recent conversation in the channel (most recent last):\n"
            f"{context.format_for_prompt()}\n\n"
            f"You (the bot) then sent {target} this message:\n"
            f'"{replied_to_content}"\n\n'
            f"{target} then wrote this, plausibly in response to it:\n"
            f'"{reply_content}"\n\n'
            "Using the conversation above for context, decide whether adding "
            "an emoji reaction to that message would feel natural here, the "
            "way a person casually reacts to a text message. Say no if the "
            "message is long, serious, sensitive, or reads like it's "
            "actually part of a different conversation your message just "
            "interrupted rather than genuinely engaging with it — a "
            "reaction would look tone-deaf or inattentive there.\n\n"
            "If you do react, choose ONLY one name from this list, and pick "
            "based on its actual meaning below, not just a general good/bad "
            "vibe (e.g. \"joy\" means laughing at something funny — do not "
            "pick it just because the moment is positive or nice):\n"
            f"{emoji_list}\n\n"
            "Respond with ONLY a single compact JSON object and nothing "
            "else, in exactly one of these two shapes:\n"
            '{"react": true, "emoji": "<one name from the list above>"}\n'
            '{"react": false, "emoji": null}'
        )

    async def _maybe_react_to_reply(
            self,
            message: discord.Message,
            reply_content: str,
            replied_to_content: str,
    ) -> None:
        """Ask the LLM whether an emoji reaction fits, and add it if so.

        Runs as a fire-and-forget background task whenever the target sends
        a message that plausibly answers one of the bot's own tracked
        messages — either an explicit Discord reply, or simply their next
        message shortly after. Reacting is treated as an optional,
        low-stakes flourish, so any failure in this path (LLM call, JSON
        parsing, Discord API) is logged and swallowed rather than raised —
        it must never affect the main reply-generation flow in
        ``handle_message()``.

        Args:
            message: The target's message (the one to react to).
            reply_content: Stripped text content of the target's message.
            replied_to_content: Text of the bot's message it's answering.
        """
        try:
            context = await self._fetch_recent_channel_messages(message)
            prompt = self._build_reaction_decision_prompt(
                reply_content, replied_to_content, context
            )

            loop = asyncio.get_running_loop()
            decision: Optional[dict] = await loop.run_in_executor(
                None, self._llm.generate_json, prompt
            )

            if not decision or not decision.get("react"):
                self.logger.debug("Reaction decision: not reacting (%s)", decision)
                return

            emoji_name = decision.get("emoji")
            emoji_char = _REACTION_EMOJI_CHOICES.get(emoji_name)
            if emoji_char is None:
                self.logger.warning(
                    "Reaction decision picked an unknown emoji name: %r — skipping",
                    emoji_name,
                )
                return

            await message.add_reaction(emoji_char)
            self.logger.info("Reacted to reply with :%s: (%s)", emoji_name, emoji_char)

        except discord.HTTPException as exc:
            self.logger.warning("Failed to add reaction: %s", exc)
        except Exception:
            self.logger.exception("Unexpected error while deciding/adding a reaction")

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _pick_task(self, config_key: str, defaults: list[str]) -> str:
        """Pick ONE random task instruction for this generation.

        Reads an optional list of task variants from the ai-reminder config
        (falling back to the built-in defaults), then substitutes the
        target/book/sender placeholders into the chosen variant.

        Picking a single task in Python — instead of asking the LLM to
        satisfy several instructions in one go — keeps each generation
        focused on one clear thing, which noticeably reduces incoherent or
        ungrammatical replies from smaller models.

        Args:
            config_key: Dot-notation key under the ai-reminder config
                (e.g. ``"reminder.contextual_tasks"``).
            defaults: Fallback list of task templates used when the key is
                absent or empty in config.

        Returns:
            A single, fully-formatted task instruction string.
        """
        tasks = self._ai_config.get(config_key, None) or defaults

        target = self._ai_config.get("reminder.target_name", "target")
        book = self._ai_config.get("reminder.book_title", "the book")
        sender = self._ai_config.get("reminder.sender_name", "my creator")

        task_template = random.choice(tasks)
        try:
            return task_template.format(
                target_name=target, book_title=book, sender_name=sender
            )
        except KeyError:
            # A misconfigured custom task used an unknown placeholder —
            # fall back to the raw template rather than crashing generation.
            self.logger.warning(
                "Unknown placeholder in task template for %s; using it as-is",
                config_key,
            )
            return task_template

    def _build_mention_prompt(
            self, trigger_message: str, context: _ConversationContext
    ) -> str:
        """Build an LLM prompt for a direct-mention reply.

        The target explicitly pinged the bot (or a role it has), so the
        reply should feel like a direct, personal response rather than a
        canned reminder. The book is woven in only where it fits naturally.

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
        task = self._pick_task("reminder.mention_tasks", _DEFAULT_MENTION_TASKS)

        return (
            f'You are a reminder bot whose job is to nudge {target} to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f"{target} just directly mentioned (pinged) you and wrote:\n"
            f'"{trigger_message}"\n\n'
            f"Task for this reply: {task}\n\n"
            f"Write ONE SHORT (1-2 sentences) reply. Sound like a real "
            f"person texting, not a bot, and take the conversation context "
            f"shown above into account.\n\n"
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
        task = self._pick_task("reminder.contextual_tasks", _DEFAULT_CONTEXTUAL_TASKS)

        return (
            f'You remind {target} to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f'{target} just wrote on Discord:\n'
            f'"{trigger_message}"\n\n'
            f"Task for this reply: {task}\n\n"
            f"Write ONE SHORT (1-2 sentences) casual reply. Sound like a "
            f"real person texting, not a bot. Take the recent conversation "
            f"into account if it's relevant, but don't force a connection "
            f"that doesn't make sense.\n\n"
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
        task = self._pick_task("reminder.response_tasks", _DEFAULT_RESPONSE_TASKS)

        return (
            f'You just sent {target} a reminder to read "{book}". '
            f"{target} is {gender} — always use grammatically correct "
            f"forms for a {gender} person.\n\n"
            f"{context_section}"
            f"They replied with:\n"
            f'"{trigger_message}"\n\n'
            f"Task for this reply: {task}\n\n"
            f"Write ONE SHORT (1-2 sentences) witty comeback. Sound like a "
            f"real person texting back, not a bot, and don't be warmer or "
            f"colder than the conversation shown above warrants.\n\n"
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
