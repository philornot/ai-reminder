"""Reply-reaction feature: tracking reactable bot messages and deciding whether to react.

Split out of ``contextual_reminder.py``. ``_ReactionMixin`` is mixed into
``ContextualReminder`` (see ``contextual.reminder``) and assumes the host
class provides ``self._cache``, ``self._ai_config``, ``self._llm``,
``self.logger``, ``self._reactable_messages_path``,
``self.reactable_lookback_minutes``, and ``self.reaction_context_limit``.

The whole feature can be disabled deployment-wide via
``contextual_reminder.enable_reactions`` in the listener config — see
``ContextualReminder.__init__`` and ``handle_message()`` in
``contextual.reminder``, which skips tracking/matching entirely when that
flag is off.

Whenever the target sends a message that's plausibly answering one of the
bot's own sent messages (mention, contextual, or response-window) — either an
explicit Discord "reply", or simply the next thing they write shortly after —
this module asks the LLM a small yes/no question: does adding an emoji
reaction actually fit here, or would it look tone-deaf (e.g. the bot's
message landed in the middle of an unrelated conversation the target is
really continuing)? Only when the LLM says yes does the bot add a reaction,
picking one emoji from a small hardcoded pool (see ``_REACTION_EMOJI_CHOICES``
in ``contextual.constants``). This runs as a best-effort background task and
never blocks or affects the normal reply-generation flow.

A tracked bot message can only ever be matched to ONE reply-reaction
decision: as soon as a target message is matched to it, it's marked
"consumed" (see ``_mark_reactable_consumed``) so a burst of several messages
from the target right after a single bot message doesn't cause every one of
them to get reacted to.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord

from .constants import (
    _MAX_REACTABLE_MESSAGES,
    _REACTION_EMOJI_CHOICES,
    _REACTION_EMOJI_MEANINGS,
)
from .context import ConversationContext as _ConversationContext


class _ReactionMixin:
    """Tracks reactable bot messages and drives the react/don't-react decision."""

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
        if not self.enable_reactions:
            return

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
            Parsed list of ``{"message_id", "content", "sent_at", "consumed"}``
            entries, or an empty list if the file is missing or unreadable.
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
        ``self.reactable_lookback_minutes``) to make that a reasonable guess.

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
        if age > timedelta(minutes=self.reactable_lookback_minutes):
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
            self, trigger_message: discord.Message, limit: Optional[int] = None
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
            limit: Maximum number of prior messages to fetch. Defaults to
                ``self.reaction_context_limit`` (config-overridable via
                ``contextual_reminder.reaction_context_limit``) when omitted.

        Returns:
            A ``_ConversationContext`` with up to ``limit`` messages, oldest
            first, or an empty one if history couldn't be read.
        """
        if limit is None:
            limit = self.reaction_context_limit

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