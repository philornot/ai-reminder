"""Per-mode reply generation and sending: mention, contextual, and response-window modes.

``_ModeHandlerMixin`` is mixed into
``ContextualReminder`` (see ``contextual.reminder``) and assumes the host
class provides ``self._llm``, ``self._webhook``, ``self._cache``,
``self.logger``, and the prompt builders from ``_PromptBuilderMixin``
(``self._build_mention_prompt`` / ``self._build_contextual_prompt`` /
``self._build_response_prompt``) as well as the cooldown/sent-state helpers
from ``_ConditionsMixin`` and ``self._remember_reactable_message`` from
``_ReactionMixin``.

Each mode follows the same three-step shape: generate the message (blocking
LLM call, run in an executor), send it via the webhook, then update whatever
state that mode is responsible for (daily-sent flag, response cooldown,
reactable-message tracking) only on success.
"""

import asyncio
from typing import Optional

from .context import ConversationContext as _ConversationContext


class _ModeHandlerMixin:
    """Generates and sends replies for mention, contextual, and response-window modes."""

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
        scheduled reminder was sent). The LLM is prompted to acknowledge
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