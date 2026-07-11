"""Prompt-construction helpers for the three reply modes.

``_PromptBuilderMixin`` is mixed into
``ContextualReminder`` (see ``contextual.reminder``) and assumes the host
class provides ``self._ai_config``, ``self._cache``, and ``self.logger``.
"""

import random

from .constants import (
    _DEFAULT_CONTEXTUAL_TASKS,
    _DEFAULT_MENTION_TASKS,
    _DEFAULT_RESPONSE_TASKS,
)
from .context import ConversationContext as _ConversationContext


class _PromptBuilderMixin:
    """Builds LLM prompts for mention/contextual/response-window replies."""

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