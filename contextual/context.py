"""Shared conversation-context data structure used across the ``contextual`` package."""

from dataclasses import dataclass, field


@dataclass
class ConversationContext:
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