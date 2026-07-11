"""Contextual book reminder package.

- ``reminder`` — the ``ContextualReminder`` orchestrator (config bootstrap,
  ``handle_message()`` dispatch, mention detection, conversation-context
  fetching).
- ``modes`` — ``_ModeHandlerMixin`` (per-mode generation + send: mention,
  contextual, response-window).
- ``prompts`` — ``_PromptBuilderMixin`` (LLM prompt construction).
- ``reactions`` — ``_ReactionMixin`` (reply-reaction tracking + decision).
- ``conditions`` — ``_ConditionsMixin`` (daily-sent / time-window / cooldown).
- ``context`` — the ``ConversationContext`` dataclass shared across the
  package.
- ``constants`` — tunables, default task lists, emoji tables.

``ContextualReminder`` is re-exported here so external callers (e.g.
``bot_listener.py``) can keep writing:

    from contextual import ContextualReminder

instead of reaching into the ``reminder`` submodule directly.
"""

from .reminder import ContextualReminder

__all__ = ["ContextualReminder"]