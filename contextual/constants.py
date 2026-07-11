"""Module-level constants and default prompt data for the ``contextual`` package.

Nothing in this file has side effects — it only defines tunables and the
default task/emoji tables used by the other modules in this package.
"""

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
# Overridable via contextual_reminder.reactable_lookback_minutes in the
# listener config; see ContextualReminder.__init__.
_DEFAULT_REACTABLE_LOOKBACK_MINUTES: int = 30

# How many messages of surrounding channel history to show the LLM when
# deciding whether a reaction fits. Two isolated sentences (the bot's message
# + the target's reply) can't reveal that the bot's message actually
# interrupted an unrelated conversation — this gives it enough to notice.
# Overridable via contextual_reminder.reaction_context_limit in the listener
# config; see ContextualReminder.__init__.
_DEFAULT_REACTION_CONTEXT_LIMIT: int = 10