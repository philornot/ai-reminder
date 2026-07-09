# AI Reminder

Automated reminder system that uses AI to generate creative messages and sends
them via Discord webhooks. Optionally extends into a live Discord listener
(`bot_listener.py`) that reacts to the target user's messages in real time.

## Components

| File                     | Purpose                                                            |
|--------------------------|--------------------------------------------------------------------|
| `main.py`                | Scheduled reminder loop — generates and sends messages on a timer  |
| `bot_listener.py`        | Discord bot — watches for messages and triggers contextual replies |
| `contextual_reminder.py` | Core logic shared by both modes                                    |
| `cache_manager.py`       | Persists pre-generated messages and response-window state          |
| `llm_client.py`          | Thin wrapper around OpenAI / Gemini / Groq APIs                    |
| `discord_webhook.py`     | Sends messages via Discord webhooks                                |
| `config_loader.py`       | Loads and validates `config/config.yaml`                           |
| `scheduler.py`           | Calculates when the next scheduled reminder should fire            |
| `logger.py`              | Coloured console + rotating file logging                           |

## Features

- Multiple LLM provider support (OpenAI, Google Gemini, Groq)
- AI-generated reminder messages with a customisable prompt template
- Scheduled reminders with randomised or fixed timing
- Message cache (maintains 10 pre-generated messages so sending never blocks)
- Response-window mode: after a reminder is sent the bot watches for the
  target's reply and fires a witty comeback
- Contextual mode: if the target writes during the reminder window the bot
  reacts to whatever they said and sneaks the book in
- Reply reactions: if the target uses Discord's native "reply" feature on
  one of the bot's own messages, the bot asks the LLM whether an emoji
  reaction fits the moment and, if so, adds one from a small hardcoded pool
- Discord webhook integration with optional debug notifications
- Coloured console logging with file rotation
- Systemd service support

## Supported LLM Providers

| Provider          | Example model             | Pricing             |
|-------------------|---------------------------|---------------------|
| **OpenAI**        | `gpt-4o`, `gpt-4`         | Paid                |
| **Google Gemini** | `gemini-1.5-flash`        | Free tier available |
| **Groq**          | `llama-3.1-70b-versatile` | Free tier available |

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
nano config/config.yaml
```

Minimum required fields: `discord.main_webhook_url`, `llm.api_key`, and the
`reminder` block (`target_name`, `sender_name`, `book_title`, `time_range`).

### 3. Run manually to test

```bash
python3 main.py
```

Press `Ctrl+C` to stop.

---

## Scheduled Reminder (`main.py`)

Runs a loop that:

1. Pre-fills the cache with AI-generated messages on startup.
2. Sends one message per day at the configured time (fixed or randomised).
3. After sending, opens a **response window** so `contextual_reminder.py`
   can reply to whatever the target writes back.
4. Refills the cache in the background after each send.

### Install as a systemd service

```bash
cp systemd/ai-reminder.example.service /etc/systemd/system/ai-reminder.service
# Edit paths (User=, WorkingDirectory=, ExecStart=) to match your setup
sudo nano /etc/systemd/system/ai-reminder.service

sudo systemctl daemon-reload
sudo systemctl enable --now ai-reminder
sudo journalctl -u ai-reminder -f
```

---

## Discord Listener (`bot_listener.py`)

A lightweight Discord bot that watches for messages from the configured target
user and passes them to `ContextualReminder`. Run this alongside `main.py`.

**Requires the Message Content privileged intent** — enable it in the Discord
Developer Portal under *Bot → Privileged Gateway Intents → Message Content Intent*.

### Configure

```bash
cp config_listener.example.yaml config_listener.yaml
nano config_listener.yaml
# Set: discord_token, target_discord_id, ai_reminder_config
```

### Run manually to test

```bash
python3 bot_listener.py
```

### Install as a systemd service

```bash
cp systemd/bot-listener.example.service /etc/systemd/system/bot-listener.service
sudo nano /etc/systemd/system/bot-listener.service
sudo systemctl daemon-reload
sudo systemctl enable --now bot-listener
sudo journalctl -u bot-listener -f
```

---

## Configuration Reference

### LLM provider

**OpenAI:**

```yaml
llm:
  provider: "openai"
  api_key: "sk-..."
  model: "gpt-4o"
```

**Google Gemini (free tier):**

```yaml
llm:
  provider: "gemini"
  api_key: "YOUR_GEMINI_API_KEY"
  model: "gemini-1.5-flash"
```

Get a key: <https://ai.google.dev/>

**Groq (free tier):**

```yaml
llm:
  provider: "groq"
  api_key: "YOUR_GROQ_API_KEY"
  model: "llama-3.1-70b-versatile"
```

Get a key: <https://console.groq.com/>

### Reminder timing

**Random time in a window (default):**

```yaml
reminder:
  randomize_time: true
  time_range:
    start: "14:00"
    end: "17:00"
```

**Fixed time:**

```yaml
reminder:
  randomize_time: false
  time_range:
    start: "15:30"
    end: "15:30"   # ignored when randomize_time is false
```

### Response window

After a scheduled reminder is sent, the bot stays in "response mode" for a
configurable number of hours, replying to whatever the target writes.

```yaml
reminder:
  response_window_hours: 3   # 0 to disable
```

### Debug webhook levels

```yaml
discord:
  debug_level: "error"   # debug | info | warning | error
```

---

## Prompt Template

The prompt is fully configurable in `config/config.yaml` under the `prompt` key.

Available placeholders:

| Placeholder         | Value                                          |
|---------------------|------------------------------------------------|
| `{sender_name}`     | Person who set this up                         |
| `{target_name}`     | Person being reminded                          |
| `{book_title}`      | Title of the book                              |
| `{language}`        | Language for LLM output                        |
| `{target_gender}`   | `female` or `male` (for grammatical agreement) |
| `{recent_messages}` | Last N sent messages (for variety)             |

---

## Cache Utilities

```bash
python3 tools/cache_utils.py inspect   # show cache contents and stats
python3 tools/cache_utils.py repair    # remove invalid / duplicate entries
python3 tools/cache_utils.py clear     # wipe the cache
```