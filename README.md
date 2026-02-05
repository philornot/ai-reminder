# AI Reminder

Automated reminder system that uses AI to generate creative messages and sends them via Discord webhooks.

## Features

- Multiple LLM provider support (OpenAI, Google Gemini, Groq)
- AI-generated reminder messages with customizable prompts
- Scheduled reminders with randomized or fixed timing
- Message caching system (maintains 10 pre-generated messages)
- Discord webhook integration with debug notifications
- Colored console logging with file rotation
- Systemd service support for Raspberry Pi
- Comprehensive error handling
- Fully configurable via YAML

## Supported LLM Providers

- **OpenAI** (gpt-4, gpt-3.5-turbo, etc.) - Paid
- **Google Gemini** (gemini-pro, gemini-1.5-flash) - Free tier available
- **Groq** (llama-3.1-70b-versatile, mixtral-8x7b-32768) - Free tier available

## Configuration

Edit `config/config.yaml` and configure:

1. **LLM Provider**: Choose between openai, gemini, or groq
2. **API Keys**: Add your API key for chosen provider
3. **Discord Webhooks**: Main webhook and optional debug webhook
4. **Reminder Settings**:
   - Names and book title
   - Time randomization (true/false)
   - Time range or fixed time
5. **Logging**: Console colors, file output, log levels
6. **Debug Level**: Choose notification threshold
7. **Prompt Template**: Customize the AI's behavior

### LLM Provider Setup

**OpenAI:**
```yaml
llm:
  provider: "openai"
  api_key: "sk-..."
  model: "gpt-4"
```

**Google Gemini (Free):**
```yaml
llm:
  provider: "gemini"
  api_key: "YOUR_GEMINI_API_KEY"
  model: "gemini-1.5-flash"
```
Get API key: https://ai.google.dev/

**Groq (Free):**
```yaml
llm:
  provider: "groq"
  api_key: "YOUR_GROQ_API_KEY"
  model: "llama-3.1-70b-versatile"
```
Get API key: https://console.groq.com/

### Time Configuration

**Random time (default):**
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
    end: "15:30"  # Ignored when randomize_time is false
```

## Testing

Run manually to test:
```bash
python3 main.py
```

Press Ctrl+C to stop.

## Install as System Service

1. Copy service file:
```bash
sudo cp systemd/ai-reminder.service.example /etc/systemd/system/
```

2. Reload systemd:
```bash
sudo systemctl daemon-reload
```

3. Enable and start service:
```bash
sudo systemctl enable ai-reminder
sudo systemctl start ai-reminder
```

4. Check status:
```bash
sudo systemctl status ai-reminder
```

5. View logs:
```bash
sudo journalctl -u ai-reminder -f
```

## Debug Webhook Levels

Configure `discord.debug_level` in config.yaml:
- `debug`: All messages
- `info`: Info and above
- `warning`: Warnings and errors
- `error`: Errors only (default)
