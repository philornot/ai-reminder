"""LLM client for generating reminder messages."""

import json
import logging
import re
from typing import Optional

import requests


class LLMClient:
    """Client for interacting with LLM API to generate messages."""

    PROVIDER_CONFIGS = {
        'openai': {
            'default_base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-4'
        },
        'gemini': {
            'default_base_url': 'https://generativelanguage.googleapis.com',
            'default_model': 'gemini-1.5-flash'
        },
        'groq': {
            'default_base_url': 'https://api.groq.com/openai/v1',
            'default_model': 'llama-3.1-70b-versatile'
        }
    }

    def __init__(
            self,
            provider: str,
            api_key: str,
            model: Optional[str] = None,
            base_url: Optional[str] = None,
            max_tokens: int = 500,
            temperature: float = 0.9,
            logger: Optional[logging.Logger] = None
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider name (openai, gemini, groq)
            api_key: API key for LLM service
            model: Model name to use (uses provider default if not specified)
            base_url: Base URL for API (uses provider default if not specified)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            logger: Logger instance
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)

        if self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {list(self.PROVIDER_CONFIGS.keys())}"
            )

        provider_config = self.PROVIDER_CONFIGS[self.provider]
        self.model = model or provider_config['default_model']
        self.base_url = base_url or provider_config['default_base_url']

        self.logger.info(
            "Initialized LLM client: provider=%s, model=%s", self.provider, self.model
        )

        # Gemini uses a custom REST client; OpenAI-compatible providers (openai,
        # groq) import the openai SDK lazily inside _generate_openai_compatible
        # so that Groq users don't need the openai package installed.
        if self.provider == 'gemini':
            self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Initialise Gemini client placeholder (uses raw requests, no SDK needed)."""
        self.client = None

    def _clean_message(self, raw_message: str) -> Optional[str]:
        """Clean and validate LLM output.

        Some models tend to generate multiple examples or add extra formatting.
        This method extracts only the actual reminder message.

        Args:
            raw_message: Raw output from LLM

        Returns:
            Cleaned message or None if invalid
        """
        if not raw_message:
            return None

        message = raw_message.strip()

        if len(message) > 200:
            self.logger.warning(
                "LLM returned long message (%d chars), attempting cleanup", len(message)
            )
            self.logger.debug("Original message: %s", message)

        example_patterns = [
            r'Lub:',
            r'Albo:',
            r'Lub tak:',
            r'Przykład:',
            r'Przykładowe',
            r'Może:',
            r'I jeszcze:',
            r'Następnie:',
            r'Lub też:',
            r'Ewentualnie:',
            r'Wersja \d+:',
            r'Opcja \d+:',
        ]

        has_examples = any(
            re.search(pattern, message, re.IGNORECASE) for pattern in example_patterns
        )

        if has_examples:
            self.logger.warning("Message contains example indicators, extracting first variant")
            separators = [
                '\n\nLub:', '\n\nAlbo:', '\n\nMoże:', '\n\nPrzykład:',
                '\nLub:', '\nAlbo:', '\nMoże:', '\nNastępnie:', '\nI jeszcze:',
                '\n\nLub tak:', '\nLub też:', '\nEwentualnie:',
            ]
            for separator in separators:
                if separator in message:
                    message = message.split(separator)[0].strip()
                    self.logger.info(
                        "Extracted first variant, reduced from %d to %d chars",
                        len(raw_message), len(message),
                    )
                    break

        message = re.sub(r'\*\*(.+?)\*\*', r'\1', message)
        message = re.sub(r'\*(.+?)\*', r'\1', message)
        message = re.sub(r'`(.+?)`', r'\1', message)
        message = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', message)
        message = message.strip()

        if len(message) > 500:
            self.logger.warning(
                "Message still too long after cleanup (%d chars), taking first sentence",
                len(message),
            )
            sentences = re.split(r'[.!?]+\s+', message)
            if sentences:
                if len(sentences[0]) < 50 and len(sentences) > 1:
                    message = sentences[0] + '. ' + sentences[1] + '.'
                else:
                    message = sentences[0] + (
                        '' if sentences[0].endswith(('.', '!', '?')) else '.'
                    )
                message = message.strip()

        if not message or len(message) < 10:
            self.logger.error("Message too short after cleanup: %r", message)
            return None

        if len(message) < len(raw_message) * 0.5:
            self.logger.info(
                "Significantly reduced message length: %d → %d chars",
                len(raw_message), len(message),
            )
            self.logger.debug("Cleaned message: %s", message)

        return message

    def generate_message(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate a reminder message using LLM.

        Runs the raw completion through ``_clean_message()`` to strip
        Markdown formatting, trim overly long output, and drop any
        "Lub:"/"Albo:"-style extra variants some models tack onto the reply.

        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retry attempts for API errors

        Returns:
            Generated message or None if failed
        """
        raw_message = self._generate_raw(prompt, max_retries=max_retries)
        if not raw_message:
            return None

        cleaned_message = self._clean_message(raw_message)
        if not cleaned_message:
            self.logger.error("Message cleanup failed, rejecting output")
            return None

        self.logger.info(
            "Successfully generated message (%d chars)", len(cleaned_message)
        )
        return cleaned_message

    def generate_json(self, prompt: str, max_retries: int = 2) -> Optional[dict]:
        """Generate a small structured JSON decision from the LLM.

        Intended for short yes/no-plus-choice decisions (e.g. "should the bot
        react to this message, and with which emoji") where the reminder-text
        cleanup pipeline in ``_clean_message()`` would do more harm than good
        (it is tuned to strip Markdown and split off "Lub:"/"Albo:" variants
        from free-form reminder sentences, not to preserve a JSON payload).

        The raw completion is scanned for the first ``{...}`` block, which
        tolerates models that wrap the JSON in a Markdown code fence or add a
        short comment before/after it.

        Args:
            prompt: Prompt instructing the LLM to answer with a single JSON
                object and nothing else.
            max_retries: Maximum number of retry attempts for API errors.

        Returns:
            The parsed JSON object, or None if the call failed or the
            response did not contain valid JSON.
        """
        raw_message = self._generate_raw(prompt, max_retries=max_retries)
        if not raw_message:
            return None
        return self._parse_json_object(raw_message)

    def _generate_raw(self, prompt: str, max_retries: int) -> Optional[str]:
        """Call the configured provider and return its raw text completion.

        Shared by ``generate_message()`` and ``generate_json()``. Retries on
        transient capacity/rate-limit errors with exponential backoff; any
        other error is re-raised immediately.

        Args:
            prompt: Prompt to send to the LLM.
            max_retries: Maximum number of retry attempts for API errors.

        Returns:
            Raw, unprocessed text returned by the provider, or None on an
            empty response.

        Raises:
            Exception: Re-raises the last provider error once retries (for
                capacity/rate-limit errors) are exhausted, or immediately for
                any non-transient error.
        """
        import time

        for attempt in range(max_retries):
            try:
                self.logger.debug(
                    "Sending prompt to LLM (provider=%s, model=%s, attempt=%d/%d)",
                    self.provider, self.model, attempt + 1, max_retries,
                )

                if self.provider == 'gemini':
                    return self._generate_gemini(prompt)
                return self._generate_openai_compatible(prompt)

            except Exception as exc:
                error_msg = str(exc).lower()
                is_capacity_error = any(indicator in error_msg for indicator in [
                    'over capacity', '503', 'rate limit', '429',
                    'too many requests', 'service unavailable', 'internal_server_error',
                ])

                if is_capacity_error and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    self.logger.warning(
                        "API capacity error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_retries, exc, wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                self.logger.error(
                    "Error generating message (attempt %d/%d): %s",
                    attempt + 1, max_retries, exc,
                )
                if attempt == max_retries - 1:
                    self.logger.error("All retry attempts exhausted")
                raise

        return None

    def _parse_json_object(self, raw_text: str) -> Optional[dict]:
        """Extract and parse the first JSON object found in raw text.

        Tolerates Markdown code fences and any leading/trailing commentary a
        model might add around the actual JSON payload.

        Args:
            raw_text: Raw text returned by the LLM.

        Returns:
            The parsed object, or None if no valid JSON object was found.
        """
        match = re.search(r"\{.*}", raw_text, re.DOTALL)
        if not match:
            self.logger.warning(
                "No JSON object found in LLM response: %s", raw_text[:200]
            )
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            self.logger.warning("Could not parse JSON from LLM response: %s", exc)
            return None

    def _generate_openai_compatible(self, prompt: str) -> Optional[str]:
        """Generate message using OpenAI-compatible API (OpenAI or Groq).

        The ``openai`` SDK is imported here rather than at module level so
        that users who only use Groq or Gemini are not forced to install it.

        Args:
            prompt: Prompt to send

        Returns:
            Generated message string, or None on empty response.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the openai/groq providers.\n"
                "Install it with:  pip install openai"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            message = response.choices[0].message.content.strip()
            self.logger.debug(
                "Raw LLM response: %s",
                (message[:200] + "...") if len(message) > 200 else message,
            )
            return message

        except openai.APIError as exc:
            self.logger.error("API error: %s", exc)
            raise
        except openai.RateLimitError as exc:
            self.logger.error("Rate limit exceeded: %s", exc)
            raise
        except openai.APIConnectionError as exc:
            self.logger.error("API connection error: %s", exc)
            raise

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """Generate message using Google Gemini REST API (no SDK required).

        Args:
            prompt: Prompt to send

        Returns:
            Generated message string, or None on empty response.
        """
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    message = parts[0]["text"].strip()
                    self.logger.debug(
                        "Raw LLM response: %s",
                        (message[:200] + "...") if len(message) > 200 else message,
                    )
                    return message

            self.logger.error("Unexpected Gemini API response structure: %s", result)
            raise ValueError("Invalid response from Gemini API")

        except requests.exceptions.RequestException as exc:
            self.logger.error("Gemini API request error: %s", exc)
            raise
        except (KeyError, IndexError) as exc:
            self.logger.error("Error parsing Gemini response: %s", exc)
            raise