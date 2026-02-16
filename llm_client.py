"""LLM client for generating reminder messages."""

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

        # Validate provider
        if self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.PROVIDER_CONFIGS.keys())}")

        provider_config = self.PROVIDER_CONFIGS[self.provider]

        # Set model and base_url with fallbacks to defaults
        self.model = model or provider_config['default_model']
        self.base_url = base_url or provider_config['default_base_url']

        self.logger.info(f"Initialized LLM client: provider={self.provider}, model={self.model}")

        # Initialize client based on provider
        if self.provider == 'gemini':
            self._init_gemini_client()
        else:
            # OpenAI and Groq use OpenAI-compatible API - lazy import
            self._init_openai_compatible_client()

    def _init_openai_compatible_client(self):
        """Initialize OpenAI-compatible client (OpenAI, Groq)."""
        import openai
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _init_gemini_client(self):
        """Initialize Gemini client (uses REST API directly)."""
        # Gemini uses a different API structure, we'll handle it separately
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

        # Remove leading/trailing whitespace
        message = raw_message.strip()

        # Log original for debugging
        if len(message) > 200:
            self.logger.warning(f"LLM returned long message ({len(message)} chars), attempting cleanup")
            self.logger.debug(f"Original message: {message}")

        # Patterns that indicate the LLM is generating examples/alternatives
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

        # Check if message contains example indicators
        has_examples = any(re.search(pattern, message, re.IGNORECASE) for pattern in example_patterns)

        if has_examples:
            self.logger.warning("Message contains example indicators, extracting first variant")

            # Split by common separators and take first part
            separators = ['\n\nLub:', '\n\nAlbo:', '\n\nMoże:', '\n\nPrzykład:',
                          '\nLub:', '\nAlbo:', '\nMoże:', '\nNastępnie:', '\nI jeszcze:',
                          '\n\nLub tak:', '\nLub też:', '\nEwentualnie:']

            for separator in separators:
                if separator in message:
                    parts = message.split(separator)
                    message = parts[0].strip()
                    self.logger.info(
                        f"Extracted first variant, reduced from {len(raw_message)} to {len(message)} chars")
                    break

        # Remove markdown formatting if present
        message = re.sub(r'\*\*(.+?)\*\*', r'\1', message)  # Bold
        message = re.sub(r'\*(.+?)\*', r'\1', message)  # Italic
        message = re.sub(r'`(.+?)`', r'\1', message)  # Code

        # Remove leading numbers/bullets (1., -, *, etc.)
        message = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', message)

        # Final cleanup
        message = message.strip()

        # Validate length - reminder should be reasonably short
        if len(message) > 500:
            self.logger.warning(f"Message still too long after cleanup ({len(message)} chars), taking first sentence")
            # Take first 1-2 sentences
            sentences = re.split(r'[.!?]+\s+', message)
            if sentences:
                # Take first sentence, or first two if first is very short
                if len(sentences[0]) < 50 and len(sentences) > 1:
                    message = sentences[0] + '. ' + sentences[1] + '.'
                else:
                    message = sentences[0] + ('.' if not sentences[0].endswith(('.', '!', '?')) else '')
                message = message.strip()

        # Validate not empty
        if not message or len(message) < 10:
            self.logger.error(f"Message too short after cleanup: '{message}'")
            return None

        # Log if we made significant changes
        if len(message) < len(raw_message) * 0.5:
            self.logger.info(f"Significantly reduced message length: {len(raw_message)} → {len(message)} chars")
            self.logger.debug(f"Cleaned message: {message}")

        return message

    def generate_message(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate a reminder message using LLM.

        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retry attempts for API errors

        Returns:
            Generated message or None if failed
        """
        import time

        for attempt in range(max_retries):
            try:
                self.logger.debug(
                    f"Sending prompt to LLM (provider: {self.provider}, model: {self.model}, attempt: {attempt + 1}/{max_retries})")

                if self.provider == 'gemini':
                    raw_message = self._generate_gemini(prompt)
                else:
                    raw_message = self._generate_openai_compatible(prompt)

                if not raw_message:
                    return None

                # Clean and validate the message
                cleaned_message = self._clean_message(raw_message)

                if not cleaned_message:
                    self.logger.error("Message cleanup failed, rejecting output")
                    return None

                self.logger.info(f"Successfully generated message ({len(cleaned_message)} chars)")
                return cleaned_message

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a capacity/rate limit error (503, 429)
                is_capacity_error = any(indicator in error_msg for indicator in [
                    'over capacity',
                    '503',
                    'rate limit',
                    '429',
                    'too many requests',
                    'service unavailable',
                    'internal_server_error'
                ])

                if is_capacity_error and attempt < max_retries - 1:
                    # Exponential backoff: 2, 4, 8 seconds
                    wait_time = 2 ** (attempt + 1)
                    self.logger.warning(f"API capacity error (attempt {attempt + 1}/{max_retries}): {e}")
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed or non-retryable error
                    self.logger.error(f"Error generating message (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        self.logger.error("All retry attempts exhausted")
                    raise

        return None

    def _generate_openai_compatible(self, prompt: str) -> Optional[str]:
        """Generate message using OpenAI-compatible API (OpenAI, Groq).

        Args:
            prompt: Prompt to send

        Returns:
            Generated message
        """
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            message = response.choices[0].message.content.strip()
            self.logger.debug(
                f"Raw LLM response: {message[:200]}..." if len(message) > 200 else f"Raw LLM response: {message}")

            return message

        except openai.APIError as e:
            self.logger.error(f"API error: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit exceeded: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.error(f"API connection error: {e}")
            raise

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """Generate message using Google Gemini API.

        Args:
            prompt: Prompt to send

        Returns:
            Generated message
        """
        try:
            # Gemini API endpoint
            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

            headers = {
                "Content-Type": "application/json"
            }

            # Add API key as query parameter for Gemini
            params = {
                "key": self.api_key
            }

            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                }
            }

            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            # Extract text from Gemini response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    message = candidate['content']['parts'][0]['text'].strip()
                    self.logger.debug(f"Raw LLM response: {message[:200]}..." if len(
                        message) > 200 else f"Raw LLM response: {message}")
                    return message

            self.logger.error("Unexpected Gemini API response structure")
            raise ValueError("Invalid response from Gemini API")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Gemini API request error: {e}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            raise
