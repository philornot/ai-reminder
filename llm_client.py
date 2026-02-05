"""LLM client for generating reminder messages."""

import openai
from typing import Optional
import logging
import requests
import json


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
            # OpenAI and Groq use OpenAI-compatible API
            self._init_openai_compatible_client()

    def _init_openai_compatible_client(self):
        """Initialize OpenAI-compatible client (OpenAI, Groq)."""
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _init_gemini_client(self):
        """Initialize Gemini client (uses REST API directly)."""
        # Gemini uses a different API structure, we'll handle it separately
        self.client = None

    def generate_message(self, prompt: str) -> Optional[str]:
        """Generate a reminder message using LLM.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            Generated message or None if failed
        """
        try:
            self.logger.debug(f"Sending prompt to LLM (provider: {self.provider}, model: {self.model})")

            if self.provider == 'gemini':
                return self._generate_gemini(prompt)
            else:
                return self._generate_openai_compatible(prompt)

        except Exception as e:
            self.logger.error(f"Error generating message: {e}")
            raise

    def _generate_openai_compatible(self, prompt: str) -> Optional[str]:
        """Generate message using OpenAI-compatible API (OpenAI, Groq).

        Args:
            prompt: Prompt to send

        Returns:
            Generated message
        """
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
            self.logger.info("Successfully generated message from LLM")
            self.logger.debug(f"Generated message: {message}")

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
                    self.logger.info("Successfully generated message from Gemini")
                    self.logger.debug(f"Generated message: {message}")
                    return message

            self.logger.error("Unexpected Gemini API response structure")
            raise ValueError("Invalid response from Gemini API")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Gemini API request error: {e}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            raise