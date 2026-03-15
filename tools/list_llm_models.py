#!/usr/bin/env python3
"""Utility script to list available models from configured LLM providers.

Reads API keys directly from ../config/config.yaml and queries each provider's
models endpoint, displaying results in a human-readable format.

Usage:
    python tools/list_models.py
    python tools/list_models.py --provider groq
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import requests
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Make sure you run this script from the project root directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Provider fetchers
# ---------------------------------------------------------------------------

def fetch_groq_models(api_key: str, base_url: str) -> list[str]:
    """Fetch available model IDs from the Groq API.

    Args:
        api_key: Groq API key.
        base_url: Groq API base URL (e.g. https://api.groq.com/openai/v1).

    Returns:
        Sorted list of model ID strings.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}/models"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return sorted(m["id"] for m in data.get("data", []))


def fetch_openai_models(api_key: str, base_url: str) -> list[str]:
    """Fetch available model IDs from the OpenAI API.

    Args:
        api_key: OpenAI API key.
        base_url: OpenAI API base URL (e.g. https://api.openai.com/v1).

    Returns:
        Sorted list of model ID strings.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}/models"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return sorted(m["id"] for m in data.get("data", []))


def fetch_gemini_models(api_key: str, base_url: str) -> list[str]:
    """Fetch available model IDs from the Google Gemini API.

    Gemini uses a query-parameter API key instead of a Bearer token and
    returns model names in the form ``models/<id>``, so we strip the prefix.

    Args:
        api_key: Gemini API key.
        base_url: Gemini API base URL
            (e.g. https://generativelanguage.googleapis.com).

    Returns:
        Sorted list of model ID strings (without the ``models/`` prefix).

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}/v1beta/models"
    response = requests.get(url, params={"key": api_key}, timeout=10)
    response.raise_for_status()
    data = response.json()
    # Strip the "models/" prefix so IDs are consistent with other providers.
    return sorted(
        m["name"].removeprefix("models/") for m in data.get("models", [])
    )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_FETCHERS = {
    "groq": fetch_groq_models,
    "openai": fetch_openai_models,
    "gemini": fetch_gemini_models,
}

_DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}


def get_models_for_provider(
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
) -> list[str]:
    """Fetch models for a given provider using the appropriate fetcher.

    Args:
        provider: One of ``groq``, ``openai``, or ``gemini``.
        api_key: API key for the provider.
        base_url: Optional override for the API base URL.

    Returns:
        List of model ID strings.

    Raises:
        ValueError: If the provider is not supported.
        requests.HTTPError: On API errors.
    """
    if provider not in _FETCHERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported: {list(_FETCHERS.keys())}"
        )
    resolved_url = base_url or _DEFAULT_BASE_URLS[provider]
    return _FETCHERS[provider](api_key, resolved_url)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def extract_provider_config(
        config: dict, provider: str
) -> tuple[str, Optional[str]]:
    """Extract API key and optional base URL for a provider from config.

    The config's ``llm`` section is used only when the configured provider
    matches. For other providers we look for a top-level key under ``llm``
    with the provider name (e.g. ``llm.groq.api_key``), falling back to the
    main ``llm.api_key`` when the configured provider matches.

    Args:
        config: Parsed configuration dictionary.
        provider: Provider name to extract config for.

    Returns:
        Tuple of (api_key, base_url). base_url may be None.

    Raises:
        KeyError: If the API key cannot be found for the provider.
    """
    llm_cfg = config.get("llm", {})
    configured_provider = llm_cfg.get("provider", "").lower()

    # Check for a provider-specific sub-section first (e.g. llm.groq.api_key)
    provider_section = llm_cfg.get(provider, {})
    if isinstance(provider_section, dict):
        api_key = provider_section.get("api_key")
        base_url = provider_section.get("base_url")
    else:
        api_key = None
        base_url = None

    # Fall back to the top-level llm.api_key when this is the active provider
    if not api_key and configured_provider == provider:
        api_key = llm_cfg.get("api_key")
        if not base_url:
            # Also check the provider sub-section for base_url only
            sub = llm_cfg.get(provider, {})
            if isinstance(sub, dict):
                base_url = sub.get("base_url")

    if not api_key:
        raise KeyError(
            f"No API key found for provider '{provider}' in config. "
            f"Add it under llm.api_key (if '{provider}' is your active provider) "
            f"or under llm.{provider}.api_key."
        )

    return api_key, base_url


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_models(provider: str, models: list[str]) -> None:
    """Print a formatted list of models for a provider.

    Args:
        provider: Provider name (used as section header).
        models: List of model ID strings to display.
    """
    header = f"  {provider.upper()} ({len(models)} models)"
    print("=" * 60)
    print(header)
    print("=" * 60)
    for model in models:
        print(f"  - {model}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="List available models from configured LLM providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/list_models.py                 # all providers in config\n"
            "  python tools/list_models.py --provider groq # only Groq\n"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=list(_FETCHERS.keys()),
        default=None,
        help=(
            "Fetch models only from this provider. "
            "Defaults to the provider set in config.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        default="../config/config.yaml",
        help="Path to config YAML file (default: ../config/config.yaml).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_providers",
        help="Query all supported providers (requires API keys for each).",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine which providers to query
    if args.all_providers:
        providers_to_query = list(_FETCHERS.keys())
    elif args.provider:
        providers_to_query = [args.provider]
    else:
        # Default: only the provider currently set in config
        active = config.get("llm", {}).get("provider", "").lower()
        if not active:
            print("No provider set in config. Use --provider or --all.", file=sys.stderr)
            sys.exit(1)
        providers_to_query = [active]

    print(f"\nQuerying {len(providers_to_query)} provider(s)...\n")

    exit_code = 0
    for provider in providers_to_query:
        try:
            api_key, base_url = extract_provider_config(config, provider)
            models = get_models_for_provider(provider, api_key, base_url)
            print_models(provider, models)
        except KeyError as exc:
            print(f"[{provider.upper()}] Config error: {exc}\n", file=sys.stderr)
            exit_code = 1
        except requests.HTTPError as exc:
            print(
                f"[{provider.upper()}] API error {exc.response.status_code}: "
                f"{exc.response.text[:200]}\n",
                file=sys.stderr,
            )
            exit_code = 1
        except requests.RequestException as exc:
            print(f"[{provider.upper()}] Network error: {exc}\n", file=sys.stderr)
            exit_code = 1
        except ValueError as exc:
            print(f"[{provider.upper()}] {exc}\n", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
