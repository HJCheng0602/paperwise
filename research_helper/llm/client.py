from __future__ import annotations
import os
from research_helper import config

_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "deepseek":  "deepseek-chat",
    "qwen":      "qwen-plus",
}


def _get_base_url(provider: str) -> str | None:
    """Return base URL for provider. Checks EMBEDDING_BASE_URL (generic),
    then {PROVIDER}_BASE_URL env var, falls back to hardcoded default.
    Returns None for openai (uses SDK default)."""
    generic = os.getenv("EMBEDDING_BASE_URL")
    if generic and provider in ("qwen", "openai"):
        return generic
    env_key = f"{provider.upper()}_BASE_URL"
    return os.getenv(env_key) or _DEFAULT_BASE_URLS.get(provider)


def complete(system: str, user: str, max_tokens: int = 4096) -> str:
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or _DEFAULT_MODELS.get(provider, "")

    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens)
    elif provider in ("openai", "deepseek", "qwen"):
        return _openai_compat(system, user, model, max_tokens, provider)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            "Choose from: anthropic, openai, deepseek, qwen"
        )


def _anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic
    from research_helper.utils import cost_tracker

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    cost_tracker.record_llm(model, msg.usage.input_tokens, msg.usage.output_tokens)
    return msg.content[0].text


def _openai_compat(system: str, user: str, model: str, max_tokens: int, provider: str) -> str:
    from openai import OpenAI
    from research_helper.utils import cost_tracker

    api_key = _api_key(provider)
    base_url = _get_base_url(provider)

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    if resp.usage:
        cost_tracker.record_llm(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content or ""


def _api_key(provider: str) -> str:
    keys = {
        "openai":   config.OPENAI_API_KEY,
        "deepseek": config.DEEPSEEK_API_KEY,
        "qwen":     config.QWEN_API_KEY,
    }
    key = keys.get(provider, "")
    if not key:
        raise EnvironmentError(
            f"{provider.upper()}_API_KEY is not set. Add it to .env or environment."
        )
    return key
