"""DeepSeek requests for synthesis routes and structural descriptions."""
import os

import requests


def create_session(config):
    session = requests.Session()
    proxy = config.get("proxy_url", "")
    session.trust_env = config["http_trust_env"] and not proxy
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


def validate_api(config):
    environment_key = "DEEPSEEK_API_KEY"
    if not config.get("api_key"):
        raise ValueError(f"Set api.api_key in config.json or the {environment_key} environment variable")
    if not config.get("model", "").strip():
        raise ValueError("Set api.model in config.json to an API model available to your account")
    if not config.get("api_base_url", "").startswith("https://"):
        raise ValueError("api.api_base_url must be an HTTPS API base URL")
    if config["api_key"].startswith(("sk-proj-", "sk-svcacct-")):
        raise ValueError("DeepSeek requires its own API key; an OpenAI key cannot be sent to DeepSeek")


def api_settings(config):
    settings = dict(config)
    environment_key = "DEEPSEEK_API_KEY"
    settings["api_key"] = os.environ.get(environment_key, "").strip() or settings["api_key"].strip()
    return settings


def redact_config(value):
    if isinstance(value, dict):
        return {k: "<redacted>" if k == "api_key" else redact_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config(v) for v in value]
    return value


def complete_chat(session, api_key, model, base_url, messages, max_output_tokens,
                  timeout, reasoning_effort=""):
    validate_api({"api_key": api_key, "model": model, "api_base_url": base_url})
    request = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    request.update(max_tokens=max_output_tokens, thinking={"type": "disabled" if reasoning_effort == "none" else "enabled"})
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    response = session.post(base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + api_key}, json=request, timeout=timeout)
    response.raise_for_status()
    return response.json()
