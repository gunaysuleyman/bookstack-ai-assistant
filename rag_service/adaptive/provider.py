import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger("RAGProvider")

RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class LLMUsage:
    provider: str
    model: str
    purpose: str
    prompt_tokens_est: int
    completion_tokens_est: int
    prompt_tokens_actual: Optional[int]
    completion_tokens_actual: Optional[int]
    latency_ms: float
    attempts: int


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    def __init__(self, max_calls: int, deadline_s: float, now: Optional[Callable[[], float]] = None):
        self.max_calls = max_calls
        self.deadline_s = deadline_s
        self._now = now or time.monotonic
        self.started = self._now()
        self.calls = 0

    def check_time(self) -> None:
        if self._now() - self.started > self.deadline_s:
            raise BudgetExhausted("deadline")

    def charge_model_call(self) -> None:
        self.check_time()
        if self.calls >= self.max_calls:
            raise BudgetExhausted("model_calls")
        self.calls += 1


_REQUEST_BUDGET: contextvars.ContextVar[Optional[CallBudget]] = contextvars.ContextVar("request_llm_budget", default=None)


@contextmanager
def bind_budget(budget: Optional[CallBudget]):
    token = _REQUEST_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _REQUEST_BUDGET.reset(token)


def redact_secrets(text: str, secrets: List[str]) -> str:
    cleaned = text or ""
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def complete_llm(
    *,
    provider: str,
    model: str,
    fallbacks: List[str],
    api_key: str,
    system_instruction: str,
    user_prompt: str,
    purpose: str,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    sleeper: Callable[[float], None] = time.sleep,
    transport: Optional[httpx.BaseTransport] = None,
    openai_model: str = "",
    openai_key: str = "",
    budget: Optional[CallBudget] = None,
) -> LLMResult:
    started = time.perf_counter()
    prompt_est = _estimate(system_instruction) + _estimate(user_prompt)
    attempts = 0
    last_error: Optional[Exception] = None
    active_budget = budget if budget is not None else _REQUEST_BUDGET.get()
    secrets = [api_key, openai_key]

    if provider == "openai":
        models = [openai_model or model]
        key = openai_key or api_key
    else:
        models = [model] + [item for item in fallbacks if item and item != model]
        key = api_key

    for model_name in models:
        for retry in range(max_retries + 1):
            if active_budget is not None:
                active_budget.charge_model_call()
            attempts += 1
            try:
                text, actual_prompt, actual_completion = _call_once(
                    provider=provider,
                    model_name=model_name,
                    api_key=key,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    timeout_s=timeout_s,
                    transport=transport,
                )
                usage = LLMUsage(
                    provider=provider,
                    model=model_name,
                    purpose=purpose,
                    prompt_tokens_est=prompt_est,
                    completion_tokens_est=_estimate(text),
                    prompt_tokens_actual=actual_prompt,
                    completion_tokens_actual=actual_completion,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attempts=attempts,
                )
                return LLMResult(text=text, usage=usage)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                logger.warning("LLM %s failed status=%s attempt=%s", model_name, status, retry)
                if status not in RETRY_STATUSES or retry >= max_retries:
                    break
                sleeper(min(2.0, 0.2 * (2**retry)))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "LLM %s transport error attempt=%s: %s",
                    model_name,
                    retry,
                    redact_secrets(str(exc), secrets),
                )
                if retry >= max_retries:
                    break
                sleeper(min(2.0, 0.2 * (2**retry)))
    raise RuntimeError(f"LLM call failed for {purpose}") from None


def _call_once(
    *,
    provider: str,
    model_name: str,
    api_key: str,
    system_instruction: str,
    user_prompt: str,
    timeout_s: float,
    transport: Optional[httpx.BaseTransport],
) -> tuple:
    with httpx.Client(timeout=timeout_s, transport=transport) as client:
        if provider == "openai":
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            return text, usage.get("prompt_tokens"), usage.get("completion_tokens")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        response = client.post(
            url,
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": f"{system_instruction}\n\n{user_prompt}"}]}]},
        )
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata") or {}
        return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no json object")
    return json.loads(cleaned[start : end + 1])
