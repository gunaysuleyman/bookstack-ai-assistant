import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger("RAGProvider")

# USD per 1,000,000 tokens. Stored on each turn so a later model change does not rewrite history.
MODEL_USD_PER_MTOK = {
    "gpt-6-luna": (0.1, 0.5),
}

_TURN_USAGES: contextvars.ContextVar[Optional[List["LLMUsage"]]] = contextvars.ContextVar("turn_usages", default=None)

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
class FunctionCall:
    name: str
    args: Dict[str, Any]
    call_id: str
    thought_signature: str = ""


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage
    function_calls: List[FunctionCall]
    model_content: Optional[Dict[str, Any]] = None


class BudgetExhausted(RuntimeError):
    pass


@contextmanager
def collect_turn_usages():
    bucket: List[LLMUsage] = []
    token = _TURN_USAGES.set(bucket)
    try:
        yield bucket
    finally:
        _TURN_USAGES.reset(token)


def summarize_turn(usages: List[LLMUsage]) -> Dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    priced = True
    models: List[str] = []
    rates: Dict[str, tuple] = {}
    for usage in usages:
        incoming = usage.prompt_tokens_actual if usage.prompt_tokens_actual is not None else usage.prompt_tokens_est
        outgoing = usage.completion_tokens_actual if usage.completion_tokens_actual is not None else usage.completion_tokens_est
        incoming = int(incoming or 0)
        outgoing = int(outgoing or 0)
        input_tokens += incoming
        output_tokens += outgoing
        if usage.model not in models:
            models.append(usage.model)
        rate = MODEL_USD_PER_MTOK.get(usage.model)
        if rate is None:
            priced = False
            continue
        rates[usage.model] = rate
        cost += (incoming / 1_000_000) * rate[0] + (outgoing / 1_000_000) * rate[1]
    input_rate = output_rate = None
    if len(rates) == 1:
        input_rate, output_rate = next(iter(rates.values()))
    return {
        "provider": usages[-1].provider if usages else "",
        "model": ",".join(models),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_usd_per_mtok": input_rate,
        "output_usd_per_mtok": output_rate,
        "cost_usd": round(cost, 8) if priced and usages else None,
    }


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
    contents: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_config: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 800,
    reasoning_effort: str = "",
) -> LLMResult:
    started = time.perf_counter()
    prompt_est = _estimate(system_instruction) + _estimate(user_prompt)
    attempts = 0
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
                parsed = _call_once(
                    provider=provider,
                    model_name=model_name,
                    api_key=key,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    timeout_s=timeout_s,
                    transport=transport,
                    contents=contents,
                    tools=tools,
                    tool_config=tool_config,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
                usage = LLMUsage(
                    provider=provider,
                    model=model_name,
                    purpose=purpose,
                    prompt_tokens_est=prompt_est,
                    completion_tokens_est=_estimate(parsed["text"]),
                    prompt_tokens_actual=parsed["prompt_tokens"],
                    completion_tokens_actual=parsed["completion_tokens"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attempts=attempts,
                )
                bucket = _TURN_USAGES.get()
                if bucket is not None:
                    bucket.append(usage)
                return LLMResult(
                    text=parsed["text"],
                    usage=usage,
                    function_calls=parsed["function_calls"],
                    model_content=parsed["model_content"],
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.warning("LLM %s failed status=%s attempt=%s", model_name, status, retry)
                if status not in RETRY_STATUSES or retry >= max_retries:
                    break
                sleeper(min(2.0, 0.2 * (2**retry)))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
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
    contents: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_config: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 800,
    reasoning_effort: str = "",
) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout_s, transport=transport) as client:
        if provider == "openai":
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=_responses_payload(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    contents=contents,
                    tools=tools,
                    tool_config=tool_config,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                ),
            )
            response.raise_for_status()
            return _parse_responses_output(response.json())

        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents
            or [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }
        if tools:
            payload["tools"] = tools
        if tool_config:
            payload["toolConfig"] = tool_config
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        response = client.post(
            url,
            headers={"x-goog-api-key": api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return _parse_gemini_candidate(data)


def _parse_gemini_candidate(data: Dict[str, Any]) -> Dict[str, Any]:
    candidate = (data.get("candidates") or [{}])[0]
    content = candidate.get("content") or {"role": "model", "parts": []}
    parts = content.get("parts") or []
    texts = []
    calls: List[FunctionCall] = []
    for part in parts:
        text = part.get("text")
        if text:
            texts.append(text)
        raw_call = part.get("functionCall")
        if raw_call:
            calls.append(
                FunctionCall(
                    name=str(raw_call.get("name") or ""),
                    args=_coerce_args(raw_call.get("args")),
                    call_id=str(raw_call.get("id") or ""),
                    thought_signature=str(part.get("thoughtSignature") or ""),
                )
            )
    usage = data.get("usageMetadata") or {}
    return {
        "text": "\n".join(texts).strip(),
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "function_calls": calls,
        "model_content": content,
    }


def _coerce_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _responses_payload(
    *,
    model_name: str,
    system_instruction: str,
    user_prompt: str,
    contents: Optional[List[Dict[str, Any]]],
    tools: Optional[List[Dict[str, Any]]],
    tool_config: Optional[Dict[str, Any]],
    max_output_tokens: int,
    reasoning_effort: str = "",
) -> Dict[str, Any]:
    effort = (reasoning_effort or "").strip().lower()
    payload: Dict[str, Any] = {
        "model": model_name,
        "instructions": system_instruction,
        "input": _contents_to_responses_input(contents, user_prompt),
        "store": False,
        "max_output_tokens": max_output_tokens,
    }
    if effort in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        payload["reasoning"] = {"effort": effort}
    openai_tools = _gemini_tools_to_responses(tools)
    if openai_tools:
        payload["tools"] = openai_tools
        mode = ((tool_config or {}).get("functionCallingConfig") or {}).get("mode")
        payload["tool_choice"] = "none" if mode == "NONE" else "auto"
    return payload


def _gemini_tools_to_responses(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted = []
    for tool in tools or []:
        for declaration in tool.get("functionDeclarations") or []:
            converted.append(
                {
                    "type": "function",
                    "name": declaration.get("name"),
                    "description": declaration.get("description") or "",
                    "parameters": declaration.get("parameters") or {"type": "object", "properties": {}},
                    "strict": False,
                }
            )
    return converted


def _contents_to_responses_input(contents: Optional[List[Dict[str, Any]]], user_prompt: str) -> Any:
    if not contents:
        return user_prompt
    items: List[Dict[str, Any]] = []
    for content in contents:
        role = "assistant" if content.get("role") == "model" else "user"
        for part in content.get("parts") or []:
            item = part.get("responsesItem")
            if item:
                items.append(item)
                continue
            if part.get("text"):
                items.append({"role": role, "content": part["text"]})
            raw_call = part.get("functionCall")
            if raw_call:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(raw_call.get("id") or ""),
                        "name": str(raw_call.get("name") or ""),
                        "arguments": json.dumps(raw_call.get("args") or {}, ensure_ascii=False),
                    }
                )
            raw_response = part.get("functionResponse")
            if raw_response:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(raw_response.get("id") or ""),
                        "output": json.dumps(raw_response.get("response") or {}, ensure_ascii=False),
                    }
                )
    return items


def _parse_responses_output(data: Dict[str, Any]) -> Dict[str, Any]:
    texts = []
    calls: List[FunctionCall] = []
    parts = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "function_call":
            name = str(item.get("name") or "")
            args = _coerce_args(item.get("arguments"))
            call_id = str(item.get("call_id") or "")
            calls.append(FunctionCall(name=name, args=args, call_id=call_id))
            parts.append(
                {
                    "functionCall": {"name": name, "args": args, "id": call_id},
                    "responsesItem": item,
                }
            )
            continue
        if kind == "message":
            message_text = [
                str(block.get("text"))
                for block in item.get("content") or []
                if isinstance(block, dict) and block.get("text")
            ]
            if message_text:
                texts.append("\n".join(message_text))
        parts.append({"responsesItem": item})
    if not texts and isinstance(data.get("output_text"), str) and data.get("output_text"):
        texts.append(data["output_text"])
    text = "\n".join(texts).strip()
    usage = data.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "function_calls": calls,
        "model_content": {"role": "model", "parts": parts or ([{"text": text}] if text else [])},
    }


def function_response_content(calls_and_results: List[tuple]) -> Dict[str, Any]:
    parts = []
    for call, payload in calls_and_results:
        body = {"name": call.name, "response": payload}
        if call.call_id:
            body["id"] = call.call_id
        parts.append({"functionResponse": body})
    return {"role": "user", "parts": parts}


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
