"""DeepSeek Responses API client with custom function tools and multi-turn chat.

The endpoint is stateless — `previous_response_id` and `conversation` are not
supported — so `Conversation` keeps the whole input item list locally and resends
it on every request.

    export DEEPSEEK_API_KEY=sk-...
    uv run python -m aihealthcare.agents.deepseek

Usage:

    from aihealthcare.agents.deepseek import Conversation, ResponsesClient

    def get_weather(city: str) -> str:
        '''Look up the current weather.

        Args:
            city: Name of the city, e.g. "Hangzhou".
        '''
        return f"{city}: 24C, sunny"

    chat = Conversation(ResponsesClient(), tools=[get_weather])
    print(chat.ask("What's the weather in Hangzhou?"))
    print(chat.ask("And how about tomorrow?"))

Only the standard library is used, so this module can be lifted out of the
project as-is.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import types
import typing
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class DeepSeekError(RuntimeError):
    """An API request failed, or the tool loop could not be completed."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


# --------------------------------------------------------------------------- #
# Function tools
# --------------------------------------------------------------------------- #

_SCALAR_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_ARG_LINE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SECTION = re.compile(r"^(args|arguments|parameters|returns|raises|yields|examples?|notes?)\s*:$", re.I)


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    if annotation is None or annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        variants = [a for a in args if a is not type(None)]
        if len(variants) == 1:
            return _json_schema_for(variants[0])
        return {"anyOf": [_json_schema_for(a) for a in variants]}
    if origin in (list, set, frozenset, tuple):
        return {"type": "array", "items": _json_schema_for(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation in _SCALAR_TYPES:
        return {"type": _SCALAR_TYPES[annotation]}
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and Google-style ``Args:`` entries."""
    if not doc:
        return "", {}

    summary: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current = ""

    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        section = _SECTION.match(stripped)
        if section:
            in_args = section.group(1).lower() in ("args", "arguments", "parameters")
            current = ""
            continue
        if not in_args:
            summary.append(stripped)
            continue
        if not stripped:
            continue
        match = _ARG_LINE.match(stripped)
        if match:
            current = match.group(1).lstrip("*")
            params[current] = match.group(2).strip()
        elif current:
            params[current] = f"{params[current]} {stripped}".strip()

    return "\n".join(summary).strip(), params


def _schema_from_signature(fn: Callable[..., Any], descriptions: dict[str, str]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs should not break tool registration
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _json_schema_for(hints.get(name))
        if name in descriptions:
            schema = {**schema, "description": descriptions[name]}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return schema


@dataclass(frozen=True)
class Tool:
    """A Python callable exposed to the model as a `function` tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        if isinstance(fn, Tool):
            return fn
        declared = getattr(fn, "__deepseek_tool__", {})
        summary, arg_docs = _parse_docstring(inspect.getdoc(fn))
        resolved = cls(
            name=name or declared.get("name") or fn.__name__,
            description=description or declared.get("description") or summary,
            parameters=(
                parameters
                or declared.get("parameters")
                or _schema_from_signature(fn, arg_docs)
            ),
            fn=fn,
        )
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", resolved.name):
            raise ValueError(f"invalid tool name {resolved.name!r}: must match ^[a-zA-Z0-9_-]{{1,128}}$")
        return resolved

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def call(self, arguments: str) -> str:
        """Run the tool. Failures come back as text so the model can recover."""
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc})"
        if not isinstance(kwargs, dict):
            return "ERROR: arguments must be a JSON object"
        try:
            result = self.fn(**kwargs)
        except TypeError as exc:
            return f"ERROR: bad arguments for {self.name} ({exc})"
        except Exception as exc:
            return f"ERROR: {self.name} raised {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Attach tool metadata to a function, overriding what is inferred from it.

    The function stays directly callable; pass it to `Conversation(tools=[...])`.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        target.__deepseek_tool__ = {  # type: ignore[attr-defined]
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        return target

    return decorate(fn) if fn is not None else decorate


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #


def output_text(response: dict[str, Any]) -> str:
    """Concatenate the `output_text` parts of a response, like `response.output_text`."""
    chunks = [
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    return "".join(chunks)


def reasoning_text(response: dict[str, Any]) -> str:
    chunks = [
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "reasoning"
        for part in item.get("content", [])
        if part.get("type") == "reasoning_text"
    ]
    return "".join(chunks)


def function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


class ResponsesClient:
    """Thin wrapper over `POST {base_url}/responses`."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("no API key: pass api_key= or set DEEPSEEK_API_KEY")
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def create(
        self,
        *,
        input: str | Sequence[dict[str, Any]] | None = None,
        instructions: str | None = None,
        tools: Iterable[Any] | None = None,
        tool_choice: Any = None,
        model: str | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Any:
        """Create a response. Returns the response dict, or an event iterator if streaming."""
        if input is None and instructions is None:
            raise ValueError("at least one of input and instructions is required")

        payload: dict[str, Any] = {"model": model or self.model}
        if input is not None:
            payload["input"] = list(input) if not isinstance(input, str) else input
        if instructions is not None:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = [
                t.spec() if isinstance(t, Tool) else t if isinstance(t, dict) else Tool.from_function(t).spec()
                for t in tools
            ]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update({k: v for k, v in extra.items() if v is not None})

        if stream:
            payload["stream"] = True
            return self._stream(payload)
        return self._post(payload)

    def _request(self, payload: dict[str, Any], *, stream: bool):
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code in RETRY_STATUS and attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise _http_error(exc.code, body) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise DeepSeekError(f"request to {self.base_url} failed: {exc.reason}") from exc
        raise DeepSeekError(f"request failed after {self.max_retries + 1} attempts: {last_error}")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request(payload, stream=False) as response:
            body = json.loads(response.read().decode())
        if body.get("status") == "failed":
            error = body.get("error") or {}
            raise DeepSeekError(error.get("message", "response failed"), code=error.get("code"))
        return body

    def _stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._request(payload, stream=True) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                event = json.loads(data)
                yield event
                if event.get("type") == "response.failed":
                    error = (event.get("response") or {}).get("error") or {}
                    raise DeepSeekError(error.get("message", "response failed"), code=error.get("code"))


def _http_error(status: int, body: str) -> DeepSeekError:
    message, code = body.strip(), None
    try:
        parsed = json.loads(body).get("error") or {}
        message = parsed.get("message", message)
        code = parsed.get("code")
    except (json.JSONDecodeError, AttributeError):
        pass
    return DeepSeekError(f"HTTP {status}: {message}", status=status, code=code)


# --------------------------------------------------------------------------- #
# Multi-turn conversation
# --------------------------------------------------------------------------- #


class Conversation:
    """Multi-turn chat that runs registered tools until the model replies with text."""

    def __init__(
        self,
        client: ResponsesClient | None = None,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        model: str | None = None,
        max_tool_rounds: int = 8,
        keep_reasoning: bool = False,
        on_tool_call: Callable[[str, str, str], None] | None = None,
        **params: Any,
    ):
        self.client = client or ResponsesClient()
        self.instructions = instructions
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.keep_reasoning = keep_reasoning
        self.on_tool_call = on_tool_call
        self.params = params
        self.history: list[dict[str, Any]] = []
        self.last_response: dict[str, Any] | None = None
        self.tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    def register(self, fn: Any, **kwargs: Any) -> Tool:
        resolved = Tool.from_function(fn, **kwargs)
        self.tools[resolved.name] = resolved
        return resolved

    def reset(self) -> None:
        self.history.clear()
        self.last_response = None

    def ask(self, message: str | dict[str, Any] | None = None) -> str:
        """Send a user message, resolve any tool calls, and return the reply text."""
        if message is not None:
            self.history.append(_user_item(message))

        for _ in range(self.max_tool_rounds + 1):
            response = self.client.create(
                input=self.history,
                instructions=self.instructions,
                tools=[t.spec() for t in self.tools.values()] or None,
                model=self.model,
                **self.params,
            )
            self.last_response = response
            self._absorb(response)
            if not self._run_tools(response):
                return output_text(response)

        raise DeepSeekError(f"tool loop did not settle within {self.max_tool_rounds} rounds")

    def ask_stream(self, message: str | dict[str, Any] | None = None) -> Iterator[str]:
        """Same as `ask`, yielding text deltas. Tool rounds are silent."""
        if message is not None:
            self.history.append(_user_item(message))

        for _ in range(self.max_tool_rounds + 1):
            response: dict[str, Any] | None = None
            for event in self.client.create(
                input=self.history,
                instructions=self.instructions,
                tools=[t.spec() for t in self.tools.values()] or None,
                model=self.model,
                stream=True,
                **self.params,
            ):
                if event.get("type") == "response.output_text.delta":
                    yield event.get("delta", "")
                elif event.get("type") in ("response.completed", "response.incomplete"):
                    response = event.get("response")

            if response is None:
                raise DeepSeekError("stream ended without a terminal response event")
            self.last_response = response
            self._absorb(response)
            if not self._run_tools(response):
                return

        raise DeepSeekError(f"tool loop did not settle within {self.max_tool_rounds} rounds")

    def _absorb(self, response: dict[str, Any]) -> None:
        """Append the model's output to the history, since the API stores nothing."""
        for item in response.get("output", []):
            kind = item.get("type")
            if kind == "message":
                text = "".join(
                    part.get("text", "")
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
                self.history.append(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                )
            elif kind == "function_call":
                self.history.append(
                    {
                        "type": "function_call",
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": item.get("arguments", "{}"),
                    }
                )
            elif kind == "reasoning" and self.keep_reasoning:
                self.history.append({"type": "reasoning", "content": item.get("content", [])})
            elif kind == "web_search_call":
                self.history.append(item)

    def _run_tools(self, response: dict[str, Any]) -> bool:
        """Execute every function call in the response. Returns True if any ran."""
        calls = function_calls(response)
        for call in calls:
            name, arguments = call["name"], call.get("arguments", "{}")
            target = self.tools.get(name)
            result = (
                target.call(arguments)
                if target is not None
                else f"ERROR: unknown tool {name!r}; available tools: {', '.join(self.tools) or 'none'}"
            )
            if self.on_tool_call is not None:
                self.on_tool_call(name, arguments, result)
            self.history.append(
                {"type": "function_call_output", "call_id": call["call_id"], "output": result}
            )
        return bool(calls)


def _user_item(message: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {"role": "user", "content": message}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #


def _demo_tools() -> list[Callable[..., Any]]:
    from datetime import datetime, timezone

    def get_current_time(timezone_offset_hours: int = 0) -> str:
        """Return the current UTC time, optionally shifted by a whole-hour offset.

        Args:
            timezone_offset_hours: Hours to add to UTC, e.g. 8 for Beijing time.
        """
        now = datetime.now(timezone.utc)
        shifted = now.timestamp() + timezone_offset_hours * 3600
        return datetime.fromtimestamp(shifted, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def get_weather(city: str, unit: typing.Literal["celsius", "fahrenheit"] = "celsius") -> dict[str, Any]:
        """Look up the current weather for a city (stub data for the demo).

        Args:
            city: Name of the city.
            unit: Temperature unit to report.
        """
        celsius = 24
        return {
            "city": city,
            "temperature": celsius if unit == "celsius" else round(celsius * 9 / 5 + 32),
            "unit": unit,
            "condition": "sunny",
        }

    return [get_current_time, get_weather]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat with DeepSeek through the Responses API.")
    parser.add_argument("prompt", nargs="*", help="Prompt; omit for an interactive session")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--instructions", default="You are a helpful assistant.")
    parser.add_argument("--effort", default=None, help="Thinking effort: none/low/medium/high/max")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--quiet-tools", action="store_true", help="Do not print tool calls")
    args = parser.parse_args()

    def show_tool_call(name: str, arguments: str, result: str) -> None:
        print(f"\n[tool] {name}({arguments}) -> {result}\n", flush=True)

    chat = Conversation(
        ResponsesClient(model=args.model),
        instructions=args.instructions,
        tools=_demo_tools(),
        on_tool_call=None if args.quiet_tools else show_tool_call,
        reasoning={"effort": args.effort} if args.effort else None,
    )

    def turn(prompt: str) -> None:
        if args.no_stream:
            print(chat.ask(prompt))
            return
        for delta in chat.ask_stream(prompt):
            print(delta, end="", flush=True)
        print()

    if args.prompt:
        turn(" ".join(args.prompt))
        return

    print("DeepSeek chat. Ctrl-C or an empty line to quit.")
    while True:
        try:
            prompt = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            return
        print("\nbot > ", end="", flush=True)
        turn(prompt)


if __name__ == "__main__":
    main()
