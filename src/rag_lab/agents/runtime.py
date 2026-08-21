"""Stateful tool-use agent loop (plan §Phase 8, Step 8.3).

Genuinely new code, not a wrapper around ``llm.call_llm``. ``call_llm`` (Phase
7, Step 7.0) is a one-shot ``prompt -> text`` call with no conversation state
and no ``tool_use``/``tool_result`` blocks; its own docstring says as much
("Deliberately outside ``agents/`` -- Phase 8 reserves that package for its
tool-use agent loop"). What this module reuses from it is the *convention*,
not the function: a lazy ``anthropic`` import (so a ``core``-only install
stays import-safe) and a deterministic mock mode.

The central seam is ``ModelCaller``: a plain callable of
``(messages, system, tools) -> ModelTurn``. ``run_agent_loop`` itself never
knows or cares whether that callable talks to the real Anthropic API or
returns a scripted response -- ``anthropic_model_caller`` below builds the
real one; ``agents/router.py`` and ``agents/optimizer.py`` each build their
own deterministic mock one for ``--mock-llm``. This is what keeps the loop's
control flow (tool execution, malformed-call handling, budget caps,
transcript logging) written exactly once, shared by both agents, and
independent of mock-vs-real.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class ToolError(Exception):
    """Raised by a tool handler for an expected, reportable failure -- becomes
    a ``tool_result`` with ``is_error=True`` fed back to the model, exactly
    like a schema-validation failure, rather than propagating and killing the
    whole agent loop."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    # A small JSON-Schema-like subset (`type`/`required`/`properties.*.type`)
    # -- see `_validate_args`'s docstring for exactly how much of JSON Schema
    # this actually checks.
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    """One assistant turn: free text plus zero or more tool calls. Zero tool
    calls means the model is done -- the loop stops."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# (messages so far, system prompt, available tools) -> the next assistant
# turn. `anthropic_model_caller` below and every mock caller in
# `agents/router.py`/`agents/optimizer.py` share this exact signature.
ModelCaller = Callable[[list[dict[str, Any]], str, list["Tool"]], ModelTurn]


@dataclass(frozen=True)
class Budget:
    """Stopping conditions, checked *before* each model call so a breach never
    starts a turn it can't afford. A breach stops the loop cleanly (partial
    results returned, ``stopped_reason`` set) -- it is never raised as an
    exception. Plan §Phase 8, Step 8.2's "budget caps" design point."""

    max_steps: int = 5
    max_total_tokens: int | None = None
    max_wall_clock_s: float | None = None
    max_usd: float | None = None


@dataclass(frozen=True)
class AgentLoopResult:
    final_text: str
    transcript: list[dict[str, Any]]  # every request/response pair -- feeds --explain and the optimizer trace
    tool_results: list[dict[str, Any]]  # every tool call + its outcome, in order
    total_input_tokens: int
    total_output_tokens: int
    steps_used: int
    stopped_reason: str  # "end_turn" | "max_steps" | "budget_exceeded" | "repeated_invalid_tool_call"


# --------------------------------------------------------------------------- #
# Cost estimation -- new; nothing in this project tracked $ cost before Phase 8
# --------------------------------------------------------------------------- #

# (input, output) USD per million tokens. A small, deliberately static table:
# no existing code in this project tracks real $ cost anywhere (plan §Phase 8's
# risk register calls this out explicitly), so there is no dynamic pricing
# source to defer to instead.
PRICE_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5": (0.8, 4.0),
}
DEFAULT_PRICE_PER_MTOK_USD = (3.0, 15.0)  # unknown model name -> assume Sonnet-tier pricing


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICE_PER_MTOK_USD.get(model, DEFAULT_PRICE_PER_MTOK_USD)
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


# --------------------------------------------------------------------------- #
# Minimal argument validation -- deliberately not a new `jsonschema` dependency
# --------------------------------------------------------------------------- #

_SCHEMA_TYPE_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _validate_args(schema: dict[str, Any], args: Any) -> list[str]:
    """Structural validation only -- ``required`` keys present, and each
    given key's value matches its declared ``type``. This is a hand-rolled
    subset of JSON Schema, not a general validator: it's enough for this
    module's own tool schemas (all written by hand, all shallow objects), and
    ``jsonschema`` is not a dependency anywhere else in this project, so
    adding it just for this would be new dependency weight for one caller.
    """
    if not isinstance(args, dict):
        return [f"arguments must be an object, got {type(args).__name__}"]
    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"missing required argument {key!r}")
    properties = schema.get("properties", {})
    for key, value in args.items():
        prop = properties.get(key)
        if prop is None:
            continue  # unknown extra key -- ignored, not an error; models over-specify sometimes
        expected = _SCHEMA_TYPE_TO_PYTHON.get(prop.get("type"))
        if expected is not None and not isinstance(value, expected):
            errors.append(
                f"argument {key!r} must be of type {prop['type']}, got {type(value).__name__}"
            )
    return errors


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def run_agent_loop(
    *,
    system: str,
    user_message: str,
    tools: list[Tool],
    model_caller: ModelCaller,
    model: str = "",
    budget: Budget | None = None,
) -> AgentLoopResult:
    """Drive one tool-use conversation to completion.

    ``model`` is used only for USD cost estimation against ``budget.max_usd``
    -- it plays no role in how ``model_caller`` is invoked, since the caller
    already has whatever model identity it needs baked in.

    A tool call that fails argument validation, names an unknown tool, or
    whose handler raises ``ToolError`` becomes an ``is_error`` tool_result fed
    back to the model rather than an exception -- the model gets one chance to
    recover on the next step. Two such failures *in a row* (not two total --
    a valid call in between resets the counter) stop the loop early with
    ``stopped_reason="repeated_invalid_tool_call"`` rather than burning the
    rest of the step budget on a model that is stuck.
    """
    budget = budget or Budget()
    tools_by_name = {t.name: t for t in tools}
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    transcript: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    consecutive_invalid = 0
    started = time.monotonic()

    stopped_reason = "max_steps"
    final_text = ""

    for step in range(budget.max_steps):
        elapsed = time.monotonic() - started
        if budget.max_wall_clock_s is not None and elapsed >= budget.max_wall_clock_s:
            stopped_reason = "budget_exceeded"
            break
        if budget.max_total_tokens is not None and (total_input + total_output) >= budget.max_total_tokens:
            stopped_reason = "budget_exceeded"
            break
        if budget.max_usd is not None:
            spent = estimate_cost_usd(model, total_input, total_output)
            if spent >= budget.max_usd:
                stopped_reason = "budget_exceeded"
                break

        turn = model_caller(messages, system, tools)
        total_input += turn.input_tokens
        total_output += turn.output_tokens
        transcript.append(
            {
                "step": step,
                "text": turn.text,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
            }
        )

        if not turn.tool_calls:
            final_text = turn.text
            stopped_reason = "end_turn"
            break

        assistant_content: list[dict[str, Any]] = []
        if turn.text:
            assistant_content.append({"type": "text", "text": turn.text})
        for call in turn.tool_calls:
            assistant_content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        messages.append({"role": "assistant", "content": assistant_content})

        tool_result_blocks: list[dict[str, Any]] = []
        any_invalid_this_step = False
        for call in turn.tool_calls:
            tool = tools_by_name.get(call.name)
            result: Any = None
            error_text: str | None = None
            if tool is None:
                error_text = f"unknown tool {call.name!r}; available: {sorted(tools_by_name)}"
            else:
                errors = _validate_args(tool.input_schema, call.arguments)
                if errors:
                    error_text = "invalid arguments: " + "; ".join(errors)
                else:
                    try:
                        result = tool.handler(call.arguments)
                    except ToolError as exc:
                        error_text = str(exc)
                    except Exception as exc:  # a bug in a handler -- reported, not raised
                        error_text = f"tool {call.name!r} raised {type(exc).__name__}: {exc}"
                        log.warning("tool_handler_raised", tool=call.name, error=str(exc))

            is_error = error_text is not None
            any_invalid_this_step = any_invalid_this_step or is_error
            tool_results.append(
                {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": is_error,
                    **({"error": error_text} if is_error else {"result": result}),
                }
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": error_text if is_error else _stringify(result),
                    **({"is_error": True} if is_error else {}),
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

        consecutive_invalid = consecutive_invalid + 1 if any_invalid_this_step else 0
        if consecutive_invalid >= 2:
            stopped_reason = "repeated_invalid_tool_call"
            final_text = turn.text
            break
    else:
        stopped_reason = "max_steps"
        final_text = transcript[-1]["text"] if transcript else ""

    return AgentLoopResult(
        final_text=final_text,
        transcript=transcript,
        tool_results=tool_results,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        steps_used=len(transcript),
        stopped_reason=stopped_reason,
    )


# --------------------------------------------------------------------------- #
# The real (non-mock) ModelCaller
# --------------------------------------------------------------------------- #


def anthropic_model_caller(model: str, max_tokens: int = 1024) -> ModelCaller:
    """Build the real ``ModelCaller``: one ``anthropic.Anthropic().messages.create``
    call per turn.

    ``anthropic`` is imported lazily inside the returned closure, not at
    module import time -- importing ``agents.runtime`` (and building a mock
    ``ModelCaller``, which is all ``--mock-llm`` needs) never requires the
    ``agents`` extra, the same convention ``llm.call_llm`` established.
    """

    def _call(messages: list[dict[str, Any]], system: str, tools: list[Tool]) -> ModelTurn:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        usage = getattr(response, "usage", None)
        return ModelTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    return _call


__all__ = [
    "DEFAULT_PRICE_PER_MTOK_USD",
    "PRICE_PER_MTOK_USD",
    "AgentLoopResult",
    "Budget",
    "ModelCaller",
    "ModelTurn",
    "Tool",
    "ToolCall",
    "ToolError",
    "anthropic_model_caller",
    "estimate_cost_usd",
    "run_agent_loop",
]
