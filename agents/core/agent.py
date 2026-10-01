"""The generic agent: a loop around an LLM.

This file is the whole idea of an "agent". There is no framework magic:

  1. A *system prompt* gives the agent its role and rules.
  2. A list of *tools* (plain Python functions) is described to the model.
  3. We call the model. If it asks to use a tool, we run it, append the result to
     the conversation, and call the model again. Repeat.
  4. The agent finishes by calling a special ``submit_result`` tool whose input
     must validate against a pydantic schema. That is how a free-form LLM turns
     into a typed, machine-readable output that the next agent (or code) can use.

Agents have no memory between runs; anything they should "remember" must be put
into the prompt by the caller. Hard limits (turns, dollars) are enforced here, in
code, not by asking the model to behave.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .pricing import DEFAULT_MODEL, cost_usd

SUBMIT_TOOL = "submit_result"
MAX_TOOL_RESULT_CHARS = 20_000


class AgentError(RuntimeError):
    """The agent could not finish (refusal, truncation, too many turns...)."""


class BudgetExceeded(AgentError):
    """The agent hit its per-run dollar cap."""


@dataclass
class Tool:
    """A Python function the model may call."""
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0

    def add(self, other: "Usage") -> None:
        for f in ("input_tokens", "output_tokens", "cache_read_tokens",
                  "cache_write_tokens", "cost_usd", "api_calls"):
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class AgentResult:
    output: BaseModel
    transcript: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0


def _block_to_dict(block: Any) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in vars(block).items() if not k.startswith("_")}


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]"


class Agent:
    def __init__(
        self,
        name: str,
        system: str,
        output_model: Type[BaseModel],
        tools: Optional[list[Tool]] = None,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_turns: int = 12,
        max_cost_usd: float = 1.0,
        max_tokens: int = 16000,
        client: Any = None,
    ):
        self.name = name
        self.system = system
        self.output_model = output_model
        self.tools = {t.name: t for t in (tools or [])}
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so tests need no API key
            self._client = anthropic.Anthropic()
        return self._client

    def _tool_specs(self) -> list[dict]:
        specs = [t.spec() for t in self.tools.values()]
        specs.append({
            "name": SUBMIT_TOOL,
            "description": ("Submit your final answer. Call this exactly once when you are "
                            "done. The input must match the schema; if it is rejected, fix "
                            "the problems and call it again."),
            "input_schema": self.output_model.model_json_schema(),
        })
        return specs

    def _call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool. Returns (text_for_model, is_error). Never raises: a failing
        tool becomes an error message the model can react to."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available: {sorted(self.tools)}", True
        try:
            return _truncate(json.dumps(tool.fn(**args), default=str)), False
        except Exception as err:  # noqa: BLE001 - tool code can fail in any way
            return f"Tool '{name}' failed: {type(err).__name__}: {err}", True

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        transcript: list[dict] = [{"role": "user", "content": task}]
        usage = Usage()
        nudged = False

        for turn in range(1, self.max_turns + 1):
            if usage.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"{self.name}: spent ${usage.cost_usd:.3f} >= cap ${self.max_cost_usd:.2f}")

            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self._tool_specs(),
                messages=messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},  # cache the growing prefix
            )

            u = resp.usage
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            usage.add(Usage(
                input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                cost_usd=cost_usd(self.model, u.input_tokens, u.output_tokens,
                                  cache_read, cache_write),
                api_calls=1))

            # Echo the assistant turn back unchanged (thinking blocks must be preserved).
            messages.append({"role": "assistant", "content": resp.content})
            transcript.append({"role": "assistant", "turn": turn,
                               "content": [_block_to_dict(b) for b in resp.content]})

            if resp.stop_reason == "refusal":
                raise AgentError(f"{self.name}: model refused the request")
            if resp.stop_reason == "max_tokens":
                raise AgentError(f"{self.name}: response truncated at max_tokens")

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                if nudged:
                    raise AgentError(f"{self.name}: finished without calling {SUBMIT_TOOL}")
                nudged = True
                nudge = f"You must finish by calling the {SUBMIT_TOOL} tool with your final answer."
                messages.append({"role": "user", "content": nudge})
                transcript.append({"role": "user", "content": nudge})
                continue

            results, output = [], None
            for tu in tool_uses:
                if tu.name == SUBMIT_TOOL:
                    try:
                        output = self.output_model.model_validate(tu.input)
                        text, is_error = "Accepted.", False
                    except ValidationError as err:
                        text, is_error = f"Rejected, fix and resubmit:\n{err}", True
                else:
                    text, is_error = self._call_tool(tu.name, dict(tu.input))
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": text, "is_error": is_error})

            # All tool results from one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})
            transcript.append({"role": "user", "content": results})

            if output is not None:
                return AgentResult(output=output, transcript=transcript, usage=usage, turns=turn)

        raise AgentError(f"{self.name}: no result after {self.max_turns} turns")
