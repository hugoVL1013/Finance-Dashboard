"""Agent loop tests using a scripted fake client (no network, no API key)."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from agents.core.agent import Agent, AgentError, BudgetExceeded, Tool
from agents.core.tools import compute_correlations, compute_price_stats, make_market_tools
from agents.roles import em_analyst


class Out(BaseModel):
    answer: str


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def reply(*blocks, stop="tool_use", inp=100, out=50):
    usage = NS(input_tokens=inp, output_tokens=out,
               cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=list(blocks), stop_reason=stop, usage=usage)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: the loop keeps appending
        return self.replies.pop(0)


def echo_tool():
    return Tool("echo", "echo back", {"type": "object", "properties": {"x": {"type": "string"}}},
                lambda x: {"echoed": x})


def make(replies, **kw):
    client = FakeClient(replies)
    return Agent("t", "sys", Out, tools=[echo_tool()], client=client, **kw), client


def test_tool_then_submit():
    agent, client = make([
        reply(tool_use("1", "echo", x="hi")),
        reply(tool_use("2", "submit_result", answer="done"), stop="tool_use"),
    ])
    res = agent.run("go")
    assert res.output.answer == "done" and res.turns == 2
    assert res.usage.api_calls == 2 and res.usage.cost_usd > 0
    # tool result was fed back to the model on the second call
    last_user = client.calls[1]["messages"][-1]
    assert last_user["content"][0]["content"] == '{"echoed": "hi"}'


def test_invalid_submit_is_rejected_then_retried():
    agent, client = make([
        reply(tool_use("1", "submit_result", wrong="field")),
        reply(tool_use("2", "submit_result", answer="ok")),
    ])
    assert agent.run("go").output.answer == "ok"
    rejected = client.calls[1]["messages"][-1]["content"][0]
    assert rejected["is_error"] and "Rejected" in rejected["content"]


def test_unknown_and_failing_tools_become_errors():
    boom = Tool("boom", "fails", {"type": "object", "properties": {}}, lambda: 1 / 0)
    client = FakeClient([
        reply(tool_use("1", "nope"), tool_use("2", "boom")),
        reply(tool_use("3", "submit_result", answer="ok")),
    ])
    Agent("t", "sys", Out, tools=[boom], client=client).run("go")
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["is_error"] for r in results] == [True, True]
    assert "Unknown tool" in results[0]["content"] and "ZeroDivisionError" in results[1]["content"]


def test_nudge_when_model_forgets_to_submit():
    agent, _ = make([
        reply(text("here you go"), stop="end_turn"),
        reply(tool_use("1", "submit_result", answer="ok")),
    ])
    assert agent.run("go").output.answer == "ok"


def test_fails_after_second_missing_submit():
    agent, _ = make([reply(text("a"), stop="end_turn"), reply(text("b"), stop="end_turn")])
    with pytest.raises(AgentError, match="without calling"):
        agent.run("go")


def test_max_turns():
    agent, _ = make([reply(tool_use(str(i), "echo", x="a")) for i in range(5)], max_turns=3)
    with pytest.raises(AgentError, match="no result after 3 turns"):
        agent.run("go")


def test_budget_cap():
    agent, _ = make([reply(tool_use("1", "echo", x="a"), inp=2_000_000)] * 3, max_cost_usd=1.0)
    with pytest.raises(BudgetExceeded):
        agent.run("go")


def test_refusal_and_truncation():
    agent, _ = make([reply(text("no"), stop="refusal")])
    with pytest.raises(AgentError, match="refused"):
        agent.run("go")
    agent, _ = make([reply(text("..."), stop="max_tokens")])
    with pytest.raises(AgentError, match="truncated"):
        agent.run("go")


# --- market tools -----------------------------------------------------------

def synthetic(n=300, drift=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-30", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.01, n)), index=idx)


def test_price_stats_known_values():
    idx = pd.bdate_range(end="2026-09-30", periods=300)
    close = pd.Series(100.0, index=idx)
    close.iloc[-1] = 110.0
    s = compute_price_stats(close)
    assert s["last_close"] == 110.0 and s["ret_1m"] == 0.1 and s["max_drawdown_1y"] == 0.0
    assert s["pct_below_1y_high"] == 0.0 and s["last_date"] == "2026-09-30"


def test_price_stats_drawdown_and_short_history():
    idx = pd.bdate_range(end="2026-09-30", periods=100)
    close = pd.Series(np.r_[np.linspace(100, 200, 50), np.linspace(200, 100, 50)], index=idx)
    assert compute_price_stats(close)["max_drawdown_1y"] == pytest.approx(-0.5, abs=1e-3)
    with pytest.raises(ValueError):
        compute_price_stats(close.iloc[:10])


def test_correlations_symmetric_and_bounded():
    c = compute_correlations({"A": synthetic(seed=1), "B": synthetic(seed=2)})
    assert c["matrix"]["A"]["A"] == 1.0
    assert c["matrix"]["A"]["B"] == c["matrix"]["B"]["A"] and abs(c["matrix"]["A"]["B"]) <= 1


def test_market_tools_validate_tickers_and_isolate_failures():
    def fetch(t, period):
        if t == "BAD":
            raise RuntimeError("no data")
        return synthetic()
    tools = {t.name: t for t in make_market_tools(fetch)}
    out = tools["get_price_stats"].fn(["eem", "BAD"])
    assert "last_close" in out["EEM"] and "error" in out["BAD"]
    with pytest.raises(ValueError):
        tools["get_price_stats"].fn(["EEM; DROP TABLE"])


def test_em_analyst_end_to_end_with_scripted_model():
    memo = dict(headline="h", macro_view="m", key_risks=["r"], data_caveats=[],
                candidates=[dict(ticker="EEM", view="buy", conviction=3, rationale="x")])
    client = FakeClient([
        reply(tool_use("1", "get_price_stats", tickers=["EEM", "VWO"])),
        reply(tool_use("2", "submit_result", **memo)),
    ])
    agent = em_analyst.build_agent(client=client, fetch=lambda t, p: synthetic())
    res = agent.run(em_analyst.build_task())
    assert res.output.candidates[0].ticker == "EEM"
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert not tool_result["is_error"] and "EEM" in tool_result["content"]
    # long-only mandate is in the system prompt; submit schema is exposed as a tool
    assert "LONG-ONLY" in client.calls[0]["system"]
    assert client.calls[0]["tools"][-1]["name"] == "submit_result"
