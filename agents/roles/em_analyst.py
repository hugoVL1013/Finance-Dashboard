"""Emerging-market equity analyst: the first agent in the firm (Phase 0)."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..core.agent import Agent
from ..core.tools import make_market_tools

SEED_UNIVERSE = ["EEM", "VWO", "IEMG", "INDA", "EWZ", "FXI", "EWT", "EWY", "EZA", "EWW"]

SYSTEM = """\
You are a senior emerging-market (EM) equity analyst at a small long-only investment firm.
You report to the EM portfolio manager. Your job is a daily research memo: which USD-listed
EM equity ETFs deserve capital right now, and why.

Rules you must follow:
- The firm is LONG-ONLY. Views are buy / hold / reduce / avoid. Never suggest shorting,
  leverage, leveraged or inverse ETFs, options, or non-USD listings.
- Every number you cite must come from a tool call in this session. Never quote prices,
  returns, volatility or fund metadata from memory. If a tool fails or returns an
  error, say so under data_caveats instead of guessing.
- Look at several candidates and at how they relate (use correlations) rather than
  assessing each in isolation. Prefer liquid, low-cost, diversified vehicles unless you
  can justify a country tilt with evidence from the data.
- Be concise and specific. Conviction is 1 (weak) to 5 (strong); most ideas should
  not be 5. State what would change your mind in key_risks.
- Finish by calling submit_result exactly once."""


class Candidate(BaseModel):
    ticker: str
    view: Literal["buy", "hold", "reduce", "avoid"]
    conviction: int = Field(ge=1, le=5)
    rationale: str = Field(description="Two or three sentences citing tool-derived figures")


class AnalystMemo(BaseModel):
    headline: str = Field(description="One-sentence takeaway")
    macro_view: str = Field(description="Short read on the EM backdrop supported by the data")
    candidates: list[Candidate] = Field(min_length=1, max_length=8)
    key_risks: list[str]
    data_caveats: list[str]


def build_task(today: Optional[date] = None, universe: Optional[list[str]] = None) -> str:
    today = today or date.today()
    universe = universe or SEED_UNIVERSE
    return (f"Today is {today.isoformat()}. Produce your daily EM equity memo.\n"
            f"Start by reviewing this reference set with your tools: {', '.join(universe)}. "
            f"You may nominate other USD-listed EM equity ETFs if the data supports them, "
            f"but verify each with the tools before recommending it.")


def build_agent(client: Any = None, fetch=None, **overrides) -> Agent:
    tools = make_market_tools(fetch) if fetch else make_market_tools()
    kwargs = dict(name="em_analyst", system=SYSTEM, output_model=AnalystMemo,
                  tools=tools, client=client)
    kwargs.update(overrides)
    return Agent(**kwargs)
