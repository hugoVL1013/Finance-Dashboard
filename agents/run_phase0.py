"""Phase 0 demo: run a single EM analyst agent end to end and print its memo.

    export ANTHROPIC_API_KEY=...
    python -m agents.run_phase0 [--model claude-opus-5-5] [--effort medium] [--max-cost 1.0]

The full transcript (every model turn and tool call) is saved to agents/runs/ so you
can read exactly how the agent reasoned and what each tool returned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .core.agent import AgentError
from .core.pricing import DEFAULT_MODEL
from .roles import em_analyst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-cost", type=float, default=1.0, help="hard USD cap for this run")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    agent = em_analyst.build_agent(model=args.model, effort=args.effort,
                                   max_cost_usd=args.max_cost, max_turns=args.max_turns)
    try:
        result = agent.run(em_analyst.build_task())
    except AgentError as err:
        print(f"Agent failed: {err}", file=sys.stderr)
        return 1

    memo = result.output
    print(f"\n=== {memo.headline} ===\n{memo.macro_view}\n")
    for c in memo.candidates:
        print(f"  {c.ticker:6} {c.view:7} conviction {c.conviction}  {c.rationale}")
    print("\nRisks:", *[f"\n  - {r}" for r in memo.key_risks])
    print("Data caveats:", *[f"\n  - {r}" for r in memo.data_caveats])
    u = result.usage
    print(f"\n{result.turns} turns, {u.api_calls} API calls, {u.input_tokens} in / "
          f"{u.output_tokens} out tokens, cache read {u.cache_read_tokens}, cost ${u.cost_usd:.3f}")

    out_dir = Path(__file__).parent / "runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"em_analyst_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"memo": memo.model_dump(), "usage": u.__dict__,
                                "transcript": result.transcript}, indent=2, default=str))
    print(f"Transcript saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
