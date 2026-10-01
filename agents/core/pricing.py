"""Per-model token prices (USD per 1M tokens) used to meter and cap agent spend.

Cache writes are billed at 1.25x the input price; cache reads at the listed rate.
Prices are a local snapshot -- update when Anthropic changes them.
"""

PRICING = {
    # model: (input, output, cache_read)
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
DEFAULT_MODEL = "claude-opus-5-5"


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Dollar cost of one API call. Unknown models are priced like the default
    (a conservative guess) so budget caps still bite."""
    inp, out, read = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (input_tokens * inp
            + cache_write_tokens * inp * 1.25
            + cache_read_tokens * read
            + output_tokens * out) / 1_000_000
