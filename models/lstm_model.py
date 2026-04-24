from __future__ import annotations


class LstmResearchPlaceholder:
    """Reserved for research. Keep neural nets out of the runtime path until proven."""

    def __init__(self) -> None:
        raise RuntimeError(
            "LSTM experiments should live behind an optional research dependency group. "
            "Promote only a trained, validated artifact into submission."
        )
