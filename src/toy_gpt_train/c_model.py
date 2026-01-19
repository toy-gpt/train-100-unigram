"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for unigram (no context).
  A unigram models P(next) - just word frequencies, ignoring all context.

Responsibilities:
- Represent a simple parameterized model that outputs the same
  probability distribution regardless of input.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight vector (1D: just next_token scores)
- one forward computation that ignores input
- no learning here

Training is handled in a different module.
"""

import logging
import math
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

__all__ = ["SimpleNextTokenModel"]

LOG: logging.Logger = get_logger("MODEL", level="INFO")


class SimpleNextTokenModel:
    """A minimal next-token prediction model (unigram - no context).

    Unigram ignores all context and predicts based solely on
    corpus word frequencies: P(next).
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the model with a given vocabulary size."""
        self.vocab_size: Final[int] = vocab_size

        # Weight matrix: 1 row x vocab_size columns
        # Unigram has only ONE row because predictions don't depend on input.
        # We store as list[list[float]] with 1 row for artifact compatibility.
        self.weights: list[list[float]] = [[0.0 for _ in range(vocab_size)]]

        LOG.info(f"Model initialized with vocabulary size {vocab_size} (unigram).")

    def forward(self, current_id: int | None = None) -> list[float]:
        """Perform a forward pass.

        Args:
            current_id: Ignored for unigram - included for API consistency.

        Returns:
            Probability distribution over next tokens (same for all inputs).
        """
        # Unigram ignores current_id - always returns the same distribution
        _ = current_id
        scores: list[float] = self.weights[0]
        return self._softmax(scores)

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        max_score: float = max(scores)
        exp_scores: list[float] = [math.exp(s - max_score) for s in scores]
        total: float = sum(exp_scores)
        return [s / total for s in exp_scores]


def main() -> None:
    """Demonstrate a forward pass of the simple unigram model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train.a_tokenizer import SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo (Unigram - No Context)")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.info("No tokens available for demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Forward pass (unigram ignores input).
    probs: list[float] = model.forward()

    # Step 5: Inspect results.
    LOG.info("Unigram ignores input - same predictions for any context:")
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
