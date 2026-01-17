"""b_vocab.py - Vocabulary construction module.

Builds a vocabulary from tokenized text data.

Responsibilities:
- Identify unique tokens in the corpus
- Assign each token a unique integer ID
- Compute token frequency counts
- Provide token-to-id and id-to-token mappings

This module bridges raw tokens and numerical representations
required by statistical and neural models.
"""

from collections import Counter
import logging

from datafun_toolkit.logger import get_logger, log_header

__all__ = ["Vocabulary"]

LOG: logging.Logger = get_logger("VOCAB", level="INFO")


class Vocabulary:
    """Build token-to-id mappings and token frequency counts from a token list."""

    def __init__(self, tokens: list[str]) -> None:
        """Initialize the vocabulary from a list of tokens.

        Args:
            tokens: List of tokens from which to build the vocabulary.
        """
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.token_freq: dict[str, int] = {}

        self._build_vocab(tokens)
        LOG.info(f"Vocabulary initialized with {self.vocab_size()} unique tokens.")

    def _build_vocab(self, tokens: list[str]) -> None:
        """Build mappings and token frequency counts.

        Args:
            tokens: List of tokens to process.
        """
        counts: Counter[str] = Counter(tokens)

        for idx, token in enumerate(sorted(counts.keys())):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.token_freq[token] = counts[token]

        LOG.debug(f"Built vocabulary with {len(self.token_to_id)} tokens.")

    def vocab_size(self) -> int:
        """Return the number of unique tokens."""
        return len(self.token_to_id)

    def get_token_id(self, token: str) -> int | None:
        """Return the integer ID for token, or None if not found."""
        return self.token_to_id.get(token)

    def get_id_token(self, idx: int) -> str | None:
        """Return the token for integer ID, or None if not found."""
        return self.id_to_token.get(idx)

    def get_token_frequency(self, token: str) -> int:
        """Return frequency count for token (0 if not found)."""
        return self.token_freq.get(token, 0)


def main() -> None:
    """Demonstrate vocabulary construction from the project corpus.

    Flow:
    1) Tokenize corpus text -> tokens (list[str])
    2) Build vocabulary -> token_to_id, id_to_token, token_freq
    3) Demonstrate one lookup (token -> id, token -> frequency)
    """
    from toy_gpt_train.a_tokenizer import SimpleTokenizer

    log_header(LOG, "Vocabulary Demo")

    # 1) Tokenize - start with corpus and turn it into tokens
    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    # 2) Build vocabulary (the set of unique tokens) from the list of tokens
    vocab: Vocabulary = Vocabulary(tokens)
    LOG.info(f"Vocabulary size: {vocab.vocab_size()}")

    if tokens:
        sample_token: str = tokens[0]
        sample_id: int | None = vocab.get_token_id(sample_token)
        sample_freq: int = vocab.get_token_frequency(sample_token)

        LOG.info(
            f"Sample token: {sample_token!r} "
            f"| ID: {sample_id} "
            f"| Frequency: {sample_freq}"
        )
    else:
        LOG.info("No tokens found; cannot demonstrate vocabulary lookup.")


if __name__ == "__main__":
    main()
