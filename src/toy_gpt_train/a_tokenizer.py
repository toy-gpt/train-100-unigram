"""a_tokenizer.py - A simple tokenizer for toy language-model training.

This module reads text from a corpus file and converts it into a list of tokens.

Concepts:
- token: a discrete unit of text used by a model (in this project, a token is a word).
- tokenize: the process of splitting text into tokens.

Notes:
- This tokenizer uses whitespace splitting for clarity and inspectability.
- Real language models often use subword tokenizers (breaking a word into subparts).
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

__all__ = ["SimpleTokenizer"]

LOG: logging.Logger = get_logger("TOKEN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
CORPUS_DIR: Final[Path] = BASE_DIR / "corpus"


class SimpleTokenizer:
    """Class that reads a corpus file and splits its text into tokens.

    In this project, tokenization is implemented as whitespace splitting.
    """

    def __init__(self, corpus_path: Path | None = None) -> None:
        """Initialize the tokenizer and load tokens from the corpus file.

        Args:
            corpus_path: Path to a plain-text corpus file. If None, automatically
                finds the single file in the corpus directory.

        Raises:
            FileNotFoundError: If corpus directory is missing or contains no files.
            ValueError: If corpus directory contains more than one file.
        """
        self.corpus_path: Path = corpus_path or self._find_corpus_file()
        self.tokens: list[str] = self._load_and_tokenize_corpus()
        LOG.info(f"Tokenizer initialized with {len(self.tokens)} tokens.")

    def _find_corpus_file(self) -> Path:
        """Find the single corpus file in the corpus directory.

        Returns:
            Path to the corpus file.

        Raises:
            FileNotFoundError: If corpus directory is missing or contains no files.
            ValueError: If corpus directory contains more than one file.
        """
        if not CORPUS_DIR.exists():
            msg = f"Corpus directory not found: {CORPUS_DIR}"
            raise FileNotFoundError(msg)

        files: list[Path] = list(CORPUS_DIR.iterdir())
        if len(files) == 0:
            msg = f"No files found in corpus directory: {CORPUS_DIR}"
            raise FileNotFoundError(msg)
        if len(files) > 1:
            msg = f"Expected exactly one file in corpus directory, found {len(files)}: {CORPUS_DIR}"
            raise ValueError(msg)

        LOG.debug(f"Found corpus file: {files[0].name}")
        return files[0]

    def _load_and_tokenize_corpus(self) -> list[str]:
        """Load corpus text and tokenize it into a list of tokens.

        Returns:
            Tokens in the order they appear in the corpus.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
        """
        if not self.corpus_path.exists():
            msg = f"Corpus file not found: {self.corpus_path}"
            raise FileNotFoundError(msg)

        text: str = self.corpus_path.read_text(encoding="utf-8")
        tokens: list[str] = text.split()
        LOG.debug(f"Tokenized text into {len(tokens)} tokens.")
        return tokens

    def get_tokens(self) -> list[str]:
        """Return the token list.

        Returns:
            The list of tokens loaded from the corpus.
        """
        return self.tokens


def main() -> None:
    """Demonstrate tokenization on the default corpus file."""
    import statistics

    log_header(LOG, "Tokenizer Demo")

    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    LOG.info(f"First 10 tokens: {tokens[:10]}")
    LOG.info(f"Total number of tokens: {len(tokens)}")

    if tokens:
        avg_token_length: float = statistics.mean(len(token) for token in tokens)
        LOG.info(f"Average token length: {avg_token_length:.2f}")
    else:
        LOG.info("No tokens available to calculate average length.")


if __name__ == "__main__":
    main()
