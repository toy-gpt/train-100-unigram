"""prompts.py - Prompt parsing and context selection utilities.

This module provides shared, model-agnostic helpers for working with
prompt text during inference.

Responsibilities:
- Normalize raw prompt text in a predictable way
- Tokenize prompt text consistently with training-time expectations
- Select the final N tokens needed to condition a model with a given
  context window (unigram, bigram, context-2, context-3, etc.)
- Define a small, explicit CLI argument surface for prompt-driven inference

Non-responsibilities:
- No vocabulary lookup or token-id mapping
- No model or weight-matrix assumptions
- No corpus-specific logic

Design note:
This module is intentionally pure and reusable so it can be imported
unchanged by derived repositories (animals, llm_glossary, etc.).
"""

import argparse
from dataclasses import dataclass

__all__ = [
    "PromptContext",
    "normalize_prompt_text",
    "prompt_to_tokens",
    "select_context_tokens",
    "parse_args",
]


@dataclass(frozen=True)
class PromptContext:
    """Normalized context tokens extracted from a prompt.

    Attributes:
        tokens: The final N tokens (after normalization), where
            N is determined by the model's context window.
            Tokens are ordered from oldest to most recent.
    """

    tokens: tuple[str, ...]


# === PROMPT NORMALIZATION & TOKENIZATION ===


def normalize_prompt_text(prompt: str) -> str:
    """Normalize raw prompt text prior to tokenization.

    Current policy:
    - strip leading/trailing whitespace
    - lowercase text

    This must remain consistent with training-time tokenization.
    """
    return prompt.strip().lower()


def prompt_to_tokens(prompt: str) -> list[str]:
    """Convert prompt text into a list of normalized tokens.

    Args:
        prompt: Raw prompt string.

    Returns:
        List of tokens, or an empty list if the prompt is empty.
    """
    text = normalize_prompt_text(prompt)
    return text.split() if text else []


# === CONTEXT SELECTION ===


def select_context_tokens(*, prompt: str, context_window: int) -> PromptContext:
    """Select the final context tokens needed to condition a model.

    Args:
        prompt: Full prompt text provided by the user.
        context_window: Number of tokens required by the model
            (e.g., 1 for unigram, 2 for bigram, 3 for context-3).

    Returns:
        PromptContext containing up to the last `context_window` tokens.

    Raises:
        ValueError: If context_window is less than 1.
    """
    if context_window < 1:
        raise ValueError("context_window must be >= 1")

    tokens = prompt_to_tokens(prompt)
    if not tokens:
        return PromptContext(tokens=())

    return PromptContext(tokens=tuple(tokens[-context_window:]))


# === COMMAND-LINE INTERFACE ===


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-driven inference.

    This parser is intentionally model-agnostic. Interpretation of the
    prompt and context length is handled by the caller based on model kind.
    """
    parser = argparse.ArgumentParser(
        description="Toy GPT inference from saved artifacts."
    )

    parser.add_argument(
        "--prompt",
        dest="prompt",
        default="",
        help=(
            "Prompt text used to condition generation. "
            "The last N tokens are selected based on the model context window."
        ),
    )

    parser.add_argument(
        "--start",
        dest="start_token",
        default="",
        help=(
            "Explicit start token for generation. If provided, this overrides --prompt."
        ),
    )

    parser.add_argument(
        "--num",
        dest="num_tokens",
        type=int,
        default=10,
        help="Number of tokens to generate (not counting the start token).",
    )

    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=3,
        help="Show top-k next-token probabilities for the conditioning context.",
    )

    return parser.parse_args()
