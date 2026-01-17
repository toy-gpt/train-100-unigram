"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using a unigram context (current token only).

Responsibilities:
- Create (current_token -> next_token) training pairs from the corpus
- Run a basic gradient-descent training loop
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss
  - think descending to find the bottom of a valley in a landscape
  - where the valley floor corresponds to lower prediction error

Notes:
- This is intentionally simple: no deep learning framework, no Transformer.
- The model is a softmax regression classifier per input token ID.
- Training updates only the row of weights for the current token ID.
- token_embeddings.csv is a visualization-friendly projection for levels 100-400;
  in later repos (500+), embeddings become a first-class learned table.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.io_artifacts import (
    RowLabeler,
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax, cross_entropy_loss

__all__ = [
    "make_training_pairs",
    "row_labeler_unigram",
    "token_row_index_unigram",
    "train_model",
]

LOG: logging.Logger = get_logger("TRAIN", level="INFO")


def token_row_index_unigram(token_id: int, vocab_size: int) -> int:
    """Return the row index for a unigram token ID.

    In a unigram model, each token maps directly to one row.
    """
    return token_id


def row_labeler_unigram(vocab: VocabularyLike, vocab_size: int) -> RowLabeler:
    """Map a unigram row index to a token label."""

    def label(row_idx: int) -> str:
        token_id: int = row_idx
        tok: str | None = vocab.get_id_token(token_id)
        return tok if tok is not None else f"id_{token_id}"

    return label


def make_training_pairs(token_ids: list[int]) -> list[tuple[int, int]]:
    """Convert a sequence of token IDs into (current, next) training pairs.

    Language models learn to predict "what comes next" given "what came before".
    This function creates supervised training examples by pairing each token
    with the token that follows it in the corpus.

    Args:
        token_ids: Sequence of integer token IDs from the corpus.

    Returns:
        List of (input_id, target_id) tuples. Each tuple represents one
        training example: given input_id, the model should predict target_id.

    Example:
        Token sequence "the cat sat" with IDs [3, 1, 2] produces:
        [(3, 1), (1, 2)]
        Meaning: after token 3, predict token 1; after token 1, predict token 2.
    """
    pairs: list[tuple[int, int]] = []
    for i in range(len(token_ids) - 1):
        pairs.append((token_ids[i], token_ids[i + 1]))
    return pairs


def train_model(
    model: "SimpleNextTokenModel",
    pairs: list[tuple[int, int]],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the model using gradient descent on softmax cross-entropy.

    Training proceeds in epochs (full passes through all training pairs).
    For each pair, we:
    1. Compute the model's predicted probabilities (forward pass).
    2. Measure how wrong the prediction was (loss).
    3. Adjust weights to reduce the loss (gradient descent).

    Args:
        model: The model to train (weights will be modified in place).
        pairs: List of (input_id, target_id) training pairs.
        learning_rate: Step size for gradient descent. Larger values learn
            faster but may overshoot; smaller values are more stable but slower.
        epochs: Number of complete passes through the training data.

    Returns:
        List of per-epoch metrics dictionaries containing epoch number,
        average loss, and accuracy.
    """
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for input_id, target_id in pairs:
            # Forward pass: get probability distribution over next tokens.
            probs: list[float] = model.forward(input_id)

            # Compute loss: how surprised is the model by the correct answer?
            loss: float = cross_entropy_loss(probs, target_id)
            total_loss += loss

            # Check if the model's top prediction matches the correct answer.
            pred_id: int = argmax(probs)
            if pred_id == target_id:
                correct += 1

            # Backward pass: compute gradients and update weights.
            #
            # For softmax cross-entropy, the gradient has an elegant form:
            #   gradient[j] = predicted_prob[j] - true_prob[j]
            #
            # Since true_prob is one-hot (1.0 for target, 0.0 elsewhere):
            #   - For the target token: gradient = prob - 1.0 (negative, so weight increases)
            #   - For other tokens: gradient = prob - 0.0 (positive, so weight decreases)
            #
            # This pushes probability mass toward the correct token.
            row: list[float] = model.weights[input_id]
            for j in range(model.vocab_size):
                y: float = 1.0 if j == target_id else 0.0
                grad: float = probs[j] - y
                row[j] -= learning_rate * grad

        # Compute epoch-level metrics.
        avg_loss: float = total_loss / len(pairs) if pairs else float("nan")
        accuracy: float = correct / len(pairs) if pairs else 0.0

        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "avg_loss": avg_loss,
            "accuracy": accuracy,
        }
        history.append(metrics)

        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run a simple training demo end-to-end."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Softmax Regression")

    base_dir: Final[Path] = Path(__file__).resolve().parents[2]
    outputs_dir: Final[Path] = base_dir / "outputs"
    train_log_path: Final[Path] = outputs_dir / "train_log.csv"

    # Step 0: Identify the corpus file (single file rule).
    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.error("No tokens found. Check corpus file.")
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error(f"Token not found in vocabulary: {tok!r}")
            return
        token_ids.append(tok_id)

    # Step 4: Create training pairs (input -> target).
    pairs: list[tuple[int, int]] = make_training_pairs(token_ids)
    LOG.info(f"Created {len(pairs)} training pairs.")

    # Step 5: Initialize model with random weights.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(train_log_path, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=base_dir,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="unigram",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_unigram(vocab, vocab.vocab_size()),
    )

    # Step 8: Qualitative check - what does the model predict after first token?
    current_token: str = tokens[0]
    current_id: int | None = vocab.get_token_id(current_token)
    if current_id is not None:
        probs: list[float] = model.forward(current_id)
        best_next_id: int = argmax(probs)
        best_next_tok: str | None = vocab.get_id_token(best_next_id)
        LOG.info(
            f"After training, most likely next token after {current_token!r} "
            f"is {best_next_tok!r} (ID: {best_next_id})."
        )


if __name__ == "__main__":
    main()
