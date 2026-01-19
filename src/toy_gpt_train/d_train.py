"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using unigram (no context - just word frequencies).

A unigram models P(next) - the probability of each word based purely
on how often it appears in the corpus, ignoring all context.

Responsibilities:
- Count token frequencies in the corpus
- Train a single row of weights to predict based on frequency
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- unigram: predict next token using only corpus frequencies (no context)
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss

Notes:
- This is intentionally simple: no deep learning framework, no Transformer.
- The model has only ONE row of weights (predictions are context-independent).
- Training updates the same single row for every example.
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
    "make_training_targets",
    "row_labeler_unigram",
    "train_model",
]

LOG: logging.Logger = get_logger("TRAIN", level="INFO")


def row_labeler_unigram(vocab: VocabularyLike, vocab_size: int) -> RowLabeler:
    """Map a unigram row index to a label.

    Unigram has only one row, labeled to indicate it's context-free.
    """
    _ = vocab  # unused - unigram doesn't label by token
    _ = vocab_size

    def label(row_idx: int) -> str:
        # Only one row in unigram - label it descriptively
        return "(no context)"

    return label


def make_training_targets(token_ids: list[int]) -> list[int]:
    """Extract training targets for unigram model.

    For unigram, we don't need (input, target) pairs because
    the model ignores input. We just need the list of all tokens
    that appear in the corpus - each one is a target to predict.

    Args:
        token_ids: Sequence of integer token IDs from the corpus.

    Returns:
        List of target token IDs (all tokens in corpus).

    Example:
        Token sequence "the cat sat" with IDs [3, 1, 2] produces:
        [3, 1, 2]
        Meaning: the model should learn to predict these tokens
        based on their frequency (3 appears once, 1 appears once, etc.)
    """
    return token_ids


def train_model(
    model: "SimpleNextTokenModel",
    targets: list[int],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the unigram model using gradient descent on softmax cross-entropy.

    Unigram training learns corpus frequencies. The model has a single row
    of weights that gets updated for every token in the corpus.

    Training proceeds in epochs (full passes through all tokens).
    For each token, we:
    1. Compute the model's predicted probabilities (forward pass).
    2. Measure how wrong the prediction was (loss).
    3. Adjust weights to reduce the loss (gradient descent).

    Args:
        model: The model to train (weights will be modified in place).
        targets: List of target token IDs from the corpus.
        learning_rate: Step size for gradient descent.
        epochs: Number of complete passes through the training data.

    Returns:
        List of per-epoch metrics dictionaries containing epoch number,
        average loss, and accuracy.
    """
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for target_id in targets:
            # Forward pass: get probability distribution (same for all inputs).
            probs: list[float] = model.forward()

            # Compute loss: how surprised is the model by this token?
            loss: float = cross_entropy_loss(probs, target_id)
            total_loss += loss

            # Check if the model's top prediction matches the target.
            pred_id: int = argmax(probs)
            if pred_id == target_id:
                correct += 1

            # Backward pass: update the single row of weights.
            #
            # For softmax cross-entropy, the gradient is:
            #   gradient[j] = predicted_prob[j] - true_prob[j]
            #
            # This pushes probability mass toward frequently-seen tokens.
            row: list[float] = model.weights[0]  # unigram has only one row
            for j in range(model.vocab_size):
                y: float = 1.0 if j == target_id else 0.0
                grad: float = probs[j] - y
                row[j] -= learning_rate * grad

        # Compute epoch-level metrics.
        avg_loss: float = total_loss / len(targets) if targets else float("nan")
        accuracy: float = correct / len(targets) if targets else 0.0

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

    log_header(LOG, "Training Demo: Unigram (Frequency-Based) Model")

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

    # Step 4: Create training targets (just the tokens themselves for unigram).
    targets: list[int] = make_training_targets(token_ids)
    LOG.info(f"Created {len(targets)} training targets.")

    # Step 5: Initialize model (unigram has only 1 row of weights).
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        targets=targets,
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

    # Step 8: Qualitative check - what does the model predict?
    probs: list[float] = model.forward()
    best_id: int = argmax(probs)
    best_tok: str | None = vocab.get_id_token(best_id)
    LOG.info(
        f"After training, most likely token (based on frequency) "
        f"is {best_tok!r} (ID: {best_id})."
    )


if __name__ == "__main__":
    main()
