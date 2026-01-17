"""io.py - Input/output and training-artifact utilities used by the models.

This module is responsible for persisting and describing the results of
model training in a consistent, inspectable format.

It does not perform training.
It:
- Writes artifacts produced by training (weights, vocabulary, logs, metadata)
- Enforces a fixed repository layout for reproducibility
- Provides small helper utilities shared across training and inference

The directory structure is intentionally fixed:
- artifacts/ contains all inspectable model outputs
- corpus/ contains exactly one training text file
- outputs/ contains training logs and diagnostics

External callers should treat paths as implementation details and interact
only through the functions provided here.


Concepts
--------

Artifact
    A concrete file written to disk that captures some aspect of training.
    In this project, artifacts are designed to be:
    - Human-readable (CSV / JSON)
    - Stable across model variants (unigram, bigram, context-3, etc.)
    - Reusable by inference without retraining

Epoch
    One epoch is one complete pass through all training examples.
    Training typically consists of multiple epochs so the model can
    gradually improve its predictions by repeatedly adjusting weights.

Training Log
    A CSV file recording per-epoch metrics such as:
    - average loss
    - accuracy
    This allows learning behavior to be inspected after training.

Vocabulary
    A mapping between token strings and integer token IDs.
    The vocabulary defines:
    - the size of the model output space
    - the meaning of each row and column in the weight tables

Row Labeler
    A small function that maps a numeric row index in the model's weight table
    to a human-readable label.
    For example, as the number of context tokens increases, the row labeler
    produces context strings such as:
    - unigram:        "cat"
    - bigram:         "the|cat"
    - context-3:      "the|black|cat"

    Row labels are written into CSV artifacts to make model structure visible.

Model Weights
    Numeric parameters learned during training.
    Conceptually:
    - each row corresponds to an input context
    - each column corresponds to a possible next token
    Weights are written verbatim so learning can be inspected or reused.

Token Embeddings (Derived)
    A simple 2D projection derived from model weights for visualization.
    These are not learned embeddings yet.
    In later stages (500+), embeddings become first-class learned parameters.

Reproducibility Metadata
    The 00_meta.json file records:
    - which corpus was used
    - how it was hashed
    - which model variant was trained
    - what training settings were applied
    This allows results to be traced and compared across runs and repositories.


Design Notes
------------
- This module is shared unchanged across model levels (100-400).
- More advanced pipelines (embeddings, attention, batching) build on
  the same artifact-writing concepts.
- Centralizing I/O logic prevents drift across repositories
  and keeps training code focused on learning.
"""

from collections.abc import Callable
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Final, Protocol

from datafun_toolkit.logger import get_logger

from toy_gpt_train.c_model import SimpleNextTokenModel

__all__ = [
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "RowLabeler",
    "VocabularyLike",
    "find_single_corpus_file",
    "repo_name_from_base_dir",
    "sha256_of_bytes",
    "sha256_of_file",
    "write_artifacts",
    "write_meta_json",
    "write_model_weights_csv",
    "write_token_embeddings_csv",
    "write_training_log",
    "write_vocabulary_csv",
]

LOG: logging.Logger = get_logger("IO", level="INFO")


type RowLabeler = Callable[[int], str]
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class VocabularyLike(Protocol):
    """Protocol for vocabulary-like objects used in training and artifacts."""

    def vocab_size(self) -> int:
        """Return the total number of unique tokens in the vocabulary."""
        ...

    def get_token_id(self, token: str) -> int | None:
        """Return the integer ID for a given token, or None if not found."""
        ...

    def get_id_token(self, idx: int) -> str | None:
        """Return the token string for a given integer ID, or None if not found."""
        ...

    def get_token_frequency(self, token: str) -> int:
        """Return the frequency count for a given token."""
        ...


_BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
_OUTPUTS_DIR: Final[Path] = _BASE_DIR / "outputs"
_ARTIFACTS_DIR: Final[Path] = _BASE_DIR / "artifacts"
_META_PATH: Final[Path] = _ARTIFACTS_DIR / "00_meta.json"
_VOCAB_PATH: Final[Path] = _ARTIFACTS_DIR / "01_vocabulary.csv"
_WEIGHTS_PATH: Final[Path] = _ARTIFACTS_DIR / "02_model_weights.csv"
_EMBEDDINGS_PATH: Final[Path] = _ARTIFACTS_DIR / "03_token_embeddings.csv"


def find_single_corpus_file(corpus_dir: Path) -> Path:
    """Find the single corpus file in corpus/ (same rule as SimpleTokenizer)."""
    if not corpus_dir.exists():
        msg = f"Corpus directory not found: {corpus_dir}"
        raise FileNotFoundError(msg)

    files = [p for p in corpus_dir.iterdir() if p.is_file()]
    if len(files) == 0:
        msg = f"No files found in corpus directory: {corpus_dir}"
        raise FileNotFoundError(msg)
    if len(files) > 1:
        msg = f"Expected exactly one file in corpus directory, found {len(files)}: {corpus_dir}"
        raise ValueError(msg)

    return files[0]


def repo_name_from_base_dir(base_dir: Path) -> str:
    """Infer repository name from base directory."""
    return base_dir.resolve().name


def sha256_of_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest for given bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 digest for a file."""
    data = path.read_bytes()
    return sha256_of_bytes(data)


def write_artifacts(
    *,
    base_dir: Path,
    corpus_path: Path,
    vocab: VocabularyLike,
    model: SimpleNextTokenModel,
    model_kind: str,
    learning_rate: float,
    epochs: int,
    row_labeler: RowLabeler,
) -> None:
    """Write all training artifacts to artifacts/.

    Args:
        base_dir: Repository base directory.
        corpus_path: Corpus file used for training.
        vocab: VocabularyLike instance.
        model: Trained model (weights already updated).
        model_kind: Human-readable model kind (e.g., "unigram", "bigram").
        learning_rate: Training learning rate.
        epochs: Number of training passes.
        row_labeler: Function that maps a model weight-row index to a label
            written in the first column of 02_model_weights.csv.
    """
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    write_vocabulary_csv(_VOCAB_PATH, vocab)
    write_model_weights_csv(_WEIGHTS_PATH, vocab, model, row_labeler=row_labeler)
    write_token_embeddings_csv(_EMBEDDINGS_PATH, model, row_labeler=row_labeler)
    write_meta_json(
        _META_PATH,
        base_dir=base_dir,
        corpus_path=corpus_path,
        vocab_size=vocab.vocab_size(),
        model_kind=model_kind,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def write_meta_json(
    path: Path,
    *,
    base_dir: Path,
    corpus_path: Path,
    vocab_size: int,
    model_kind: str,
    learning_rate: float,
    epochs: int,
) -> None:
    """Write 00_meta.json describing corpus, model, and training settings.

    This file is the authoritative, human-readable summary of a training run.
    It records:
    - what corpus was used
    - what model architecture was trained
    - how training was configured
    - which artifacts were produced

    The intent is transparency and reproducibility.

    Args:
        path: Output JSON path (artifacts/00_meta.json).
        base_dir: Repository base directory.
        corpus_path: Corpus file used for training.
        vocab_size: Number of unique tokens.
        model_kind: Human-readable model kind
            (e.g., "unigram", "bigram", "context2", "context3").
        learning_rate: Training learning rate.
        epochs: Number of epochs (full passes over the training pairs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    repo_name = repo_name_from_base_dir(base_dir)
    corpus_rel = str(corpus_path.resolve().relative_to(base_dir.resolve()))
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_lines = [ln for ln in corpus_text.splitlines() if ln.strip()]

    meta: JsonObject = {
        "repo_name": repo_name,
        "model_kind": model_kind,
        "vocab_size": vocab_size,
        "training": {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "epoch_definition": (
                "One epoch is a complete pass through all training pairs. "
                "Each pair contributes one gradient update."
            ),
        },
        "corpus": {
            "path": corpus_rel,
            "filename": corpus_path.name,
            "sha256": sha256_of_file(corpus_path),
            "num_lines": len(corpus_lines),
            "num_chars": len(corpus_text),
            "description": (
                "The corpus is tokenized sequential text. "
                "Training pairs are derived by sliding a fixed-size context window "
                "over the token stream."
            ),
        },
        "artifacts": {
            "00_meta.json": path.name,
            "01_vocabulary.csv": _VOCAB_PATH.name,
            "02_model_weights.csv": _WEIGHTS_PATH.name,
            "03_token_embeddings.csv": _EMBEDDINGS_PATH.name,
        },
        "concepts": {
            "token": "An atomic symbol produced by the tokenizer (e.g., a word).",
            "vocabulary": (
                "The set of all unique tokens observed in the corpus, "
                "mapped to integer IDs."
            ),
            "context": (
                "The fixed number of preceding tokens used as input "
                "to predict the next token."
            ),
            "softmax": (
                "A function that converts raw scores into probabilities "
                "that sum to 1.0."
            ),
            "cross_entropy_loss": (
                "A measure of how well the predicted probability distribution "
                "matches the correct next token."
            ),
            "gradient_descent": (
                "An optimization process that incrementally adjusts weights "
                "to reduce prediction error."
            ),
        },
        "notes": [
            "This is an intentionally inspectable training pipeline.",
            "Models are trained using softmax regression with cross-entropy loss.",
            "Weights are updated incrementally via gradient descent.",
            "Token embeddings are a derived 2D projection for visualization only "
            "in levels 100-400.",
            "In later stages (500+), embeddings are a learned parameter table.",
        ],
    }

    # WHY: Ensure the file always ends with a newline so pre-commit's
    # end-of-file-fixer does not modify generated artifacts in CI.
    rendered = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")

    LOG.info(f"Wrote meta to {path}")


def write_model_weights_csv(
    path: Path,
    vocab: VocabularyLike,
    model: SimpleNextTokenModel,
    *,
    row_labeler: RowLabeler,
) -> None:
    """Write 02_model_weights.csv with token-labeled columns.

    Shape:
        - first column: input_token
        - remaining columns: one per output token (header names are tokens)

    Args:
        path: Output CSV path.
        vocab: Vocabulary instance (must provide vocab_size(), get_id_token()).
        model: Trained model (must provide vocab_size and weights).
        row_labeler: Function that maps a model weight-row index to a label
            written in the first column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    out_tokens: list[str] = []
    for j in range(vocab.vocab_size()):
        tok = vocab.get_id_token(j)
        out_tokens.append(tok if tok is not None else f"id_{j}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_token"] + out_tokens)

        for row_idx, row in enumerate(model.weights):
            input_token = row_labeler(row_idx)
            writer.writerow([input_token] + [f"{w:.8f}" for w in row])

    LOG.info(f"Wrote model weights to {path}")


def write_token_embeddings_csv(
    path: Path,
    model: SimpleNextTokenModel,
    *,
    row_labeler: RowLabeler,
) -> None:
    """Write 03_token_embeddings.csv as a simple 2D projection for plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "label", "x", "y"])

        for row_idx, row in enumerate(model.weights):
            x = row[0] if len(row) >= 1 else 0.0
            y = row[1] if len(row) >= 2 else 0.0
            writer.writerow([row_idx, row_labeler(row_idx), f"{x:.8f}", f"{y:.8f}"])


def write_training_log(path: Path, history: list[dict[str, float]]) -> None:
    """Write per-epoch training metrics to a CSV file.

    Args:
        path: Output file path.
        history: List of per-epoch metrics dictionaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = ["epoch", "avg_loss", "accuracy"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "avg_loss": f"{row['avg_loss']:.8f}",
                    "accuracy": f"{row['accuracy']:.6f}",
                }
            )

    LOG.info(f"Wrote training log to {path}")


def write_vocabulary_csv(path: Path, vocab: VocabularyLike) -> None:
    """Write 01_vocabulary.csv: token_id, token, frequency.

    Args:
        path: Output CSV path.
        vocab: Vocabulary instance (must provide vocab_size(), get_id_token(), get_token_frequency()).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "frequency"])

        for token_id in range(vocab.vocab_size()):
            token = vocab.get_id_token(token_id)
            if token is None:
                continue
            freq = vocab.get_token_frequency(token)
            writer.writerow([token_id, token, freq])

    LOG.info(f"Wrote vocabulary to {path}")
