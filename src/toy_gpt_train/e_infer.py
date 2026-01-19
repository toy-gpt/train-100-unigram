"""e_infer.py - Inference module (artifact-driven).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load inspectable training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv
- Reconstruct a vocabulary-like interface and model weights
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection

Notes:
- This module does NOT retrain by default.
- If artifacts are missing, run d_train.py first.

Unigram inference:
    The model ignores all context and predicts based solely on
    corpus word frequencies. Every call to forward() returns the
    same probability distribution.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.math_training import argmax

__all__ = [
    "ArtifactVocabulary",
    "generate_tokens_unigram",
    "load_meta",
    "load_model_weights_csv",
    "load_vocabulary_csv",
    "parse_args",
    "require_artifacts",
    "top_k",
]

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

LOG: logging.Logger = get_logger("INFER", level="INFO")


@dataclass(frozen=True)
class ArtifactVocabulary:
    """Vocabulary reconstructed from artifacts/01_vocabulary.csv.

    Provides the same surface area used by inference:
    - vocab_size()
    - get_token_id()
    - get_id_token()
    - get_token_frequency()
    """

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    token_freq: dict[str, int]

    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def get_token_id(self, token: str) -> int | None:
        """Return the token ID for a given token, or None if not found."""
        return self.token_to_id.get(token)

    def get_id_token(self, idx: int) -> str | None:
        """Return the token for a given token ID, or None if not found."""
        return self.id_to_token.get(idx)

    def get_token_frequency(self, token: str) -> int:
        """Return the frequency count for a given token, or 0 if not found."""
        return self.token_freq.get(token, 0)


def require_artifacts(
    *,
    meta_path: Path,
    vocab_path: Path,
    weights_path: Path,
    train_hint: str,
) -> None:
    """Fail fast with a helpful message if artifacts are missing."""
    missing: list[Path] = []
    for p in [meta_path, vocab_path, weights_path]:
        if not p.exists():
            missing.append(p)

    if missing:
        LOG.error("Missing training artifacts:")
        for p in missing:
            LOG.error(f"  - {p}")
        LOG.error("Run training first:")
        LOG.error(f"  {train_hint}")
        raise SystemExit(2)


def load_meta(path: Path) -> JsonObject:
    """Load 00_meta.json."""
    with path.open("r", encoding="utf-8") as f:
        data: JsonObject = json.load(f)
    return data


def load_vocabulary_csv(path: Path) -> ArtifactVocabulary:
    """Load 01_vocabulary.csv -> ArtifactVocabulary."""
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    token_freq: dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"token_id", "token", "frequency"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"Unexpected vocabulary header. Expected {sorted(expected)} "
                f"but got {reader.fieldnames}"
            )

        for row in reader:
            token_id = int(row["token_id"])
            token = row["token"]
            freq = int(row["frequency"])

            token_to_id[token] = token_id
            id_to_token[token_id] = token
            token_freq[token] = freq

    return ArtifactVocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        token_freq=token_freq,
    )


def load_model_weights_csv(path: Path, vocab_size: int) -> list[list[float]]:
    """Load 02_model_weights.csv -> weights matrix.

    For unigram, expects exactly 1 row (context-independent predictions).
    """
    weights: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Weights CSV is empty.")
        if len(header) < 2 or header[0] != "input_token":
            raise ValueError("Weights CSV must start with header 'input_token'.")

        num_outputs = len(header) - 1
        if num_outputs != vocab_size:
            raise ValueError(
                f"Weights CSV output width mismatch. Expected {vocab_size} output columns "
                f"but found {num_outputs}."
            )

        for row in reader:
            if not row:
                continue
            if len(row) != vocab_size + 1:
                raise ValueError(
                    f"Invalid weights row length. Expected {vocab_size + 1} columns but found {len(row)}."
                )
            # row[0] is input token label; row[1:] are numeric weights
            weights.append([float(x) for x in row[1:]])

    # Unigram has exactly 1 row
    if len(weights) != 1:
        raise ValueError(
            f"Unigram weights CSV should have 1 row but found {len(weights)}."
        )

    return weights


def top_k(probs: list[float], k: int) -> list[tuple[int, float]]:
    """Return top-k (token_id, probability) pairs sorted by probability."""
    pairs: list[tuple[int, float]] = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def generate_tokens_unigram(
    model: SimpleNextTokenModel,
    vocab: ArtifactVocabulary,
    num_tokens: int,
) -> list[str]:
    """Generate tokens using unigram (no context - same prediction every time).

    Note: Unigram ignores all context, so there's no start token.
    Every generated token will be the same (the most frequent word).
    """
    generated: list[str] = []

    for _ in range(num_tokens):
        probs: list[float] = model.forward()  # no input - unigram ignores context
        next_id: int = argmax(probs)
        next_token: str | None = vocab.get_id_token(next_id)

        if next_token is None:
            LOG.error(f"Generated invalid token ID: {next_id}")
            break

        generated.append(next_token)

    return generated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference using saved training artifacts (unigram)."
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10,
        help="Number of tokens to generate (default: 10).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text (Unigram)")

    base_dir: Final[Path] = Path(__file__).resolve().parents[2]
    artifacts_dir: Final[Path] = base_dir / "artifacts"
    meta_path: Final[Path] = artifacts_dir / "00_meta.json"
    vocab_path: Final[Path] = artifacts_dir / "01_vocabulary.csv"
    weights_path: Final[Path] = artifacts_dir / "02_model_weights.csv"
    require_artifacts(
        meta_path=meta_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        train_hint="uv run python src/toy_gpt_train/d_train.py",
    )

    meta: JsonObject = load_meta(meta_path)
    vocab: ArtifactVocabulary = load_vocabulary_csv(vocab_path)

    v: int = vocab.vocab_size()
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=v)
    model.weights = load_model_weights_csv(weights_path, vocab_size=v)

    args: argparse.Namespace = parse_args()

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {v}")
    LOG.info("Unigram model: predictions are the same regardless of input.")

    # Show predictions (same for any input)
    probs: list[float] = model.forward()
    LOG.info("Top next-token predictions (based on corpus frequency):")
    for tok_id, prob in top_k(probs, k=max(1, args.topk)):
        tok: str | None = vocab.get_id_token(tok_id)
        LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    generated: list[str] = generate_tokens_unigram(
        model=model,
        vocab=vocab,
        num_tokens=max(0, args.num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
