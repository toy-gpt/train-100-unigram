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
from toy_gpt_train.d_train import argmax

LOG: logging.Logger = get_logger("INFER", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"
META_PATH: Final[Path] = ARTIFACTS_DIR / "00_meta.json"
VOCAB_PATH: Final[Path] = ARTIFACTS_DIR / "01_vocabulary.csv"
WEIGHTS_PATH: Final[Path] = ARTIFACTS_DIR / "02_model_weights.csv"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


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


def require_artifacts() -> None:
    """Fail fast with a helpful message if artifacts are missing."""
    missing: list[Path] = []
    for p in [META_PATH, VOCAB_PATH, WEIGHTS_PATH]:
        if not p.exists():
            missing.append(p)

    if missing:
        LOG.error("Missing training artifacts:")
        for p in missing:
            LOG.error(f"  - {p}")
        LOG.error("Run training first:")
        LOG.error("  uv run python src/toy_gpt_train/d_train.py")
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

    Expected shape:
    - one row per input token
    - one column per output token (after the first 'input_token' column)
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

    if len(weights) != vocab_size:
        raise ValueError(
            f"Weights CSV row count mismatch. Expected {vocab_size} rows but found {len(weights)}."
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
    start_token: str,
    num_tokens: int,
) -> list[str]:
    """Generate tokens using a unigram context (current token only)."""
    generated: list[str] = [start_token]
    current_id: int | None = vocab.get_token_id(start_token)

    if current_id is None:
        LOG.error(f"Start token not in vocabulary: {start_token!r}")
        return generated

    for _ in range(num_tokens):
        probs: list[float] = model.forward(current_id)
        next_id: int = argmax(probs)
        next_token: str | None = vocab.get_id_token(next_id)

        if next_token is None:
            LOG.error(f"Generated invalid token ID: {next_id}")
            break

        generated.append(next_token)
        current_id = next_id

    return generated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Toy GPT inference from saved artifacts."
    )
    parser.add_argument(
        "--start",
        dest="start_token",
        default="",
        help="Start token for generation. If omitted, uses the first token in the vocabulary.",
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
        help="Show top-k next-token probabilities for the start token.",
    )
    return parser.parse_args()


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text")

    require_artifacts()

    meta = load_meta(META_PATH)
    vocab = load_vocabulary_csv(VOCAB_PATH)

    model = SimpleNextTokenModel(vocab_size=vocab.vocab_size())
    model.weights = load_model_weights_csv(WEIGHTS_PATH, vocab_size=vocab.vocab_size())

    args = parse_args()

    # Choose a start token.
    start_token = args.start_token
    if not start_token:
        # Deterministic fallback: smallest token_id present
        first_id = min(vocab.id_to_token.keys())
        start_token = vocab.id_to_token[first_id]

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {vocab.vocab_size()}")
    LOG.info(f"Start token: {start_token!r}")

    start_id = vocab.get_token_id(start_token)
    if start_id is not None:
        probs: list[float] = model.forward(current_id=start_id)
        LOG.info(f"Top next-token predictions after {start_token!r}:")
        for tok_id, prob in top_k(probs, k=max(1, args.topk)):
            tok = vocab.get_id_token(tok_id)
            LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    generated = generate_tokens_unigram(
        model=model,
        vocab=vocab,
        start_token=start_token,
        num_tokens=max(0, args.num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info("  " + " ".join(generated))


if __name__ == "__main__":
    main()
