"""math_training.py - Mathematical utilities used during model training.

This module contains reusable math functions used by the training and
inference code in this repository.

Scope:
- Pure functions with no model, vocabulary, or artifact assumptions.
- Reusable across unigram, bigram, and higher-context variants.

These functions are intentionally simple and explicit to support inspection
and debugging.
"""

import math

__all__ = ["argmax", "cross_entropy_loss"]


def argmax(values: list[float]) -> int:
    """Return the index of the maximum value in a list.

    Concept:
        argmax is the argument (index) at which a function reaches its maximum.

    Common uses:
        - Measuring accuracy during training (pick the most likely token)
        - Greedy decoding during inference (choose the top prediction)

    In training and inference:
        - A model outputs a probability distribution over possible next tokens.
        - The token with the highest probability is the model's most confident prediction.
        - argmax selects that token.

    Example:
        values = [0.1, 0.7, 0.2] has index values of 0,1, 2 respectively.
        argmax(values) -> 1 (since 0.7 is the largest value)

    Args:
        values: A non-empty list of numeric values (typically logits or probabilities).

    Returns:
        The index of the largest value in the list.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("argmax() requires a non-empty list")

    best_idx: int = 0
    best_val: float = values[0]

    for i in range(1, len(values)):
        v = values[i]
        if v > best_val:
            best_val = v
            best_idx = i

    return best_idx


def cross_entropy_loss(probs: list[float], target_id: int) -> float:
    """Compute cross-entropy loss for a single training example.

    Concept: Cross-Entropy Loss
        Cross-entropy measures how well a predicted probability distribution
        matches the true outcome.

        In next-token prediction:
        - The true distribution is "one-hot" which means we encode it as either 1 or 0:
            - Probability = 1.0 for the correct next token
            - Probability = 0.0 for all others
        - The model predicts a probability distribution over all tokens.

        Cross-entropy answers the question:
            "How well does the predicted probability distribution align with the true outcome?"

    Formula:
        loss = -log(p_correct)

        - If the model assigns high probability to the correct token,
          the loss is small.
        - If the probability is near zero, the loss is large.

    Numerical safety:
        log(0) is undefined, so we clamp probabilities to a small minimum
        (1e-12). This does not change learning behavior in practice,
        but prevents runtime errors.

    In training:
        - This loss value drives gradient descent.
        - Lower loss means better predictions.

    Args:
        probs: A probability distribution over the vocabulary (sums to 1.0).
        target_id: The integer ID of the correct next token.

    Returns:
        A non-negative floating-point loss value.
        - 0.0 means a perfect prediction
        - Larger values indicate worse predictions
    Raises:
        ValueError: If target_id is out of range for probs.
    """
    if target_id < 0 or target_id >= len(probs):
        raise ValueError(
            f"target_id out of range: target_id={target_id} len(probs)={len(probs)}"
        )

    p: float = probs[target_id]

    # Guard against log(0), which would produce -infinity
    p = max(p, 1e-12)

    return -math.log(p)
