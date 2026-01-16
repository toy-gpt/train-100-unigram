"""math_training.py - Mathematical utilities used during model training.

This module contains reusable mathematical functions that appear
throughout the training process of language models.

Common themes:
- These functions are not specific to any one model.
- They are reused unchanged across unigram, bigram, and higher-context models.
- Keeping them here avoids duplication and keeps training code readable.

As models become more complex (embeddings, attention, batching),
these core ideas remain the same.
"""

import math

__all__ = ["argmax", "cross_entropy_loss"]


def argmax(values: list[float]) -> int:
    """Return the index of the maximum value in a list.

    Concept:
        argmax is the argument (index) at which a function reaches
        its maximum value.

    In training and inference:
        - A model outputs a probability distribution over possible next tokens.
        - The token with the highest probability is the model's most confident prediction.
        - argmax selects that token.

    Example:
        values = [0.1, 0.7, 0.2] has index values of 0,1, 2 respectively.
        argmax(values) -> 1 (since 0.7 is the largest value)

    This is used for:
        - Measuring accuracy during training
        - Greedy decoding during inference

    Args:
        values: A list of numeric values (typically probabilities).

    Returns:
        The index of the largest value in the list.

    Raises:
        ValueError: If the list is empty.

    """
    best_idx: int = 0
    best_val: float = values[0]

    for i in range(1, len(values)):
        if values[i] > best_val:
            best_val = values[i]
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
    """
    p: float = probs[target_id]

    # Guard against log(0), which would produce -infinity
    p = max(p, 1e-12)

    return -math.log(p)
