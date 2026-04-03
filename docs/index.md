# Toy GPT Training Repository

This repo is an example of a pre-trained next-token prediction model,
illustrating how language models are trained and used.

## Navigation

- `README.md` in the repository root for the home page
- `SE_MANIFEST.toml` for intent, scope, and declared training corpus

## py.typed (PEP 561)

Python projects may include a `py.typed` marker file (defined by PEP 561).

Type checkers (Pyright, Mypy) trust inline type hints when this marker is present.
The file is typically empty; comments are allowed.

## API Reference

Selected public APIs are provided below.

### Training Math

::: toy_gpt_train.math_training

### Models

::: toy_gpt_train.c_model

### Training Pipeline

::: toy_gpt_train.d_train

### Inference

::: toy_gpt_train.e_infer

### Artifacts and I/O

::: toy_gpt_train.io_artifacts
