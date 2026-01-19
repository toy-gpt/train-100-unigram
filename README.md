# Toy-GPT: train-100-unigram

[![PyPI version](https://img.shields.io/pypi/v/toy-gpt-train-100-unigram)](https://pypi.org/project/toy-gpt-train-100-unigram/)
[![Latest Release](https://img.shields.io/github/v/release/toy-gpt/train-100-unigram)](https://github.com/toy-gpt/train-100-unigram/releases)
[![Docs](https://img.shields.io/badge/docs-live-blue)](https://toy-gpt.github.io/train-100-unigram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![CI](https://github.com/toy-gpt/train-100-unigram/actions/workflows/ci-python-mkdocs.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-100-unigram/actions/workflows/ci-python-mkdocs.yml)
[![Deploy-Docs](https://github.com/toy-gpt/train-100-unigram/actions/workflows/deploy-mkdocs.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-100-unigram/actions/workflows/deploy-mkdocs.yml)
[![Check Links](https://github.com/toy-gpt/train-100-unigram/actions/workflows/links.yml/badge.svg)](https://github.com/toy-gpt/train-100-unigram/actions/workflows/links.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/toy-gpt/train-100-unigram/security)

> Demonstrates, at very small scale, how a language model is trained.

This repository is part of a series of toy training repositories plus a companion client repository:

- [**Training repositories**](https://github.com/toy-gpt) produce pretrained artifacts (vocabulary, weights, metadata).
- A [**web app**](https://toy-gpt.github.io/toy-gpt-chat/) loads the artifacts and provides an interactive prompt.

## Contents

- a small, declared text corpus
- a tokenizer and vocabulary builder
- a simple next-token prediction model
- a repeatable training loop
- committed, inspectable artifacts for downstream use

## Scope

This is:

- an intentionally inspectable training pipeline
- a next-token predictor trained on an explicit corpus

This is not:

- a production system
- a full Transformer implementation
- a chat interface
- a claim of semantic understanding

## Outputs

This repository produces and commits pretrained artifacts under `artifacts/`.

Training logs and evidence are written under `outputs/`
(for example, `outputs/train_log.csv`).

## Quick start

See `SETUP.md` for full setup and workflow instructions.

Run the full training script:

```shell
uv run python src/toy_gpt_train/d_train.py
```

Run individually:

- a/b/c are demos (can be run alone if desired)
- d_train produces artifacts
- e_infer consumes artifacts

```shell
uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py
```

## Provenance and Purpose

The primary corpus used for training is declared in `SE_MANIFEST.toml`.

This repository commits pretrained artifacts so the client can run
without retraining.

## Annotations

[ANNOTATIONS.md](./ANNOTATIONS.md) - REQ/WHY/OBS annotations used

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)

## SE Manifest

[SE_MANIFEST.toml](./SE_MANIFEST.toml) - project intent, scope, and role
