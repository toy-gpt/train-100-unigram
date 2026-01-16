# Training Artifacts

This directory contains inspectable artifacts produced by the training pipeline.
These files are written by `d_train.py` and consumed by `e_infer.py`.

They are intended for transparency, inspection, and comparison across model variants.
They are not optimized for performance or storage efficiency.

## Artifact Files

### 00_meta.json - Training metadata

Describes the training run and its inputs, including:

- repository and model kind
- corpus path and hash (the training text)
- vocabulary size
- training hyperparameters (learning rate and number of training passes)
- list of generated artifacts (files written by the training run)

This file provides provenance and context for all other artifacts.

### 01_vocabulary.csv - Symbol system

Lists the discrete tokens observed in the corpus, including:

- token ID
- token string
- frequency in the corpus

This defines the symbol system used by the model.

### 02_model_weights.csv - Instrument parameters

The learned weight matrix for next-token prediction.

- Rows correspond to input tokens
- Columns correspond to output tokens
- Values are raw learned weights

This file represents the trained model parameters.

### 03_token_embeddings.csv - Observable structure

A simple 2D projection derived from the learned weights.

For levels 100–400, this is a visualization-friendly view intended for
plotting and qualitative inspection.
In later model levels, embeddings become a first-class learned structure.

## Notes

- Artifacts are deterministic given the corpus and hyperparameters.
- All files are designed to be human-readable and diffable (changes can be inspected line-by-line using standard version control tools).
