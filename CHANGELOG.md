# Changelog

All notable changes to this project will be documented in this file.

The format is based on **[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)**
and this project adheres to **[Semantic Versioning](https://semver.org/spec/v2.0.0.html)**.

## [Unreleased]

### Added

- (placeholder) Notes for the next release.

---

## [0.9.6] - 2026-01-18

### Updated

- Unigram predicts simply based on frequency.

---

## [0.9.5] - 2026-01-18

### Updated

- Extracted prompts

---

## [0.9.3] - 2026-01-17

### Updated

- Paths updated for reuse

---

## [0.9.1] - 2026-01-16

### Updated

- EOL in meta JSON

---

## [0.9.0] - 2026-01-16

### Added

- Initial public release
- CI/CD: GitHub Actions for checks, docs, and automated releases

---

## Notes on versioning and releases

- We use **SemVer**:
  - **MAJOR** – breaking schema/OpenAPI changes
  - **MINOR** – backward-compatible additions
  - **PATCH** – clarifications, docs, tooling
- Versions are driven by git tags via `setuptools_scm`. Tag `vX.Y.Z` to release.
- Docs are deployed per version tag and aliased to **latest**.
- Sample commands:

```shell
# if deleting is required
git tag -d v0.9.0
git push origin :refs/tags/v0.9.0
# adding a tag example
git tag v0.9.0 -m "0.9.0"
git push origin v0.9.0
```

[Unreleased]: https://github.com/toy-gpt/train-100-unigram/compare/v0.9.6...HEAD
[0.9.6]: https://github.com/toy-gpt/train-100-unigram/releases/tag/v0.9.6
[0.9.5]: https://github.com/toy-gpt/train-100-unigram/releases/tag/v0.9.5
[0.9.3]: https://github.com/toy-gpt/train-100-unigram/releases/tag/v0.9.3
[0.9.1]: https://github.com/toy-gpt/train-100-unigram/releases/tag/v0.9.1
[0.9.0]: https://github.com/toy-gpt/train-100-unigram/releases/tag/v0.9.0
