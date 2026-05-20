# Contributing to tt-emule

Thanks for your interest in tt-emule. Contributions of any size — bug
reports, doc fixes, tests, new features — are welcome.

## Reporting issues

Use [GitHub Issues](https://github.com/tenstorrent/tt-emule/issues) for
bug reports, feature requests, and questions. Search first to avoid
duplicates.

For **security** issues, do **not** open a public issue. See
[SECURITY.md](SECURITY.md).

## Submitting a pull request

1. Fork the repo and create a topic branch off `main`.
2. Make focused commits. Each commit should compile and pass tests where
   possible.
3. Add or update tests for any behavior change.
4. Run the regression suites locally before pushing — see
   [README.md](README.md) for the build commands and
   `run_regression.sh` / `run_d2m_regression.sh` for the test drivers.
5. Open a PR against `main`. Fill in the PR template.

### CI

Two pipelines run on every PR:

- **PR Regression** — builds tt-metal against your PR's tt-emule and runs
  the gtest unit-tests suite.
- **PR D2M Regression** — builds tt-mlir against your PR's tt-emule and
  runs the D2M pytest suite.

Both must be green before merge. New failures must either be fixed or
added to `.github/known-failures.txt` / `.github/known-failures-d2m.txt`
with justification in the PR description.

### Review

A maintainer will review your PR. Expect feedback within a few business
days. Be responsive to comments; reviewers may push small touch-ups
directly if the change is otherwise good.

## Coding conventions

- C++ targets clang-20, C++20.
- New source files must carry SPDX headers — see existing files for the
  format (REUSE 3.0):
  ```cpp
  // SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
  //
  // SPDX-License-Identifier: Apache-2.0
  ```
- Match the surrounding style; don't reformat unrelated code in the same
  PR.

## Adding third-party dependencies

tt-emule has no vendored third-party code today. If you need to add a
dependency:

- Confirm the dependency's license is **Apache-2.0 compatible** (MIT,
  BSD-2/3, ISC, and Apache-2.0 itself are fine; GPL/AGPL/LGPL are not).
- Note the dependency, its source, version, and license in the PR
  description.
- A maintainer will perform a license-compatibility review before merge.

## Contributor license terms

By submitting a PR, you agree that your contribution is licensed under
the Apache License 2.0 (see [LICENSE](LICENSE)). You confirm that you
have the right to submit the contribution under that license.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
Reports of unacceptable behavior should go to ospo@tenstorrent.com.
