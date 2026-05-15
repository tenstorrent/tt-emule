# Security Policy

## Reporting a Vulnerability

If you find a security issue in tt-emule, please report it privately
rather than opening a public issue.

The preferred channel is **GitHub Private Vulnerability Reporting**:

1. Go to the repository on GitHub.
2. Click the **Security** tab.
3. Click **Report a vulnerability**.

If GitHub PVR is unavailable, email **ospo@tenstorrent.com** instead.
Include a description of the issue, steps to reproduce, and the
affected version or commit SHA. Encrypted mail is welcome; ask for a
PGP key in your initial mail if you need one.

Please do not include personally identifying information for third
parties in your report.

## Response Process

You should receive an acknowledgement within **3 business days**. We
will work with you to confirm the issue, determine impact, and
coordinate a fix.

We aim to disclose and ship a fix within **90 days** of the initial
report. Reporters who request credit will be acknowledged in release
notes; reporters who prefer to remain anonymous will be respected.

## Scope

In scope:

- Bugs in tt-emule code (this repository) that affect correctness,
  isolation, or safety of code executed by the emulator.
- Build / packaging issues that could allow supply-chain compromise of
  downstream consumers.

Out of scope (please use regular issues instead):

- Bugs in upstream `tt-metal`, `tt-mlir`, `tt-umd`, or other
  Tenstorrent repositories. Report those to the respective project.
- Test failures that don't indicate a security vulnerability.
- Theoretical issues without a demonstrated impact.

## Supported Versions

Security fixes are applied to the `main` branch. We do not currently
maintain a stable release line; users should consume tt-emule by
pinning to a specific commit and updating as needed.
