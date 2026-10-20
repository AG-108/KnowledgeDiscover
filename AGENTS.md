# AGENTS.md

## Project Context

This repository implements a benchmark for ODE/PDE
symbolic equation discovery.

The goal is not merely to run algorithms, but to
produce scientifically valid, fair, and reproducible
comparisons suitable for academic publication.

## General Principles

- Prefer minimal, verifiable changes.
- Understand existing implementations before modifying them.
- Do not silently change experimental protocols.
- Do not modify unrelated modules unless necessary.
- Distinguish implemented features from validated features.
- Never fabricate tests, results, or completion claims.

## Scientific Integrity

For any change affecting experiments:

- Identify its impact on benchmark fairness.
- Explain changes to preprocessing or evaluation.
- Preserve reproducibility through configs and logs.
- Distinguish original baseline behavior from adaptations.
- Report assumptions and limitations explicitly.

## Task Completion Protocol

After completing a substantive task, report:

1. What changed, and why?
2. Which files and functions were modified?
3. How does the implementation work?
4. What tests or experiments were actually executed?
5. Which claims are verified, and which remain uncertain?
6. How could this change affect scientific conclusions?

Provide file paths, commands, and evidence.

## Researcher Handoff

For substantive algorithm, evaluation, or experiment
changes, create or update docs/agent_handoff.md.

Write for a researcher who does not have access to
the current conversation.

Explain important concepts in clear technical language.
Avoid unexplained jargon.

Include evidence references and open questions.

Do not claim scientific correctness based solely on
successful code execution.
