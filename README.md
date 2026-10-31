# Evaluating TRACE against extraction campaigns

This workspace evaluates how an account-level harness restricts access to a teacher model across extraction strategies and corpora, and how it affects legitimate clients. It measures requests, released responses and tokens, escalation, and service costs.

The supported domains are AG News, SQuAD, GSM8K, and coding. Campaign schedules cover multiplexing, sequential domains, cover traffic, slow collection, account splitting, and independent sources. The default campaign budget is 12,000 HTTP attempts.

## Setup

Enter `devenv shell`, copy `.env.example` to `.env`, set the teacher settings and review identifier, and run `uv sync --all-packages --extra eval --extra embed`. A vLLM installation and GPU are needed to serve the teacher. Encoder dependencies are needed for MMD calibration and replay.

## Run

See [the evaluation workflow](docs/evaluation.md) for deployment, corpus preparation, calibration, protected campaign collection, and interpretation of results. Inspect available options with `uv run distill --help`, `uv run distill gateway --help`, and `uv run harness --help`.

Existing observations and their limitations are in [readings](readings/v4/README.md). The harness limits what an account can receive; that is not evidence that it prevents distillation. See [what budgets establish](docs/harness.md#what-budgets-establish).
