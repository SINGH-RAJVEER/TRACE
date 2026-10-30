# Evaluating an account-level harness against multi-domain model distillation

This project studies whether TRACE (Traffic-based Response Access Control for Extraction), an account-level harness, stops useful distillation from a language-model API without blocking legitimate users. One gateway serves AG News, SQuAD 1.1, GSM8K, and Python coding through a single teacher. Each attack campaign trains one student on every response it receives, and the student is scored on all four domains.

## Setup

Enter `devenv shell`, which provides Python 3.14, uv, Bun, just, Tectonic, and the Linux coding sandbox. Then:

```bash
cp .env.example .env    # set TEACHER_TERMS_REVIEW_ID
just setup
```

## Run

```bash
just start      # full study: calibration, defended and undefended campaigns, training, evaluation
just distill    # undefended collection and shared student training only
just diagnose   # replay one extractor against benign calibrations
```

Both launchers default to four domains, seeds 42/1/20, and 6,000 submissions per campaign. Outputs go to `artifacts/`. For a teacher on another machine:

```bash
just teacher-host                                  # on the teacher machine
just distill --teacher-url http://TEACHER_IP:8000  # on the student machine
```

## Evaluate

```bash
just benchmark configs/eval.ag_news.toml before
just benchmark configs/eval.ag_news.toml after
just dashboard-build
just dashboard --results-dir outputs/tasks/ag_news   # http://127.0.0.1:8002
just benchmark-demo                                  # simulated preview, no models
```

## Develop

```bash
just test
just check
just format
just paper      # compile paper/main.pdf
```

## Documentation

- [Shared study protocol](docs/multi-domain-study.md)
- [Launcher options](docs/run-scripts.md)
- [Metrics and denominators](docs/benchmarks.md)
- [Harness](docs/harness.md)
- [Coding task](docs/coding-task.md)
- [Manuscript](paper/main.tex)

Passing tests and configured experiments do not establish defense effectiveness. No teacher-student result has been measured yet.
