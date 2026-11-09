# v7 student benchmark panel

Validated recorded inference on the same frozen 746-item GSM8K development suite. No new inference, training or teacher requests were made to assemble this report.

| Model / condition | Training examples | Correct | Accuracy | Wilson 95% interval |
|---|---:|---:|---:|---:|
| base |  | 386 / 746 | 51.74% | [48.16, 55.31]% |
| teacher |  | 619 / 746 | 82.98% | [80.11, 85.50]% |
| A2/full | 193 | 527 / 746 | 70.64% | [67.28, 73.80]% |
| A2/instantaneous | 95 | 528 / 746 | 70.78% | [67.41, 73.93]% |
| A5/full | 197 | 523 / 746 | 70.11% | [66.73, 73.28]% |
| A5/instantaneous | 98 | 548 / 746 | 73.46% | [70.18, 76.50]% |
| unprotected | 1473 | 548 / 746 | 73.46% | [70.18, 76.50]% |

## Scope and uncertainty

- Recorded inference is reused; this report does not perform a fresh inference run.
- One training seed (42), GSM8K development suite only; not a full-test leaderboard result.
- Intervals condition on the trained models and resample items, not training or campaign seeds.
- Items may share source families; item-level intervals can understate dependence.
- Arms differ in training membership, size and campaign history; comparisons are descriptive.
- Paired intervals are pointwise. McNemar p-values include Holm adjustment across all pairs.
- No matched quota students or general distillation-prevention qualification are provided.

All 21 pairwise comparisons are retained in student-panel.json, including paired item-bootstrap intervals, exact McNemar p-values and Holm-adjusted p-values. Positive differences favor the first named model. These are exploratory comparisons.

## Reproduction

Run `uv run python -m paper.figures.student_panel` with the original local artifacts, or `uv run python -m paper.figures.student_panel --verify paper/figures/student-panel.json` to verify every statistic from the published correctness vectors alone. The snapshot contains ordered item IDs and artifact hashes, without prompts, completions, account files, logs or credentials.
