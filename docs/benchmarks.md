# Benchmarks and dashboard

The benchmark setup evaluates the teacher and base student before distillation, then reevaluates the teacher and evaluates the distilled student after training. The teacher stays untrained; its first score is the fixed gap-closure reference, and its second score measures drift. It does not start collection or training. Run each phase yourself when its models and data are ready. The dashboard reads completed evaluation artifacts; it never triggers inference.

## Public benchmark suite

| Benchmark | Capability | Dataset and split | Recorded quality metric |
| --- | --- | --- | --- |
| GSM8K | Multi-step arithmetic reasoning | `openai/gsm8k`, `main`, `test` | Numeric exact-match accuracy |
| ARC-Challenge | Grade-school science reasoning | `allenai/ai2_arc`, `ARC-Challenge`, `test` | Generated-choice accuracy |
| MMLU | Knowledge across 57 subjects, including computer science | `cais/mmlu`, `all`, `test` | Generated-choice accuracy; separate subject scores |
| TruthfulQA | Recognizing truthful answers to misconception-prone questions | Authors' `TruthfulQA.csv`, binary task added in 2025, all questions | Generated-choice accuracy with randomized answer positions |
| HumanEval | Python function synthesis | `openai/openai_humaneval`, `openai_humaneval`, `test` | Executable pass@1 |
| MBPP | Short Python programming tasks | `google-research-datasets/mbpp`, `sanitized`, `test` | Executable pass@1 on provided assertions and available challenge tests |

The [GSM8K authors](https://github.com/openai/grade-school-math) specify a final answer after `####`. The [ARC dataset](https://huggingface.co/datasets/allenai/ai2_arc) includes alphabetic and numeric original answer labels, which the freezer remaps to alphabetic labels. The [MMLU authors](https://github.com/hendrycks/test) provide subject-based knowledge questions. [TruthfulQA's authors](https://github.com/sylinrl/TruthfulQA) recommend the newer binary task over the original MC1 and MC2 tasks. Its two answers come directly from the authors' pinned CSV; the freezer does not invent distractors. See the [HumanEval authors](https://github.com/openai/human-eval) and [MBPP authors](https://github.com/google-research/google-research/tree/master/mbpp) for the original code tasks and tests.

All models receive the same prompts and exactly one completion per question. The multiple-choice tasks require the generated letter of the selected option. They do not compute answer likelihoods, ARC `acc_norm`, TruthfulQA MC1/MC2 probabilities, or standard few-shot leaderboard scores. This is a zero-shot generated-answer comparison for the hosted-teacher contract. Report the protocol with the scores rather than comparing them directly against incompatible leaderboard numbers.

HumanEval asks for the complete function, includes the original function prompt, and withholds the tests from the model. MBPP includes its public assertion examples in the prompt, as part of its task contract; available challenge tests stay outside the prompt. These suites evaluate Python. They do not measure other programming languages or repository-level software engineering.

## Install and freeze

Use Python 3.14 and `uv` for the benchmark tools, and `bun` for the dashboard. The local-model backend also needs the optional training stack. Coding evaluations require Docker and a locally available Python container image.

```bash
uv sync --all-packages --extra eval
uv run --extra eval distill benchmark freeze \
	--output data/benchmarks/suite.json \
	--limit 200 \
	--seed 42
```

`--limit` is a maximum per benchmark. Smaller datasets contribute all available questions. `--limit 0` uses the full chosen splits. The default suite includes all six benchmarks. To select a smaller scope, use `--benchmarks gsm8k,arc_challenge,mmlu,truthfulqa` or `--benchmarks humaneval,mbpp`. MMLU sampling spreads the limit across shuffled subjects; other benchmarks also spread it across their available categories. A limit smaller than the number of subjects cannot cover them all. Sampled scores describe that balanced subset, while full-test scores weight subjects by their actual question counts.

The freezer resolves Hugging Face dataset revisions to commit hashes and downloads the authors' TruthfulQA CSV at the pinned commit in `suites.py`. It records the repository, revision, configuration, split, license, available count, selected count, and source URL. A SHA-256 covers the source metadata, sampling seed, scorer version, and every question, prompt, choice, reference answer, and test. Loading a suite verifies its hash and item counts. An existing suite file cannot be overwritten. Freeze a new path when changing the protocol.

Keep the suite out of transfer training and model selection. `--limit 2` is enough to validate the import plumbing; it cannot support a model-quality conclusion. Use the full splits for final reporting. Benchmark overlap checks cover the supplied transfer corpora, not unknown pretraining data, paraphrases, or semantic duplicates.

## Run the comparison later

Review `configs/eval.example.toml`. Its four slots are `teacher`, `student_before`, `teacher_after`, and `student_after`. The `teacher` slot records the before phase. The included pairing uses the pinned Qwen3.5-9B teacher and Qwen3.5-2B-Base student. Local student runs use the same base revision, tokenizer, chat template, precision, four-bit setting, and `<|im_end|>` stopping token in both phases. Only `student_after` loads the adapter. Models load one at a time.

```bash
uv sync --all-packages --extra train --extra eval
docker pull python:3.12-slim

# Start your teacher separately, then record the baseline.
uv run --extra train distill benchmark run \
	--config configs/eval.example.toml \
	--stage before

# After you have completed training separately, run the second phase.
uv run --extra train distill benchmark run \
	--config configs/eval.example.toml \
	--stage after
```

`--stage before` evaluates `teacher` and `student_before`. `--stage after` evaluates `teacher_after` and `student_after`. `--stage all` evaluates all four slots when every required model is already available. Running the before phase before training preserves the base-student baseline. The teacher stays untrained. Its second evaluation checks changes in serving behavior or model identity across phases, separately from student gains.

The remote backends are `teacher_api`, for this repository's wrapper, and `openai`, for a chat-completions server. Set `base_url`, `model`, `revision`, and optionally `api_key_env`. The `revision` for a remote model is your declaration of the served snapshot; the evaluator cannot attest to the server's weights. It records the actual returned model identity and request ID. OpenAI-compatible endpoints must support `chat_template_kwargs.enable_thinking=false`; the teacher wrapper forwards this option. A local backend requires an immutable 40-character model commit. Optional `tokenizer_model` needs its own `tokenizer_revision`. An explicit `chat_template_path` must be identical in both student phases. The after adapter's manifest must match the base model, revision, and training file hashes. When its manifest records a chat-template hash, the local backend also verifies it.

Set `training_paths` to every transfer training and validation JSONL file when those files exist. The evaluator hashes them and checks their `prompt` or `messages` text for normalized exact benchmark questions, including a benchmark question followed by extra answer instructions. Adapter evaluation automatically adds the training and validation files named in its training manifest. Missing or changed files fail preflight. Without supplied corpora, the report explicitly notes that contamination checking is incomplete.

Coding evaluations resolve `code_image` to its immutable local Docker image ID before any inference request. Containers run with no network, a read-only root filesystem, no additional mounts, one CPU, 256 MiB memory, no added capabilities, 64-process limits, a non-root user, and a 16 MiB temporary filesystem. `code_timeout_seconds` limits the complete test container invocation, including startup. The runner removes timed-out containers. A normal exit without completed assertions does not pass. These limits are part of the protocol; a resource-heavy correct solution may fail them. Container infrastructure failures abort the run instead of becoming coding-quality failures.

Requests run sequentially. Generation uses temperature zero, a fixed seed, and the same `max_output_tokens` for every slot and benchmark. API failures count as failures and receive no hidden retries or replacement samples. Rerun an affected phase after fixing infrastructure; each rerun receives a new ID. The report selects the latest completed run for each slot. Results retain failed responses and successful earlier attempts in a sample journal if a run stops, but interrupted journals are not complete score reports and are not resumable.

## Quality metrics

Every quality score is on a 0–100 scale, where higher is better. A completed run must contain each frozen question exactly once. `n` is the complete question count in the benchmark or subject slice, including failed API requests, invalid outputs, truncations, and code test failures. `correct` is the count of successful, correctly scored answers.

| Recorded field | Formula or rule | Interpretation |
| --- | --- | --- |
| `scores[slot][benchmark].n` | Number of frozen questions in the benchmark | Denominator for all benchmark quality results |
| `correct` | Sum of per-question `correct` booleans | Integer count of correct answers or passing code submissions |
| `score` | `100 × correct / n` | Benchmark accuracy or coding pass@1; unavailable if the input set is empty |
| `ci95` | 95% Wilson interval for `correct` successes in `n` Bernoulli trials, multiplied by 100 | Sampling uncertainty for this question set; endpoints remain within 0–100 |
| `statuses` | Count of samples in each response or test status | Operational and output-format outcomes, not an alternative quality score |
| `slices[benchmark/category][slot]` | The same summary fields, restricted to the frozen category | MMLU subjects and other task categories; retain sample counts when interpreting small slices |
| `macro[slot]` | Arithmetic mean of that slot's benchmark `score` values | Equal benchmark weighting, independent of dataset size; no aggregate confidence interval is claimed |
| `gap_closure_percent` | `100 × (macro.student_after − macro.student_before) / (macro.teacher − macro.student_before)` | Fraction of the initial teacher-to-student quality gap closed; requires all three runs and a positive initial gap |

Gap closure can be negative after a regression and can exceed 100% when the student surpasses the initial teacher. It is unavailable when the initial gap is zero or negative, or required slots are absent. The dashboard summary always uses the entire suite. Benchmark filters affect the detailed charts, not those full-suite macro values.

For Wilson intervals, write `p = correct / n`, `z = 1.959963984540054`, and `d = 1 + z²/n`. The center is `(p + z²/(2n)) / d`; the radius is `z × sqrt(p(1−p)/n + z²/(4n²)) / d`. Multiply the clipped center-minus-radius and center-plus-radius endpoints by 100. Even all-correct and all-wrong samples retain uncertainty. Questions are treated as independent Bernoulli observations; correlated templates weaken that interpretation. These intervals do not measure variation across training seeds or served model versions.

### Per-question scoring

| Task metric | Accepted response and correctness rule |
| --- | --- |
| `numeric_exact_match` | GSM8K accepts a bare signed integer or decimal, or the numeric text after the final `####`. Correct thousands separators and whitespace are allowed. Decimal normalization makes `1200`, `1,200`, and `1200.00` equivalent. Explanations without the requested final marker, units after the number, malformed commas, and multiple final answers fail extraction. The evaluator never guesses from the last number in a rationale. |
| `choice_accuracy` | Trim surrounding whitespace, then require exactly one uppercase letter within the question's choice range. Compare that letter to the frozen reference. An explanation containing the right letter does not pass the output contract. |
| `pass_at_1` | Extract a single optional Python Markdown fence, parse the Python syntax, and run the candidate with the frozen assertions in the isolated container. HumanEval can also accept a function-body continuation after the original prompt. One candidate produces one binary success. With one candidate per task, pass@1 is the mean of those successes; no pass@k estimate is recorded. |

A wrong but well-formed general answer has `status="ok"` and `correct=false`. Coding test failure uses its own status. Truncated outputs score zero even if they contain the reference answer. Missing or nonterminal finish reasons also score zero. Responses containing `<think>` or `</think>` score zero under this non-thinking protocol.

| Sample `status` | Meaning | Quality contribution |
| --- | --- | --- |
| `ok` | Completed output passed the format contract; for coding, all tests passed | Correctness determines 1 or 0; coding OK is 1 |
| `invalid` | Unparseable answer, invalid Python syntax, missing or unsupported terminal reason, or a thinking trace | 0 |
| `truncated` | Provider reports `finish_reason="length"` | 0 |
| `error` | Inference request failed or returned an invalid provider response | 0; no provider error body is saved |
| `test_failed` | Candidate did not complete the assertions successfully within the container | 0 |
| `timeout` | Coding container exceeded its wall-clock budget | 0 |

Each `statuses` entry is a count, not a percentage. A status absent from the map has count zero. If a rate is needed, divide its count by the summary's `n`. Counts sum to `n`. Inspect API errors before interpreting low scores as model-quality failures.

## Paired comparison metrics

Pairing uses the exact question ID set, so shuffled result order does not change the comparison. Incompatible or incomplete sets cannot produce paired metrics. The available comparison groups are:

| Group | First run | Second run | Meaning of a positive `delta_pp` |
| --- | --- | --- | --- |
| `student_gain` | Student before | Student after | Distilled student scores higher |
| `teacher_drift` | Teacher before | Teacher after | Teacher scores higher on reevaluation |
| `teacher_student_before` | Teacher before | Student before | Base student scores higher than the teacher |
| `teacher_student_after` | Teacher after | Student after | Distilled student scores higher than the contemporaneous teacher |

Each group stores one comparison per benchmark. Define `aᵢ` and `bᵢ` as the first and second run's binary correctness on question `i`, then `dᵢ = bᵢ − aᵢ`.

| Recorded field | Definition | Units and missing-data behavior |
| --- | --- | --- |
| `n` | Number of matched frozen questions | Count; includes failed attempts |
| `delta_pp` | `100 × mean(dᵢ)` | Percentage points, not percent relative improvement |
| `ci95` | Percentile endpoints of 2,000 bootstrap resamples of the paired `dᵢ`, using seed 42 | Percentage points; the saved lower and upper entries are indices 49 and 1949 in the sorted bootstrap draws |
| `improved` | Count where the first run is wrong and the second is correct | Count |
| `regressed` | Count where the first run is correct and the second is wrong | Count |
| `both_correct` | Count where both runs are correct | Count |
| `both_wrong` | Count where both runs are wrong | Count; includes operational failures |
| `answer_agreement` | `100 × count(first prediction = second prediction) / agreement_n` | Percent, on parseable non-code answers only; unavailable when `agreement_n` is zero |
| `agreement_n` | Number of question pairs where both predictions are parseable, excluding code tasks | Count; report this coverage alongside agreement |

The four paired outcome counts sum to `n`, and `delta_pp = 100 × (improved − regressed) / n`. Bootstrap resampling uses equivalent multinomial draws of negative, zero, and positive differences rather than allocating full question arrays for every draw. It preserves pairing and remains reproducible. Small or constant difference sets can yield discrete or zero-width bootstrap intervals; these do not establish certainty beyond the evaluated sample. There are no p-values or claims of significance across multiple benchmark comparisons.

Teacher agreement measures output similarity, including shared wrong answers. It does not measure correctness, truthfulness, or distillation success. Code tasks have no prediction-equality agreement because different correct programs may implement the same function. Non-code agreement requires two parseable predictions; parseable content can still come from a response that failed the completion contract, so consult statuses as well.

## Latency and throughput metrics

The evaluator times the awaited generation call using a monotonic clock. The timer includes client overhead, network time, server queuing, prefill, and generation. It excludes model loading, answer scoring, Docker startup, and code tests. Local decoding happens inside the timed call. Successful requests with invalid or truncated outputs still contribute observed latency. Failed inference requests save elapsed attempt time per sample but do not enter summary latency or throughput.

| Recorded field | Definition | Units and interpretation |
| --- | --- | --- |
| `samples[].latency_seconds` | Monotonic elapsed time around one generation attempt | Seconds; saved even for inference errors |
| `latency_p50` | 50th percentile of generation times for non-error requests | Seconds; unavailable if there are no non-error requests |
| `latency_p95` | 95th percentile of the same times | Seconds; unavailable if there are no non-error requests |
| `tokens_per_second` | Sum of known output-token counts divided by the sum of generation times for those same non-error requests | Output tokens per second; end-to-end observed rate, not decoder-only speed |
| `token_coverage` | Number of non-error requests with known output-token counts divided by `n` | Fraction from 0–1; the dashboard displays it as a percentage |
| `samples[].usage.input_tokens` | Provider input-token count or the local tokenized prompt length | Tokens; null when the provider omits it |
| `samples[].usage.output_tokens` | Provider output-token count or the number of locally generated tokens | Tokens; null when the provider omits it |

Percentiles sort the observations and linearly interpolate at index `(count − 1) × quantile`. They are descriptive estimates without confidence intervals. A known count of zero tokens is still known usage; it is not treated as missing. A throughput result is unavailable when no usable timed requests have known output counts or their combined duration is zero. Local output counts include generated terminal tokens, while decoded answer text omits special tokens. Different providers may count tokens differently.

These sequential runs do not record requests per second under concurrency, time to first token, decoder-only speed, code-test duration, GPU peak memory, power usage, monetary cost, calibrated probabilities, or token-level distribution agreement. Missing measurements must not be inferred from the charts.

## Saved artifacts and provenance

Results live under `output_dir`, which defaults to `outputs/benchmarks`. Frozen suites and experiment outputs are ignored by version control. Run files contain responses and public test code, so keep output directories local unless you intend to share them.

| Artifact | Contents and use |
| --- | --- |
| `suite-<sha256>.json` | Copy of the verified frozen questions and source metadata; required to build comparisons |
| `<run-id>.samples.jsonl` | One appended sample per completed attempt; useful for inspecting interrupted runs; not a resumable or complete report |
| `<run-id>.json` | Completed evaluation run with configuration, provenance, and all question outcomes; dashboard input |
| Exported comparison JSON | Frozen suite, chosen completed runs, scores, subject slices, paired metrics, macro values, warnings, and answer details |

Every sample records `item_id`, `benchmark`, `category`, raw `response`, extracted `prediction`, binary `correct`, `status`, `latency_seconds`, `usage`, `finish_reason`, returned `served_model`, and provider `request_id` when available. Code predictions are `"passed"` for passing programs and null otherwise. Inference-error samples have empty response text, null predictions, and no token counts or served model. Error response bodies and API credentials are not recorded.

Each completed run records its schema `version`, unique `id`, `experiment`, `slot`, UTC `started_at` and `completed_at`, `suite_sha256`, `scorer_version`, `generation` settings, declared `model` configuration, resolved `code_image_id`, `code_timeout_seconds`, supplied `training_sha256`, local `artifact_sha256`, installed `packages`, operating-system `platform`, and Jujutsu `code_revision`. The adapter artifact hashes include its immediate files, such as weights, adapter configuration, tokenizer files, and training manifest. An explicit chat-template file also receives a hash. Environment variable names may appear in configuration; secret values do not.

`demo` distinguishes synthetic fixtures from measured runs. Synthetic and measured runs always form separate comparisons. The comparison `id` hashes the experiment name, suite hash, scorer version, generation configuration, code-image ID, code-test timeout, and demo flag. Model identity is retained per slot and differences are noted; student base or inference changes prevent attributing gains to distillation alone. Remote revision declarations need independent server-side verification. The code revision records the current Jujutsu working-copy commit ID, so reruns after source edits retain a different evaluation revision.

Training artifacts are checked against their manifest but do not become benchmark metrics. Training loss, validation loss, or best-checkpoint selection is evidence about optimization, not a held-out task quality score.

## Dashboard outline and commands

The dashboard uses React, shadcn-style controls, and Recharts. It ships a single dark theme. Every chart carries a title, a plain-language note explaining how to read it, a legend, labelled axes with units, and a hover tooltip with exact values.

Color identifies the model: teacher is blue and student is orange. Phase also uses shape: before runs have outlined translucent bars, dashed lines, or hollow dots; after runs are solid.

The page opens with a one-sentence result summary and six headline numbers: student before and after, student change in percentage points, teacher before and after, and teacher gap closed. A sticky toolbar holds the view tabs, a benchmark-scope filter, and per-run toggles. The views are:

- Accuracy: benchmark accuracy bars with Wilson intervals, student gain and teacher drift bars with paired bootstrap intervals, question-by-question outcome stacks (gained, lost, right both times, wrong both times), student–teacher answer agreement, and a per-subject heatmap table.
- Speed: median response time, response-time spread, accuracy versus median latency, and generation speed in tokens per second, plus a timing and reliability table with p95, token coverage, and non-scorable answers.
- Answers: question search, outcome filters, the reference answer, and every run's response, parsed answer, status, and latency, with the official tests for coding tasks.
- Method: scoring rules, dataset sources and licenses, model revisions, adapters, run timestamps, generation settings, checked training corpora, and run notes.

Every Speed chart has a **Log scale** switch, because response times and throughput are positive and can differ by multiples between the teacher and student. In log mode, axes use 1-2-5 ticks and the axis title says "log scale". The response-time spread switches to geometric bins. Accuracy, change, and outcome charts stay linear. Accuracy is bounded at 0–100%, change can be negative, and stacked shares would be distorted on a log axis.

Experiment selection separates incompatible suites and protocols. Missing phases stay unavailable rather than becoming zeros. Full-suite macro summaries remain visible when detail charts are filtered. The dashboard can export the selected complete comparison as JSON.

```bash
cd apps/dashboard
bun install
bun run build
cd ../..
uv run distill benchmark dashboard
```

Open `http://127.0.0.1:8002`. The server binds to loopback. `GET /api/experiments` lists compatible comparisons and corrupt-run notices; `GET /api/reports/<comparison-id>` returns the report. For dashboard development, run the API and `bun run dev` in separate terminals. Vite proxies `/api` to port 8002.

Preview the dashboard without training, inference, or Docker:

```bash
uv run distill benchmark demo
uv run distill benchmark dashboard --results-dir outputs/benchmark-demo
```

The preview creates explicitly labeled synthetic scores in a separate directory and refuses to overwrite existing artifacts. It is a UI fixture, not a benchmark result.

For a JSON report without the dashboard:

```bash
uv run distill benchmark report --results-dir outputs/benchmarks
uv run distill benchmark report \
	--results-dir outputs/benchmarks \
	--comparison COMPARISON_ID \
	--output outputs/comparison.json
```

## Test the setup

```bash
uv run pytest
uv run ruff check .
uv run mypy apps packages tests
cd apps/dashboard
bun test src
bun run build
```

The automated checks use fixed trusted fixtures and mocked inference. They cover answer extraction, failure denominators, Wilson intervals, paired comparisons, protocol separation, suite integrity, training overlap, adapter provenance, coding assertion completion, container isolation flags, the read-only API, and chart data with missing phases. They do not load model weights or run distillation. Actual GPU inference, Docker runtime behavior, and final model scores require your later evaluation runs.
