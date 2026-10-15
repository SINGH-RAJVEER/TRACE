# Prompt corpus

The teacher only transfers what the prompts ask about, so the prompt set decides what the student learns. `distill corpus build` assembles that set from pinned public datasets and prompts written for this repository. It writes the prompts with their grouped splits, plus an answer key that `distill review` uses to reject wrong or malformed teacher answers before training.

```bash
uv sync --all-packages --extra eval
uv run --extra eval distill corpus build --profile standard
# or: just corpus standard
```

The command writes three files to `data/prompts/`, which is gitignored:

| File | Contents |
| --- | --- |
| `all-splits.jsonl` | Prompt records in the collector's input format, with `train` or `validation` splits |
| `references.jsonl` | One answer key per prompt, keyed by `example_id` |
| `manifest.json` | Profile, seed, per-source counts and filter reasons, dataset revisions and licenses, the held-out benchmark sources, and SHA-256 hashes of the outputs |

The build is deterministic. With the same profile, seed, and `data/authored/prompts.jsonl`, it writes identical files, so an interrupted collection can resume against a rebuilt corpus. Changing any of those inputs changes some prompts; start a new ledger when that happens, because resume rejects prompts that differ from the ledger.

## Profiles

| Profile | Prompts | Train | Validation | Estimated collection time on an RTX 4000 Ada (Qwen3-14B teacher) | Estimated training time (Qwen3-4B student) |
| --- | --- | --- | --- | --- | --- |
| `quick` | 7,208 | 6,952 | 256 | 4 to 8 hours | 1 to 2 hours |
| `standard` | 36,104 | 34,740 | 1,364 | 20 to 40 hours | 4 to 8 hours |
| `full` | 75,944 | 73,147 | 2,797 | 45 to 85 hours | 9 to 16 hours |

The counts were measured. The times are estimates, not measurements, made for the earlier Qwen3-14B teacher and Qwen3-4B-Base student: an FP8 teacher at `--concurrency 8`, an average of about 250 answer tokens per prompt (multiple-choice letters are a few tokens, open-ended answers several hundred), and a rank-128 4B adapter. The current Qwen3.5-9B teacher and Qwen3.5-2B-Base student are smaller and the teacher serves more requests at once, so expect both phases to take less time, but they have not been re-estimated. Time a `quick` run to plan a larger one. Collection resumes after interruption, so it can run over several sessions.

`standard` is the intended run. `quick` checks the whole pipeline on real data in an evening. `full` takes every eligible prompt from the smaller sources and larger samples from the big ones.

## Sources

| Family | Source | License | Prompts in `standard` | Answer key |
| --- | --- | --- | --- | --- |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main`, train split | MIT | 7,469 | Final number after `####` |
| `aqua_rat` | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat) `raw`, train and validation | Apache-2.0 | 3,000 | Choice letter |
| `arc_challenge` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Challenge`, train and validation | CC BY-SA 4.0 | 1,404 | Choice letter |
| `arc_easy` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Easy`, train and validation | CC BY-SA 4.0 | 2,815 | Choice letter |
| `commonsense_qa` | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa), train and validation | MIT | 3,000 | Choice letter |
| `mbpp` | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) `full`, train, validation, and prompt splits | CC BY 4.0 | 466 | Assertion tests |
| `self_oss_instruct` | [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | ODC-By | 4,000 | Python code must parse |
| `dolly` | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | 6,000 | None |
| `oasst2` | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), English opening prompts | Apache-2.0 | 5,000 | None |
| `authored` | `data/authored/prompts.jsonl`, written for this repository | Apache-2.0 | 238 | Mixed |
| `python_functions` | Generated from 52 function specifications | Apache-2.0 | 104 | Assertion tests |
| `word_problems` | Generated from 21 arithmetic templates | Apache-2.0 | 1,200 | Final number after `####` |
| `json_extraction` | Generated from 3 record templates | Apache-2.0 | 500 | Exact JSON object |
| `text_operations` | Generated from 8 list and text operations | Apache-2.0 | 500 | Exact answer |
| `constrained_writing` | Generated from 102 topics and 4 format rules | Apache-2.0 | 408 | Word, bullet, and phrase constraints |

Every Hub dataset is pinned to a commit in `DATASETS` in `packages/distillation/src/distillation/corpus.py`, and the manifest records each revision and license. Only prompts are used; the teacher writes every answer. Check the licenses again before publishing a model or dataset. The CC BY-SA sources require attribution and share-alike terms for derived datasets.

Sources were chosen to cover the six evaluated capabilities without touching their held-out splits:

- Math reasoning: GSM8K train and AQuA-RAT, plus generated word problems with computed answers.
- Science and commonsense multiple choice: ARC train and validation, and CommonsenseQA.
- Python: MBPP's non-test splits, self-oss-instruct, and generated function tasks whose tests come from reference implementations.
- Instruction following and general chat: Dolly, OpenAssistant, the authored prompts, and the constrained-writing and extraction families. The student is a base model, so it learns the chat format and following instructions entirely from these.

Sources left out:

- MMLU, TruthfulQA, and HumanEval: the benchmark uses their test splits, or all of TruthfulQA, and the other splits are too small or are the benchmark itself.
- The MATH competition dataset: it was withdrawn from the Hub.
- `HuggingFaceH4/no_robots`: its license is non-commercial.
- OpenBookQA: its Hub card declares no license.
- Corpora generated by commercial APIs whose terms restrict training competing models.

## How prompts are built

- **Evaluation formats.** GSM8K-style prompts end with the benchmark's instruction to finish with `####` and the number. About half the ARC and CommonsenseQA prompts use the benchmark's "Return only the letter of the correct answer." The rest ask for brief reasoning followed by an `Answer: X` line, which teaches both the short format the scorer parses and the reasoning behind it. AQuA uses the reasoning form for 85% of prompts. Generated Python tasks come in a HumanEval-style form (complete the signature and docstring) and an MBPP-style form (a description plus visible assertions). The variant is chosen by a seeded hash of the example ID.
- **Overlap filtering.** The builder downloads every held-out question from the six benchmarks, about 17,700, not only the sampled frozen suite. It drops any prompt that contains a held-out question shorter than 13 words or shares a 13-word span with a longer one, after the same whitespace and case normalization the evaluator's training audit uses. MBPP task IDs are also checked against the sanitized test split. The `standard` build dropped 165 prompts this way. The check cannot detect paraphrases.
- **Length.** Prompts longer than 3,000 characters are dropped (`--max-prompt-chars`). The longest remaining prompt is 1,026 student tokens and the median is 76, so a prompt plus a 1,024-token answer fits the 3,072-token training limit and a 4,096-token vLLM context.
- **Deduplication.** Prompts whose normalized text matches an earlier prompt are dropped.
- **Splits.** Each prompt goes to `validation` when a seeded hash of its `(corpus, group_id)` falls in the lowest 4% (`--validation-fraction`), otherwise to `train`. Groups keep related prompts together: one OpenAssistant conversation tree, one MBPP task, one generated function in both prompt styles, or one generation template. Because a template is one group, a whole generated family can land in one split. Preparation still enforces the split boundary.
- **Sampling.** When a family has a cap, the builder keeps the first eligible prompts in a seeded hash order, so the sample does not depend on the source's row order.

Use `--families gsm8k,authored` to build a subset and `--seed` for a different sample. The same seed always produces the same corpus.

## Authored and generated prompts

`data/authored/prompts.jsonl` holds 238 hand-written prompts in 15 categories: rewriting, summarizing and reading comprehension over passages written for this file, drafting messages, explanations, logic puzzles, math, coding help (debugging, SQL, regex, shell, and other languages), structured output, classification with fixed labels, planning, creative writing, format following, multi-turn conversations, and requests the assistant should decline or redirect safely. Each row has an `id`, a `task`, either `user` text or a `messages` list ending with a user turn, an optional `system` message, an optional `group`, and an optional `reference`:

```json
{"id": "logic-001", "task": "logic", "user": "Ana, Ben, and Caro each own exactly one pet ... Who owns the cat? Return only the name.", "reference": {"kind": "exact", "answer": "Caro"}}
```

Add rows to this file to extend the corpus. Avoid questions from the evaluated benchmarks and close paraphrases of them, because the overlap filter only catches shared word spans.

The generated families in `packages/distillation/src/distillation/synthetic.py` compute their answers instead of storing them. Python function tests come from running reference implementations, word-problem answers from the template arithmetic, and JSON targets from the values written into the text. Template prompts add volume with checkable answers rather than diversity, so they stay under a tenth of the corpus.

## Answer keys and review

`references.jsonl` holds one record per prompt:

| `kind` | Accepted when |
| --- | --- |
| `numeric` | The number after the last `####` equals `answer` |
| `choice` | The whole response is the correct letter |
| `choice_reasoning` | The last `Answer: X` line names the correct letter |
| `python_tests` | The code parses, defines `entry_point`, and, with `--code-image`, passes `tests` in Docker |
| `python_code` | The first code block, or the whole response, parses as Python |
| `json_equals` | The response parses as JSON equal to `answer` |
| `json_valid` | The response parses as JSON |
| `exact` | The response matches `answer`, ignoring case, surrounding quotes, backticks, a final period, and spacing around commas |
| `general` | Optional `constraints` hold: `min_words`, `max_words`, `bullet_count`, `must_include`, `must_exclude` |

The launchers build the corpus and run the review automatically when given `--corpus`; see [run-scripts.md](run-scripts.md#full-run-with-the-prompt-corpus). To run it by hand, see [distillation-pipeline.md](distillation-pipeline.md#collection-and-preparation). Review rejects teacher answers that are wrong against the key, so the student never trains on them. Expect a share of the AQuA and CommonsenseQA prompts to be rejected, because those datasets contain some incorrect keys and the teacher will sometimes disagree with them. The review summary reports rejections by reason and accepted records by task, which shows how much of each capability survived.
