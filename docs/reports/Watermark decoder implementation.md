# Watermark decoder implementation for TRACE

Status: the keyed KGW core and detector now exist as a CPU-only prototype, described in [the watermark probe](../watermark-probe.md). Everything below that needs a serving engine, a model or a GPU remains unexecuted.

Research-only decision, 2026-11-05. Workspace `/home/rajveer/.t3/worktrees/TRACE/t3-1b07588c`, existing base `2b426d3d633a18674ca1317a7be4d054409ad580`. [Full cited research and gaps](../research_notes/Watermark%20decoder%20implementation/research.md).

## Recommended first prototype

First qualify vLLM `v0.31.0` native Gumbel watermarking on Qwen/Qwen3.5-9B with a shared nonsecret fixture key, nonzero-temperature sampling and speculative decoding disabled. It has an actual sampler and detector in official tagged code, which makes it the smallest credible decoder-control probe. This is a feasibility experiment, not a production recommendation for tenant attribution. Native SynthID Text is explicitly planned rather than implemented. [Native documentation](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/features/watermarking.md), [official exports](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/watermarking/__init__.py).

Then compare a keyed soft KGW processor with gamma 0.25, delta 0.5/1/2 and context widths 1/2/4 on the pinned MRV2 interface. This is the recommended first custom implementation if per-tenant keys and controllable bias strength are requirements. Maintain a reference detector and explicit algorithm version. HF's seeded green-list implementation and a new GPU keyed-hash implementation are different compatibility profiles. [KGW authors](https://github.com/jwkirchenbauer/lm-watermarking), [HF pinned implementation](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/logits_process.py), [MRV2 guide](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/custom_logitsprocs.md).

## Hard compatibility blockers

- TRACE has no serving-version lock or local serving environment in this workspace. The Python 3.14 app lock contains Transformers 5.19.0 and Torch 2.14.1 for optional embedding work. Candidate vLLM 0.31.0 requires Transformers >=5.10.4,<5.18.0 and pins CUDA Torch 2.13.0. Use its own Python 3.12 uv project/lock; do not install it into the app environment. Python 3.14 itself is not universally unsupported by this newer vLLM tag. [Local lock](../../uv.lock), [environment](../../devenv.nix), [vLLM requirements](https://github.com/vllm-project/vllm/blob/v0.31.0/requirements/common.txt), [CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.31.0/requirements/cuda.txt), [Python metadata](https://github.com/vllm-project/vllm/blob/v0.31.0/pyproject.toml).
- The current adapter does not send watermark controls, custom xargs, token-ID requests or streaming fields. The request defaults to temperature zero. Native Gumbel skips greedy output. A separately named stochastic experiment is necessary; silently changing historical generation would invalidate the comparison. [Adapter](../../packages/distillation/src/distillation/teacher.py), [schemas](../../packages/distillation/src/distillation/schemas.py), [native behavior](https://docs.vllm.ai/en/v0.31.0/features/watermarking/).
- The teacher endpoint has shared authentication and no trusted tenant key identity. Native watermark keys are engine-level. Efficient tenant-specific attribution needs a trusted selector and custom generation support, or costly separate key-bound engines. [Endpoint](../../apps/teacher-api/src/teacher_api/app.py), [native configuration](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/watermarking.py).
- vLLM's plugin API differs between older V1 and MRV2 and carries a change warning. Select the exact runner and version before writing code. HF `LogitsProcessor(input_ids,scores)` is not a compatible vLLM callback. [V1 guide](https://github.com/vllm-project/vllm/blob/v0.22.0/docs/features/custom_logitsprocs.md), [MRV2 guide](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/custom_logitsprocs.md).
- No GPU/model verification occurred. Syntax, model-family support and official code are insufficient to establish this exact model revision, native sampler, flags, driver and batching combination works. [Qwen serving documentation](https://huggingface.co/Qwen/Qwen3.5-9B), [current launch command](../../scripts/study.sh).

## Technical design and fallback

Keep generation-side model dependencies in the separate serving environment. A later adapter should transmit only a trusted profile/key ID, with the server resolving secrets locally. Bind generation and detection to algorithm version, tokenizer revision/hash, template hash, special-token policy, key version and generation config. Retain retired detector keys for historical samples and bind a key version across retries. Never reuse API tokens or put watermark secrets in manifests. This is proposed technical design, not existing behavior.

For a custom MRV2 processor, maintain request state by slot and map each logits row through the step context. Reset recycled slots. Preserve grammar masks and disabled rows. Test batching, cancellation and reordering before model evaluation. Implement streaming at sampling time, with one state per logical request. TRACE currently returns whole responses. For older V1, state updates and `is_argmax_invariant=False` are separate obligations. [MRV2 guide](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/custom_logitsprocs.md), [V1 guide](https://github.com/vllm-project/vllm/blob/v0.22.0/docs/features/custom_logitsprocs.md).

If vLLM cannot qualify, use a separate HF Qwen text-generation backend and HF `WatermarkingConfig` or `SynthIDTextWatermarkingConfig`. HF support does not make vLLM support automatic. Google's reference mixins target Gemma/GPT-2 and its present package adds JAX/Flax detector dependencies; evaluate that separately. A direct SynthID vLLM port must preserve processing order, repeated-context behavior and detector equivalence. [Google integration](https://ai.google.dev/responsible/docs/safeguards/synthid), [reference code](https://github.com/google-deepmind/synthid-text), [tournament specification](https://github.com/google-deepmind/synthid-text/blob/main/docs/tournament.md).

Kuditipudi's distortion-free sampler is a worthwhile research comparator but requires control over sampling randomness and alignment detection. SemStamp requires sentence rejection and embeddings, so defer it for TRACE's mixed code/math/JSON workload and streaming requirements. Headers, metadata and prompt rewrites do not replace decoder watermarking. [Distortion-free paper](https://arxiv.org/abs/2307.15593), [SemStamp](https://arxiv.org/abs/2310.03991).

## Proposed commands and evidence contract

The following commands are future implementation acceptance specifications. The `watermark_probe` module, fixture paths and new tests do not exist yet. Do not run them as present repository commands. All model/GPU commands need a later authorized evaluation stage; none ran during this research.

Existing environment checks for that stage, before any model download:

```bash
mkdir -p artifacts/watermark
uv pip freeze --python .venv-vllm/bin/python > artifacts/watermark/runtime-freeze.txt
.venv-vllm/bin/vllm --version > artifacts/watermark/vllm-version.txt
.venv-vllm/bin/vllm serve --help > artifacts/watermark/serve-help.txt
```

Proposed native fixture-key smoke command after a frozen compatible environment and approved model availability. `42` is public test material only; never substitute a production secret in a logged command:

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
	--revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
	--host 127.0.0.1 --port 8001 --language-model-only \
	--max-model-len 8192 --gpu-memory-utilization 0.90 \
	--default-chat-template-kwargs '{"enable_thinking": false}' \
	--watermark-config '{"algorithm":"gumbel","key":42,"context_width":4,"deduplicate_contexts":"single_turn"}'
```

A future probe must explicitly send temperature 0.7 or 1 for the native-positive cell and zero for its documented skip control. Also compare a stochastic unwatermarked cell at identical temperature and truncation settings. Capture watermark application/skip status, not merely a 200 response. Source for the native startup flag: [tagged guide](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/features/watermarking.md).

Proposed evaluation command contract, using a separate uv project and nonsecret fixtures. The future fixture manifest must contain frozen prompt identities and source-disjoint splits, not hydrated historical readings:

```bash
uv run --project experiments/watermark --frozen python -m watermark_probe vectors --fixture fixtures/watermark/reference.json --output artifacts/watermark/vectors.json
uv run --project experiments/watermark --frozen python -m watermark_probe replay --manifest fixtures/watermark/prompts.jsonl --profiles fixtures/watermark/profiles.json --temperatures 0,0.7,1 --lengths 16,32,64,128,256,512,1024 --concurrency 1,4,16 --output artifacts/watermark/generations.jsonl
uv run --project experiments/watermark --frozen python -m watermark_probe calibrate --split calibration --generations artifacts/watermark/generations.jsonl --familywise-alpha 0.001 --output artifacts/watermark/calibration.json
uv run --project experiments/watermark --frozen python -m watermark_probe evaluate --split test --calibration artifacts/watermark/calibration.json --generations artifacts/watermark/generations.jsonl --output artifacts/watermark/report.json
```

Proposed artifact schema, with tabs as indentation. These are field contracts, not observed data:

```text
manifest.json
	schema_version, run_id, created_at, workspace_base, serving_lock_sha256
	model_id, model_revision, tokenizer_revision, tokenizer_files_sha256
	chat_template_sha256, special_token_ids, thinking, effective_generation_config
	vllm_version, source_commit, runner, torch_version, transformers_version
	gpu_model, driver, cuda_version, attention_backend, dtype, quantization
	algorithm_id, algorithm_version, processor_digest, prf_version
	key_id, key_epoch, context_width, dedup_policy, gamma, delta
	fixture_manifest_sha256, split_hashes, seed_schedule, concurrency, batch_invariance
	secret_material: forbidden

generations.jsonl
	request_id, prompt_id, split, task, profile_id, key_id, seed, retry_id
	actual_parameters, prompt_token_ids, generated_token_ids, exact_text_sha256
	text_only_roundtrip_match, finish_reason, output_tokens, scored_tokens
	watermark_application_status, skip_reason, stream_mode, batch_condition
	ttft_ms, inter_token_latency_ms, latency_ms, tokens_per_second, peak_memory_bytes
	utility_score, exact_answer_match, code_pass_at_1, json_valid, error

detections.jsonl
	request_id, detector_version, candidate_key_id, algorithm_profile_id
	text_transform, attack_budget, scored_tokens, unique_contexts, green_count
	z_score_or_native_statistic, raw_p_value, adjusted_p_value, tested_hypotheses
	decision: detected | not_detected | insufficient_evidence | unsupported
	boundary_policy, normalization_version, retained_utility

calibration.json and report.json
	frozen_thresholds, minimum_effective_lengths, held_out_split_hash
	fpr, tpr, confidence_intervals, coverage, abstentions, per_task_and_length_results
	teacher_utility_deltas, throughput_and_latency_deltas, failures, limitations
```

Measure exact-token and text-only detection separately. Exclude prompt, padding and special tokens under a frozen rule; record whether initial contexts require the prompt. Deduplicate correlated events, correct multiple key/window/profile tests, and abstain for insufficient effective length. A proposed familywise alpha of 1e-3 is an evaluation target, not a measured property. With zero false positives, roughly 3/n is the upper 95% binomial bound; a 1e-5 claim needs roughly 300,000 independent null cases even before multiplicity. Do not count repeated answers as independent evidence.

Require paired task utility and confidence intervals for AG News, SQuAD, GSM8K, executable code and JSON/short-answer stress cases. Report original greedy, stochastic control and stochastic marked results separately. Greedy KGW can change answers; native Gumbel skips greedy. Grammar validity does not ensure task correctness, and fully forced output offers no token-choice channel. No sample-length floor or overhead number is yet qualified.

Attack evaluation must include paraphrase, translation, truncation, mixed text, code formatting/comment stripping, wrong keys and spoofing, with retained utility and attack budgets. Student inheritance needs a separate matched-training study on disjoint prompts, including unwatermarked controls and different contamination fractions. Teacher-output detection is not a student-membership test. Inheritance can help detection and also enable imitation; no universal attribution or extraction-prevention claim follows. [Reliability](https://arxiv.org/abs/2306.04634), [stealing](https://arxiv.org/abs/2402.19361), [qualified removal limits](https://arxiv.org/abs/2311.04378), [radioactivity](https://arxiv.org/abs/2402.14904), [distillation spoofing](https://arxiv.org/abs/2510.10987).

The deliverable is a research recommendation and an unexecuted evaluation contract. No implementation, GPU or model evidence exists for this proposal.
