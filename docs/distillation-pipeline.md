# Distillation pipeline

## Scope

This repository implements a black-box teacher to open-weight student experiment. The teacher is a hosted chat model available through an OpenAI-compatible `POST /chat/completions` endpoint. The student is a Hugging Face causal language model trained with LoRA or QLoRA.

The current method is sequence-level distillation. The collector saves the teacher's generated text, and TRL trains the student with cross-entropy on completion tokens. It does not use teacher logits.

The protection wrapper discussed in the paper is outside this implementation.

## Adopted model pairing

- Student: `Qwen/Qwen3-4B-Base` at revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` (Apache 2.0, verified 2026-09-05).
- Teacher: `Qwen/Qwen3-14B` instruct at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, run in non-thinking mode.

Both are Qwen3, so they share the tokenizer and chat template and there is no cross-family template mismatch. The 4B to 14B gap leaves real headroom for the student to close while the 4B trains comfortably under QLoRA on a 16GB card. The 14B teacher serves at roughly 28GB in BF16 or 10GB quantized, so one 24GB card with a Q4 quant handles collection. Re-verify both revisions against the Hub before a paper run; model repos can move.

Self-hosting the teacher with vLLM keeps provider terms out of the picture entirely:

```bash
vllm serve Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
	--host 127.0.0.1 --port 8001
```

Point `TEACHER_UPSTREAM_BASE_URL` at `http://127.0.0.1:8001/v1`. After starting it, confirm responses contain no thinking traces before the main collection run, and keep transfer prompts inside the student's 32K native context.

## Authorization gate

Do not collect data until the hosted provider has explicitly authorized using its output to train the planned student. Output ownership does not necessarily grant model-training rights.

Record the approval, contract amendment, or provider permission in `TEACHER_TERMS_REVIEW_ID`. The collector separately requires `--terms-review-id` and writes it into every ledger record. This is an audit reference, not an enforcement mechanism.

The prompt source also requires a corpus name, source ID, and license. The example fixture uses text authored for this repository under Apache-2.0.

## Teacher inference API

Configure `.env` from `.env.example`, then run:

```bash
uv sync --all-packages
uv run teacher-api
```

The service exposes:

- `GET /healthz`, which reports `ready` or `unconfigured`.
- `POST /v1/teacher/generate`, which validates a provider-neutral request and calls the configured hosted teacher.
- `/docs`, FastAPI's generated OpenAPI interface.

Set `TEACHER_API_TOKEN` to require a bearer token on generation requests. Bind to `127.0.0.1` by default. Add network TLS and an authenticated gateway before exposing the service outside a trusted machine.

Example request:

```json
{
	"messages": [
		{"role": "system", "content": "Answer accurately and concisely."},
		{"role": "user", "content": "Why is the sky blue?"}
	],
	"temperature": 0,
	"max_output_tokens": 256,
	"seed": 42
}
```

The upstream adapter sends `max_output_tokens` as the widely supported `max_tokens` chat-completions field. Add a provider adapter if the chosen authorized provider uses a different protocol. API keys remain in the API process and are never returned to collectors.

## Prompt input

The collector accepts JSONL. Each row has a stable ID, a split assigned before teacher collection, task metadata, source rights, and structured messages:

```json
{
	"example_id": "support-0001",
	"split": "train",
	"task": "support-answer",
	"source": {
		"corpus": "licensed-support-v1",
		"source_id": "row-1",
		"license": "internal-approved"
	},
	"prompt": [
		{"role": "user", "content": "How do I reset the device?"}
	]
}
```

Use grouped train, validation, and test splits. Related templates, documents, users, or generated variants belong in the same split. Do not collect training responses for the frozen test split.

## Collection and preparation

Run the teacher API first. Then collect responses:

```bash
uv run distill collect \
	--input data/examples/prompts.jsonl \
	--output data/teacher/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--concurrency 4 \
	--max-attempts 3
```

Collection is resumable by `example_id`. It rejects duplicate input IDs, retries teacher failures, and appends each successful response from its worker, so a later failure does not discard completed results. The ledger records teacher identity, the actual upstream endpoint, request parameters, provider request ID, token usage, source rights, approval reference, finish reason, and basic checks. A sibling `ledger.attempts.jsonl` file records every successful and failed call, including prompts whose retries all failed. Test prompts are skipped unless `--include-test` is set for a controlled teacher evaluation.

The basic checks reject blank content and nonterminal finish reasons. Passing responses remain `pending`; the collector does not call them training-ready. Store review decisions in a separate JSONL file so the raw ledger stays unchanged:

```json
{"example_id":"support-0001","status":"accepted","checks":["schema","grounding","human_review"],"reviewer":"reviewer-1"}
```

Task validators and human review must produce these decisions before a paper run. For example, compile generated code, validate structured output against a schema, check citations against supplied text, and audit every task slice.

Project accepted training rows into TRL's conversational prompt-completion schema:

```bash
uv run distill prepare \
	--input data/teacher/ledger.jsonl \
	--reviews data/reviews/reviews.jsonl \
	--output data/processed/train.jsonl \
	--split train
```

Preparation keeps reviewed and accepted rows from the requested split. It drops duplicate prompts within the split and prompts found in another split. It refuses to prepare the frozen test split. `--accept-pending` bypasses review for a local plumbing test and should not be used to produce paper results. Raw ledger files and prepared datasets are ignored by version control because they can contain licensed, private, or costly generated data.

## Student training

Install the optional training stack:

```bash
uv sync --all-packages --extra train
```

Review `configs/train.example.toml`. The included 0.5B checkpoint makes a cheap plumbing test, not a strong student. Pin `base_revision` to an immutable Hugging Face commit for a paper run. Confirm that the base model license allows fine-tuning and the intended distribution.

Start training:

```bash
uv run --extra train distill train --config configs/train.example.toml
```

The trainer:

- Loads conversational JSONL with `datasets`.
- Uses completion-only loss through TRL `SFTTrainer`.
- Applies LoRA to all linear layers.
- Uses NF4, double quantization, and BF16 compute when `qlora = true`.
- Saves the adapter, tokenizer, chat template state, configuration, dataset SHA-256, and package versions.

The Qwen example sets `eos_token = "<|im_end|>"` to match its chat template rather than the pretrained tokenizer's `<|endoftext|>` token.

Set `qlora = false` for BF16 or FP16 LoRA when memory permits. Change `precision` to match the GPU. Four-bit training requires CUDA and bitsandbytes. CPU execution is suitable for API and pipeline tests, not the example QLoRA run.

If the base tokenizer has no chat template, set `chat_template_path` to a reviewed template supported by TRL. Treat that template as part of the trained artifact. The trainer tokenizes every record before training and stops if any sequence exceeds `max_length`; filter those records or increase the limit rather than truncating teacher answers.

## Evaluation

Freeze evaluation prompts before collection. Compare the base student, trained adapter, and authorized teacher under the same task contract. Use executable task metrics where possible. Report quality by task and source, safety failures, truncation, latency, throughput, and peak memory.

Do not use training loss as the main result. Sequence-level distillation aims to transfer behavior on held-out prompts. Add DPO only after SFT if a measured failure has reliable preference pairs. The current code intentionally stops at the SFT baseline.

## Reproducibility limits

Provider models can change behind aliases. Pin a dated snapshot where available and preserve provider request IDs. Hosted generation may remain nondeterministic with a fixed seed. The local training manifest captures package versions and dataset content hash, but a publication should also record GPU model, driver, CUDA version, operating system, exact base commit, prompt-set hash, evaluation code commit, and random seeds.
