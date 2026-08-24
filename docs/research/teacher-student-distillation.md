# Black-box teacher to open-weight student

Practical research note, 2026-08-25. This is an engineering recommendation, not legal advice.

## Recommendation

Use **offline sequence-level knowledge distillation**, implemented as supervised fine-tuning on teacher-generated completions. In practical LLM tooling this is ordinary prompt-completion SFT, not token-level knowledge distillation. Start with one reviewed teacher completion per prompt and train a LoRA or QLoRA adapter with completion-only loss. Add DPO only after SFT if evaluation identifies a behavior that good preference pairs can address.

Put a provider-authorization gate before collection. Current standard OpenAI and Anthropic commercial terms generally prohibit using API output to train competing AI models. An open-weight generative student is likely inside that restriction. Do not start collection under those standard terms. Use a provider and contract that expressly permits this training use, obtain a written exception, or use a teacher whose weights and license permit local generation.

The proposed first version is:

```text
licensed prompts + frozen evaluation split
                   |
                   v
        authorized hosted teacher
                   |
                   v
 immutable response ledger (raw JSONL + provenance)
                   |
                   v
 validation, task checks, deduplication, human audit
                   |
                   v
 TRL prompt-completion dataset
                   |
                   v
 base model + chat template + QLoRA/LoRA + SFTTrainer
                   |
                   v
 task metrics + human review + safety and regression suites
                   |
                   v
 optional preference collection and DPOTrainer
```

## Why this is the right method without logits

Classic knowledge distillation trains the student against the teacher's softened class distribution. Hinton, Vinyals, and Dean define soft targets from teacher logits or probabilities and use temperature-scaled cross-entropy. A text-only API response does not expose that distribution, so this objective cannot be reconstructed from the returned string. Token log probabilities, when an API offers only selected values, are still not equivalent to the full teacher distribution unless the full vocabulary distribution is available at every position.

Sequence-level knowledge distillation is the applicable approximation. Kim and Rush replace the teacher's intractable sequence distribution with a generated sequence, then train the student with cross-entropy on the generated dataset. Their procedure is exactly: generate teacher outputs over a transfer set, then train the student on those outputs. For a hosted LLM, prompts are the transfer inputs and returned completions are hard sequence targets.

Self-Instruct provides a later LLM-scale example of generation, filtering, and supervised instruction tuning. It also shows why raw synthetic output should not go straight into training. In its 200-example audit, only 54% of sampled instruction, input, and output triples had all fields marked valid. Its pipeline filtered invalid, duplicate, and overly similar generations before fine-tuning.

TRL's experimental `GKDTrainer` is not a shortcut for a black-box teacher. Its interface requires a local `teacher_model`, and its source performs a teacher forward pass, reads `teacher_outputs.logits`, and computes a divergence over the full teacher and student vocabulary distributions. The source also requires matching teacher and student vocabulary sizes. Its `seq_kd` option can generate teacher sequences, but training still computes the logit-based JSD loss. Use `SFTTrainer` for an API teacher.

## Legal and policy gate

Run this review before choosing a teacher or sending any data:

1. Record the provider, product, account type, contracting entity, applicable agreement, effective date, service-specific terms, usage policy, and any order-form amendments.
2. Obtain written confirmation that API outputs may be used to train the planned open-weight generative model and may be redistributed in model weights or adapters. Output ownership language alone is not enough if a separate use restriction applies.
3. Confirm rights to every prompt, reference document, human label, and benchmark item. The OpenAI and Anthropic terms both place responsibility for input rights on the customer.
4. Review the student's model license for fine-tuning, redistribution, acceptable use, attribution, and naming requirements. Preserve the exact base model repository and revision in the training manifest.
5. Decide whether prompts or responses may contain personal, confidential, regulated, copyrighted, or export-controlled data. Match API retention and regional processing to that decision.
6. Save a dated approval artifact such as `terms_review_id` in every collection manifest. Re-review before a new teacher model, endpoint, provider, or release.

### Current examples that block the proposed use

OpenAI's Services Agreement, effective 2026-01-01, says the customer owns output to the extent permitted by law, but section 3.3(e) separately prohibits using output to develop AI models that compete with OpenAI products and services except for a defined "Permitted Exception." That exception covers certain non-distributed categorization, classification, or organization models and fine-tuning models offered by OpenAI. It does not describe general-purpose external open-weight model training.

Anthropic's Commercial Terms, effective 2025-06-17, likewise say the customer owns outputs, then section D.4 prohibits accessing the services to build a competing product or service, including training competing AI models, unless Anthropic expressly approves it.

These clauses make standard-account collection from either provider a no-go for this project unless counsel confirms the use is non-competing or the provider gives express approval. Terms change, and a negotiated order form can change the result. Check the agreement that actually governs the account rather than relying on this note.

### Data handling examples

OpenAI's API data-control documentation says API data is not used to train OpenAI models unless the customer opts in. It also documents endpoint-specific storage and default abuse-monitoring retention of up to 30 days. The Batch endpoint is not eligible for Zero Data Retention in the cited table, and the Batch guide says output files are automatically deleted 30 days after completion.

Anthropic's API retention documentation says it does not train on retained customer data without express permission. ZDR can cover eligible synchronous Messages requests, but Message Batches are not ZDR-eligible and have 29-day retention. Flagged content or legal holds can override the normal arrangement. The governing contract remains the source of truth.

Privacy controls do not grant training rights. Treat contractual use permission and data retention as separate checks.

## First-version data plan

### 1. Define the task distribution first

Do not ask the teacher to invent the whole curriculum in version one. Start from prompts that represent intended production traffic or a licensed task corpus. Assign each prompt a task family, difficulty stratum, language, source, and stable ID.

Split by source groups before calling the teacher:

- `train` supplies teacher-generated training examples.
- `validation` selects checkpoints and training settings.
- `test` remains frozen and must not contribute prompts, completions, preference labels, filtering thresholds, or teacher feedback to training.

Group related templates, documents, users, entities, or problem generators into one split. A random row split can leak near-duplicates. Keep public benchmark test items out of the generation prompt and training corpus.

For a base model, the prompt set must cover instruction-following basics as well as the target specialty. If it covers only a narrow specialist task, expect a specialist rather than a robust general chat model.

### 2. Collect teacher responses as an auditable ledger

Use the provider's asynchronous batch API if the approved contract permits it. OpenAI and Anthropic both document a 50% batch discount and stable `custom_id` fields. Both warn that results may return out of input order, so join on the stable ID, never file position.

For each prompt in version one:

- Request one high-quality completion with a fixed system instruction, bounded output format, and enough output tokens to avoid truncation.
- Pin a model snapshot when the provider exposes one. Avoid a moving `latest` alias.
- Record every request parameter. A nominal temperature of zero does not remove the need for provenance or repeatability checks.
- Keep the raw request and response unchanged. Produce cleaned training projections separately.
- Retry transport, rate-limit, and server failures idempotently under the same example ID and a new attempt ID.
- Reject truncated outputs unless truncation is valid for the task.
- Do not request hidden chain-of-thought. If a task benefits from explanations and the provider permits their reuse, ask for a concise, user-facing rationale in a declared field. "Distilling Step-by-Step" supports rationale supervision for the task-specific classification and reasoning setup it studied, but that evidence does not make long free-form rationales a default for general chat SFT.

One completion per prompt is the cheapest sound baseline. Collect two to four candidates only for tasks where selection, disagreement analysis, or later preference pairs justify the extra cost.

### Canonical collection record

Keep provenance beside the content in an immutable JSONL or Parquet ledger. This is a project schema, not a TRL input schema.

```json
{
	"example_id": "ex_01J...",
	"attempt_id": "att_01J...",
	"split": "train",
	"task": "support_answer",
	"source": {
		"corpus": "licensed-corpus-v1",
		"source_id": "row-1842",
		"license": "internal-approved",
		"content_sha256": "..."
	},
	"prompt": [
		{"role": "system", "content": "Answer from the supplied policy text."},
		{"role": "user", "content": "..."}
	],
	"teacher": {
		"provider": "approved-provider",
		"model": "pinned-model-snapshot",
		"endpoint": "messages",
		"request_id": "provider-request-id",
		"batch_id": "provider-batch-id",
		"generated_at": "2026-08-25T12:00:00Z",
		"parameters": {"temperature": 0, "max_output_tokens": 1024},
		"terms_review_id": "legal-2026-08-01"
	},
	"completion": [
		{"role": "assistant", "content": "..."}
	],
	"finish_reason": "stop",
	"usage": {"input_tokens": 312, "output_tokens": 146},
	"quality": {
		"status": "accepted",
		"checks": ["schema", "grounding", "dedupe"],
		"reviewer": "human-or-versioned-checker-id"
	}
}
```

Do not put secrets, API keys, provider headers, or personal reviewer data into distributable datasets.

### 3. Validate and filter

The minimum acceptance pipeline is:

1. Parse the expected structure and reject empty, malformed, or truncated completions.
2. Run task-native validators. Examples include JSON Schema, code compilation and tests, exact-answer checks, citation verification against supplied context, and allowed-label checks.
3. Reject prompt leakage, system-prompt disclosure, irrelevant refusals, unsupported claims in grounded tasks, and responses that violate the target safety policy.
4. Normalize only transport artifacts. Preserve meaningful whitespace and code formatting.
5. Deduplicate exact and near-duplicate prompts and completions. Check against validation and test sets before training.
6. Scan for personal data, secrets, memorized passages, and content outside the approved license or policy scope.
7. Human-audit a stratified sample from every task, language, source, and filter outcome. Audit rejected examples too, since an over-aggressive filter can narrow the learned behavior.

Store rejection reasons. They are needed to diagnose coverage loss and to build better prompts later.

### 4. Project into TRL schemas

TRL `SFTTrainer` accepts standard or conversational prompt-completion records. Use the explicit conversational form so roles stay structured and the student's chat template controls serialization:

```json
{
	"prompt": [
		{"role": "system", "content": "Answer from the supplied policy text."},
		{"role": "user", "content": "..."}
	],
	"completion": [
		{"role": "assistant", "content": "..."}
	]
}
```

Prompt-completion data makes the desired loss explicit. `SFTTrainer` computes loss on completion tokens by default for this dataset type. Set `completion_only_loss=True` rather than relying on an implicit default that could change. Use `assistant_only_loss=True` only if the chosen chat template supports TRL's generation masks and this has been tested.

Keep provider-specific role markers and serialized prompt text out of the training data. Transformers documents that different chat models use different control tokens even when they share a base model, and that the wrong tokens hurt performance. Choose one student chat template, save it with the tokenizer, apply it through TRL or `tokenizer.apply_chat_template`, and align the EOS token with the template. If the base checkpoint has no template, supply a stable template through `chat_template_path` and treat it as part of the model artifact.

## SFT with LoRA or QLoRA

### Student choice

Choose the strongest base checkpoint that fits the deployment and training budget, has a suitable context window and tokenizer, and has a license that permits the intended release. A base checkpoint gives control over the post-training recipe but requires enough broad instruction data and a deliberate chat template. An instruction checkpoint is a lower-risk first experiment if starting from a base model is not a hard requirement.

Compare at least two student sizes if possible. Distillation cannot guarantee that a student with too little capacity will represent the teacher's behavior.

### LoRA versus QLoRA

Use LoRA in BF16 if the model, optimizer, activations, and target sequence length fit comfortably. It is simpler and avoids quantization error. Use QLoRA when frozen base-weight memory is the constraint.

LoRA freezes the pretrained weights and learns low-rank update matrices. QLoRA stores the frozen base in 4-bit form, dequantizes for computation, and backpropagates only into LoRA parameters. The QLoRA paper reports that NF4 plus double quantization matched its 16-bit LoRA or full-fine-tuning baselines in the evaluated settings, but this is an empirical result, not a guarantee for every model and dataset.

A practical QLoRA starting configuration is:

| Setting | First value to try | Reason |
| --- | --- | --- |
| base storage | 4-bit | QLoRA memory reduction |
| quantization type | `nf4` | Transformers recommends NF4 for training 4-bit base models |
| compute dtype | BF16 when hardware supports it | documented QLoRA compute path |
| nested quantization | enabled when memory is tight | saves about 0.4 bits per parameter in Transformers docs |
| preparation | `prepare_model_for_kbit_training` | PEFT's documented preprocessing step |
| LoRA targets | `all-linear` | PEFT's QLoRA-style setting; the QLoRA paper found broad layer coverage important |
| rank | 16 as a baseline, then compare 8 and 32 | a modest experiment range, not a universal optimum |
| alpha | 16 or 32, tuned with rank and learning rate | adapter scaling |
| dropout | 0 to 0.05 | tune on validation data |
| task type | causal language modeling | decoder-only student |

Transformers warns that 8-bit and 4-bit training supports only extra parameters, which is why the quantized base remains frozen. PEFT documents calling `prepare_model_for_kbit_training` before attaching the adapter. PEFT also recommends `target_modules="all-linear"` for QLoRA-style training.

### Training controls

- Train only on accepted examples and compute loss only on the teacher completion.
- Pick `max_length` from the tokenized length distribution and deployment context. Report the fraction truncated. Silent truncation of answers poisons targets.
- Start with one to three epochs, an effective batch size set through gradient accumulation, gradient checkpointing, and adapter learning rates around the TRL-documented order of `1e-4`. Select by validation metrics, not training loss alone.
- Enable packing only after verifying completion masks, EOS boundaries, and one-example loss behavior. It is an efficiency option, not a quality requirement.
- Save the adapter, tokenizer, chat template, exact base revision, dataset manifest and hashes, package versions, random seeds, and full training arguments.
- Keep an unmerged adapter for reproducibility. If deployment requires merged weights, evaluate the merged artifact independently.
- Run a small full-BF16 LoRA comparison when hardware permits. It reveals whether 4-bit storage is costing quality on this task.

## Optional preference data and DPO

Preference optimization is not needed to transfer a good teacher response. SFT already does that. Add preference data when the desired behavior is comparative, such as preferring grounded over plausible-sounding answers, concise over verbose answers, or policy-compliant over unsafe answers.

Build pairs after the first SFT model exists:

1. Sample two plausible responses to the same held-out training prompt. Useful sources are two student samples, one student and one approved teacher sample, or multiple teacher candidates.
2. Apply task validators first. Human labelers then compare the remaining pair under a written rubric. A separately approved teacher judge may help triage, but should not be the sole authority.
3. Randomize response order, blind source identity, record ties, and retain only pairs with a defensible preference. Pairs where both answers are bad or both are good add ambiguous supervision.
4. Avoid making every `chosen` response a teacher output and every `rejected` response a student output. The model can learn source style, verbosity, or formatting rather than the desired property.

TRL's recommended explicit conversational preference schema is:

```json
{
	"prompt": [{"role": "user", "content": "..."}],
	"chosen": [{"role": "assistant", "content": "..."}],
	"rejected": [{"role": "assistant", "content": "..."}]
}
```

Run DPO from the SFT checkpoint. DPO needs no teacher logits. It computes log probabilities from the local trainable policy and a frozen reference policy. TRL's `DPOTrainer` uses the initial policy as the reference when `ref_model` is omitted. The original DPO paper also starts preference optimization from an SFT model and uses that model as the reference. Start with TRL's sigmoid DPO objective and a small learning rate, then require an evaluation win over SFT before keeping the stage.

Preference training roughly doubles completion scoring and introduces a reference-policy memory cost. TRL can precompute reference log probabilities to save training memory, subject to the documented trainer constraints. PEFT adapters are supported directly.

## Evaluation plan

### Systems to compare

Evaluate the same frozen test prompts with the same decoding contract for:

- Untuned student base model.
- SFT student.
- DPO student, if built.
- Teacher, when the approved contract permits evaluation calls.
- A simple task baseline where one exists.

The goal is not textual imitation. The goal is a student that meets task, safety, latency, and cost requirements.

### Metrics

Use task-native, executable metrics first:

- Classification: accuracy, macro F1, calibration, and per-class recall.
- Extraction or structured output: parse rate, schema validity, exact match, and field-level F1.
- Code: compile rate and isolated unit-test pass rate.
- Grounded QA: answer correctness, citation validity, support in the supplied context, and abstention quality.
- Generation: blind human pairwise preference under a task rubric, with error categories.

Add cross-cutting checks for safety-policy compliance, privacy leakage, memorized strings, instruction hierarchy, prompt injection, multilingual performance, answer length, refusal rate, latency, throughput, and peak memory.

Teacher-forced validation loss is useful for checkpoint selection but is not the product metric. Kim and Rush observed sequence-distilled models with worse perplexity but better BLEU, which is a direct warning against using perplexity alone for hard-output distillation.

### Judge models and human review

If an LLM judge is contractually permitted, calibrate it against human labels on the project's rubric. Blind model names, randomize answer order, evaluate both orders for a sample, and report disagreement. The QLoRA paper found a significant ordering effect in GPT-4 scoring and recommended averaging both orders. It also found model and human rankings could disagree, so retain human review for release decisions and high-risk slices.

### Experimental design

- Freeze the test set, prompts, decoding settings, and metrics before the main training run.
- Report confidence intervals or paired bootstrap intervals, not only point estimates.
- Break results down by task, source, language, difficulty, length, and safety category.
- Audit failures jointly with the training-data provenance. Determine whether each failure came from teacher error, filter error, missing coverage, truncation, student capacity, or optimization.
- Run data ablations such as 10%, 25%, 50%, and 100% of accepted SFT data. Synthetic data quality and coverage matter more than raw call count.
- Set release gates in advance. Example gates are no statistically credible regression on critical tasks, a target fraction of the teacher-to-base quality gap recovered, safety non-regression, and deployment budget compliance.

## What not to do in version one

- Do not use a provider whose applicable terms prohibit external model training, even if the API assigns ownership of output.
- Do not call the process token-level KD or claim soft-target transfer. Only hard generated sequences are available.
- Do not use TRL `GKDTrainer` with an API wrapper. It requires local teacher forward passes and full logits.
- Do not train directly on raw batch results. Preserve raw responses, then create a validated projection.
- Do not mix provider chat markup with the student's chat format.
- Do not train on evaluation prompts or use the test set to tune filtering and generation prompts.
- Do not add DPO before SFT has a measured baseline and the project has reliable preference labels.
- Do not use the same teacher as generator, sole filter, sole preference labeler, and sole evaluator without human calibration. Correlated errors will look like agreement.
- Do not request or retain hidden reasoning traces. Use concise visible rationales only when they are useful, approved, and evaluated.

## Suggested milestones

1. **Authorization and benchmark.** Approve teacher and data rights, freeze the grouped test set, and score the base model.
2. **Pilot collection.** Collect 1,000 to 5,000 responses, run validators, and human-audit every task slice. Stop if the accepted data is not clearly better than the base student's outputs.
3. **SFT baseline.** Train one QLoRA adapter and one smaller data ablation. Compare against base and teacher.
4. **Scale by evidence.** Add prompts in failure-heavy or uncovered slices rather than multiplying all prompts uniformly.
5. **Preference experiment.** Build a small human-labeled pair set only for a measured SFT weakness. Keep DPO only if it beats SFT on frozen tests without regressions.
6. **Release review.** Recheck provider terms, student license, dataset lineage, privacy, safety, and merged-artifact evaluation.

## Primary sources

### Distillation and synthetic instruction data

- Hinton, Vinyals, and Dean, *Distilling the Knowledge in a Neural Network*. Defines temperature-based soft-target distillation from teacher probabilities or logits. <https://arxiv.org/abs/1503.02531>
- Kim and Rush, *Sequence-Level Knowledge Distillation*. Defines generated-sequence targets followed by ordinary cross-entropy student training. <https://arxiv.org/abs/1606.07947>
- Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. Covers seed tasks, synthetic instruction and instance generation, filtering, SFT, and a manual quality audit. <https://arxiv.org/abs/2212.10560>
- Hsieh et al., *Distilling Step-by-Step*. Studies generated rationales as extra task-specific supervision. <https://arxiv.org/abs/2305.02301>

### LoRA, QLoRA, and Hugging Face implementation

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*. <https://arxiv.org/abs/2106.09685>
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*. Covers NF4, double quantization, paged optimizers, broad adapter coverage, and evaluation caveats. <https://arxiv.org/abs/2305.14314>
- PEFT LoRA guide, including `target_modules="all-linear"` for QLoRA-style training. <https://huggingface.co/docs/peft/en/developer_guides/lora>
- PEFT quantization guide, including `prepare_model_for_kbit_training`. <https://huggingface.co/docs/peft/en/developer_guides/quantization>
- Transformers bitsandbytes guide, including NF4, BF16 compute, nested quantization, and the extra-parameters-only training warning. <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>
- Transformers chat-template guide. <https://huggingface.co/docs/transformers/en/chat_templating>
- TRL dataset formats. <https://huggingface.co/docs/trl/en/dataset_formats>
- TRL `SFTTrainer`, including prompt-completion formats, completion-only loss, packing, PEFT integration, chat-template handling, and EOS alignment. <https://huggingface.co/docs/trl/en/sft_trainer>
- TRL `GKDTrainer` documentation. <https://huggingface.co/docs/trl/en/gkd_trainer>
- TRL `GKDTrainer` source, showing local teacher forward passes and full-logit JSD. <https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py>

### Preference optimization

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. <https://arxiv.org/abs/2305.18290>
- TRL `DPOTrainer`, including explicit preference schemas, reference-model behavior, and PEFT support. <https://huggingface.co/docs/trl/en/dpo_trainer>

### Provider terms, collection, and retention

- OpenAI Services Agreement, updated 2025-12-01 and effective 2026-01-01. See sections 3.3 and 4, plus the definition of "Permitted Exception." <https://openai.com/policies/services-agreement/>
- OpenAI API data controls and endpoint retention. <https://platform.openai.com/docs/guides/your-data>
- OpenAI Batch API. <https://platform.openai.com/docs/guides/batch>
- Anthropic Commercial Terms, effective 2025-06-17. See sections B and D.4. <https://www.anthropic.com/legal/commercial-terms>
- Anthropic API and data retention. <https://platform.claude.com/docs/en/manage-claude/api-and-data-retention>
- Anthropic Message Batches. <https://platform.claude.com/docs/en/build-with-claude/batch-processing>
