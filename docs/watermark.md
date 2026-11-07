# Keyed output watermark

The watermark marks each principal's teacher output with a key of its own, so that a model or text suspected of coming from TRACE output can be tested against the principals who received it. It supports attribution after the fact. It does not stop collection or training.

## How it works

The `unigram-v1` scheme gives every key a fixed secret green list: a `gamma` share of the token IDs below `eligible_vocab`. The teacher adds `delta` to those tokens' logits through the upstream `logit_bias` field, which llama-server and vLLM both accept, so no custom sampler is required. The green list does not depend on context. Low token IDs are the most common subwords in BPE vocabularies, so limiting the list to `eligible_vocab` IDs keeps each request's bias map small.

Keys are HMAC-SHA256 of a master secret and a public key ID. The gateway derives each principal's key ID from `HARNESS_WATERMARK_NAMESPACE` and the principal, and the teacher holds the secret. A client cannot choose a key: the gateway refuses requests that carry `watermark_key`.

## Configuration

On the teacher API, `TEACHER_WATERMARK_KEY_FILE` holds a hex secret of at least 16 bytes. Generate one with `openssl rand -hex 32` and keep it outside readings and calibration directories; `study.sh` redacts variables named `*_KEY*` from recorded environments. `TEACHER_WATERMARK_GAMMA` (0.25), `TEACHER_WATERMARK_DELTA` (2.0) and `TEACHER_WATERMARK_ELIGIBLE_VOCAB` (32,768) set the scheme.

On the harness, `HARNESS_WATERMARK=true` sends each principal's key ID with its requests. The teacher must report that it applied the watermark; otherwise the response is withheld as a failed generation, never released unmarked. The flag is removed from client responses. The setting enters the policy revision. Noise responses at L4 are not watermarked.

## Detection

```bash
distill watermark key-id alice
distill watermark detect --texts suspect.jsonl --key-file artifacts/watermark.key \
	--principal alice --principal bob --tokenizer Qwen/Qwen3.5-9B \
	--tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --output artifacts/detect.json
```

`suspect.jsonl` has a `text` field per line, such as a suspect student's answers to held-out prompts. The detector tokenizes each text with the teacher's tokenizer, counts each text's distinct eligible tokens once, pools the counts, and computes a one-sided z-score against `gamma` for each candidate. Candidates are corrected jointly with Bonferroni at `--alpha` (default 0.001). Pools with fewer than `--minimum-tokens` (200) eligible tokens report `insufficient_evidence`. The report records the scheme digest and tokenizer, never the secret.

## What this does not establish

- **Student inheritance is unmeasured.** Fixed green lists are the easiest watermark for a student to learn, which is why this scheme was chosen, but whether a student distilled from this teacher carries a detectable bias needs the protected and unprotected student runs.
- **Utility cost is unmeasured.** The bias also changes greedy answers. Measure task accuracy with and without the watermark before enabling it, and keep it off in arms whose student results are compared.
- **Short answers carry little evidence.** A 16-token classification answer is too short to test on its own, so detection pools many texts and abstains when they are too few.
- **The vocabulary must match.** The GGUF and the Hugging Face tokenizer must share token IDs. Record the tokenizer revision with every report.
- **Detection is not proof.** A positive result shows a statistical bias toward a key's tokens. It does not show training, intent or which data was used, and paraphrasing, translation or mixing with other data weakens it. The key IDs of a cluster's principals can be tested together, but each test adds to the correction.

## Relation to the decoder watermark probe

`experiments/watermark` ([docs](watermark-probe.md)) prototypes a context-dependent keyed watermark: each green list depends on the preceding tokens, which is harder to learn or strip but needs a logits processor inside the engine. `unigram-v1` is the scheme that can be served now, because `llama-server` and vLLM both accept `logit_bias`. Both test detection the same way in spirit, counting each independent event once (a distinct token here, a distinct context there) so that repeats do not inflate the evidence. Their keys and green lists are not interchangeable, and a text marked with one is not detectable with the other.
