# A2 interim collection and undefended student benchmark

On 2026-10-30, the bounded shared-gateway study was intentionally interrupted during A2 protected collection to prioritize an undefended student benchmark. The protected campaign is incomplete and cannot be resumed in place with its original policy state. Its retained artifacts are local to `artifacts/gateway-bounded-v1/A2/s42-b6000/full/`.

At interruption, the partial protected campaign recorded 2,222 successful attempts and 2,223 level-0 serve decisions; the extra decision is not evidence of a completed response. No completed protected campaign summary or protected utility measurement exists. These partial observations do not establish final defense effectiveness.

The completed A2 undefended campaign recorded 6,000 successful responses, zero failures, and 588,309 released output tokens. The usability review accepted 4,816 examples (AG News 1,500; coding 1,450; GSM8K 367; SQuAD 1,499) and rejected 1,184 because the collector reported the output-length limit. This filtering, particularly for GSM8K, limits interpretation.

A shared Qwen3.5-2B-Base LoRA student is training on the accepted responses using the development-selected learning rate 0.0001 and three epochs. Training artifacts and the source ledger hash are stored locally under `artifacts/undefended-a2-benchmark/`. Benchmark scores are not yet available; model improvement must be measured against the base student on held-out suites before drawing conclusions.
