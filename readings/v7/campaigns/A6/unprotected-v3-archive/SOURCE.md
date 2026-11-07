# Archived unprotected A6 ledger (v3)

This campaign was not collected by v7. It is a copy of the v3 undefended A6 ledger from the `archive` branch (`443e5d63baaec3dcbb40d8a0018b3f221e527b2d`), path `readings/v3-2026-10-29-gateway-replay/ledgers/A6/s42-b6000/undefended/`, kept here as the unprotected reference for the v7 student comparison.

It was chosen as the best unprotected GSM8K ledger in the archive: of the archive's full 6,000-request undefended ledgers, it has the most GSM8K replies that finished (383 of 1,500) and were correct against the gold answers (379). The other ledgers from the same teacher finish 342 to 371, and the v4 Kaggle ledger has 450 usable GSM8K replies after review.

Differences from the v7 arms:

- Teacher: full-precision `Qwen/Qwen3.5-9B` served by vLLM (`self-hosted-vllm`), not the v7 Q4_K_M GGUF on llama-server. About 75% of its GSM8K replies stop at the 1,024-token limit, against 2% in v7 `A6/quota`.
- Prompts: same attack, seed 42 and 6,000-request budget, but drawn from the v3 corpus build. It shares 1,037 of its 6,000 example IDs with v7 `A6/quota`.
- Contract: `gateway-execution-v2`, the same as the v7 campaigns.

None of its 1,500 GSM8K prompts appear in `data/math/development-suite.json`, by source ID or by question text. The files are byte-for-byte copies of the archive versions.
