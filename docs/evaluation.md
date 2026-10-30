# Teacher and harness evaluation

## Scope and interpretation

The study measures response access and legitimate-user impact under explicitly specified campaign schedules. An unprotected policy admits all valid requests by design; this is a policy reference, not an observed 100% response-success rate. Failed, timed-out, and incomplete requests remain distinct from confirmed blocks.

## Prepare and deploy

1. Enter `devenv shell`, configure `.env` from `.env.example`, and run `just setup`.
2. Serve an OpenAI-compatible teacher. For the historical teacher, install vLLM separately and run `vllm serve Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --host 127.0.0.1 --port 8001 --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}'`. Select quantization and hardware settings appropriate to the host.
3. Run `just teacher`, then verify its health and authenticated generation. Set the teacher review identifier and credentials as needed.
4. Build each domain pool with `uv run --extra eval distill tasks build --task ag_news`, and repeat for `squad`, `gsm8k`, and `coding`. Dataset downloads and the coding sandbox have their own resource requirements. Use pinned dataset revisions and keep source groups disjoint.
5. Generate separate benign fitting, score-calibration, policy-calibration, and held-out test schedules with `uv run distill gateway benign --help`. Use `distill gateway detector-input` on each schedule. These files contain domain-tagged detector text, identities, and source groups. Check the source separation yourself before calibration. The gateway serializes the tagged turn according to its domain.
6. Run `uv run --extra embed harness calibrate --fitting FIT.jsonl --score-calibration SCORE.jsonl --policy-calibration POLICY.jsonl --output-dir artifacts/harness --task gateway`. Inspect the calibration counts and thresholds. Minimum empirical resolution is not proof of population false-positive control. A new configuration or workload requires a fresh calibration and output directory.
7. Configure account keys and gateway caps in `.env`, then start `just harness`. Use independent credentials and account identities for the experiment. See [harness settings and behavior](harness.md).

## Execute and report

Freeze each attack schedule with `uv run distill gateway schedule --help`. The `external` command prepares independent-source pools for A3 and A7. Keep the request inventory, seed, campaign budget, deadline, and account mapping fixed when comparing policies. `uv run distill gateway collect --help` executes a schedule against the protected endpoint. It writes `campaign.json`, an event log, a response ledger, and `campaign-timing.json`. Use a fresh output directory and fresh account state for each condition.

Report per campaign and domain: submitted attempts, received responses, explicit blocks, challenges, failures, deadlines and pending attempts, and released output tokens. Proof redemption consumes an HTTP attempt. An L1 alert still serves requests; a reach is not necessarily a restriction. Use timing and account-level outcomes to show how much data arrived before intervention. Missing token usage and incomplete decision attribution must remain unknown, not zero. Assess legitimate profiles separately, including short and specialist accounts, and show both all-horizon and scored-horizon rates. Replay is policy-only and does not measure delivered answers or solve challenges. Use account- or source-level uncertainty rather than treating correlated requests as independent.

A fresh protected run is required to measure effectiveness. Existing observations and software tests do not establish current performance.

For a small authored traffic suite, `distill traffic freeze`, `distill traffic run --config configs/traffic.example.toml`, and `distill traffic report --path RUN.json` provide a separate protected-only measurement. The report labels actors by the frozen suite and includes decision coverage. This synthetic suite is not a substitute for corpus-based campaign and legitimate-client evaluation.
