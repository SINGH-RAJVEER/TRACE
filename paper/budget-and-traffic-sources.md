# Campaign budget and traffic mix: evidence and sources

Notes for the manuscript, compiled 2026-11-02. Repository figures come from the v3 readings and the v4 audit ([REPORT.md](../readings/v4/analysis/REPORT.md)). External figures are quoted from the cited sources. Where a source does not report a quantity, these notes say so.

## Effectiveness of the 6,000-prompt A1 campaign

A1 sent 6,000 prompts from one account: 1,500 each from AG News, SQuAD, GSM8K and coding, interleaved.

### As a test of the harness

- The campaign samples each domain uniformly, so every 100-request window resembles the pooled benign reference more closely than most single-user windows do. A1 window p-values were 0.947 to 1.0. A uniform shuffle of held-out benign prompts scored 0.935 to 0.995 (audit F1).
- The leaky accumulator rises only when p < e^-1, about 0.37. A1 gave it no evidence, so the full MMD policy stayed at L0 and released all 6,000 responses, 588,653 output tokens (audit F2).
- An offline replay predicted this result in minutes. The live protected A1 runs took about 15.5 hours (audit F3).
- Reordering the same 6,000 prompts by topic evaded every tested MMD variant, including lower-tail and two-sided tests that catch the interleaved order (`readings/v4/results/evasion.json`).
- Only the volume limit had an effect. The 1,215-request quota released 1,215 responses and refused 4,785.
- The campaign is one run at one seed. Its prompts were inspected and reordered during the audit, so it cannot serve as an untouched confirmatory test. Task builds now seal a confirmatory attacker partition for that reason (see `docs/evaluation.md`).

### As distillation data

Estimated output-token share by domain, using the audit's per-domain mean attacker output tokens:

| Domain | Mean output tokens | Share of about 588k |
|---|---:|---:|
| GSM8K | 246 | about 63% |
| Coding | 136 | about 35% |
| SQuAD | 7.3 | about 2% |
| AG News | 2.4 | under 1% |

- Transfer has been measured only for AG News: +20.0 percentage points from 500 development responses (95% CI 16.4 to 23.8).
- No student trained on this campaign has been evaluated for GSM8K or coding. Those two domains carry about 98% of the tokens.
- Without a learning curve, it is unknown whether 1,500 AG News responses add anything beyond 500, or whether the 312 released under the quota already provide most of the gain.

## Would a 10,000-prompt budget help?

A balanced 10,000-prompt campaign requests 2,500 prompts per domain, against 1,500 at 6,000. The repository default is already 12,000 HTTP attempts.

- **Detection.** No change is expected. Per-window evidence for a broad, same-distribution campaign is near zero, so additional windows do not accumulate risk.
- **Quota.** The 1,215-request quota still releases 1,215 responses; a larger budget only adds refusals. An evenly split multi-account campaign needs ceil(10,000 / 1,215) = 9 accounts instead of 5.
- **Student capability.** Unknown, and likely subject to diminishing returns. Krishna et al. report extracted-model accuracy against query budget, expressed as fractions of the original training set size [1]:

| Task (1x size) | Queries | 0.1x | 0.2x | 0.5x | 1x | 2x | 5x | 10x |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SQuAD 1.1 F1 (87,599) | WIKI | 72.4 | 79.6 | 83.8 | 86.1 | 87.4 | 88.4 | 89.4 |
| SQuAD 1.1 F1 (87,599) | RANDOM | 60.6 | 68.5 | 75.8 | 79.1 | 81.9 | 84.8 | 85.8 |
| SST2 accuracy (67,349) | WIKI | 89.6 | 90.6 | 91.7 | 91.4 | 91.6 | 91.2 | 91.4 |
| MNLI accuracy (392,702) | WIKI | 68.0 | 71.6 | 75.9 | 77.8 | 78.9 | 79.7 | 79.3 |

The victim reached 90.6 F1 on SQuAD when trained on the full set. The authors conclude that "while more queries is usually better, accuracy gains quickly diminish" [1]. Their models and tasks differ from TRACE's, so the table shows the shape of the curve, not TRACE's values.

- **Pool capacity.** At seed 42, the sealed GSM8K attacker partitions hold 1,697 exploratory and 1,692 confirmatory prompts. A balanced 10,000-prompt campaign exceeds both, and the schedule command refuses it. A 6,000-prompt campaign, at 1,500 per domain, fits.
- **Realism.** Reported real campaigns are far larger than either budget:
	- Model Leeching collected 83,335 usable SQuAD answers from ChatGPT-3.5-Turbo for $50 in API cost, out of about 100,000 requests [2].
	- Google reported one Gemini extraction campaign of more than 100,000 prompts [3].
	- Anthropic attributed more than 16 million exchanges, through about 24,000 fraudulent accounts, to three laboratories: over 150,000 for DeepSeek, 3.4 million for Moonshot and 13 million for MiniMax [4].

	Realism therefore argues for more accounts at a similar per-account volume rather than a larger single-account budget.

**Recommendation.** Raise the budget only if a learning curve shows a gain between 1,500 and 2,500 prompts per domain, and reduce GSM8K's share if it is raised. Until then, 6,000 prompts support the same defense conclusions at lower cost.

## Benign and attack traffic in this study

The benign share depends on the denominator:

| Scope | Benign | Attack |
|---|---|---|
| A1's own account, the detector's decision unit | 0% | 100% (6,000 prompts) |
| A3 cover-traffic campaign | 50% in-distribution cover by default (`cover_share=0.5`) | 50% |
| Whole study | about 24,000 requests per benign role, across 125 accounts | 6,000 per campaign |

The whole-study row works out to roughly 80% benign. In v3, however, benign utility ran after each attack campaign on separate accounts, not concurrently, so the study never observed a real mixture. Within the benign population, the generator makes about 10% of accounts generalists (`generalist_share=0.1`); in each role, 112 of 125 accounts are single-domain specialists (audit F1).

## Real-world traffic splits: what sources support

No public source located for these notes reports the share of real API traffic that is extraction rather than benign use.

- **Anthropic [4]** describes one proxy network "mixing distillation traffic with unrelated customer requests to make detection harder" while managing more than 20,000 fraudulent accounts at once. It says distillation stood out by "volume, structure, and focus", including "massive volume concentrated in a few areas, highly repetitive structures". It does not give a share of total traffic. This is the provider's own account of its telemetry and has not been independently verified.
- **Google GTIG [3]** reports the campaign size only. It says nothing on mixing with legitimate traffic, the traffic share, or the detection method.
- **Liu, Guo and Dong [5]** are the closest academic reference, but their attacker shares are experimental settings, not observations. Using 1,500-query windows, the MMD detector's true-positive rate was 59.0% at a 5% attacker share, 93.7% at 10%, and 100% at 25%, 50% and 100%, with 0.3% benign false positives at 100%. Their limitations section states that "low-fraction mixed traffic remains challenging".

**What can be cited.** Anthropic supports the claim that real campaigns mix extraction with legitimate traffic and spread it across many accounts. Liu et al. support the claim that MMD detection degrades below about a 10% attacker share in their setting. Any specific benign-to-attack ratio in TRACE is a design assumption and should be stated as one. Sweeping the attacker share, for example across 5%, 10%, 25% and 50% as Liu et al. do, is easier to defend than a single ratio.

## Sources

1. Krishna, Tomar, Parikh, Papernot, Iyyer. Thieves on Sesame Street! Model Extraction of BERT-based APIs. ICLR 2020. https://arxiv.org/abs/1910.12366 (Table 3)
2. Birch, Hackett, Trawicki, Suri, Garraghan. Model Leeching: An Extraction Attack Targeting LLMs. CAMLIS 2023. https://arxiv.org/abs/2309.10544
3. Google Threat Intelligence Group. Distillation, experimentation, and integration: AI adversarial use. 2026-02-12. https://cloud.google.com/blog/topics/threat-intelligence/distillation-experimentation-integration-ai-adversarial-use
4. Anthropic. Detecting and preventing distillation attacks. 2026-02-23. https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks
5. Liu, Guo, Dong. An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic. arXiv:2606.05725 (EMNLP 2026). https://arxiv.org/html/2606.05725
