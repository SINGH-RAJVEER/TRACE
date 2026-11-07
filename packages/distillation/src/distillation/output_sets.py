"""Hashed output arms, adaptive training sets and matched clean controls.

Callbacks own model invocation, answer extraction, task validation and exact tokenization.
No teacher or model is loaded here. Every recipe operates on one frozen acquisition cache.
"""

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from distillation.acquisition_manifest import (
	_sha256_json,
	load_manifest,
	select_rows,
	write_new,
)
from distillation.schemas import LedgerRecord, Message

# Verbatim, dehyphenated from the local paper's Appendix A.2. E.3 studies model sizes.
DENSE_INSTRUCTION = (
	"Your task is to process a provided chain of logical deductions and reformulate it "
	"utilizing an esoteric, formal, and densely technical lexicon, thereby obfuscating its "
	"clarity for a non-specialist audience. Refrain from any extraneous discourse."
)
PAPER_SHA256 = "d6e94d58be2e75a03d1d32b223be0faa40c6671ec827784ec71289b70d1d2762"
ARM_VERSION = "output-arm-v1"
OUTPUT_ARMS = ("clean", "compact_verified", "final_answer_only", "paraphrase", "dense_locked")
ADAPTIVE_ARMS = (
	"final_answer_extraction",
	"cleaned_explanations",
	"paraphrased_protected",
	"protected_loss",
)
CONTROL_ARMS = (
	"clean_matched_queries",
	"clean_matched_tokens",
	"clean_truncated",
	"clean_subsampled",
)
INSTRUCTIONS = {
	"compact_verified": "Write a concise correct solution. Preserve the final answer.",
	"paraphrase": "Paraphrase the explanation in ordinary language. Preserve the final answer.",
	"dense_locked": DENSE_INSTRUCTION,
	"cleaned_explanations": "Simplify and clean this explanation. Preserve the final answer.",
	"paraphrased_protected": "Paraphrase this explanation. Preserve the final answer.",
}


# A development value, not a measured boundary between rewriting and copying.
MAX_COPIED_FRACTION = 0.8
_COPY_SHINGLE = 5


def _copied_share(words: list[str], other: list[str]) -> float:
	"""The share of `words` found in `other`, as word 5-grams or whole when shorter."""
	if len(words) < _COPY_SHINGLE:
		return 1.0 if f" {' '.join(words)} " in f" {' '.join(other)} " else 0.0
	seen = {tuple(other[i : i + _COPY_SHINGLE]) for i in range(len(other) - _COPY_SHINGLE + 1)}
	grams = [tuple(words[i : i + _COPY_SHINGLE]) for i in range(len(words) - _COPY_SHINGLE + 1)]
	return sum(gram in seen for gram in grams) / len(grams)


def copied_fraction(candidate: str, source: str) -> float:
	"""How much of either text is the other, so padding cannot hide a complete copy."""
	words, original = candidate.casefold().split(), source.casefold().split()
	return max(_copied_share(words, original), _copied_share(original, words))


class Codec(Protocol):
	def encode(self, text: str) -> Sequence[int]: ...
	def decode(self, tokens: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class Rewrite:
	text: str
	queries: int
	money: float
	seconds: float


@dataclass(frozen=True)
class Adapters:
	"""IDs must identify frozen code/model/tokenizer revisions, including validator rules.

	extract_answer returns the releasable answer form, such as "#### 42", and must return the
	same string when applied to it: the final-answer arms release exactly what it returns.
	"""

	codec: Codec
	extract_answer: Callable[[LedgerRecord, str], str | None]
	valid: Callable[[LedgerRecord, str], bool]
	identity: dict[str, str]
	rewrite: Callable[[LedgerRecord, str, str], Rewrite] | None = None


@dataclass(frozen=True)
class Recipe:
	arm: str
	fallback: Literal["withhold", "final_answer"] = "withhold"
	max_tokens: int | None = None
	loss: Literal["ordinary", "filter", "clip"] = "ordinary"
	loss_threshold: float | None = None
	max_copied_fraction: float = MAX_COPIED_FRACTION


def verify_artifact(artifact: dict[str, Any]) -> None:
	if artifact.get("digest") != _sha256_json({k: v for k, v in artifact.items() if k != "digest"}):
		raise ValueError("artifact digest mismatch")


def _require_version(artifact: dict[str, Any], version: str) -> None:
	verify_artifact(artifact)
	if artifact.get("version") != version:
		raise ValueError(f"expected a {version} artifact")


def _final_answer(adapters: Adapters, record: LedgerRecord, text: str) -> str:
	"""The extractor's output, refused when it is not itself a releasable answer."""
	answer = adapters.extract_answer(record, text)
	if answer and adapters.extract_answer(record, answer) != answer:
		raise ValueError("extract_answer must return a releasable form it extracts unchanged")
	return answer or ""


def _seal(artifact: dict[str, Any]) -> dict[str, Any]:
	artifact["digest"] = _sha256_json(artifact)
	return artifact


def loss_values(losses: Sequence[float], recipe: Recipe) -> list[float]:
	"""Per-token objective before reduction; filtering zeros high-loss token gradients.

	For tensor training use tensor_loss below so clipping also clips its gradient. Filtering
	retains the original token denominator, making the objective reproducible across batches.
	"""
	_validate_loss(recipe)
	if any(not math.isfinite(x) or x < 0 for x in losses):
		raise ValueError("token losses must be finite and nonnegative")
	threshold = recipe.loss_threshold
	if recipe.loss == "ordinary":
		return list(losses)
	assert threshold is not None
	return [
		min(x, threshold) if recipe.loss == "clip" else x if x <= threshold else 0.0 for x in losses
	]


def tensor_loss(token_losses: Any, recipe: Recipe) -> Any:
	"""Differentiable PyTorch-compatible recipe hook; call on unmasked completion tokens."""
	_validate_loss(recipe)
	if recipe.loss == "clip":
		return token_losses.clamp(max=recipe.loss_threshold).mean()
	if recipe.loss == "filter":
		return (token_losses * (token_losses.detach() <= recipe.loss_threshold)).mean()
	return token_losses.mean()


def _validate_loss(recipe: Recipe) -> None:
	if recipe.loss not in {"ordinary", "filter", "clip"}:
		raise ValueError("unknown loss recipe")
	if recipe.loss != "ordinary" and (
		recipe.loss_threshold is None
		or not math.isfinite(recipe.loss_threshold)
		or recipe.loss_threshold <= 0
	):
		raise ValueError("filter and clip require a positive finite loss_threshold")


def _records(manifest: dict[str, Any]) -> dict[str, LedgerRecord]:
	records = {}
	for entry in manifest["inputs"]:
		data = Path(entry["path"]).read_bytes()
		if hashlib.sha256(data).hexdigest() != entry["sha256"]:
			raise ValueError("acquisition ledger changed")
		for line in data.splitlines():
			if line.strip():
				record = LedgerRecord.model_validate_json(line)
				records[record.example_id] = record
	for row in manifest["examples"]:
		if (
			hashlib.sha256(records[row["example_id"]].model_dump_json().encode()).hexdigest()
			!= row["record_sha256"]
		):
			raise ValueError("cached record hash mismatch")
	return records


def build_arm(
	manifest_path: Path,
	recipe: Recipe,
	adapters: Adapters,
	*,
	protected: dict[str, Any] | None = None,
	admissions: dict[str, Any] | None = None,
	include_pending: bool = False,
) -> dict[str, Any]:
	"""Build all accepted training rows. Selection happens afterwards using released counts.

	Dense rewriting uses exactly A.2 as its instruction and checks its extracted answer.
	Answer locking is a stricter variant than the paper, not a paper replication claim.
	A failed callback must return costed failure metadata, or raise and abort this build.
	include_pending also admits unreviewed responses, as fresh collection writes them, and is
	recorded in the artifact. Rejected responses never enter.
	"""
	_validate_loss(recipe)
	if recipe.loss != "ordinary" and recipe.arm != "protected_loss":
		raise ValueError("only protected_loss may change the training loss")
	if protected is not None and recipe.arm not in ADAPTIVE_ARMS:
		raise ValueError("only adaptive arms take protected outputs")
	if admissions is not None and recipe.arm not in {"quota", "quota_tokens"}:
		raise ValueError("only quota arms take observed admissions")
	if recipe.arm not in (*OUTPUT_ARMS, *ADAPTIVE_ARMS, *CONTROL_ARMS, "quota", "quota_tokens"):
		raise ValueError("unknown output arm")
	if recipe.arm == "protected_loss" and recipe.loss == "ordinary":
		raise ValueError("protected_loss requires filter or clip")
	if recipe.fallback not in {"withhold", "final_answer"}:
		raise ValueError("fallback must be withhold or final_answer")
	if recipe.arm == "clean_truncated" and (recipe.max_tokens is None or recipe.max_tokens < 1):
		raise ValueError("clean_truncated requires positive max_tokens")
	if not all(adapters.identity.get(k) for k in ("codec", "answer_extractor", "validator")):
		raise ValueError("frozen codec, answer_extractor and validator identities are required")
	if recipe.arm in INSTRUCTIONS and (
		adapters.rewrite is None or not adapters.identity.get("rewriter")
	):
		raise ValueError("this arm requires a pluggable rewriter and frozen rewriter identity")
	if not 0 < recipe.max_copied_fraction <= 1:
		raise ValueError("max_copied_fraction must be in (0, 1]")
	manifest = load_manifest(manifest_path, require_ledgers=True)
	if manifest["response_policy"] != "clean":
		raise ValueError("build arms from one clean acquisition manifest")
	records = _records(manifest)
	accepted_ids = None
	baseline_settings = None
	if recipe.arm in {"quota", "quota_tokens"}:
		if (
			admissions is None
			or admissions.get("manifest_digest") != manifest["digest"]
			or admissions.get("policy") != recipe.arm
		):
			raise ValueError("quota sets require manifest-bound observed admissions")
		evidence = Path(admissions["evidence_path"]).read_bytes()
		if hashlib.sha256(evidence).hexdigest() != admissions["evidence_sha256"]:
			raise ValueError("admission evidence hash mismatch")
		accepted_ids = set(admissions["accepted_example_ids"])
		if not accepted_ids <= records.keys():
			raise ValueError("admissions contain unknown cached examples")
		baseline_settings = admissions["settings"]
	parents = {}
	if protected is not None:
		_require_version(protected, ARM_VERSION)
		if protected["manifest_digest"] != manifest["digest"]:
			raise ValueError("protected outputs use another acquisition manifest")
		parents = {r["example_id"]: r for r in protected["rows"]}
	if recipe.arm in ADAPTIVE_ARMS and protected is None:
		raise ValueError("adaptive arms require protected outputs")
	eligible = {"accepted", "pending"} if include_pending else {"accepted"}
	rows = []
	for source in manifest["examples"]:
		if source["split"] != "train" or source["quality_status"] not in eligible:
			continue
		record = records[source["example_id"]]
		clean = "\n".join(m.content for m in record.completion)
		answer = adapters.extract_answer(record, clean)
		parent = parents.get(record.example_id)
		if recipe.arm in ADAPTIVE_ARMS and parent is None:
			raise ValueError("protected arm missing a cached training row")
		draft = parent["released_text"] if recipe.arm in ADAPTIVE_ARMS and parent else clean
		text = draft
		fallback = None
		failure = None
		cost = Rewrite("", 0, 0.0, 0.0)
		if not draft:
			text = ""
			failure = "source withheld"
		elif recipe.arm in {"final_answer_only", "final_answer_extraction"}:
			text = _final_answer(adapters, record, draft)
		elif recipe.arm == "clean_truncated":
			text = adapters.codec.decode(adapters.codec.encode(clean)[: recipe.max_tokens])
		elif recipe.arm in INSTRUCTIONS:
			assert adapters.rewrite is not None
			rewrite_record = record
			if recipe.arm in ADAPTIVE_ARMS:
				rewrite_record = record.model_copy(
					update={"completion": [Message(role="assistant", content=draft)]}
				)
			cost = adapters.rewrite(rewrite_record, draft, INSTRUCTIONS[recipe.arm])
			if (
				isinstance(cost.queries, bool)
				or not isinstance(cost.queries, int)
				or cost.queries < 1
				or not all(math.isfinite(x) and x >= 0 for x in (cost.money, cost.seconds))
			):
				raise ValueError("rewriter must record queries, money and elapsed seconds")
			text = cost.text
			if not text.strip():
				failure = "empty rewrite"
			elif (
				recipe.arm in OUTPUT_ARMS
				and copied_fraction(text, draft) >= recipe.max_copied_fraction
			):
				failure = "rewrite copies the clean draft"
			elif answer is None or adapters.extract_answer(record, text) != answer:
				failure = "answer mismatch or missing locked answer"
			elif not adapters.valid(record, text):
				failure = "task validity failed"
			if failure:
				# Record the fallback that happened, which is withholding when no valid
				# answer exists to release.
				final = _final_answer(adapters, record, clean) if answer else ""
				releasable = recipe.fallback == "final_answer" and adapters.valid(record, final)
				fallback = "final_answer" if final and releasable else "withhold"
				text = final if fallback == "final_answer" else ""
		if accepted_ids is not None and record.example_id not in accepted_ids:
			text = ""
			failure = "withheld by observed quota admission"
		valid = bool(text) and adapters.valid(record, text)
		if (
			recipe.arm in {"compact_verified", "final_answer_only", "final_answer_extraction"}
			and not valid
		):
			failure = failure or "task validity failed"
			fallback = "withhold"
			text = ""
		rows.append(
			{
				"example_id": record.example_id,
				"source_family": source["source_family"],
				"record_sha256": source["record_sha256"],
				"prompt_sha256": source["prompt_sha256"],
				"clean_completion_sha256": source["completion_sha256"],
				"parent_row_sha256": _sha256_json(parent) if parent else None,
				"released_text": text,
				"released_sha256": hashlib.sha256(text.encode()).hexdigest(),
				"released_tokens": parent["released_tokens"]
				if recipe.arm in ADAPTIVE_ARMS and parent
				else len(adapters.codec.encode(text))
				if text
				else 0,
				"training_tokens": len(adapters.codec.encode(text)) if text else 0,
				"answer_equal": answer is not None
				and adapters.extract_answer(record, text) == answer
				if text
				else False,
				"task_valid": valid,
				"fallback": fallback,
				"failure": failure,
				"rewriter_cost": {k: v for k, v in asdict(cost).items() if k != "text"},
				"sft": {
					"prompt": [m.model_dump(mode="json") for m in record.prompt],
					"completion": [{"role": "assistant", "content": text}],
				}
				if text
				else None,
			}
		)
	if not rows:
		raise ValueError(
			"the manifest has no eligible training rows; fresh collection is pending review, "
			"so pass include_pending to build from unreviewed responses"
		)
	return _seal(
		{
			"version": ARM_VERSION,
			"manifest_digest": manifest["digest"],
			"recipe": asdict(recipe),
			"include_pending": include_pending,
			"adapters": adapters.identity,
			"published_instruction": {
				"text": DENSE_INSTRUCTION,
				"paper_sha256": PAPER_SHA256,
				"section": "A.2; same-size rewriter discussion E.3",
			}
			if recipe.arm == "dense_locked"
			else None,
			"parent_digest": protected["digest"] if protected else None,
			"rows": rows,
			"baseline_settings": baseline_settings,
			"admissions": admissions,
			"released_tokens": sum(row["released_tokens"] for row in rows),
			"fallbacks": sum(row["fallback"] is not None for row in rows),
		}
	)


def select_training_set(
	arm: dict[str, Any],
	*,
	by: Literal["queries", "tokens"],
	budget: int,
	seed: int,
) -> dict[str, Any]:
	"""Select a nested seeded prefix with exact released tokens, never upstream usage.

	Withheld rows still consume their acquisition query. No new calls are made when a short
	output uses more cached examples. Exhaustion and discrete token slack remain explicit.
	"""
	_require_version(arm, ARM_VERSION)
	# Reuse the manifest selector's ordering and prefix rules on the *released* token counts.
	# The clean label states that these counts are what the client received, which is why the
	# selector's refusal of a transformed policy's teacher-reported tokens does not apply.
	selection = select_rows(
		{
			"response_policy": "clean",
			"digest": arm["manifest_digest"],
			"examples": [
				{
					"example_id": row["example_id"],
					"split": "train",
					"quality_status": "accepted",
					"output_tokens": row["released_tokens"],
				}
				for row in arm["rows"]
			],
		},
		by=by,
		budget=budget,
		seed=seed,
	)
	ids = selection["example_ids"]
	by_id = {row["example_id"]: row for row in arm["rows"]}
	return _seal(
		{
			"version": "student-training-set-v1",
			"manifest_digest": arm["manifest_digest"],
			"arm_digest": arm["digest"],
			"recipe": arm["recipe"],
			"adapters": arm["adapters"],
			"baseline_settings": arm.get("baseline_settings"),
			"parent_digest": arm.get("parent_digest"),
			"selection": selection,
			"rows": [by_id[key] for key in ids],
			"trainable_rows": sum(by_id[key]["sft"] is not None for key in ids),
		}
	)


def write_training_set(artifact: dict[str, Any], path: Path) -> None:
	verify_artifact(artifact)
	write_new(path, json.dumps(artifact, indent="\t") + "\n")
