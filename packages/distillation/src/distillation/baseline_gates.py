"""Machine-checked decisions that a task has headroom and a clean student can learn it.

An incapable student makes every defense look effective, so these gates run before any defended
student is compared. The headroom gate (Phase 0) reads base and teacher per-problem scores on one
suite. The clean gate (Phase 1) reads a learning-curve directory. Both write a versioned report
with the SHA-256 of every input and a digest of the report itself. Model identities in score files
and the `clean` label on a manifest are the operator's assertions; hashes show byte identity only.
Pure data and hashing, with no model or network, except the optional evaluation helpers below.
"""

import asyncio
import hashlib
import importlib
import json
import math
import random
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx

from distillation.acquisition_manifest import load_manifest, write_new
from distillation.benchmarks.models import BenchmarkItem, FrozenSuite
from distillation.benchmarks.scoring import evaluate_answer
from distillation.io import canonical_json, read_jsonl
from distillation.learning_curve import (
	MATH_RECIPE,
	digest,
	load_plan,
	math_family,
	validate_scores,
	write,
)
from distillation.schemas import (
	Message,
	PromptRecord,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
)
from distillation.transfer_metrics import SuppressionGate, _bounds, _decimal

GATE_VERSION = "clean-baseline-gate-v1"
SCORES_VERSION = "problem-scores-v1"
HEADROOM = "phase0-headroom"
CLEAN = "phase1-clean"
# The sizes and the seed floor are the gate's definition, so neither has an option to relax it.
REQUIRED_SIZES = (300, 1000)
MIN_SEEDS = 2
MIN_REPLICATES = 1000
MARGIN_STATUS = "Declared by the operator for this gate; a preregistered choice, not established."
METHOD = "percentile, paired resampling of task families"
SEEDS_NOTE = (
	"Training seeds are not part of the bootstrap; each declared seed must pass on its own."
)
FIX_HEADROOM = "Change the domain or the base before collecting or training anything."
FIX_LEARNING = (
	"Fix the learning experiment: an incapable student makes every defense look effective."
)
ONE_SEED = "One training seed is development evidence only and cannot pass this gate."


def read_items(path: Path) -> list[BenchmarkItem]:
	"""Items of a frozen suite file, or of the suite.json a learning-curve plan freezes."""
	value = json.loads(path.read_text())
	if isinstance(value, list):
		return [BenchmarkItem.model_validate(item) for item in value]
	suite = FrozenSuite.model_validate(value)
	suite.verify()
	return suite.items


def read_gate(path: Path | None, manifest_digest: str) -> dict[str, Any]:
	"""Require an intact, passing clean gate bound to this acquisition manifest."""
	if path is None:
		raise ValueError("missing clean-baseline gate report")
	gate = load_gate(path)
	if (
		gate.get("gate") != CLEAN
		or gate.get("passed") is not True
		or gate.get("manifest_digest") != manifest_digest
	):
		raise ValueError("clean-baseline gate must pass and bind this acquisition manifest")
	return {"path": str(path.resolve()), "sha256": digest(path), "report": gate}


def suite_digest(items: Sequence[BenchmarkItem]) -> str:
	"""A hash of the problems themselves, equal for a frozen suite and a plan's copy of it."""
	rows = [item.model_dump(mode="json") for item in items]
	return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


def suite_task(items: Sequence[BenchmarkItem]) -> str:
	tasks = {item.benchmark for item in items}
	if len(tasks) != 1:
		raise ValueError("a gate covers one task; the suite mixes several")
	return tasks.pop()


def _families(items: Sequence[BenchmarkItem]) -> dict[str, list[str]]:
	"""Item IDs by task family, for the one task whose family key a suite can reproduce.

	The GSM8K key is the normalized question, so a family is usually one problem and the
	bootstrap then resamples problems. Other tasks need their own family rule first.
	"""
	if suite_task(items) != "gsm8k":
		raise ValueError("gates define task families for gsm8k only")
	families: dict[str, list[str]] = {}
	for item in items:
		families.setdefault(math_family(item.question), []).append(item.id)
	if len(families) < 2:
		raise ValueError("need at least two task families")
	return dict(sorted(families.items()))


def _gain(
	panel: Sequence[dict[str, Any]],
	reference: dict[str, Any],
	families: dict[str, list[str]],
	replicates: int,
	seed: int,
	label: str,
) -> dict[str, Any]:
	"""Exact gain in points of the mean of a score panel, with a family-resampled interval."""
	numerators = [
		sum(int(scores[i]) - int(reference[i]) for scores in panel for i in ids)
		for ids in families.values()
	]
	sizes = [len(panel) * len(ids) for ids in families.values()]
	rng = random.Random(f"{seed}:{label}")
	indices = range(len(numerators))
	draws = []
	for _ in range(replicates):
		chosen = rng.choices(indices, k=len(numerators))
		draws.append(
			Fraction(sum(numerators[i] for i in chosen), sum(sizes[i] for i in chosen)) * 100
		)
	point = Fraction(sum(numerators), sum(sizes)) * 100
	lower, upper = _bounds(draws)
	return {
		"gain_pp": float(point),
		"ci95_pp": [float(lower), float(upper)],
		# Exact, so a zero endpoint counts as zero rather than as a rounding residue.
		"positive_excluding_zero": point > 0 and lower > 0,
		"_exact": point,
	}


def _public(entry: dict[str, Any]) -> dict[str, Any]:
	return {key: value for key, value in entry.items() if not key.startswith("_")}


def _parameters(replicates: int, seed: int) -> dict[str, Any]:
	if replicates < MIN_REPLICATES:
		raise ValueError(f"a gate needs at least {MIN_REPLICATES} bootstrap replicates")
	return {"replicates": replicates, "seed": seed, "method": METHOD}


def _seal(report: dict[str, Any]) -> dict[str, Any]:
	"""Key the report to its inputs and parameters, then digest the whole of it."""
	report["inputs_sha256"] = hashlib.sha256(
		canonical_json({"inputs": report["inputs"], "parameters": report["parameters"]}).encode()
	).hexdigest()
	report["digest"] = hashlib.sha256(canonical_json(report).encode()).hexdigest()
	return report


def _read_scores(path: Path, role: str, suite: str, ids: set[str]) -> dict[str, Any]:
	value = json.loads(path.read_text())
	if (
		not isinstance(value, dict)
		or value.get("version") != SCORES_VERSION
		or value.get("role") != role
	):
		raise ValueError(f"{path} is not a {SCORES_VERSION} score file for the {role}")
	if value.get("suite_sha256") != suite:
		raise ValueError(f"{path} scores a different suite")
	validate_scores(value["scores"], ids)
	if not value.get("model") or not value.get("revision"):
		raise ValueError(f"{path} does not name the model and revision it scored")
	return dict(value)


def headroom_gate(
	base_scores: Path,
	teacher_scores: Path,
	suite_path: Path,
	*,
	margin_points: float,
	replicates: int = 2000,
	seed: int = 42,
) -> dict[str, Any]:
	"""Pass only if teacher minus base meets the declared margin and its interval excludes zero."""
	if (
		not isinstance(margin_points, int | float)
		or not math.isfinite(margin_points)
		or margin_points <= 0
	):
		raise ValueError("the headroom margin must be a positive number of points")
	parameters = {
		"margin_points": margin_points,
		"margin_status": MARGIN_STATUS,
		**_parameters(replicates, seed),
	}
	items = read_items(suite_path)
	suite = suite_digest(items)
	ids = {item.id for item in items}
	base = _read_scores(base_scores, "base", suite, ids)
	teacher = _read_scores(teacher_scores, "teacher", suite, ids)
	families = _families(items)
	headroom = _gain([teacher["scores"]], base["scores"], families, replicates, seed, "headroom")
	reasons = []
	if headroom["_exact"] < _decimal(margin_points):
		reasons.append(
			f"teacher minus base is {headroom['gain_pp']:.2f} points, below the declared "
			f"{margin_points} point margin"
		)
	if not headroom["positive_excluding_zero"]:
		reasons.append("the teacher-base interval does not exclude zero")
	return _seal(
		{
			"version": GATE_VERSION,
			"gate": HEADROOM,
			"task": suite_task(items),
			"base": {"model": base["model"], "revision": base["revision"]},
			"teacher": {"model": teacher["model"], "revision": teacher["revision"]},
			"student_revision": None,
			"suite_sha256": suite,
			"inputs": {
				"base_scores": digest(base_scores),
				"teacher_scores": digest(teacher_scores),
				"suite": digest(suite_path),
			},
			"parameters": parameters,
			"items": len(ids),
			"families": len(families),
			"results": {
				"base_accuracy": sum(base["scores"].values()) / len(ids),
				"teacher_accuracy": sum(teacher["scores"].values()) / len(ids),
				"headroom": _public(headroom),
			},
			"status": "failed" if reasons else "passed",
			"passed": not reasons,
			"reasons": reasons,
			"action": FIX_HEADROOM if reasons else None,
		}
	)


def student_identity(plan: dict[str, Any], items: Sequence[BenchmarkItem]) -> dict[str, str]:
	"""What a clean gate covers: the task, the base, and the student recipe as a revision.

	A recipe change is a different student, so the revision hashes the plan version and recipe.
	A defended student planned with the same recipe from the same base has the same identity.
	"""
	recipe = plan["recipe"]
	return {
		"task": suite_task(items),
		"base_model": recipe["base_model"],
		"base_revision": recipe["revision"],
		"student_revision": hashlib.sha256(
			canonical_json({"version": plan["version"], "recipe": recipe}).encode()
		).hexdigest(),
	}


def plan_identity(curve: Path) -> dict[str, str]:
	"""The identity of the student frozen in a learning-curve directory, hashes checked."""
	return student_identity(load_plan(curve), read_items(curve / "suite.json"))


def _cell(
	curve: Path, name: str, contract: str, n: int | None, seed: int, arm: str, ids: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
	path = curve / "scores" / f"{name}.json"
	if not path.is_file():
		return None, None
	result = json.loads(path.read_text())
	if (
		result.get("contract") != contract
		or result.get("n") != n
		or result.get("seed") != seed
		or result.get("arm", "teacher") != arm
	):
		raise ValueError(f"{path} belongs to a different plan or cell")
	validate_scores(result["scores"], ids)
	return result["scores"], digest(path)


def _arm_results(
	curve: Path,
	plan: dict[str, Any],
	arm: str,
	base: dict[str, Any],
	families: dict[str, list[str]],
	contract: str,
	ids: set[str],
	inputs: dict[str, str],
	replicates: int,
	seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
	"""One row per required size with a gain per declared seed and pooled over them."""
	rows, failures = [], []
	prefix = "gold-" if arm == "gold" else ""
	for n in REQUIRED_SIZES:
		seeds: dict[str, Any] = {}
		panel = []
		for training_seed in plan["seeds"]:
			name = f"{prefix}n{n}-s{training_seed}"
			scores, sha = _cell(curve, name, contract, n, training_seed, arm, ids)
			if scores is None:
				seeds[str(training_seed)] = None
				failures.append(f"{arm} scores for n={n}, seed {training_seed} are missing")
				continue
			inputs[f"scores/{name}.json"] = str(sha)
			panel.append(scores)
			label = f"{arm}:{n}:{training_seed}"
			entry = _gain([scores], base, families, replicates, seed, label)
			seeds[str(training_seed)] = _public(entry)
			if not entry["positive_excluding_zero"]:
				failures.append(
					f"{arm} n={n}, seed {training_seed}: gain {entry['gain_pp']:.2f} points "
					f"with interval {entry['ci95_pp']} is not positive excluding zero"
				)
		row: dict[str, Any] = {"n": n, "seeds": seeds, "pooled": None}
		if panel:
			pooled = _gain(panel, base, families, replicates, seed, f"{arm}:{n}:pooled")
			row["pooled"] = _public(pooled)
			if not pooled["positive_excluding_zero"]:
				failures.append(
					f"{arm} n={n}: pooled gain {pooled['gain_pp']:.2f} points with interval "
					f"{pooled['ci95_pp']} is not positive excluding zero"
				)
		rows.append(row)
	return rows, failures


def _headroom_reasons(report: dict[str, Any], identity: dict[str, str], suite: str) -> list[str]:
	if report["gate"] != HEADROOM:
		return ["the headroom report is not a headroom gate"]
	reasons = []
	if report["passed"] is not True:
		reasons.append("the headroom gate did not pass")
	if report["task"] != identity["task"] or report["base"] != {
		"model": identity["base_model"],
		"revision": identity["base_revision"],
	}:
		reasons.append("the headroom gate covers a different task or base")
	if report["suite_sha256"] != suite:
		reasons.append("the headroom gate scored different held-out problems")
	return reasons


def _clean_ledger_reasons(plan: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
	reasons = []
	if manifest["response_policy"] != "clean":
		reasons.append(f"the manifest declares response policy {manifest['response_policy']!r}")
	covered = {entry["sha256"] for entry in manifest["inputs"]}
	ledgers = plan.get("training_ledgers")
	if not ledgers:
		reasons.append("the plan does not name its training ledgers")
	elif any(plan["inputs"][ledger] not in covered for ledger in ledgers):
		reasons.append("a training ledger of this plan is not in the clean manifest")
	return reasons


def clean_gate(
	curve: Path,
	headroom: Path,
	manifest: Path,
	*,
	replicates: int = 2000,
	seed: int = 42,
) -> dict[str, Any]:
	"""Pass only if the clean student beats the base at 300 and 1,000 examples on every seed.

	Each declared seed must have a positive gain whose family-resampled interval excludes zero,
	at both sizes, and so must the pooled gain. A single declared seed can never pass. The gold
	student must be present and is reported as a diagnostic: it says whether the recipe can learn
	the task at all, but its gain does not stand in for the clean gain. A passing headroom gate
	on the same task, base and problems and a clean acquisition manifest of the training ledger
	are required.
	"""
	parameters = {
		"required_sizes": list(REQUIRED_SIZES),
		"min_seeds": MIN_SEEDS,
		"seeds_note": SEEDS_NOTE,
		**_parameters(replicates, seed),
	}
	plan = load_plan(curve)
	contract = digest(curve / "plan.json")
	items = read_items(curve / "suite.json")
	identity = student_identity(plan, items)
	suite = suite_digest(items)
	ids = {item.id for item in items}
	families = _families(items)
	inputs = {
		"plan": contract,
		"headroom_gate": digest(headroom),
		"acquisition_manifest": digest(manifest),
	}
	reasons = _headroom_reasons(load_gate(headroom), identity, suite)
	acquisition = load_manifest(manifest)
	reasons += _clean_ledger_reasons(plan, acquisition)
	seeds = plan["seeds"]
	single_seed = len(seeds) < MIN_SEEDS
	if single_seed:
		reasons.append(ONE_SEED)
	missing = [n for n in REQUIRED_SIZES if n not in plan["sizes"]]
	if missing:
		reasons.append(f"the plan does not include the required sizes {missing}")
	base, sha = _cell(curve, "base", contract, None, seeds[0], "teacher", ids)
	results: dict[str, Any] = {"clean": [], "gold": []}
	gold_failures = ["the base scores are missing"]
	if base is None:
		reasons.append("the base scores are missing")
	else:
		inputs["scores/base.json"] = str(sha)
		results["clean"], clean_failures = _arm_results(
			curve, plan, "teacher", base, families, contract, ids, inputs, replicates, seed
		)
		results["gold"], gold_failures = _arm_results(
			curve, plan, "gold", base, families, contract, ids, inputs, replicates, seed
		)
		reasons += clean_failures
		reasons += [f for f in gold_failures if f.endswith("are missing")]
	passed = not reasons
	status = "passed" if passed else "development-only" if single_seed else "failed"
	report = {
		"version": GATE_VERSION,
		"gate": CLEAN,
		"task": identity["task"],
		"base": {"model": identity["base_model"], "revision": identity["base_revision"]},
		"student_revision": identity["student_revision"],
		"suite_sha256": suite,
		"manifest_digest": acquisition["digest"],
		"inputs": inputs,
		"parameters": parameters,
		"items": len(ids),
		"families": len(families),
		"declared_seeds": seeds,
		"results": results,
		"gold_diagnostic": {
			"learned": not gold_failures,
			"note": "The gold arm shows whether the recipe can learn the task from trusted "
			"answers. If it learned and the clean arm did not, suspect the teacher data.",
		},
		"status": status,
		"passed": passed,
		"reasons": reasons,
		"action": None if passed else FIX_LEARNING,
	}
	return _seal(report)


def write_gate(report: dict[str, Any], output: Path) -> str:
	"""Publish a new report; a gate is never overwritten. Returns the file's SHA-256."""
	write_new(output, json.dumps(report, indent="\t", allow_nan=False) + "\n")
	return digest(output)


def load_gate(path: Path) -> dict[str, Any]:
	"""A gate report whose version and digest check out; an edited report is refused."""
	report = json.loads(path.read_text())
	if not isinstance(report, dict):
		raise ValueError(f"{path} is not an intact {GATE_VERSION} report")
	sealed = {key: value for key, value in report.items() if key != "digest"}
	if (
		report.get("version") != GATE_VERSION
		or report.get("digest") != hashlib.sha256(canonical_json(sealed).encode()).hexdigest()
	):
		raise ValueError(f"{path} is not an intact {GATE_VERSION} report")
	return report


def covering_gate(report: dict[str, Any], identity: dict[str, str]) -> SuppressionGate:
	"""Whether a verified report licenses suppression for a student of this identity."""
	reason = None
	if report["gate"] != CLEAN:
		reason = "only a clean-student gate covers suppression; a headroom gate does not"
	elif report["passed"] is not True:
		reason = f"the clean-student gate has status {report['status']}: not a passing gate"
	elif (
		report["task"] != identity["task"]
		or report["base"]["model"] != identity["base_model"]
		or report["base"]["revision"] != identity["base_revision"]
		or report["student_revision"] != identity["student_revision"]
	):
		reason = "the clean-student gate covers a different task, base or student revision"
	return SuppressionGate(
		covered=reason is None, reason=reason, digest=report["digest"], identity=dict(identity)
	)


# Evaluation entry points for the GPU host. They write the score files the headroom gate reads.


def _score_file(
	role: str,
	model: str,
	revision: str,
	items: Sequence[BenchmarkItem],
	suite_path: Path,
	scores: dict[str, Any],
	**extra: Any,
) -> dict[str, Any]:
	validate_scores(scores, {item.id for item in items})
	return {
		"version": SCORES_VERSION,
		"role": role,
		"model": model,
		"revision": revision,
		"task": suite_task(items),
		"suite_sha256": suite_digest(items),
		"suite_file_sha256": digest(suite_path),
		"evaluation_max_new_tokens": MATH_RECIPE["evaluation_max_new_tokens"],
		"scores": scores,
		**extra,
	}


def evaluate_base(
	suite_path: Path,
	output: Path,
	*,
	seed: int = 42,
	runner: Callable[[Path, dict[str, Any], int | None, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
	"""Score the pinned untrained base through the same runner the curve uses."""
	if runner is None:
		from distillation.learning_curve_runner import run_cell

		runner = run_cell
	items = read_items(suite_path)
	with tempfile.TemporaryDirectory() as work:
		write(Path(work) / "suite.json", [item.model_dump(mode="json") for item in items])
		scores = runner(Path(work), MATH_RECIPE, None, seed)
	result = _score_file(
		"base", MATH_RECIPE["base_model"], MATH_RECIPE["revision"], items, suite_path, scores
	)
	write_new(output, json.dumps(result, indent="\t") + "\n")
	return result


async def _teacher_scores(
	items: Sequence[BenchmarkItem],
	*,
	url: str,
	token: str,
	model: str,
	concurrency: int,
	timeout_seconds: float,
	transport: httpx.AsyncBaseTransport | None,
) -> tuple[dict[str, int], Counter[str]]:
	semaphore = asyncio.Semaphore(concurrency)
	scores: dict[str, int] = {}
	finishes: Counter[str] = Counter()
	failures: list[str] = []
	headers = {"Authorization": f"Bearer {token}"} if token else {}
	async with httpx.AsyncClient(
		base_url=url.rstrip("/") + "/",
		headers=headers,
		timeout=timeout_seconds,
		transport=transport,
	) as client:

		async def score(item: BenchmarkItem) -> None:
			request = TeacherGenerationRequest(
				messages=item.messages,
				max_output_tokens=MATH_RECIPE["evaluation_max_new_tokens"],
			)
			async with semaphore:
				try:
					response = await client.post(
						"v1/teacher/generate", json=request.model_dump(mode="json")
					)
					if "x-harness-decision" in response.headers:
						raise ValueError("response came through a harness, not the direct teacher")
					response.raise_for_status()
					generation = TeacherGeneration.model_validate(response.json())
					if generation.model != model:
						raise ValueError(f"teacher answered as {generation.model}, not {model}")
				except (httpx.HTTPError, ValueError) as error:
					failures.append(f"{item.id}: {error}")
					return
			result = await evaluate_answer(
				item, generation.message.content, generation.finish_reason
			)
			scores[item.id] = int(result.correct)
			finishes[generation.finish_reason or "unknown"] += 1

		await asyncio.gather(*(score(item) for item in items))
	if failures:
		# Counting a failed request as wrong would understate the teacher and the headroom.
		raise ValueError(
			f"teacher failed on {len(failures)} of {len(items)} problems; no scores written. "
			f"First: {failures[0]}"
		)
	return scores, finishes


def evaluate_teacher(
	suite_path: Path,
	output: Path,
	*,
	url: str,
	model: str,
	revision: str,
	token: str = "",
	concurrency: int = 4,
	timeout_seconds: float = 300,
	transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
	"""Score a direct teacher under the students' token budget, with one attempt per problem."""
	items = read_items(suite_path)
	scores, finishes = asyncio.run(
		_teacher_scores(
			items,
			url=url,
			token=token,
			model=model,
			concurrency=concurrency,
			timeout_seconds=timeout_seconds,
			transport=transport,
		)
	)
	result = _score_file(
		"teacher",
		model,
		revision,
		items,
		suite_path,
		scores,
		endpoint_sha256=hashlib.sha256(url.encode()).hexdigest(),
		finish_reasons=dict(finishes),
	)
	write_new(output, json.dumps(result, indent="\t") + "\n")
	return result


def collect_clean(
	pool_path: Path,
	output: Path,
	*,
	count: int,
	seed: int,
	url: str,
	terms_review_id: str,
	token: str = "",
	concurrency: int = 4,
	max_output_tokens: int = MATH_RECIPE["evaluation_max_new_tokens"],
	transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int, int]:
	"""Collect one task's fresh responses from a direct teacher, resuming a partial ledger.

	Takes the first `count` training prompts by sha256(seed:example_id), so a rerun selects the
	same prompts. `gateway schedule` needs several domains to keep its detector mix, and this
	collection has no detector in the path. The ledger is what the manifest and the plan read.
	"""
	from distillation.collection import CollectionOptions, collect
	from distillation.teacher import TeacherAPIGateway

	prompts = [row for row in read_jsonl(pool_path, PromptRecord) if row.split == "train"]
	prompts.sort(key=lambda row: hashlib.sha256(f"{seed}:{row.example_id}".encode()).hexdigest())
	if len(prompts) < count:
		raise ValueError(f"{pool_path} has {len(prompts)} training prompts, fewer than {count}")
	selected = output.with_name(output.name + ".prompts.jsonl")
	output.parent.mkdir(parents=True, exist_ok=True)
	selected.write_text("".join(row.model_dump_json() + "\n" for row in prompts[:count]))
	return asyncio.run(
		collect(
			input_path=selected,
			output_path=output,
			teacher=TeacherAPIGateway(base_url=url, token=token, transport=transport),
			options=CollectionOptions(
				terms_review_id=terms_review_id,
				concurrency=concurrency,
				max_output_tokens=max_output_tokens,
			),
		)
	)


def pilot_student(
	output: Path,
	suite_path: Path,
	pool_path: Path,
	gold_path: Path,
	*,
	n: int = 32,
	items: int = 16,
	seed: int = 42,
	runner: Callable[[Path, dict[str, Any], int | None, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
	"""Time one small training cell on the longest gold examples, the realistic worst case.

	Uses trusted gold answers because no teacher response exists yet. The result sizes the
	machine and is not evidence of learning; its scores are discarded.
	"""
	from distillation.tasks import GoldRecord

	if output.exists():
		raise ValueError("pilot output must be a new directory")
	torch = None
	if runner is None:
		from distillation.learning_curve_runner import run_cell

		runner = run_cell
		torch = importlib.import_module("torch")
	gold = {row.example_id: row.completion for row in read_jsonl(gold_path, GoldRecord)}
	rows = [
		{
			"example_id": record.example_id,
			"sft": SFTRecord(
				prompt=record.prompt,
				completion=[Message(role="assistant", content=gold[record.example_id])],
			).model_dump(mode="json"),
		}
		for record in read_jsonl(pool_path, PromptRecord)
		if record.example_id in gold
	]
	rows.sort(key=lambda row: -len(json.dumps(row["sft"], sort_keys=True)))
	suite = read_items(suite_path)[:items]
	if len(rows) < n or not suite:
		raise ValueError("the pool or suite is too small for this pilot")
	output.mkdir(parents=True)
	write(output / f"subset-{n}.json", rows[:n])
	write(output / "suite.json", [item.model_dump(mode="json") for item in suite])
	started = time.monotonic()
	runner(output, MATH_RECIPE, n, seed)
	seconds = time.monotonic() - started
	result = {
		"version": "student-pilot-v1",
		"n": n,
		"evaluation_items": len(suite),
		"seconds": seconds,
		"peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if torch else None,
		"peak_memory_reserved_bytes": torch.cuda.max_memory_reserved() if torch else None,
		"recipe": MATH_RECIPE,
		"note": "Machine sizing on the longest gold examples; not evidence of learning.",
	}
	write(output / "pilot.json", result)
	return result
