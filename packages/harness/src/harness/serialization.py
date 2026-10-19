"""Detector serialization: the task content of a request, without fixed service instructions.

The paper embeds the current user turn excluding the service's own instructions (Section
III-B), so the detector compares what clients ask about, not the shared prompt template. The
markers below mirror the task prompts in `distillation.tasks`; a test keeps them in step.
"""

import json
from typing import Literal, cast

Task = Literal["ag_news", "squad", "gsm8k", "coding", "text", "gateway"]

# Recorded in every calibration artifact. Change it whenever the extraction rules change.
SERIALIZATION = "detector-v2"

ARTICLE = "\n\nArticle: "
PASSAGE = "\n\nPassage: "
QUESTION = "\n\nQuestion: "
CODING_PREFIX = (
	"Implement the Python function described below. Return only a Python code block."
	"\n\nSpecification: "
)
GSM8K_SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."


def detector_text(task: Task, turn: str) -> str:
	"""The text the detector embeds for one user turn.

	- ag_news: the article.
	- squad: the question, then the passage, so encoder truncation removes passage text first.
	- gsm8k: the problem, without the fixed solving instruction.
	- text: the whole turn, for services without a fixed template.

	A turn that does not follow its task's template is embedded whole: an off-template request
	is itself part of the observed traffic.
	"""
	if task == "gateway":
		value = json.loads(turn)
		domain = value["domain"]
		if domain not in {"ag_news", "squad", "gsm8k", "coding"}:
			raise ValueError("unsupported gateway domain")
		return f"{domain}\n{detector_text(cast(Task, domain), value['turn'])}"
	if task == "ag_news":
		head, marker, article = turn.rpartition(ARTICLE)
		return article if marker and head else turn
	if task == "squad":
		head, marker, question = turn.rpartition(QUESTION)
		prefix, separator, passage = head.rpartition(PASSAGE)
		if marker and separator and prefix:
			return f"{question}\n\n{passage}"
		return turn
	if task == "coding":
		if turn.startswith(CODING_PREFIX):
			content, marker, signature = turn[len(CODING_PREFIX) :].rpartition("\n\nSignature: ")
			return content if marker and signature else turn
		return turn
	if task == "gsm8k":
		return turn.removesuffix(GSM8K_SUFFIX)
	return turn


def gateway_turn(domain: str, turn: str) -> str:
	"""Bind the declared domain to content identically in calibration and serving."""
	return json.dumps({"domain": domain, "turn": turn}, sort_keys=True)


def valid_gateway_turn(domain: str, turn: str) -> bool:
	"""The controlled gateway accepts one declared task template, never inferred substrings."""
	if domain == "ag_news":
		prefix = (
			"Classify the news article into one of these topics: "
			"World, Sports, Business, Sci/Tech. "
			"Answer with only the topic name.\n\nArticle: "
		)
		return turn.startswith(prefix) and bool(turn[len(prefix) :].strip())
	if domain == "squad":
		prefix = (
			"Answer the question with a short span copied from the passage. "
			"Return only the answer.\n\nPassage: "
		)
		passage, marker, question = turn[len(prefix) :].rpartition(QUESTION)
		return turn.startswith(prefix) and bool(passage.strip() and marker and question.strip())
	if domain == "gsm8k":
		return turn.endswith(GSM8K_SUFFIX) and bool(turn.removesuffix(GSM8K_SUFFIX).strip())
	if domain == "coding":
		specification, marker, signature = turn[len(CODING_PREFIX) :].rpartition("\n\nSignature: ")
		return turn.startswith(CODING_PREFIX) and bool(
			specification.strip() and marker and signature.strip()
		)
	return False
