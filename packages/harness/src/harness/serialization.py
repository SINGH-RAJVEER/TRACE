"""Detector serialization: the task content of a request, without fixed service instructions.

The paper embeds the current user turn excluding the service's own instructions (Section
III-B), so the detector compares what clients ask about, not the shared prompt template. The
markers below mirror the task prompts in `distillation.tasks`; a test keeps them in step.
"""

from typing import Literal

Task = Literal["ag_news", "squad", "gsm8k", "text"]

# Recorded in every calibration artifact. Change it whenever the extraction rules change.
SERIALIZATION = "detector-v1"

ARTICLE = "\n\nArticle: "
PASSAGE = "\n\nPassage: "
QUESTION = "\n\nQuestion: "
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
	if task == "ag_news":
		head, marker, article = turn.rpartition(ARTICLE)
		return article if marker and head else turn
	if task == "squad":
		head, marker, question = turn.rpartition(QUESTION)
		prefix, separator, passage = head.rpartition(PASSAGE)
		if marker and separator and prefix:
			return f"{question}\n\n{passage}"
		return turn
	if task == "gsm8k":
		return turn.removesuffix(GSM8K_SUFFIX)
	return turn
