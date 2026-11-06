"""Cheap text definitions shared by offline and bounded online session telemetry."""

import re


def answer_only(text: str) -> bool:
	return bool(
		re.search(
			r"\b(answer only|only (?:the )?answer|no explanation|just (?:the )?answer)\b",
			text.lower(),
		)
	)


def followup_language(text: str) -> bool:
	return bool(re.search(r"\b(also|instead|again|previous|that|follow.up)\b", text.lower()))


def online_answer_only(text: str) -> bool:
	"""Recognize supported answer-format requests without changing frozen offline features."""
	return answer_only(text) or bool(
		re.search(
			r"\b(?:return|output|respond|answer)(?:\s+with)?\s+(?:only|just)\s+"
			r"(?:the\s+)?(?:final\s+)?(?:numeric\s+)?(?:answer|result|number|topic\s+name)\b"
			r"|\breply\s+with\s+(?:one|a\s+single)\s+word\b"
			r"|\bend\s+with\s+####\s+followed\s+by\s+the\s+final\s+numeric\s+answer\b",
			text,
			re.I,
		)
	)


def online_contextual_language(text: str) -> bool:
	"""Window-level follow-up context, retaining the v2 prefix vocabulary."""
	return bool(
		re.match(
			r"\s*(?:and|also|why|how about|what about|continue)\b"
			r"|\s*(?:following up on|regarding your previous|in your previous answer)\b",
			text,
			re.I,
		)
	)


def online_workflow_language(text: str) -> bool:
	"""Narrow editing instructions with explicit local-document context."""
	return bool(
		re.match(
			r"\s*(?:please\s+)?(?:edit|revise|rewrite|debug|proofread|clarify)\s+"
			r"(?:this|my|the following|your previous)\b",
			text,
			re.I,
		)
	)
