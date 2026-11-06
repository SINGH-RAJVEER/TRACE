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
