"""Parse generated Python without executing it."""

import ast
import re


def extract_code(response: str) -> str | None:
	"""Use the first Python fence, otherwise require the whole response to parse."""
	match = re.search(r"```(?:python|py)\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
	code = (match.group(1) if match else response).strip()
	if not code:
		return None
	try:
		tree = ast.parse(code)
	except SyntaxError, ValueError, RecursionError:
		return None
	return code if tree.body else None
