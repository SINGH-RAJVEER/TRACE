"""Functional agreement traces collected inside the same boundary as correctness scoring."""

import ast
import json
from dataclasses import dataclass
from functools import lru_cache

from coding.extract import extract_code
from coding.sandbox import run_python


@dataclass(frozen=True)
class Trace:
	valid: bool = False
	calls: int = 0
	inputs_sha256: str = ""
	outputs_sha256: str = ""


class _ObserveAssertions(ast.NodeTransformer):
	def visit_Assert(self, node: ast.Assert) -> ast.Try:
		# Evaluate the original expression, including candidate calls, but keep going when
		# an assertion is false or its comparison raises. Candidate exceptions are marked
		# invalid by _observe; an oracle rejecting a wrong type is not an invocation error.
		return ast.Try(
			body=[ast.Expr(node.test)],
			handlers=[
				ast.ExceptHandler(
					type=ast.Name(id="Exception", ctx=ast.Load()),
					name=None,
					body=[ast.Pass()],
				)
			],
			orelse=[],
			finalbody=[],
		)


@lru_cache(maxsize=512)
def execution_trace(code: str, tests: str, entry_point: str) -> Trace:
	"""Compare typed outputs on identical ordered calls, including shared wrong answers.

	Arbitrary objects and execution failures have no valid trace. Float equality is exact,
	including signed zero; this is independent of the correctness oracle's tolerances.
	This shares the assertion scorer's documented same-interpreter tampering limitation.
	"""
	code = extract_code(code) or ""
	if not code or not entry_point.isidentifier():
		return Trace()
	tree = ast.fix_missing_locations(_ObserveAssertions().visit(ast.parse(tests)))
	observed = ast.unparse(tree)
	driver = """
import hashlib as _hashlib, json as _json, random as _random
_random.seed(42)
try:
	import numpy as _np
	_np.random.seed(42)
except ImportError:
	_np = None
_trace_valid = True
_trace_calls = 0
_trace_inputs = _hashlib.sha256()
_trace_outputs = _hashlib.sha256()
def _encode(value):
	if _np is not None and isinstance(value, _np.ndarray):
		return ['array', str(value.dtype), list(value.shape), _encode(value.tolist())]
	if _np is not None and isinstance(value, _np.generic):
		return _encode(value.item())
	if value is None or type(value) in (bool, int, str):
		return [type(value).__name__, value]
	if type(value) is float:
		return ['float', value.hex()]
	if type(value) in (list, tuple):
		return [type(value).__name__, [_encode(v) for v in value]]
	if type(value) in (set, frozenset):
		return [type(value).__name__, sorted((_encode(v) for v in value), key=_json.dumps)]
	if type(value) is dict:
		pairs = ([_encode(k), _encode(v)] for k, v in value.items())
		return ['dict', sorted(pairs, key=_json.dumps)]
	raise TypeError('unsupported trace output type')
def _observe(*args, **kwargs):
	global _trace_valid, _trace_calls
	_trace_calls += 1
	try:
		encoded = _json.dumps(_encode([args, kwargs]), sort_keys=True) + '\\n'
		_trace_inputs.update(encoded.encode())
		globals()[_trace_entry] = _candidate
		value = _candidate(*args, **kwargs)
		_trace_outputs.update((_json.dumps(_encode(value), sort_keys=True) + '\\n').encode())
		return value
	except BaseException:
		_trace_valid = False
		raise
	finally:
		globals()[_trace_entry] = _observe
"""
	source = (
		driver
		+ f"\nexec(compile({code!r}, '<candidate>', 'exec'))\n"
		+ f"_trace_entry = {entry_point!r}\n_candidate = {entry_point}\n{entry_point} = _observe\n"
		+ f"try:\n\texec(compile({observed!r}, '<trace-tests>', 'exec'))\n"
		+ "except BaseException:\n\t_trace_valid = False\n"
		+ "print('CODING_TRACE=' + _json.dumps({'valid': bool(_trace_valid and _trace_calls), "
		+ "'calls': _trace_calls, 'inputs_sha256': _trace_inputs.hexdigest(), "
		+ "'outputs_sha256': _trace_outputs.hexdigest()}))\n"
	)
	result = run_python(source)
	if result.status != "ok" or result.returncode != 0:
		return Trace()
	for line in reversed(result.stdout.splitlines()):
		if line.startswith("CODING_TRACE="):
			try:
				return Trace(**json.loads(line.removeprefix("CODING_TRACE=")))
			except TypeError, ValueError:
				return Trace()
	return Trace()
