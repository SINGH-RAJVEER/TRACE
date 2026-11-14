"""Authenticated vLLM configuration acknowledgement for fail-closed teacher releases."""

import re
from importlib.metadata import version

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from trace_watermark.serving import (
	ACK_HEADER,
	CHALLENGE_HEADER,
	PROCESSOR,
	VLLM_VERSION,
	KgwSpec,
	acknowledgement,
	engine_secret,
)


class KgwMiddleware(BaseHTTPMiddleware):
	"""Acknowledge only successful requests against a KGW-configured engine.

	This confirms configuration, not signal strength. The engine processor validates the same
	key and profile on admission and fails generation on unsupported shapes/configurations.
	"""

	async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
		nonce = request.headers.get(CHALLENGE_HEADER)
		if nonce is None:
			return await call_next(request)
		if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
			return Response(status_code=400)
		config = request.app.state.vllm_config
		spec = KgwSpec.from_env()
		if (
			version("vllm") != VLLM_VERSION
			or PROCESSOR not in (config.model_config.logits_processors or ())
			or config.speculative_config is not None
			or config.scheduler_config.async_scheduling
			or config.model_config.get_vocab_size() != spec.vocab_size
			or config.model_config.runner_type != "generate"
		):
			return Response(status_code=503)
		response = await call_next(request)
		if 200 <= response.status_code < 300:
			response.headers[ACK_HEADER] = acknowledgement(engine_secret(), nonce, spec)
		return response
