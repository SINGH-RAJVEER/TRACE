"""FastAPI application for authorized hosted-teacher inference."""

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from distillation.schemas import TeacherGeneration, TeacherGenerationRequest
from distillation.teacher import (
	OpenAICompatibleTeacher,
	Teacher,
	TeacherReasoningError,
	TeacherUpstreamError,
	TeacherWatermarkError,
)
from distillation.watermark import WatermarkSpec, read_master_secret
from fastapi import FastAPI, Header, HTTPException, status

from teacher_api.settings import Settings


def create_app(*, settings: Settings | None = None, teacher: Teacher | None = None) -> FastAPI:
	"""Build the application with optional test dependencies."""

	resolved_settings = settings or Settings()
	resolved_teacher = teacher
	watermark = (
		WatermarkSpec(
			gamma=resolved_settings.watermark_gamma,
			delta=resolved_settings.watermark_delta,
			eligible_vocab=resolved_settings.watermark_eligible_vocab,
		)
		if resolved_settings.watermark_key_file is not None
		else None
	)
	secret = (
		read_master_secret(resolved_settings.watermark_key_file)
		if resolved_settings.watermark_key_file is not None
		else None
	)
	if resolved_teacher is None and resolved_settings.is_configured:
		resolved_teacher = OpenAICompatibleTeacher(
			base_url=resolved_settings.upstream_base_url,
			api_key=resolved_settings.upstream_api_key,
			model=resolved_settings.upstream_model,
			provider=resolved_settings.provider_name,
			timeout_seconds=resolved_settings.timeout_seconds,
			default_thinking=resolved_settings.enable_thinking,
			reasoning_budget_tokens=resolved_settings.reasoning_budget_tokens,
			watermark=watermark,
			watermark_secret=secret,
		)

	@asynccontextmanager
	async def lifespan(_: FastAPI) -> AsyncIterator[None]:
		yield
		if resolved_teacher is not None:
			await resolved_teacher.close()

	app = FastAPI(
		title="Teacher inference API",
		version="0.1.0",
		lifespan=lifespan,
	)

	@app.get("/healthz")
	async def health() -> dict[str, str]:
		state = "ready" if resolved_teacher is not None else "unconfigured"
		return {"status": state}

	@app.post("/v1/teacher/generate", response_model=TeacherGeneration)
	async def generate(
		request: TeacherGenerationRequest,
		authorization: str | None = Header(default=None),
	) -> TeacherGeneration:
		_authenticate(resolved_settings.api_token, authorization)
		if resolved_teacher is None:
			raise HTTPException(
				status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
				detail={"code": "teacher_unconfigured", "message": "teacher is not configured"},
			)
		try:
			return await resolved_teacher.generate(request)
		except TeacherWatermarkError as error:
			raise HTTPException(
				status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
				detail={"code": "watermark_unavailable", "message": str(error)},
			) from error
		except TeacherReasoningError as error:
			raise HTTPException(
				status_code=status.HTTP_502_BAD_GATEWAY,
				detail={"code": "teacher_reasoning_withheld", "message": str(error)},
			) from error
		except TeacherUpstreamError as error:
			raise HTTPException(
				status_code=status.HTTP_502_BAD_GATEWAY,
				detail={"code": "teacher_upstream_error", "message": str(error)},
			) from error

	return app


def _authenticate(expected_token: str, authorization: str | None) -> None:
	if not expected_token:
		return
	provided = ""
	if authorization and authorization.startswith("Bearer "):
		provided = authorization.removeprefix("Bearer ")
	if not hmac.compare_digest(expected_token, provided):
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail={"code": "unauthorized", "message": "invalid bearer token"},
		)
