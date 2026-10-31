"""Command-line entry point for the protected teacher API."""

import uvicorn

from harness_api.settings import Settings


def main() -> None:
	settings = Settings()
	# The noise template lives in this process, and without HARNESS_STATE_DB so do account
	# state, open challenges, and the aggregate stream, so the service runs as one worker.
	uvicorn.run(
		"harness_api.app:create_app",
		factory=True,
		host=settings.host,
		port=settings.port,
		log_level=settings.log_level,
		workers=1,
	)


if __name__ == "__main__":
	main()
