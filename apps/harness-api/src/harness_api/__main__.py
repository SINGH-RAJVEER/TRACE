"""Command-line entry point for the protected teacher API."""

import uvicorn

from harness_api.settings import Settings


def main() -> None:
	settings = Settings()
	# Account state, open challenges, and the aggregate stream live in this process, so the
	# service must run as exactly one worker.
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
