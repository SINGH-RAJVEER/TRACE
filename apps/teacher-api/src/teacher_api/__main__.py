"""Command-line entry point for the teacher API."""

import uvicorn

from teacher_api.settings import Settings


def main() -> None:
	settings = Settings()
	uvicorn.run(
		"teacher_api.app:create_app",
		factory=True,
		host=settings.host,
		port=settings.port,
		log_level=settings.log_level,
	)


if __name__ == "__main__":
	main()
