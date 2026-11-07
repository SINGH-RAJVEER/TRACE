"""Command-line entry point for the dashboard API."""

import uvicorn

from dashboard_api.settings import Settings


def main() -> None:
	settings = Settings()
	uvicorn.run(
		"dashboard_api.app:create_app",
		factory=True,
		host=settings.host,
		port=settings.port,
		log_level=settings.log_level,
	)


if __name__ == "__main__":
	main()
