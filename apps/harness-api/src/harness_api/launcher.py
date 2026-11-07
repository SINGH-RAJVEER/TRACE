"""Validated product startup and explicit legacy startup for the task runner."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness.campaign import CampaignConfig, load_campaign_config

from harness_api.settings import Settings


def prepare_settings(
	config_path: Path | None, state_db: Path | None, base_policy: str
) -> tuple[Settings, CampaignConfig]:
	"""Bind a product launch without changing the raw API's environment defaults."""
	overrides: dict[str, Any] = {"HARNESS_POLICY": base_policy}
	if config_path is not None:
		overrides["HARNESS_CAMPAIGN_CONFIG"] = config_path
	if state_db is not None:
		overrides["HARNESS_STATE_DB"] = state_db
	settings = Settings(**overrides)
	if settings.campaign_config is None:
		raise ValueError(
			"set HARNESS_CAMPAIGN_CONFIG or pass --config; create one with just campaign-config"
		)
	campaign = load_campaign_config(settings.campaign_config)
	if campaign.mode == "throttle" and not settings.campaign_allow_unqualified_enforcement:
		raise ValueError(
			"throttle mode requires HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true"
		)
	if "campaign_throttled" in settings.review_verify_codes:
		raise ValueError("campaign_throttled cannot be converted into verification")
	if settings.trust_actor_header or settings.test_clock:
		raise ValueError("product startup refuses test actor headers and test clocks")
	if settings.accounts_file is None or not settings.accounts_file.is_file():
		raise ValueError("set HARNESS_ACCOUNTS_FILE to an existing authenticated accounts JSON")
	if not (settings.calibration_dir / "calibration.json").is_file():
		raise ValueError(
			"HARNESS_CALIBRATION_DIR must contain calibration.json matching HARNESS_TASK"
		)
	return settings, campaign


def main(argv: Sequence[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)
	for name in ("serve", "validate"):
		command = commands.add_parser(name)
		command.add_argument("--config", type=Path)
		command.add_argument("--state-db", type=Path)
		command.add_argument(
			"--base-policy",
			choices=("allow", "mmd", "quota", "quota_tokens", "rate"),
			default="allow",
		)
	config = commands.add_parser("config")
	config.add_argument("output", type=Path)
	config.add_argument("--namespace", required=True)
	config.add_argument("--mode", choices=("observe", "throttle"), default="observe")
	commands.add_parser(
		"legacy", help="Serve the configured base policy with campaign detection disabled"
	)
	args = parser.parse_args(argv)
	try:
		if args.command == "config":
			campaign = CampaignConfig(namespace=args.namespace, mode=args.mode)
			args.output.parent.mkdir(parents=True, exist_ok=True)
			with args.output.open("x") as stream:
				stream.write(
					json.dumps(campaign.model_dump(mode="json", exclude_none=True), indent="\t")
					+ "\n"
				)
			print(f"Created {args.output}; mode={campaign.mode}; qualified=false")
			return
		if args.command == "legacy":
			settings = Settings(
				HARNESS_CAMPAIGN_CONFIG=None, HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=False
			)
		else:
			settings, campaign = prepare_settings(args.config, args.state_db, args.base_policy)
			session = campaign.session.version if campaign.session is not None else "disabled"
			coverage = campaign.coverage.monitor.version if campaign.coverage else "disabled"
			print(
				f"Campaign {campaign.digest}; mode={campaign.mode}; qualified=false; "
				f"base_policy={settings.policy}; session={session}; "
				f"coverage={coverage}; "
				f"lifetime={campaign.lifetime.version if campaign.lifetime else 'disabled'}"
			)
			if args.command == "validate":
				print(
					"Launch settings and prerequisite paths checked; "
					"no teacher, encoder or server started."
				)
				return
	except (ValueError, OSError) as error:
		parser.error(str(error))
	# Import model/server work only after validating settings. No automatic vLLM launch.
	import uvicorn

	from harness_api.app import create_app

	uvicorn.run(
		create_app(settings=settings),
		host=settings.host,
		port=settings.port,
		log_level=settings.log_level,
	)


if __name__ == "__main__":
	main()
