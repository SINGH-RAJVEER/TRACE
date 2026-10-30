"""Simulated outcomes for UI previews, never evidence of harness effectiveness."""

from datetime import UTC, datetime
from pathlib import Path

from distillation.benchmarks.models import save_json
from distillation.benchmarks.traffic import (
	TrafficConfig,
	TrafficRun,
	TrafficSample,
	TrafficTarget,
	freeze_traffic,
	load_traffic_suite,
)


def create_traffic_demo(directory: Path) -> None:
	suite_path = directory / "traffic-demo-suite.fixture"
	suite = load_traffic_suite(suite_path) if suite_path.exists() else freeze_traffic(suite_path)
	config = TrafficConfig(
		experiment="Simulated harness preview",
		suite_path=suite_path,
		output_dir=directory,
		model="simulated-teacher",
		model_revision="simulated-v1",
		targets={
			"before": TrafficTarget(
				base_url="http://example.invalid", service_revision="simulated"
			),
			"after": TrafficTarget(base_url="http://example.invalid", service_revision="simulated"),
		},
	)
	labels = {a.id: a.label for a in suite.actors}
	false_positive_id = next(r.id for r in suite.requests if labels[r.actor_id] == "genuine")
	for phase in ("before", "after"):
		samples = []
		for request in suite.requests:
			attack = labels[request.actor_id] == "distillation"
			block = phase == "after" and (
				(attack and request.at_seconds > 35) or request.id == false_positive_id
			)
			throttle = phase == "after" and attack and 15 < request.at_seconds <= 35
			latency = 0.02 if block else 2.5 if throttle else 0.4
			samples.append(
				TrafficSample(
					request_id=request.id,
					scheduled_seconds=request.at_seconds,
					started_seconds=request.at_seconds + 0.001,
					completed_seconds=request.at_seconds + 0.001 + latency,
					decision="block" if block else "throttle" if throttle else "allow",
					status="blocked" if block else "ok",
					http_status=403 if block else 200,
					succeeded=not block,
					output_tokens=None if block else 64,
					served_model=None if block else "simulated-teacher",
				)
			)
		run = TrafficRun(
			id=f"demo-{phase}",
			phase="before" if phase == "before" else "after",
			completed_at=datetime.now(UTC),
			suite_sha256=suite.sha256,
			config=config,
			demo=True,
			samples=samples,
		)
		save_json(directory / f"demo-{phase}.json", run)
	save_json(directory / f"traffic-suite-{suite.sha256}.json", suite)
