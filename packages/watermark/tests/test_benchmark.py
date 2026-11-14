import json
from pathlib import Path

from trace_watermark.__main__ import main


def test_quick_benchmark_reports_every_path_and_no_key_material(tmp_path: Path) -> None:
	output = tmp_path / "nested" / "bench.json"
	assert main(["bench", "--quick", "--output", str(output)]) == 0
	text = output.read_text(encoding="utf-8")
	assert "public-benchmark-key" not in text
	report = json.loads(text)
	assert report["environment"]["load_average_end"]
	for path in ("seed", "mask", "bias", "batch", "detect"):
		assert report[path]
		for case in report[path]:
			assert 0 < case["p50_us"] <= case["p95_us"]
	assert {case["batch_size"] for case in report["batch"]} == {1, 4}
	assert all(case["tokens_per_second"] > 0 for case in report["batch"])
