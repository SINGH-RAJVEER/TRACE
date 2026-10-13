import { describe, expect, test } from "bun:test";
import { histogram, logDomain, logTicks, scoreRows } from "./data";
import type { Report } from "./types";

const report = {
	scores: { student_before: { gsm8k: { score: 40, ci95: [25, 55] } } },
	samples: [
		{ item: { benchmark: "gsm8k" }, results: { student_before: { latency_seconds: 2, status: "ok" } } },
		{ item: { benchmark: "gsm8k" }, results: { student_before: { latency_seconds: 20, status: "error" } } },
		{ item: { benchmark: "mbpp" }, results: { student_before: { latency_seconds: 5, status: "ok" } } },
	],
} as unknown as Report;

describe("chart data", () => {
	test("missing phases stay null and confidence bars are asymmetric distances", () => {
		const row = scoreRows(report, ["gsm8k"])[0];
		expect(row.student_before).toBe(40);
		expect(row.student_before_error).toEqual([15, 15]);
		expect(row.student_after).toBeNull();
	});
	test("latency histogram excludes errors and respects benchmark filters", () => {
		const { rows } = histogram(report, ["gsm8k"], ["student_before"], false);
		expect(rows.reduce((sum, row) => sum + row.student_before, 0)).toBe(1);
		expect(rows.at(-1)?.student_before).toBe(1);
	});
	test("log axes get a positive 1-2-5 domain and ticks", () => {
		const domain = logDomain([0, null, 0.34, 27]);
		expect(domain).toEqual([0.2, 50]);
		expect(logTicks(domain)).toEqual([0.2, 0.5, 1, 2, 5, 10, 20, 50]);
	});
});
