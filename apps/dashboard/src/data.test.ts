import { describe, expect, test } from "bun:test";
import { checkpointPlot, curvePlot, format, strataByGroup, studyLabel, verdict } from "./data";
import type { CheckpointRow, Curve, UtilityArm } from "./types";

const file = { state: "ok" as const, path: "x", error: null, sha256: null };

function curve(summary: Curve["summary"]): Curve {
	return {
		place: { root: "artifacts", path: "c" }, plan_file: file, summary_file: file, integrity: "intact",
		version: "math-learning-curve-v1", task: "gsm8k", scope: null, sizes: [300, 1000], seeds: [42, 43],
		include_pending: true, target_fraction: 0.8, training_ledgers: 1, cells: 4, recipe: {}, summary,
	};
}

const row = (n: number, mean: number, arm = "teacher") => ({ n, arm, seed_gains_pp: [mean - 1, mean + 1], mean_gain_pp: mean, ci95_pp: [mean - 2, mean + 2] as [number, number] });

describe("format", () => {
	test("keeps unknown apart from zero and false", () => {
		expect(format(null, "int")).toBeNull();
		expect(format(undefined)).toBeNull();
		expect(format("")).toBeNull();
		expect(format(0, "int")).toBe("0");
		expect(format(false)).toBe("false");
		expect(format(0.0927, "fraction")).toBe("9.3%");
		expect(format(-1.25, "pp")).toBe("-1.3 pp");
	});
});

describe("curvePlot", () => {
	test("is unavailable without a summary", () => {
		expect(curvePlot(curve(null)).state).toBe("unavailable");
	});
	test("draws no trend from a single size", () => {
		const plot = curvePlot(curve({ results: [row(300, 5)], gold_results: null, reference_gain_pp: 5, target_gain_pp: 4, smallest_n: 300, bootstrap: null, warning: null }));
		expect(plot.state).toBe("unavailable");
	});
	test("sorts sizes and keeps only arms with two or more points", () => {
		const plot = curvePlot(curve({ results: [row(1000, 9), row(300, 5)], gold_results: [row(300, 7, "gold")], reference_gain_pp: 9, target_gain_pp: null, smallest_n: null, bootstrap: null, warning: null }));
		if (plot.state !== "available") throw new Error("expected a plot");
		expect(plot.series.map(s => s.arm)).toEqual(["teacher"]);
		expect(plot.series[0].points.map(p => p.n)).toEqual([300, 1000]);
		expect(plot.target).toBeNull();
	});
});

describe("checkpointPlot", () => {
	const base = { kind: "response", cutoff_seconds: null, acquired_count: null, trainable_count: null, seeds: null };
	test("lists censored checkpoints instead of plotting them as zero", () => {
		const rows: CheckpointRow[] = [
			{ ...base, id: "zero", status: "base", mean_accuracy: 0 },
			{ ...base, id: "late", status: "reached", mean_accuracy: 0.6 },
			{ ...base, id: "response-100", status: "censored", mean_accuracy: null },
		];
		const plot = checkpointPlot(rows);
		if (plot.state !== "available") throw new Error("expected a plot");
		expect(plot.points.map(p => p.id)).toEqual(["zero", "late"]);
		expect(plot.points[0].mean).toBe(0);
		expect(plot.censored).toEqual(["response-100"]);
	});
	test("is unavailable with one reached checkpoint", () => {
		expect(checkpointPlot([{ ...base, id: "end", status: "reached", mean_accuracy: 0.5 }]).state).toBe("unavailable");
		expect(checkpointPlot(null).state).toBe("unavailable");
	});
});

describe("strataByGroup", () => {
	test("orders groups and attaches latency only where recorded", () => {
		const metrics = { submitted: 10, served: 0, completion_rate: 0, served_only_quality: null, unconditional_utility: 0, latency_p95: null };
		const arm = {
			arm: "full", strata: [
				{ group: "client_type", value: "batch", ...metrics },
				{ group: "domain", value: "gsm8k", ...metrics },
			],
			detail: { strata_latency: { domain: { gsm8k: { n: 0, p50: null, p95: null, p99: null } } } },
		} as unknown as UtilityArm;
		const groups = strataByGroup(arm);
		expect(groups.map(g => g.group)).toEqual(["domain", "client_type"]);
		expect(groups[0].rows[0].served_only_quality).toBeNull();
		expect(groups[0].rows[0].latency?.p50).toBeNull();
		expect(groups[1].rows[0].latency).toBeNull();
	});
});

test("study and verdict labels never invent a value", () => {
	expect(studyLabel([])).toBeNull();
	expect(studyLabel(["development"])).toBe("development");
	expect(verdict(null)).toBe("unknown");
	expect(verdict(false, "failed")).toBe("fail");
	expect(verdict(false, "development-only")).toBe("development");
});
