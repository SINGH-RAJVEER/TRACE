import { describe, expect, test } from "bun:test";
import { applies, campaignTotals, checkpointPlot, compact, coverage, curvePlot, emptyIntervention, format, gaps, outcomeGrid, sortArms, strataByGroup, studyLabel, verdict } from "./data";
import type { CampaignItem, CheckpointRow, Curve, RunDetail, UtilityArm } from "./types";

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

test("an empty intervention cell claims absence only under complete attribution", () => {
	expect(emptyIntervention({ status: "complete", attribution_complete: true })).toBe("none observed");
	expect(emptyIntervention({ status: "interrupted", attribution_complete: false })).toBe("unknown");
	expect(emptyIntervention({ status: "complete", attribution_complete: null })).toBe("unknown");
	expect(emptyIntervention({ status: "gate-refused", attribution_complete: null })).toBe("not run");
	expect(emptyIntervention({ status: "complete", attribution_complete: true }, "unreached")).toBe("unreached");
});

const missing = { state: "unavailable" as const, reason: "none here", items: [] };

function campaign(attack: string, arm: string, status: string, extra: Partial<CampaignItem> = {}): CampaignItem {
	return { kind: "campaign", attack, arm, status, source: "index", outcome: null, attempts: 6000, attempt_budget: 6000, received: 1200, blocked: 4800, released_tokens: 1000, attribution_complete: true, ...extra } as CampaignItem;
}

function detail(rows: CampaignItem[]): RunDetail {
	return {
		run: { id: "0", label: "r", kind: "study", run_id: "r", readings: [], studies: [], commits: ["abc"], statuses: [], rows: {}, artifacts: {}, places: [] },
		provenance: { run_id: "r", readings: [], commits: ["abc"], commit_recorded: true, studies: [], statuses: {}, seeds: [], partitions: [], sources: [] },
		campaigns: { state: "available", reason: null, items: rows },
		utility: missing,
		teacher: { capacity: missing, headroom: missing, scores: missing },
		students: { curves: missing, checkpoints: missing, transfers: missing, index: missing },
		gates: { reports: missing, index: missing, detectability: missing },
	};
}

const refused = (attack: string, arm: string) => campaign(attack, arm, "gate-refused", { attempts: null, received: null, blocked: null, released_tokens: null, attribution_complete: null });

describe("outcomeGrid", () => {
	test("keeps refused and absent cells apart from zero", () => {
		const grid = outcomeGrid([campaign("A1", "quota", "complete", { blocked: 0 }), refused("A1", "full")], ["A2"]);
		expect(grid.attacks).toEqual(["A1", "A2"]);
		expect(grid.arms).toEqual(["full", "quota"]);
		const cell = (attack: string, arm: string) => grid.cells.find(c => c.attack === attack && c.arm === arm)!;
		expect(cell("A1", "quota").blocked).toBe(0);
		expect(cell("A1", "full")).toMatchObject({ state: "gate-refused", received: null, releasedTokens: null });
		expect(cell("A2", "quota")).toMatchObject({ state: "absent", received: null, assumed: null });
	});

	test("assumes a refused arm answered every attempt, from the same attack in the same run", () => {
		const rows = [campaign("A1", "quota", "complete", { received: 1200, released_tokens: 120000 }), campaign("A1", "full", "complete", { received: 300, released_tokens: 90000 }), refused("A1", "instantaneous"), refused("A2", "full")];
		const cell = (attack: string, arm: string) => outcomeGrid(rows).cells.find(c => c.attack === attack && c.arm === arm)!;
		expect(cell("A1", "instantaneous").releasedTokens).toBeNull();
		expect(cell("A1", "instantaneous").assumed?.tokens).toBe(600000);
		expect(cell("A2", "full").assumed).toBeNull();
		expect(cell("A1", "quota").assumed).toBeNull();
		const withUndefended = [...rows, campaign("A1", "undefended", "complete", { received: 6000, released_tokens: 700000 })];
		expect(outcomeGrid(withUndefended).cells.find(c => c.attack === "A1" && c.arm === "instantaneous")!.assumed?.tokens).toBe(700000);
	});

	test("orders arms by the fixed arm order", () => {
		expect(sortArms(["quota", "zeta", "full", "undefended", "instantaneous"])).toEqual(["undefended", "full", "instantaneous", "quota", "zeta"]);
	});
});

describe("campaignTotals", () => {
	test("counts coverage without pooling acquisition across arms", () => {
		const rows = [campaign("A1", "quota", "complete"), refused("A1", "full"), campaign("A2", "quota", "complete", { attribution_complete: null })];
		expect(campaignTotals(rows, ["A1", "A2", "A7"])).toEqual({ arms: 3, complete: 2, refused: 1, interrupted: 0, ran: 2, attributed: 1, attacks: 3, attacksRun: 2 });
	});

	test("compacts counts for direct labels", () => {
		expect(compact(806570)).toBe("807k");
		expect(compact(2784)).toBe("2.8k");
		expect(compact(null)).toBeNull();
	});
});

describe("coverage and gaps", () => {
	test("marks a run with refused arms as partial and names what is missing", () => {
		const run = detail([campaign("A1", "quota", "complete"), refused("A1", "full"), campaign("A2", "full", "interrupted", { attempts: 1537, attribution_complete: false, unattributed_queries: 1537 }), campaign("A3", "undefended", "complete", { attribution_complete: null })]);
		const cells = coverage(run);
		expect(cells.find(c => c.key === "campaigns")?.state).toBe("partial");
		expect(cells.find(c => c.key === "curves")?.state).toBe("missing");
		const text = gaps(run, ["A1", "A2", "A3", "A7"]).map(g => g.text).join("\n");
		expect(text).toContain("No campaign rows for A7");
		expect(text).toContain("1 of 4 arms refused by the detectability gate");
		expect(text).toContain("A2/full interrupted after 1,537 of 6,000 attempts");
		expect(text).toContain("Attribution incomplete for A2/full (1,537 unattributed queries)");
		expect(text).toContain("Attribution not recorded for A3/undefended");
		expect(text).toContain("No benign utility");
		expect(text).toContain("Not recorded: headroom gate, capacity preflight, problem scores, learning curves, checkpoints, benchmark panel, transfer report, student index rows, clean-student gate");
	});
});

describe("applies", () => {
	test("marks campaign evidence not applicable to a lone student plan", () => {
		const run = detail([]);
		run.run = { ...run.run, kind: "checkpoints", run_id: null };
		run.campaigns = missing;
		expect(applies("checkpoints", "Acquisition")).toBe(false);
		expect(coverage(run).find(c => c.key === "campaigns")?.state).toBe("na");
		expect(coverage(run).find(c => c.key === "checkpoints")?.state).toBe("missing");
		const text = gaps(run).map(g => g.text).join("\n");
		expect(text).not.toContain("No campaign rows");
		expect(text).toContain("Not attached to an indexed run");
	});
});
