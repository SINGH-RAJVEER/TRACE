// Pure transforms from API shapes to what the views draw. None of them fills a gap: a missing
// value stays null and a series without enough comparable points is reported as unavailable.

import type { CheckpointRow, Curve, CurveRow, Latency, Maybe, UtilityArm, UtilityMetrics } from "./types";

export function isNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** Formatted text, or null when there is nothing recorded to format. */
export function format(value: unknown, as: "int" | "fraction" | "pp" | "seconds" | "rate" | "raw" = "raw"): Maybe<string> {
	if (value === null || value === undefined || value === "") return null;
	if (typeof value === "boolean") return value ? "true" : "false";
	if (!isNumber(value)) return String(value);
	switch (as) {
		case "int":
			return Number.isInteger(value) ? integer.format(value) : value.toFixed(2);
		case "fraction":
			return `${(value * 100).toFixed(1)}%`;
		case "pp":
			return `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp`;
		case "seconds":
			return value >= 100 ? `${integer.format(value)} s` : `${value.toFixed(1)} s`;
		case "rate":
			return `${value.toFixed(2)}/s`;
		default:
			return String(value);
	}
}

export function shortHash(value: Maybe<string> | undefined, length = 12): Maybe<string> {
	return value ? value.replace(/^sha256:/, "").slice(0, length) : null;
}

export type CurvePoint = { n: number; mean: number; low: number; high: number; band: [number, number]; seeds: number[] };
export type CurveSeries = { arm: "teacher" | "gold"; label: string; points: CurvePoint[] };
export type CurvePlot =
	| { state: "available"; series: CurveSeries[]; target: Maybe<number> }
	| { state: "unavailable"; reason: string };

function points(rows: Maybe<CurveRow[]> | undefined): CurvePoint[] {
	return (rows ?? [])
		.filter(row => isNumber(row.n) && isNumber(row.mean_gain_pp) && Array.isArray(row.ci95_pp) && row.ci95_pp.every(isNumber))
		.map(row => ({ n: row.n, mean: row.mean_gain_pp, low: row.ci95_pp[0], high: row.ci95_pp[1], band: [row.ci95_pp[0], row.ci95_pp[1]] as [number, number], seeds: row.seed_gains_pp ?? [] }))
		.sort((a, b) => a.n - b.n);
}

/**
 * Gain-over-base curves from one summarized plan. Each arm is drawn only when it has at least two
 * sizes: a single size is a measurement, not a trend, and is left to the table.
 */
export function curvePlot(curve: Curve): CurvePlot {
	if (!curve.summary) return { state: "unavailable", reason: "the plan has no summary.json yet; run the learning-curve summary" };
	const series: CurveSeries[] = [];
	const clean = points(curve.summary.results);
	const gold = points(curve.summary.gold_results);
	if (clean.length >= 2) series.push({ arm: "teacher", label: "Teacher-data student", points: clean });
	if (gold.length >= 2) series.push({ arm: "gold", label: "Gold-answer student", points: gold });
	if (!series.length) return { state: "unavailable", reason: "fewer than two training-set sizes were summarized for any arm" };
	return { state: "available", series, target: curve.summary.target_gain_pp ?? null };
}

export type CheckpointPoint = { id: string; kind: string; mean: number; min: Maybe<number>; max: Maybe<number> };
export type CheckpointPlot =
	| { state: "available"; points: CheckpointPoint[]; censored: string[] }
	| { state: "unavailable"; reason: string; censored: string[] };

/** Reached checkpoints in plan order; censored ones are listed, never drawn as zero accuracy. */
export function checkpointPlot(rows: Maybe<CheckpointRow[]> | undefined): CheckpointPlot {
	const all = rows ?? [];
	const censored = all.filter(row => row.status === "censored" || !isNumber(row.mean_accuracy)).map(row => row.id);
	const reached = all
		.filter(row => row.status !== "censored" && isNumber(row.mean_accuracy))
		.map(row => ({ id: row.id, kind: row.kind, mean: row.mean_accuracy as number, min: row.min_accuracy ?? null, max: row.max_accuracy ?? null }));
	if (reached.length < 2) return { state: "unavailable", reason: "fewer than two reached checkpoints have accuracy", censored };
	return { state: "available", points: reached, censored };
}

export const STRATUM_GROUPS = ["domain", "domain_profile", "client_type"] as const;
export const STRATUM_LABELS: Record<string, string> = { domain: "Domain", domain_profile: "Domain profile", client_type: "Client type" };

export type StratumRow = UtilityMetrics & { value: string; latency: Maybe<Latency> };

/** Strata of one arm, grouped in a fixed order, with p50/p99 latency where utility.json has it. */
export function strataByGroup(arm: UtilityArm): { group: string; rows: StratumRow[] }[] {
	const known = new Set<string>(STRATUM_GROUPS);
	const groups = [...STRATUM_GROUPS, ...[...new Set(arm.strata.map(row => row.group))].filter(group => !known.has(group))];
	return groups
		.map(group => ({
			group,
			rows: arm.strata
				.filter(row => row.group === group)
				.map(row => ({ ...row, latency: arm.detail?.strata_latency?.[group]?.[row.value] ?? null })),
		}))
		.filter(entry => entry.rows.length > 0);
}

/** Label for a set of study values; an empty set was not recorded. */
export function studyLabel(values: string[]): Maybe<string> {
	return values.length ? values.join(" + ") : null;
}

/** "fail", "pass" or "unknown" for a gate-like value that may be absent. */
export function verdict(passed: Maybe<boolean> | undefined, status?: Maybe<string>): "pass" | "fail" | "development" | "unknown" {
	if (status === "development-only") return "development";
	if (passed === true) return "pass";
	if (passed === false) return "fail";
	return "unknown";
}

/**
 * What an empty intervention or level cell of a campaign row means. Absence is only "none
 * observed" when every query was attributed to a policy event; an arm that never ran has nothing.
 */
export function emptyIntervention(row: { status: Maybe<string>; attribution_complete?: unknown }, absent = "none observed"): string {
	if (row.status === "gate-refused" || row.status === "not-run") return "not run";
	return row.attribution_complete === true ? absent : "unknown";
}
