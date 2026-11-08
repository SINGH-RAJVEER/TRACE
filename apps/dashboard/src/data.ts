// Pure transforms from API shapes to what the views draw. None of them fills a gap: a missing
// value stays null and a series without enough comparable points is reported as unavailable.

import type { CampaignItem, CheckpointRow, Curve, CurveRow, Latency, Maybe, RunDetail, Section, UtilityArm, UtilityMetrics } from "./types";

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

// ---------------------------------------------------------------------------------------------
// Overview model: arm identity, evidence coverage and the specific gaps in each run.

/** Arms in a fixed reading order; each keeps one color in every run. */
export const ARM_ORDER = ["undefended", "full", "instantaneous", "quota"] as const;

export const ARM_LABELS: Record<string, string> = {
	undefended: "Undefended",
	full: "Full detector",
	instantaneous: "Instantaneous detector",
	quota: "Quota",
};

export function armColor(arm: string): string {
	return (ARM_ORDER as readonly string[]).includes(arm) ? `var(--arm-${arm})` : "var(--arm-other)";
}

export function sortArms(arms: Iterable<string>): string[] {
	const rank = (arm: string) => {
		const index = (ARM_ORDER as readonly string[]).indexOf(arm);
		return index < 0 ? ARM_ORDER.length : index;
	};
	return [...new Set(arms)].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export function sortAttacks(attacks: Iterable<string>): string[] {
	return [...new Set(attacks)].sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
}

export type CoverageState = "present" | "partial" | "missing" | "error" | "na";

export type CoverageCell = { key: string; group: string; label: string; state: CoverageState; count: number; detail: string };

export const COVERAGE_GROUPS = ["Acquisition", "Utility", "Teacher", "Students", "Gates"] as const;

function failed(value: unknown): value is { state: "error"; reason: string } {
	return !!value && typeof value === "object" && (value as { state?: string }).state === "error" && !("items" in (value as object));
}

function sectionCell<T>(key: string, group: string, label: string, section: Section<T> | undefined, noun: string): CoverageCell {
	if (!section) return { key, group, label, state: "error", count: 0, detail: "the API did not return this section" };
	if (section.state === "error") return { key, group, label, state: "error", count: 0, detail: section.reason ?? "could not be read" };
	if (section.state === "unavailable" || section.items.length === 0) return { key, group, label, state: "missing", count: 0, detail: section.reason ?? "no artifact was found" };
	return { key, group, label, state: "present", count: section.items.length, detail: `${section.items.length} ${noun}${section.items.length === 1 ? "" : "s"}` };
}

/**
 * Which evidence groups can exist for a run of this kind. A student plan or gate directory found on
 * its own never holds campaigns or utility, so those are not applicable rather than missing.
 */
export function applies(kind: string, group: string): boolean {
	if (kind === "learning-curve" || kind === "checkpoints" || kind === "student") return group === "Students";
	if (kind === "baseline-gates" || kind === "gate") return group !== "Acquisition" && group !== "Utility";
	return true;
}

/** One cell per kind of evidence the dashboard can show, in a fixed order. */
export function coverage(detail: RunDetail): CoverageCell[] {
	return rawCoverage(detail).map(cell => cell.state === "missing" && !applies(detail.run.kind, cell.group)
		? { ...cell, state: "na", detail: `not applicable to a ${detail.run.kind} directory` }
		: cell);
}

function rawCoverage(detail: RunDetail): CoverageCell[] {
	const cells: CoverageCell[] = [];
	const campaigns = detail.campaigns;
	if (failed(campaigns)) cells.push({ key: "campaigns", group: "Acquisition", label: "Campaign arms", state: "error", count: 0, detail: campaigns.reason });
	else {
		const cell = sectionCell("campaigns", "Acquisition", "Campaign arms", campaigns, "arm");
		const done = campaigns.items.filter(row => row.status === "complete").length;
		if (cell.state === "present" && done < campaigns.items.length) {
			cell.state = "partial";
			cell.detail = `${done} of ${campaigns.items.length} arms complete`;
		}
		cells.push(cell);
	}
	const utility = detail.utility;
	if (failed(utility)) cells.push({ key: "utility", group: "Utility", label: "Benign utility", state: "error", count: 0, detail: utility.reason });
	else {
		const cell = sectionCell("utility", "Utility", "Benign utility", utility, "arm");
		const withFile = utility.items.filter(arm => arm.file?.state === "ok").length;
		if (cell.state === "present" && withFile < utility.items.length) {
			cell.state = "partial";
			cell.detail = `${utility.items.length} arms; utility.json read for ${withFile}`;
		}
		cells.push(cell);
	}
	const teacher = detail.teacher;
	const students = detail.students;
	const gates = detail.gates;
	const teacherSections = failed(teacher) ? null : teacher;
	const studentSections = failed(students) ? null : students;
	const gateSections = failed(gates) ? null : gates;
	cells.push(sectionCell("headroom", "Teacher", "Headroom gate", teacherSections?.headroom, "report"));
	cells.push(sectionCell("capacity", "Teacher", "Capacity preflight", teacherSections?.capacity, "report"));
	cells.push(sectionCell("scores", "Teacher", "Problem scores", teacherSections?.scores, "file"));
	for (const [key, label, section, noun] of [["curves", "Learning curves", studentSections?.curves, "curve"], ["checkpoints", "Checkpoints", studentSections?.checkpoints, "plan"]] as const) {
		const cell = sectionCell<{ summary: unknown }>(key, "Students", label, section, noun);
		const summarized = section?.items.filter(item => item.summary).length ?? 0;
		if (cell.state === "present" && section && summarized < section.items.length) {
			cell.state = "partial";
			cell.detail = `${section.items.length} ${noun}s; ${summarized} summarized`;
		}
		cells.push(cell);
	}
	cells.push(sectionCell("transfers", "Students", "Transfer report", studentSections?.transfers, "report"));
	cells.push(sectionCell("student_rows", "Students", "Student index rows", studentSections?.index, "row"));
	cells.push(sectionCell("clean_gate", "Gates", "Clean-student gate", gateSections?.reports, "report"));
	cells.push(sectionCell("detectability", "Gates", "Detectability refusals", gateSections?.detectability, "refusal"));
	return cells;
}

export type Gap = { area: string; severity: "missing" | "partial"; text: string };

function list(values: string[], limit = 4): string {
	return values.length <= limit ? values.join(", ") : `${values.slice(0, limit).join(", ")} +${values.length - limit}`;
}

/**
 * The specific things this run did not record, one short line each. Nothing here is inferred or
 * filled in; absent artifacts are folded into one "Not recorded" line.
 */
export function gaps(detail: RunDetail, attacks: string[] = []): Gap[] {
	const out: Gap[] = [];
	const campaigns = failed(detail.campaigns) ? null : detail.campaigns;
	const utility = failed(detail.utility) ? null : detail.utility;
	const kind = detail.run.kind;
	if (!detail.run.run_id && kind !== "study") out.push({ area: "Provenance", severity: "partial", text: "Not attached to an indexed run (its name matches no run ID)" });
	if (campaigns && campaigns.state !== "available" && applies(kind, "Acquisition")) out.push({ area: "Acquisition", severity: "missing", text: "No campaign rows" });
	if (campaigns?.state === "available") {
		const rows = campaigns.items;
		const key = (row: CampaignItem) => `${row.attack}/${row.arm}`;
		const recorded = new Set(rows.map(row => row.attack));
		const skipped = sortAttacks(attacks.filter(attack => !recorded.has(attack)));
		if (skipped.length) out.push({ area: "Acquisition", severity: "partial", text: `No campaign rows for ${list(skipped)}` });
		const refused = rows.filter(row => row.status === "gate-refused");
		if (refused.length) out.push({ area: "Acquisition", severity: "partial", text: `${refused.length} of ${rows.length} arms refused by the detectability gate` });
		for (const row of rows.filter(r => r.status === "interrupted")) {
			out.push({ area: "Acquisition", severity: "partial", text: `${key(row)} interrupted after ${format(row.attempts, "int") ?? "?"}${isNumber(row.attempt_budget) ? ` of ${format(row.attempt_budget, "int")}` : ""} attempts` });
		}
		const ran = rows.filter(row => row.status !== "gate-refused" && row.status !== "not-run");
		const noAttribution = ran.filter(row => row.attribution_complete !== true).map(key);
		if (noAttribution.length) out.push({ area: "Acquisition", severity: "partial", text: `Intervention times unknown for ${list(noAttribution)} (attribution incomplete)` });
		const unknownTokens = ran.filter(row => isNumber(row.released_tokens_unknown) && row.released_tokens_unknown > 0);
		if (unknownTokens.length) out.push({ area: "Acquisition", severity: "partial", text: `Some released tokens unknown in ${list(unknownTokens.map(key))}` });
		if (utility?.state === "available") {
			const campaignArms = new Set(rows.map(row => row.arm ?? "unknown"));
			const utilityArms = new Set(utility.items.map(arm => arm.arm));
			const noCampaign = sortArms([...utilityArms].filter(arm => !campaignArms.has(arm)));
			const noUtility = sortArms([...campaignArms].filter(arm => !utilityArms.has(arm)));
			if (noCampaign.length) out.push({ area: "Acquisition", severity: "partial", text: `No campaign for ${list(noCampaign)} (utility only)` });
			if (noUtility.length) out.push({ area: "Utility", severity: "partial", text: `No utility for ${list(noUtility)}` });
		}
	}
	if (utility && utility.state !== "available" && applies(kind, "Utility")) out.push({ area: "Utility", severity: "missing", text: "No benign utility" });
	if (utility?.state === "available") {
		const why: Record<string, string> = { "lfs-pointer": "is a Git LFS pointer (run git lfs pull)", missing: "missing", malformed: "malformed", "too-large": "too large to read", refused: "refused by path checks" };
		const noFile = utility.items.filter(arm => arm.file?.state !== "ok");
		for (const state of [...new Set(noFile.map(arm => arm.file?.state ?? "missing"))]) {
			const arms = noFile.filter(arm => (arm.file?.state ?? "missing") === state).map(a => a.arm);
			out.push({ area: "Utility", severity: "partial", text: `utility.json ${why[state] ?? state} for ${list(arms)}` });
		}
		const unknown = utility.items.filter(arm => arm.validity !== "valid" && arm.validity !== "invalid");
		if (unknown.length) out.push({ area: "Utility", severity: "partial", text: `Utility validity unknown for ${unknown.length === utility.items.length ? "every arm" : list(unknown.map(a => a.arm))} (no capacity preflight)` });
	}
	const students = failed(detail.students) ? null : detail.students;
	const unsummarized = students?.checkpoints.items.filter(item => !item.summary) ?? [];
	if (unsummarized.length) out.push({ area: "Students", severity: "partial", text: `No summary.json for ${list(unsummarized.map(item => shortLabel(item.place.path)))}, so no accuracy` });
	const censored = students?.checkpoints.items.flatMap(item => (item.summary?.checkpoints ?? []).filter(row => row.status === "censored").map(row => `${shortLabel(item.place.path)}/${row.id}`)) ?? [];
	if (censored.length) out.push({ area: "Students", severity: "partial", text: `Censored checkpoints: ${list(censored)}` });
	const unsummarizedCurves = students?.curves.items.filter(curve => !curve.summary) ?? [];
	if (unsummarizedCurves.length) out.push({ area: "Students", severity: "partial", text: `No learning-curve summary for ${list(unsummarizedCurves.map(curve => shortLabel(curve.place.path)))}` });
	const absent = coverage(detail).filter(c => c.group !== "Acquisition" && c.group !== "Utility" && c.key !== "detectability" && (c.state === "missing" || c.state === "error"));
	const missing = absent.filter(c => c.state === "missing");
	if (missing.length) out.push({ area: "Not recorded", severity: "missing", text: `Not recorded: ${missing.map(c => c.label.toLowerCase()).join(", ")}` });
	for (const c of absent.filter(c => c.state === "error")) out.push({ area: c.group, severity: "missing", text: `Could not read ${c.label.toLowerCase()}: ${c.detail}` });
	const provenance = failed(detail.provenance) ? null : detail.provenance;
	if (provenance) {
		if (!provenance.commits.length) out.push({ area: "Provenance", severity: "partial", text: "No commit recorded" });
		if (!provenance.sources.length && kind === "study") out.push({ area: "Provenance", severity: "missing", text: "No run directory found; configuration and teacher identity unknown" });
		else if (kind === "study" && !provenance.sources.some(s => s.teacher.TEACHER_MODEL || s.teacher.TEACHER_UPSTREAM_MODEL)) out.push({ area: "Provenance", severity: "partial", text: "Teacher model not named in the run configuration" });
	}
	return out;
}

// ---------------------------------------------------------------------------------------------
// Chart rows. A cell that did not run, or has no row at all, carries null values and a state,
// never zeros, so a chart can draw it as a separate mark.

export type CellState = "complete" | "gate-refused" | "interrupted" | "not-run" | "absent" | "other";

export function cellState(row: CampaignItem | undefined): CellState {
	if (!row) return "absent";
	if (row.status === "complete" || row.status === "gate-refused" || row.status === "interrupted" || row.status === "not-run") return row.status;
	return "other";
}

export type OutcomeRow = {
	attack: string;
	arm: string;
	state: CellState;
	status: Maybe<string>;
	attempts: Maybe<number>;
	budget: Maybe<number>;
	received: Maybe<number>;
	challenge: Maybe<number>;
	blocked: Maybe<number>;
	failed: Maybe<number>;
	releasedTokens: Maybe<number>;
	firstAlert: Maybe<number>;
	firstRestriction: Maybe<number>;
	firstRefusal: Maybe<number>;
	attribution: Maybe<boolean>;
	row: CampaignItem | null;
};

function num(value: unknown): Maybe<number> {
	return isNumber(value) ? value : null;
}

/** Every attack by every arm of a run, including combinations with no row. */
export function outcomeGrid(rows: CampaignItem[], attacks?: string[]): { attacks: string[]; arms: string[]; cells: OutcomeRow[] } {
	const arms = sortArms(rows.map(row => row.arm ?? "unknown"));
	const allAttacks = sortAttacks([...(attacks ?? []), ...rows.map(row => row.attack ?? "unknown")]);
	const cells: OutcomeRow[] = [];
	for (const attack of allAttacks) {
		for (const arm of arms) {
			const row = rows.find(r => (r.attack ?? "unknown") === attack && (r.arm ?? "unknown") === arm);
			cells.push({
				attack,
				arm,
				state: cellState(row),
				status: row?.status ?? null,
				attempts: num(row?.attempts),
				budget: num(row?.attempt_budget),
				received: num(row?.received),
				challenge: num(row?.challenge),
				blocked: num(row?.blocked),
				failed: num(row?.failed),
				releasedTokens: num(row?.released_tokens),
				firstAlert: num(row?.first_alert_seconds),
				firstRestriction: num(row?.first_restriction_seconds),
				firstRefusal: num(row?.first_refusal_seconds),
				attribution: typeof row?.attribution_complete === "boolean" ? row.attribution_complete : null,
				row: row ?? null,
			});
		}
	}
	return { attacks: allAttacks, arms, cells };
}

export type Totals = { arms: number; complete: number; refused: number; interrupted: number; ran: number; attributed: number; attacks: number; attacksRun: number };

/**
 * Counts of what a run's campaign rows cover. Nothing is summed across arms: acquisition from
 * different policies is never pooled into one number.
 */
export function campaignTotals(rows: CampaignItem[], universe: string[] = []): Totals {
	const ran = rows.filter(row => row.status !== "gate-refused" && row.status !== "not-run");
	const attacks = new Set([...universe, ...rows.map(row => row.attack ?? "unknown")]);
	return {
		arms: rows.length,
		complete: rows.filter(row => row.status === "complete").length,
		refused: rows.filter(row => row.status === "gate-refused").length,
		interrupted: rows.filter(row => row.status === "interrupted").length,
		ran: ran.length,
		attributed: ran.filter(row => row.attribution_complete === true).length,
		attacks: attacks.size,
		attacksRun: new Set(ran.map(row => row.attack ?? "unknown")).size,
	};
}

/** Index of the sequential step (0..7) for a value in [0, 1]; null stays null. */
export function seqStep(value: Maybe<number> | undefined): Maybe<number> {
	if (!isNumber(value)) return null;
	return Math.max(0, Math.min(7, Math.floor(value * 8)));
}

/** Compact counts for direct labels: 806,570 reads 807k, 2,784 reads 2.8k. */
export function compact(value: Maybe<number> | undefined): Maybe<string> {
	if (!isNumber(value)) return null;
	const abs = Math.abs(value);
	if (abs >= 1e6) return `${(value / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
	if (abs >= 1e4) return `${Math.round(value / 1e3)}k`;
	if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
	return integer.format(value);
}

export function hours(seconds: Maybe<number> | undefined): Maybe<string> {
	if (!isNumber(seconds)) return null;
	return seconds < 3600 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(2)} h`;
}

/** The last path segment of a run label, for tight places; the full label stays in a title. */
export function shortLabel(label: string): string {
	const parts = label.split("/").filter(Boolean);
	return parts.length ? parts[parts.length - 1] : label;
}
