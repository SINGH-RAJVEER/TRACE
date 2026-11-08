// One run on one short page: a few headline figures, what is missing, and one chart per question.
// Everything else sits behind a single toggle at the bottom.

import * as React from "react";
import { armColor, campaignTotals, compact, coverage, emptyIntervention, format, gaps, isNumber, outcomeGrid, planCell, shortLabel, sortArms, type Gap, type OutcomeRow } from "../data";
import type { Benchmark, BenchmarkCell, CampaignItem, Checkpoints, GateReport, RunDetail, RunSummary, UtilityArm } from "../types";
import { cn } from "../lib";
import { ArmSwatch, CoverageMark, Figure, GroupedBars, More, Stat, Tip, TipRows, linearScale, logScale, type Bar, type LegendItem } from "./charts";
import { Hash, Unavailable, Value, Verdict } from "./evidence";
import { ProvenanceView } from "./provenance";
import { CurveChart, StudentDetails } from "./students";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

export type Loaded = { run: RunSummary; detail: RunDetail | null; error?: string };

const items = <T,>(section: unknown): T[] => section && typeof section === "object" && "items" in section ? (section as { items: T[] }).items : [];

function armLegend(arms: string[]): LegendItem[] {
	return arms.map(arm => ({ label: arm, color: armColor(arm) }));
}

// ---------------------------------------------------------------------------------------------
// Acquisition

function cellTip(cell: OutcomeRow, extra: [string, React.ReactNode][] = []) {
	return <TipRows title={`${cell.attack} / ${cell.arm}`} rows={[
		["Status", cell.status ?? "no row"],
		["Released tokens", format(cell.releasedTokens, "int") ?? "not recorded"],
		["Answered / blocked", cell.state === "complete" || cell.state === "interrupted" ? `${format(cell.received, "int") ?? "?"} / ${format(cell.blocked, "int") ?? "?"}` : "not run"],
		...extra,
	]} />;
}

function stubState(cell: OutcomeRow): Bar["state"] {
	return cell.state === "absent" ? "absent" : "no-value";
}

function TokensChart({ cells, attacks, arms }: { cells: OutcomeRow[]; attacks: string[]; arms: string[] }) {
	const values = cells.map(c => c.releasedTokens ?? c.assumed?.tokens ?? null).filter(isNumber);
	const scale = logScale(values, v => v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : `${v}`);
	return <Figure title="Released tokens per attack (log scale)" legend={[...armLegend(arms), ...(cells.some(c => c.assumed) ? [{ label: "refused by the gate, assumed let through", assumed: true }] : []), ...(cells.some(c => (c.state === "gate-refused" || c.state === "not-run") && !c.assumed) ? [{ label: "refused by the gate", hatched: true }] : []), ...(cells.some(c => c.state === "interrupted") ? [{ label: "! interrupted", color: "var(--caution)" }] : []), ...(cells.some(c => c.state === "absent") ? [{ label: "no row", outline: true }] : [])]}>
		<GroupedBars scale={scale} groups={attacks.map(attack => ({
			key: attack,
			label: attack,
			bars: arms.map(arm => {
				const cell = cells.find(c => c.attack === attack && c.arm === arm)!;
				const has = isNumber(cell.releasedTokens);
				if (!has && cell.assumed) return { key: arm, color: armColor(arm), value: cell.assumed.tokens, state: "assumed", label: compact(cell.assumed.tokens) ?? undefined, tip: <TipRows title={`${cell.attack} / ${cell.arm}: assumed, not measured`} rows={[["Status", "refused by the detectability gate"], ["Assumed tokens", `≈${format(cell.assumed.tokens, "int")}`], ["Basis", cell.assumed.basis]]} /> };
				return { key: arm, color: armColor(arm), value: cell.releasedTokens, state: has ? "value" : stubState(cell), label: compact(cell.releasedTokens) ?? undefined, flag: cell.state === "interrupted" ? "!" : undefined, tip: cellTip(cell) };
			}),
		}))} />
	</Figure>;
}

function firstIntervention(cell: OutcomeRow): number | null {
	const times = [cell.firstAlert, cell.firstRestriction, cell.firstRefusal].filter(isNumber);
	return times.length ? Math.min(...times) : null;
}

function InterventionChart({ cells, attacks, arms }: { cells: OutcomeRow[]; attacks: string[]; arms: string[] }) {
	const minutes = (s: number | null) => s === null ? null : s / 60;
	const values = cells.map(c => minutes(firstIntervention(c))).filter(isNumber);
	if (!values.length) return null;
	const scale = linearScale(values, v => `${Math.round(v)}`);
	return <Figure title="Minutes until the policy first intervened" legend={[...armLegend(arms), { label: "not run or unknown", hatched: true }, { label: "never intervened", outline: true }]}>
		<GroupedBars scale={scale} groups={attacks.map(attack => ({
			key: attack,
			label: attack,
			bars: arms.map(arm => {
				const cell = cells.find(c => c.attack === attack && c.arm === arm)!;
				const value = minutes(firstIntervention(cell));
				const empty = cell.row ? emptyIntervention(cell.row) : "no row";
				const state: Bar["state"] = isNumber(value) ? "value" : cell.state === "absent" ? "absent" : empty === "none observed" ? "none" : "no-value";
				return { key: arm, color: armColor(arm), value, state, label: isNumber(value) ? `${Math.round(value)}` : undefined, tip: cellTip(cell, [["First intervention", isNumber(value) ? `${value.toFixed(0)} min` : empty]]) };
			}),
		}))} />
	</Figure>;
}

// ---------------------------------------------------------------------------------------------
// Utility

const RATES = [
	["completion_rate", "Completion"],
	["served_only_quality", "Quality when served"],
	["unconditional_utility", "Overall utility"],
] as const;

function UtilityChart({ arms }: { arms: UtilityArm[] }) {
	const sorted = sortArms(arms.map(a => a.arm)).map(name => arms.find(a => a.arm === name)!);
	const scale = linearScale([], v => `${Math.round(v * 100)}%`, 1);
	return <Figure title="Benign workload served" legend={[...armLegend(sorted.map(a => a.arm)), { label: "not recorded", hatched: true }]}>
		<GroupedBars scale={scale} groups={RATES.map(([key, label]) => ({
			key,
			label,
			bars: sorted.map(arm => {
				const value = arm.pooled?.[key] ?? null;
				return { key: arm.arm, color: armColor(arm.arm), value, state: isNumber(value) ? "value" : "no-value", label: isNumber(value) ? `${Math.round(value * 100)}%` : undefined, tip: <TipRows title={`${arm.arm}: ${label}`} rows={[["Value", format(value, "fraction") ?? "not recorded"], ["Served / submitted", `${format(arm.pooled?.served, "int") ?? "?"} / ${format(arm.pooled?.submitted, "int") ?? "?"}`], ["Validity", arm.validity ?? "unknown"]]} /> };
			}),
		}))} />
		<p className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
			<span>p95 latency</span>
			{sorted.map(arm => <span key={arm.arm} className="inline-flex items-center gap-1.5"><ArmSwatch arm={arm.arm} className="size-2" /><Value value={arm.pooled?.latency_p95} as="seconds" /></span>)}
		</p>
	</Figure>;
}

// ---------------------------------------------------------------------------------------------
// Students

type StripRow = { key: string; label: string; value: number | null; reason: string; base: number | null; target: number | null; tip: React.ReactNode };

function stripRows(plans: Checkpoints[]): StripRow[] {
	return plans.flatMap(item => {
		const plan = shortLabel(item.place.path);
		if (!item.summary) return [{ key: plan, label: plan, value: null, reason: "no summary.json", base: null, target: item.target_accuracy, tip: <TipRows title={plan} rows={[["Accuracy", "no summary.json"], ["Target", format(item.target_accuracy, "fraction") ?? "not recorded"]]} /> }];
		const base = item.summary.base_accuracy ?? null;
		const target = item.summary.target_accuracy ?? item.target_accuracy;
		return (item.summary.checkpoints ?? []).map(row => {
			const value = row.status === "censored" ? null : row.mean_accuracy ?? null;
			return { key: `${plan}/${row.id}`, label: `${plan} · ${row.id}`, value, reason: row.status === "censored" ? "censored" : "not recorded", base, target, tip: <TipRows title={`${plan} / ${row.id}`} rows={[["Accuracy", format(value, "fraction") ?? row.status], ["Gain over base", format(row.mean_gain_pp, "pp") ?? "not recorded"], ["Seeds reaching target", format(row.seeds_reaching_target, "int") ?? "not recorded"], ["Trainable examples", format(row.trainable_count, "int") ?? "not recorded"]]} /> };
		});
	});
}

/** Accuracy on one 0 to 100% axis per row, with the base, teacher and target marked on it. */
function StudentChart({ plans, teacher }: { plans: Checkpoints[]; teacher: number | null }) {
	const rows = stripRows(plans);
	if (!rows.length) return null;
	const at = (v: number) => `${v * 100}%`;
	return <Figure title="Student accuracy" legend={[{ label: "student", color: "var(--accent)" }, { label: "untrained base", color: "var(--faint)", line: "solid" }, ...(isNumber(teacher) ? [{ label: "teacher", color: "var(--ink)", line: "solid" as const }] : []), { label: "target", color: "var(--gold)", line: "dashed" }, { label: "not recorded", hatched: true }]}>
		<div className="space-y-3">
			{rows.map(r => <div key={r.key} className="space-y-1">
				<div className="flex items-baseline justify-between gap-3 text-xs" title={r.key}><span className="num truncate text-muted">{r.label}</span>{r.value !== null && <span className="num font-medium text-ink">{format(r.value, "fraction")}</span>}</div>
				<div className="relative h-6">
					<span aria-hidden className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-track" />
					{isNumber(r.base) && <span aria-hidden className="absolute inset-y-0 border-l-2 border-faint" style={{ left: at(r.base) }} />}
					{isNumber(teacher) && <span aria-hidden className="absolute inset-y-0 border-l-2 border-ink" style={{ left: at(teacher) }} />}
					{isNumber(r.target) && <span aria-hidden className="absolute inset-y-0 border-l-2 border-dashed border-gold" style={{ left: at(r.target) }} />}
					{r.value === null
						? <span className="absolute inset-x-0 inset-y-1 flex items-center"><span className="hatched absolute inset-0 rounded-[3px]" /><span className="relative ml-2 bg-paper px-1 text-xs text-muted">{r.reason}</span></span>
						: <span className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ left: at(r.value) }}><Tip content={r.tip}><span className="block size-3.5 rounded-full border-2 border-paper bg-accent" /></Tip></span>}
				</div>
			</div>)}
			<div className="relative h-4 text-[0.62rem] text-faint">
				{[0, 0.25, 0.5, 0.75, 1].map(t => <span key={t} className={cn("num absolute", t === 0 ? "" : t === 1 ? "-translate-x-full" : "-translate-x-1/2")} style={{ left: at(t) }}>{t * 100}%</span>)}
			</div>
		</div>
	</Figure>;
}

// ---------------------------------------------------------------------------------------------
// Student benchmark panel

function cellColor(cell: BenchmarkCell): string {
	if (cell.role === "teacher") return "var(--ink)";
	if (cell.role === "base") return "var(--faint)";
	return armColor(cell.name.split("/")[1] ?? cell.name);
}

/** The paired difference of `a` over `b`, whichever order the producer stored the pair in. */
function versus(panel: Benchmark, a: string, b: string): string {
	const pair = panel.pairs.find(p => (p.a === a && p.b === b) || (p.a === b && p.b === a));
	if (!pair || !isNumber(pair.difference)) return "not recorded";
	const sign = pair.a === a ? 1 : -1;
	const pp = (v: number) => `${v * sign * 100 >= 0 ? "+" : ""}${(v * sign * 100).toFixed(1)}`;
	const ci = pair.paired_item_bootstrap95 ? ` [${[pp(pair.paired_item_bootstrap95[0]), pp(pair.paired_item_bootstrap95[1])].sort((x, y) => Number(x) - Number(y)).join(", ")}]` : "";
	const p = isNumber(pair.mcnemar_holm_p) ? `, p ${pair.mcnemar_holm_p < 0.001 ? "< 0.001" : pair.mcnemar_holm_p.toFixed(3)}` : "";
	return `${pp(pair.difference)} pp${ci}${p}`;
}

/** Each model's accuracy with its 95% interval on one shared axis, best first between teacher and base. */
function BenchmarkChart({ panel, target }: { panel: Benchmark; target: number | null }) {
	const rank = (c: BenchmarkCell) => c.role === "teacher" ? 0 : c.role === "base" ? 2 : 1;
	const rows = [...panel.cells].sort((a, b) => rank(a) - rank(b) || (b.accuracy ?? -1) - (a.accuracy ?? -1));
	const bounds = rows.flatMap(c => c.wilson95 ?? (isNumber(c.accuracy) ? [c.accuracy] : [])).concat(isNumber(target) ? [target] : []);
	const lo = Math.floor(Math.min(...bounds) * 10) / 10;
	const hi = Math.ceil(Math.max(...bounds) * 10) / 10;
	const at = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
	const ticks = Array.from({ length: Math.round((hi - lo) * 10) + 1 }, (_, i) => lo + i / 10);
	const base = rows.find(c => c.role === "base")?.accuracy ?? null;
	const teacher = rows.find(c => c.role === "teacher")?.accuracy ?? null;
	const arms = sortArms(rows.filter(c => c.role === "student").map(c => c.name.split("/")[1] ?? c.name));
	return <Figure title={`Student benchmark: ${panel.task?.toUpperCase() ?? "task"} ${panel.split ?? ""}, ${panel.items ?? "?"} items`} legend={[...arms.map(arm => ({ label: arm, color: armColor(arm) })), { label: "teacher", color: "var(--ink)" }, { label: "base", color: "var(--faint)" }, ...(isNumber(target) ? [{ label: "target", color: "var(--gold)", line: "dashed" as const }] : [])]}>
		<div className="space-y-1">
			{rows.map(c => {
				const name = c.role === "student" ? c.name : c.role ?? c.name;
				const tip = <TipRows title={c.name} rows={[
					["Accuracy", `${format(c.accuracy, "fraction") ?? "not recorded"} (${c.correct ?? "?"} / ${panel.items ?? "?"})`],
					["95% interval", c.wilson95 ? `${format(c.wilson95[0], "fraction")} to ${format(c.wilson95[1], "fraction")}` : "not recorded"],
					...(c.role === "student" ? [["Training examples", format(c.training_examples, "int") ?? "not recorded"] as [string, string], ["vs base", versus(panel, c.name, "base")] as [string, string], ["vs teacher", versus(panel, c.name, "teacher")] as [string, string]] : []),
					["Model", c.hf_repo ?? c.model ?? "not recorded"],
					...(c.consistent === false ? [["Check", "correctness does not match the count"] as [string, string]] : []),
				]} />;
				return <div key={c.name} className="grid grid-cols-[1fr_3.5rem] items-center gap-x-3 text-xs sm:grid-cols-[11.5rem_1fr_3.5rem]">
					<span className="col-span-2 min-w-0 truncate sm:col-span-1"><span className="num text-ink">{name}</span>{c.role === "student" && isNumber(c.training_examples) && <span className="text-faint"> · {compact(c.training_examples)} ex</span>}</span>
					<span className="relative h-6">
						<span aria-hidden className="absolute inset-x-0 top-1/2 border-t border-rule/70" />
						{isNumber(base) && <span aria-hidden className="absolute inset-y-0 border-l border-faint/70" style={{ left: at(base) }} />}
						{isNumber(teacher) && <span aria-hidden className="absolute inset-y-0 border-l border-ink/50" style={{ left: at(teacher) }} />}
						{isNumber(target) && <span aria-hidden className="absolute inset-y-0 border-l border-dashed border-gold" style={{ left: at(target) }} />}
						{c.wilson95 && <span aria-hidden className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full" style={{ left: at(c.wilson95[0]), width: `calc(${at(c.wilson95[1])} - ${at(c.wilson95[0])})`, background: cellColor(c) }} />}
						{isNumber(c.accuracy)
							? <span className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ left: at(c.accuracy) }}><Tip content={tip}><span className="block size-3 rounded-full border-2 border-paper" style={{ background: cellColor(c) }} /></Tip></span>
							: <span className="unrecorded absolute top-1/2 left-0 -translate-y-1/2">not recorded</span>}
					</span>
					<span className={cn("num text-right", c.consistent === false ? "text-caution" : "text-ink")}>{format(c.accuracy, "fraction") ?? "—"}</span>
				</div>;
			})}
			<div className="grid grid-cols-[1fr_3.5rem] gap-x-3 sm:grid-cols-[11.5rem_1fr_3.5rem]">
				<span className="hidden sm:block" />
				<span className="relative h-4 text-[0.62rem] text-faint">
					{ticks.map((t, i) => <span key={t} className={cn("num absolute", i === 0 ? "" : i === ticks.length - 1 ? "-translate-x-full" : "-translate-x-1/2")} style={{ left: at(t) }}>{Math.round(t * 100)}%</span>)}
				</span>
			</div>
		</div>
		<p className="text-xs text-faint">{panel.evaluation_mode === "validated_recorded_inference" ? "Recorded inference, validated. " : ""}Lines are 95% Wilson intervals over items; one training seed per student.</p>
	</Figure>;
}

// ---------------------------------------------------------------------------------------------
// Page

function Checks({ headroom, capacity, clean }: { headroom: GateReport[]; capacity: { passed: boolean | null }[]; clean: GateReport[] }) {
	const entries: [string, React.ReactNode][] = [
		...headroom.map((g, i) => [`Headroom gate${headroom.length > 1 ? ` ${i + 1}` : ""}`, <Verdict passed={g.passed} status={g.status} />] as [string, React.ReactNode]),
		...capacity.map((c, i) => [`Capacity preflight${capacity.length > 1 ? ` ${i + 1}` : ""}`, <Verdict passed={c.passed} />] as [string, React.ReactNode]),
		...clean.map((g, i) => [`Clean-student gate${clean.length > 1 ? ` ${i + 1}` : ""}`, <Verdict passed={g.passed} status={g.status} />] as [string, React.ReactNode]),
	];
	if (!entries.length) return null;
	return <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
		{entries.map(([label, badge]) => <span key={label} className="inline-flex items-center gap-2 text-muted">{label}{badge}</span>)}
	</div>;
}

export function MissingList({ items: list, limit = 5 }: { items: Gap[]; limit?: number }) {
	if (!list.length) return <p className="flex items-center gap-2 text-sm text-muted"><CoverageMark state="present" compact />Nothing the dashboard reads is missing.</p>;
	const line = (g: Gap) => <li key={g.text} className="flex items-start gap-2">
		<span className="mt-[3px] shrink-0"><CoverageMark state={g.severity === "missing" ? "missing" : "partial"} compact /></span>
		<span>{g.text}</span>
	</li>;
	return <div className="space-y-2">
		<ul className="space-y-1 text-sm">{list.slice(0, limit).map(line)}</ul>
		{list.length > limit && <More label={`${list.length - limit} more`}><ul className="space-y-1 text-sm">{list.slice(limit).map(line)}</ul></More>}
	</div>;
}

function RecordedTables({ cells, utility }: { cells: OutcomeRow[]; utility: UtilityArm[] }) {
	return <div className="space-y-8">
		{cells.length > 0 && <Table>
			<TableHeader><TableRow>{["Attack", "Arm", "Status", "Attempts", "Answered", "Challenged", "Blocked", "Failed", "Released tokens", "First intervention"].map(h => <TableHead key={h}>{h}</TableHead>)}</TableRow></TableHeader>
			<TableBody>{cells.filter(c => c.state !== "absent").map(c => <TableRow key={`${c.attack}/${c.arm}`}>
				<TableCell className="num">{c.attack}</TableCell>
				<TableCell><span className="inline-flex items-center gap-1.5"><ArmSwatch arm={c.arm} />{c.arm}</span></TableCell>
				<TableCell>{c.status}</TableCell>
				{[c.attempts, c.received, c.challenge, c.blocked, c.failed, c.releasedTokens].map((v, i) => <TableCell key={i}><Value value={v} as="int" unknown="—" /></TableCell>)}
				<TableCell>{isNumber(firstIntervention(c)) ? <Value value={firstIntervention(c)} as="seconds" /> : <span className="text-muted">{c.row ? emptyIntervention(c.row) : "—"}</span>}</TableCell>
			</TableRow>)}</TableBody>
		</Table>}
		{utility.length > 0 && <Table>
			<TableHeader><TableRow>{["Arm", "Submitted", "Served", "Completion", "Quality when served", "Overall utility", "p95 latency", "Validity"].map(h => <TableHead key={h}>{h}</TableHead>)}</TableRow></TableHeader>
			<TableBody>{utility.map(a => <TableRow key={a.arm}>
				<TableCell><span className="inline-flex items-center gap-1.5"><ArmSwatch arm={a.arm} />{a.arm}</span></TableCell>
				<TableCell><Value value={a.pooled?.submitted} as="int" /></TableCell>
				<TableCell><Value value={a.pooled?.served} as="int" /></TableCell>
				<TableCell><Value value={a.pooled?.completion_rate} as="fraction" /></TableCell>
				<TableCell><Value value={a.pooled?.served_only_quality} as="fraction" /></TableCell>
				<TableCell><Value value={a.pooled?.unconditional_utility} as="fraction" /></TableCell>
				<TableCell><Value value={a.pooled?.latency_p95} as="seconds" /></TableCell>
				<TableCell><Value value={a.validity} unknown="unknown" /></TableCell>
			</TableRow>)}</TableBody>
		</Table>}
	</div>;
}

export function RunPage({ entry, attacks }: { entry: Loaded; attacks: string[] }) {
	const { run, detail } = entry;
	return <div className="space-y-10">
		<header className="flex flex-wrap items-baseline gap-x-3 gap-y-2">
			<h2 className="num text-2xl font-semibold" style={{ fontFamily: "var(--font-sans)" }} title={run.label}>{run.label}</h2>
			{run.studies.map(study => <Badge key={study} variant={study === "confirmatory" ? "info" : "caution"}>{study}</Badge>)}
			{run.kind !== "study" && <Badge>{run.kind}</Badge>}
			<span className="text-xs text-muted">{run.commits.length ? <>commit {run.commits.map(c => <Hash key={c} value={c} />)}</> : "no commit recorded"}</span>
		</header>
		{!detail ? <Unavailable tone={entry.error ? "error" : "neutral"} reason={entry.error ?? "Reading this run's artifacts."} /> : <RunBody detail={detail} attacks={attacks} />}
	</div>;
}

function RunBody({ detail, attacks }: { detail: RunDetail; attacks: string[] }) {
	const campaignRows = items<CampaignItem>(detail.campaigns);
	const recordedAttacks = [...new Set(campaignRows.map(r => r.attack ?? "unknown"))];
	const grid = outcomeGrid(campaignRows, recordedAttacks);
	const totals = campaignTotals(campaignRows, campaignRows.length ? attacks : []);
	const utility = items<UtilityArm>(detail.utility);
	const students = "curves" in detail.students ? detail.students : null;
	const teacher = "scores" in detail.teacher ? detail.teacher : null;
	const gates = "reports" in detail.gates ? detail.gates : null;
	const plans = students?.checkpoints.items ?? [];
	const curves = (students?.curves.items ?? []).filter(c => c.summary);
	const teacherScore = teacher?.scores.items.find(s => s.role === "teacher")?.accuracy ?? null;
	const cells = coverage(detail).filter(c => c.state !== "na");
	const recorded = cells.filter(c => c.state === "present" || c.state === "partial").length;
	const list = gaps(detail, attacks);
	const panels = students?.benchmarks?.items ?? [];
	const benchmarked = new Set(panels.flatMap(b => b.cells.map(c => c.name)));
	const stripPlans = plans.filter(p => !benchmarked.has(planCell(p.place.path) ?? ""));
	const target = plans.map(p => p.summary?.target_accuracy ?? p.target_accuracy).find(isNumber) ?? null;
	const candidates = [
		...panels.flatMap(b => b.cells.filter(c => c.role === "student" && isNumber(c.accuracy)).map(c => ({ plan: c.name, value: c.accuracy as number }))),
		...stripPlans.flatMap(p => (p.summary?.checkpoints ?? []).map(r => ({ plan: shortLabel(p.place.path), value: r.status === "censored" ? null : r.mean_accuracy ?? null }))),
	].filter((r): r is { plan: string; value: number } => isNumber(r.value));
	const top = candidates.length ? Math.max(...candidates.map(c => c.value)) : null;
	const best = top === null ? null : { plan: [...new Set(candidates.filter(c => c.value === top).map(c => c.plan))].join(", "), value: top, target };
	const completion = utility.map(a => a.pooled?.completion_rate).filter(isNumber);

	return <>
		<section className="grid gap-x-12 gap-y-6 lg:grid-cols-[minmax(0,26rem)_1fr]">
			<div className="grid grid-cols-2 content-start gap-x-10 gap-y-6">
				{totals.arms > 0 && <Stat label="Arms that ran" value={<>{totals.ran}<span className="text-base text-faint"> / {totals.arms}</span></>} tone={totals.ran < totals.arms ? "caution" : undefined} />}
				{completion.length > 0 && <Stat label="Benign completion" value={completion.length > 1 ? `${format(Math.min(...completion), "fraction")}–${format(Math.max(...completion), "fraction")}` : format(completion[0], "fraction")} />}
				{best && <Stat label={`Best student${isNumber(best.target) ? ` (target ${format(best.target, "fraction")})` : ""}`} value={<>{format(best.value, "fraction")}<span className="num mt-1.5 block text-xs font-normal tracking-normal text-muted">{best.plan}</span></>} />}
				<Stat label="Evidence recorded" value={<>{recorded}<span className="text-base text-faint"> / {cells.length}</span></>} tone={recorded < cells.length ? "caution" : undefined} />
			</div>
			<div className="min-w-0 space-y-2 lg:border-l lg:border-rule lg:pl-8">
				<h3 className="text-sm font-semibold">Missing</h3>
				<MissingList items={list} />
			</div>
		</section>
		{grid.cells.length > 0 && <section className="grid gap-x-12 gap-y-10 lg:grid-cols-2">
			<TokensChart cells={grid.cells} attacks={grid.attacks} arms={grid.arms} />
			<InterventionChart cells={grid.cells} attacks={grid.attacks} arms={grid.arms} />
		</section>}
		{(utility.length > 0 || panels.length > 0 || stripPlans.length > 0 || curves.length > 0) && <section className="grid gap-x-12 gap-y-10 lg:grid-cols-2">
			{utility.length > 0 && <UtilityChart arms={utility} />}
			{panels.map(panel => <BenchmarkChart key={panel.place.path} panel={panel} target={target} />)}
			{stripPlans.length > 0 && <StudentChart plans={stripPlans} teacher={teacherScore} />}
			{curves.map(c => <Figure key={c.place.path} title={`Learning curve: ${shortLabel(c.place.path)}`}><CurveChart curve={c} /></Figure>)}
		</section>}
		<Checks headroom={teacher?.headroom.items ?? []} capacity={teacher?.capacity.items ?? []} clean={gates?.reports.items ?? []} />
		<div className={cn("border-t border-rule pt-4")}>
			<More label="Recorded values and run configuration">
				<div className="space-y-10">
					<RecordedTables cells={grid.cells} utility={utility} />
					<StudentDetails data={detail.students} />
					<ProvenanceView data={detail.provenance} />
				</div>
			</More>
		</div>
	</>;
}
