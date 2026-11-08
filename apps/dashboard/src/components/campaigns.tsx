import type * as React from "react";
import { armColor, compact, emptyIntervention, format, hours, isNumber, outcomeGrid, type OutcomeRow } from "../data";
import type { CampaignItem, Failed, Section as SectionData } from "../types";
import { cn } from "../lib";
import { ArmLegend, ArmName, HBar, Legend, More, Panel, Tip, TipRows } from "./charts";
import { Hash, Status, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

const LEVELS = [1, 2, 3, 4];

const SEGMENTS = [
	{ key: "received", label: "Received", color: "var(--out-received)" },
	{ key: "challenge", label: "Challenged", color: "var(--out-challenge)" },
	{ key: "blocked", label: "Blocked", color: "var(--out-blocked)" },
	{ key: "failed", label: "Failed", color: "var(--out-failed)" },
] as const;

/** The mark for an attack and arm that has nothing to plot, with the reason in words. */
function NoRun({ cell }: { cell: OutcomeRow }) {
	if (cell.state === "absent") return <span className="flex h-3.5 flex-1 items-center rounded-[3px] border border-dashed border-faint/60 px-2 text-[0.68rem] text-faint">no row for this arm</span>;
	const text = cell.state === "gate-refused" ? "gate refused, no requests sent" : cell.status ?? "not recorded";
	return <Tip content={<TipRows title={`${cell.attack} / ${cell.arm}`} rows={[["Status", cell.status ?? "not recorded"], ["Outcome", cell.row?.outcome ?? "not recorded"]]} />} className="flex-1">
		<span className="hatched flex h-3.5 flex-1 items-center rounded-[3px] px-2 text-[0.68rem] text-muted">{text}</span>
	</Tip>;
}

/** Small multiples, one per attack. An attack this run has no rows for collapses to one line. */
function AttackBlocks({ attacks, skipped, children }: { attacks: string[]; skipped: Set<string>; children: (attack: string) => React.ReactNode }) {
	const ran = attacks.filter(attack => !skipped.has(attack));
	const missing = attacks.filter(attack => skipped.has(attack));
	return <div className="space-y-4">
		<div className="grid gap-x-10 gap-y-6 md:grid-cols-2 2xl:grid-cols-3">
			{ran.map(attack => <div key={attack} className="min-w-0 space-y-2">
				<div className="num text-xs font-semibold text-ink">{attack}</div>
				{children(attack)}
			</div>)}
		</div>
		{missing.length > 0 && <p className="flex items-center gap-2 text-xs text-muted">
			<span className="inline-block h-3.5 w-8 rounded-[3px] border border-dashed border-faint/60" aria-hidden />
			No rows in this run for <span className="num text-ink">{missing.join(", ")}</span>, which other runs recorded.
		</p>}
	</div>;
}

function ArmRow({ arm, children }: { arm: string; children: React.ReactNode }) {
	return <div className="grid grid-cols-[6.5rem_1fr] items-center gap-3 text-xs">
		<span className="truncate text-muted"><ArmName arm={arm} /></span>
		<div className="flex min-w-0 items-center">{children}</div>
	</div>;
}

function OutcomeBar({ cell, scale }: { cell: OutcomeRow; scale: number }) {
	if (cell.state !== "complete" && cell.state !== "interrupted" && cell.state !== "other") return <NoRun cell={cell} />;
	const total = cell.attempts ?? null;
	const known = SEGMENTS.filter(s => isNumber(cell[s.key]));
	if (total === null || !known.length) return <span className="unrecorded">counts not recorded</span>;
	const tip = <TipRows title={`${cell.attack} / ${cell.arm}`} rows={[
		["Attempts", format(cell.attempts, "int") ?? "not recorded"],
		...SEGMENTS.map(s => [s.label, format(cell[s.key], "int") ?? "not recorded"] as [string, string]),
		["Status", cell.status ?? "not recorded"],
	]} />;
	return <Tip content={tip} className="flex-1">
		<span className="flex min-w-0 flex-1 items-center gap-2">
			<span className="flex h-3.5 gap-[2px]" style={{ width: `${(total / scale) * 100}%` }}>
				{known.map(s => {
					const value = cell[s.key] as number;
					if (value <= 0) return null;
					return <span key={s.key} className="h-full first:rounded-l-[3px] last:rounded-r-[4px]" style={{ width: `${(value / total) * 100}%`, minWidth: 2, background: s.color }} />;
				})}
			</span>
			<span className="num shrink-0 text-xs">{compact(cell.received)}<span className="text-faint"> / {compact(total)}</span></span>
			{cell.state === "interrupted" && <Badge variant="caution">interrupted</Badge>}
		</span>
	</Tip>;
}

const MARKS = [
	{ key: "firstAlert", label: "First alert", shape: "rounded-full" },
	{ key: "firstRestriction", label: "First restriction", shape: "rounded-none" },
	{ key: "firstRefusal", label: "First refusal", shape: "rotate-45 rounded-none" },
] as const;

function Timeline({ cell, scale }: { cell: OutcomeRow; scale: number }) {
	if (cell.state === "absent" || cell.state === "gate-refused" || cell.state === "not-run") return <NoRun cell={cell} />;
	const row = { status: cell.status, attribution_complete: cell.attribution };
	const times = MARKS.map(m => ({ ...m, value: cell[m.key] }));
	const any = times.some(t => t.value !== null);
	const empty = emptyIntervention(row);
	const tip = <TipRows title={`${cell.attack} / ${cell.arm}`} rows={times.map(t => [t.label, hours(t.value) ?? empty] as [string, string])} />;
	return <Tip content={tip} className="flex-1">
		<span className="relative flex h-4 flex-1 items-center">
			<span className="absolute inset-x-0 top-1/2 h-px bg-rule" />
			{any ? times.map(t => t.value === null ? null : <span key={t.key} aria-hidden className={cn("absolute size-2.5 -translate-x-1/2 border-2 border-paper", t.shape)} style={{ left: `${(t.value / scale) * 100}%`, background: t.key === "firstRefusal" ? "var(--ink)" : t.key === "firstRestriction" ? "var(--accent)" : "var(--faint)" }} />)
				: <span className={cn("relative z-10 bg-paper pr-2 text-[0.68rem]", empty === "none observed" ? "text-muted" : "unrecorded")}>{empty === "none observed" ? "no intervention observed (attribution complete)" : "unknown: attribution not complete"}</span>}
		</span>
	</Tip>;
}

export function CampaignsView({ data, attacks }: { data: SectionData<CampaignItem> | Failed; attacks?: string[] }) {
	if (data.state === "error") return <Unavailable reason={data.reason} tone="error" />;
	if (data.state === "unavailable") return <Unavailable reason={data.reason} />;
	const rows = data.items;
	const grid = outcomeGrid(rows, attacks);
	const cell = (attack: string, arm: string) => grid.cells.find(c => c.attack === attack && c.arm === arm)!;
	const attemptScale = Math.max(1, ...grid.cells.map(c => c.attempts ?? c.budget ?? 0));
	const tokenScale = Math.max(1, ...grid.cells.map(c => c.releasedTokens ?? 0));
	const timeScale = Math.max(1, ...grid.cells.flatMap(c => [c.firstAlert, c.firstRestriction, c.firstRefusal].filter(isNumber))) * 1.04;
	const fromDirectory = rows.some(r => r.source === "directory");
	const skipped = new Set(grid.attacks.filter(attack => grid.cells.every(c => c.attack !== attack || c.state === "absent")));

	return <div className="space-y-10">
		<div className="flex flex-wrap items-center justify-between gap-3">
			<ArmLegend arms={grid.arms} />
			{fromDirectory && <Badge variant="caution">read from run directories; not yet in the index</Badge>}
		</div>

		<Panel title="What happened to each attempt" note="Every attacker attempt, split by how the gateway answered. Bars share one scale, so a shorter bar means fewer attempts. Hover a bar for exact counts.">
			<Legend items={[...SEGMENTS.map(s => ({ label: s.label, color: s.color })), { label: "Not run (gate refused)", hatched: true }, { label: "No row", outline: true }]} />
			<AttackBlocks attacks={grid.attacks} skipped={skipped}>{attack => grid.arms.map(arm => <ArmRow key={arm} arm={arm}><OutcomeBar cell={cell(attack, arm)} scale={attemptScale} /></ArmRow>)}</AttackBlocks>
		</Panel>

		<Panel title="Released tokens" note="Output tokens the teacher released to the attacker's accounts. This is acquisition, not student accuracy; a smaller number does not by itself show that distillation was prevented.">
			<AttackBlocks attacks={grid.attacks} skipped={skipped}>{attack => grid.arms.map(arm => {
				const c = cell(attack, arm);
				if (c.state === "absent" || c.state === "gate-refused" || c.state === "not-run") return <ArmRow key={arm} arm={arm}><NoRun cell={c} /></ArmRow>;
				const unknown = isNumber(c.row?.released_tokens_unknown) && (c.row?.released_tokens_unknown as number) > 0;
				return <ArmRow key={arm} arm={arm}>
					<HBar value={c.releasedTokens} max={tokenScale} color={armColor(arm)} label={<>{compact(c.releasedTokens)}{unknown && <span className="text-caution"> + unknown</span>}</>}
						tip={<TipRows title={`${attack} / ${arm}`} rows={[["Released tokens", format(c.releasedTokens, "int") ?? "not recorded"], ["Distinct examples", format(c.row?.distinct_examples, "int") ?? "not recorded"], ["Usages unknown", format(c.row?.released_tokens_unknown, "int") ?? "not recorded"]]} />} />
				</ArmRow>;
			})}</AttackBlocks>
		</Panel>

		<Panel title="When the policy intervened" note={`Time from campaign start to the first alert, restriction and refusal, on one axis from 0 to ${hours(timeScale)}. An empty line reads "no intervention observed" only when every query was attributed; otherwise it is unknown.`}>
			<Legend items={[{ label: "First alert", color: "var(--faint)" }, { label: "First restriction", color: "var(--accent)" }, { label: "First refusal", color: "var(--ink)" }]} />
			<AttackBlocks attacks={grid.attacks} skipped={skipped}>{attack => grid.arms.map(arm => <ArmRow key={arm} arm={arm}><Timeline cell={cell(attack, arm)} scale={timeScale} /></ArmRow>)}</AttackBlocks>
		</Panel>

		{data.errors && data.errors.length > 0 && <div className="space-y-1">{data.errors.map(e => <Unavailable key={`${e.attack}/${e.arm}`} tone="error" reason={`${e.attack}/${e.arm}: ${e.error}`} />)}</div>}

		<div className="space-y-3">
			<More label={`All campaign values (${rows.length} arms)`}>
				<Table>
					<TableHeader><TableRow>
						<TableHead>Attack</TableHead><TableHead>Arm</TableHead><TableHead>Policy</TableHead><TableHead>Status</TableHead>
						<TableHead className="text-right">Attempts</TableHead><TableHead className="text-right">Received</TableHead><TableHead className="text-right">Blocked</TableHead>
						<TableHead className="text-right">Challenged</TableHead><TableHead className="text-right">Failed</TableHead><TableHead className="text-right">Released tokens</TableHead>
						<TableHead className="text-right">Distinct examples</TableHead><TableHead className="text-right">First alert</TableHead><TableHead className="text-right">First restriction</TableHead>
						<TableHead className="text-right">First refusal</TableHead><TableHead>Attribution</TableHead>
					</TableRow></TableHeader>
					<TableBody>{rows.map(row => <TableRow key={`${row.attack}/${row.arm}/${row.stratum}`}>
						<TableCell className="num">{row.attack}</TableCell>
						<TableCell><ArmName arm={row.arm ?? "unknown"} /></TableCell>
						<TableCell><Value value={[row.policy, row.accumulator].filter(Boolean).join(" / ") || null} /></TableCell>
						<TableCell>
							<Status status={row.status} />
							{row.status !== "complete" && row.outcome && <p className="mt-1 max-w-56 text-xs text-muted">{row.outcome}</p>}
						</TableCell>
						<TableCell className="text-right"><Value value={row.attempts} as="int" /></TableCell>
						<TableCell className="text-right"><Value value={row.received} as="int" /></TableCell>
						<TableCell className="text-right"><Value value={row.blocked} as="int" /></TableCell>
						<TableCell className="text-right"><Value value={row.challenge} as="int" /></TableCell>
						<TableCell className="text-right"><Value value={row.failed} as="int" /></TableCell>
						<TableCell className="text-right">
							<Value value={row.released_tokens} as="int" />
							{typeof row.released_tokens_unknown === "number" && row.released_tokens_unknown > 0 && <p className="text-xs text-caution">{row.released_tokens_unknown} usages unknown</p>}
						</TableCell>
						<TableCell className="text-right"><Value value={row.distinct_examples} as="int" /></TableCell>
						<TableCell className="text-right"><Value value={row.first_alert_seconds} as="seconds" unknown={emptyIntervention(row)} /></TableCell>
						<TableCell className="text-right"><Value value={row.first_restriction_seconds} as="seconds" unknown={emptyIntervention(row)} /></TableCell>
						<TableCell className="text-right"><Value value={row.first_refusal_seconds} as="seconds" unknown={emptyIntervention(row)} /></TableCell>
						<TableCell>
							{row.attribution_complete === false ? <Badge variant="caution">incomplete · {row.unattributed_queries ?? "?"} unattributed</Badge> : <Value value={row.attribution_complete === true ? "complete" : null} />}
						</TableCell>
					</TableRow>)}</TableBody>
					<TableCaption>An empty intervention time reads "none observed" only when every query was attributed to a policy event; otherwise it is unknown. Gate-refused arms sent no requests and keep their row.</TableCaption>
				</Table>
			</More>
			<More label="Escalation levels and identifiers">
				<p className="mb-2 max-w-3xl text-xs text-muted">J is the first attempt index whose account reached the level; Q is received responses completed strictly before that time. A level with no value is unreached only when attribution is complete; otherwise it is unknown.</p>
				<Table>
					<TableHeader><TableRow>
						<TableHead>Attack</TableHead><TableHead>Arm</TableHead>
						{LEVELS.map(level => <TableHead key={level} className="text-right">L{level} s / J / Q</TableHead>)}
						<TableHead>Seed</TableHead><TableHead>Budget</TableHead><TableHead>Partition</TableHead><TableHead>Study</TableHead><TableHead>Reading label</TableHead><TableHead>Contract</TableHead><TableHead>Ledger</TableHead>
					</TableRow></TableHeader>
					<TableBody>{rows.map(row => <TableRow key={`${row.attack}/${row.arm}`}>
						<TableCell className="num">{row.attack}</TableCell><TableCell><ArmName arm={row.arm ?? "unknown"} /></TableCell>
						{LEVELS.map(level => <TableCell key={level} className="text-right whitespace-nowrap">
							{row[`L${level}_J`] === null ? <Value value={null} unknown={emptyIntervention(row, "unreached")} /> : <span className="num"><Value value={row[`L${level}_seconds`]} as="seconds" /> · {String(row[`L${level}_J`])} · {String(row[`L${level}_Q`] ?? "?")}</span>}
						</TableCell>)}
						<TableCell><Value value={row.seed} /></TableCell>
						<TableCell><Value value={row.attempt_budget} as="int" /></TableCell>
						<TableCell><Value value={row.partition} /></TableCell>
						<TableCell><Value value={row.study} /></TableCell>
						<TableCell><Value value={row.reading_label} /></TableCell>
						<TableCell><Hash value={row.contract_sha256 as string | null} /></TableCell>
						<TableCell><Hash value={row.ledger_sha256 as string | null} /></TableCell>
					</TableRow>)}</TableBody>
				</Table>
			</More>
		</div>
	</div>;
}
