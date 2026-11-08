import { TriangleAlert } from "lucide-react";
import { COVERAGE_GROUPS, campaignTotals, compact, coverage, format, gaps, outcomeGrid, sortAttacks, studyLabel, type CoverageCell, type Gap, type OutcomeRow } from "../data";
import type { RunDetail, RunList, RunSummary } from "../types";
import { cn } from "../lib";
import { ArmSwatch, CoverageMark, Legend, Meter, Stat, Tip, TipRows } from "./charts";
import { Value } from "./evidence";

export type Loaded = { run: RunSummary; detail: RunDetail | null; error?: string };

function campaignItems(detail: RunDetail | null) {
	return detail && "items" in detail.campaigns ? detail.campaigns.items : [];
}

/** Every attack named by any run, so a run that skipped an attack shows the gap. */
export function attackUniverse(loaded: Loaded[]): string[] {
	return sortAttacks(loaded.flatMap(l => campaignItems(l.detail).map(row => row.attack ?? "unknown")));
}

function RootWarnings({ list }: { list: RunList }) {
	const missing = list.roots.filter(root => !root.exists);
	if (!missing.length && list.index.state === "ok") return null;
	return <div className="flex items-start gap-3 border-l-2 border-caution py-1 pl-4 text-sm">
		<TriangleAlert size={16} className="mt-0.5 shrink-0 text-caution" aria-hidden />
		<div className="space-y-1">
			{missing.map(root => <p key={root.name}>The configured root <span className="num">{root.name}</span> does not exist. Gates, teacher scores, capacity preflights and student artifacts are read from roots like this one, so every run reports them missing until it is created or <span className="num">DASHBOARD_ROOTS</span> points at the directory that holds them.</p>)}
			{list.index.state !== "ok" && <p>The results index <span className="num">{list.index.path}</span> is {list.index.state}{list.index.error ? `: ${list.index.error}` : ""}.</p>}
		</div>
	</div>;
}

function CoverageMatrix({ loaded, onSelect }: { loaded: Loaded[]; onSelect: (id: string) => void }) {
	const rows = loaded.map(l => ({ ...l, cells: l.detail ? coverage(l.detail) : null }));
	const columns = rows.find(r => r.cells)?.cells ?? [];
	return <div className="overflow-x-auto">
		<table className="w-full border-collapse text-xs">
			<thead>
				<tr>
					<th />
					{COVERAGE_GROUPS.map(group => {
						const span = columns.filter(c => c.group === group).length;
						return span ? <th key={group} colSpan={span} className="border-b border-rule px-2 pb-1 text-left text-[0.68rem] font-semibold tracking-wide text-faint uppercase">{group}</th> : null;
					})}
				</tr>
				<tr>
					<th className="py-2 pr-4 text-left font-medium text-muted">Run</th>
					{columns.map(c => <th key={c.key} className="px-2 py-2 text-left align-bottom font-medium text-muted">{c.label}</th>)}
				</tr>
			</thead>
			<tbody>
				{rows.map(r => <tr key={r.run.id} className="border-t border-rule/70">
					<th className="py-2.5 pr-4 text-left font-normal">
						<button type="button" onClick={() => onSelect(r.run.id)} className="num text-left text-sm text-ink underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-accent">{r.run.label}</button>
					</th>
					{r.cells ? r.cells.map(c => <td key={c.key} className="px-2 py-2.5">
						<Tip content={<TipRows title={`${r.run.label}: ${c.label}`} rows={[["State", c.state], ["Detail", <span className="font-sans whitespace-normal">{c.detail}</span>]]} />}>
							<CoverageMark state={c.state} />
						</Tip>
					</td>) : <td colSpan={columns.length || 1} className="px-2 py-2.5 text-fail">{r.error ?? "not loaded"}</td>}
				</tr>)}
			</tbody>
		</table>
	</div>;
}

function CellMark({ cell }: { cell: OutcomeRow | undefined }) {
	const state = cell?.state ?? "absent";
	const tip = cell && <TipRows title={`${cell.attack} / ${cell.arm}`} rows={[["Status", cell.status ?? "no row"], ["Received", format(cell.received, "int") ?? "not recorded"], ["Attempts", format(cell.attempts, "int") ?? "not recorded"], ["Released tokens", format(cell.releasedTokens, "int") ?? "not recorded"]]} />;
	const body = state === "absent"
		? <span className="flex h-7 w-full items-center justify-center rounded-[3px] border border-dashed border-faint/50 text-[0.65rem] text-faint">none</span>
		: state === "gate-refused" || state === "not-run"
			? <span className="hatched flex h-7 w-full items-center justify-center rounded-[3px] text-[0.65rem] text-muted">refused</span>
			: <span className={cn("num flex h-7 w-full items-center justify-center rounded-[3px] text-[0.7rem]", state === "complete" ? "bg-wash text-ink" : "bg-caution/15 text-caution")}>{state === "complete" ? compact(cell?.received) ?? "?" : state}</span>;
	return tip ? <Tip content={tip} className="w-full">{body}</Tip> : body;
}

function ArmMatrix({ loaded }: { loaded: Loaded[] }) {
	const attacks = attackUniverse(loaded);
	const runs = loaded.filter(l => campaignItems(l.detail).length).map(l => ({ ...l, grid: outcomeGrid(campaignItems(l.detail), attacks) }));
	if (!runs.length) return <p className="text-sm text-muted">No run recorded a campaign arm.</p>;
	return <div className="space-y-3">
		<Legend items={[{ label: "Complete: responses received", color: "var(--wash)" }, { label: "Interrupted", color: "color-mix(in srgb, var(--caution) 15%, transparent)" }, { label: "Refused by the detectability gate", hatched: true }, { label: "No row for this attack", outline: true }]} />
		<div className="overflow-x-auto">
			<table className="border-separate border-spacing-[3px] text-xs">
				<thead>
					<tr>
						<th />
						{runs.map(r => <th key={r.run.id} colSpan={r.grid.arms.length} className="num border-b border-rule px-1 pb-1 text-left font-medium text-ink">{r.run.label}</th>)}
					</tr>
					<tr>
						<th />
						{runs.flatMap(r => r.grid.arms.map(arm => <th key={`${r.run.id}-${arm}`} className="px-1 pt-1 text-left font-normal text-muted">
							<span className="flex items-center gap-1 whitespace-nowrap"><ArmSwatch arm={arm} className="size-2" />{arm === "instantaneous" ? "instant." : arm}</span>
						</th>))}
					</tr>
				</thead>
				<tbody>{attacks.map(attack => <tr key={attack}>
					<th className="num pr-3 text-left font-semibold">{attack}</th>
					{runs.flatMap(r => r.grid.arms.map(arm => <td key={`${r.run.id}-${arm}`} className="w-16 min-w-16 p-0">
						<CellMark cell={r.grid.cells.find(c => c.attack === attack && c.arm === arm)} />
					</td>))}
				</tr>)}</tbody>
			</table>
		</div>
	</div>;
}

const AREA_ORDER = ["Acquisition", "Utility", "Teacher", "Students", "Gates", "Provenance"];

export function GapList({ items, compact: tight = false }: { items: Gap[]; compact?: boolean }) {
	if (!items.length) return <p className="text-sm text-muted">Nothing this dashboard can read is missing for this run.</p>;
	const areas = AREA_ORDER.filter(area => items.some(g => g.area === area));
	return <div className={cn("grid gap-x-10 gap-y-5", !tight && "lg:grid-cols-2")}>
		{areas.map(area => <div key={area} className="space-y-1.5">
			<div className="text-[0.68rem] font-semibold tracking-wide text-faint uppercase">{area}</div>
			<ul className="space-y-1.5">
				{items.filter(g => g.area === area).map(g => <li key={g.text} className="flex gap-2 text-sm">
					<span className="mt-[3px] shrink-0"><CoverageMark state={g.severity === "missing" ? "missing" : "partial"} compact /></span>
					<span className="text-ink/90">{g.text}</span>
				</li>)}
			</ul>
		</div>)}
	</div>;
}

export function Overview({ list, loaded, onSelect }: { list: RunList; loaded: Loaded[]; onSelect: (id: string) => void }) {
	const attacks = attackUniverse(loaded);
	return <div className="space-y-12">
		<header className="space-y-2">
			<h2 className="display text-3xl">All runs</h2>
			<p className="max-w-3xl text-sm text-muted">What each run recorded and what it is missing. This view does not pool runs or compare them statistically; open a run for its charts.</p>
		</header>
		<RootWarnings list={list} />
		<section className="space-y-4">
			<div>
				<h3 className="display text-xl">Evidence coverage</h3>
				<p className="max-w-3xl text-sm text-muted">One row per run and one column per kind of evidence the dashboard can show. Partial means the artifact exists but some arms are refused, interrupted or lack a file. Hover a mark for the reason.</p>
			</div>
			<CoverageMatrix loaded={loaded} onSelect={onSelect} />
		</section>
		<section className="space-y-4">
			<div>
				<h3 className="display text-xl">Campaign arms by attack</h3>
				<p className="max-w-3xl text-sm text-muted">Which attack and arm combinations each run recorded, with responses received. A refused cell ran no requests; an empty cell has no row at all.</p>
			</div>
			<ArmMatrix loaded={loaded} />
		</section>
		<section className="space-y-6">
			<h3 className="display text-xl">Missing data, run by run</h3>
			{loaded.map(l => <div key={l.run.id} className="space-y-3 border-t border-rule pt-5">
				<div className="flex flex-wrap items-baseline justify-between gap-2">
					<button type="button" onClick={() => onSelect(l.run.id)} className="num text-base text-ink underline-offset-4 hover:underline">{l.run.label}</button>
					<span className="text-xs text-muted">{studyLabel(l.run.studies) ?? "study not recorded"}</span>
				</div>
				{l.detail ? <GapList items={gaps(l.detail, attacks)} /> : <p className="text-sm text-fail">{l.error ?? "not loaded"}</p>}
			</div>)}
		</section>
	</div>;
}

function CoverageStrip({ cells }: { cells: CoverageCell[] }) {
	return <div className="flex flex-wrap gap-x-5 gap-y-2">
		{cells.map(c => <Tip key={c.key} content={<TipRows title={c.label} rows={[["State", c.state], ["Detail", <span className="font-sans whitespace-normal">{c.detail}</span>]]} />}>
			<span className="inline-flex items-center gap-1.5 text-xs"><CoverageMark state={c.state} compact /><span className={cn(c.state === "missing" ? "text-faint" : "text-ink")}>{c.label}</span></span>
		</Tip>)}
	</div>;
}

/** The top of a run: headline figures, what is recorded, and what is not. */
export function RunSummaryView({ detail, attacks, list }: { detail: RunDetail; attacks: string[]; list: RunList }) {
	const rows = campaignItems(detail);
	const totals = campaignTotals(rows, rows.length ? attacks : []);
	const cells = coverage(detail).filter(c => c.state !== "na");
	const present = cells.filter(c => c.state === "present" || c.state === "partial").length;
	const items = gaps(detail, attacks);
	const utility = "items" in detail.utility ? detail.utility.items : [];
	const completion = utility.map(a => a.pooled?.completion_rate).filter((v): v is number => typeof v === "number");
	const of = (n: number, d: number) => <>{n}<span className="text-lg text-faint"> / {d}</span></>;
	return <div className="space-y-8">
		<div className="grid grid-cols-2 gap-x-8 gap-y-6 lg:grid-cols-5">
			{totals.arms > 0 && <div className="space-y-2">
				<Stat label="Arms run" value={of(totals.ran, totals.arms)} sub={`${totals.complete} complete${totals.refused ? `, ${totals.refused} gate-refused` : ""}${totals.interrupted ? `, ${totals.interrupted} interrupted` : ""}`} tone={totals.refused || totals.interrupted ? "caution" : undefined} />
				<Meter total={totals.arms} label={`${totals.complete} complete, ${totals.interrupted} interrupted, ${totals.refused} refused of ${totals.arms}`} parts={[{ value: totals.complete, color: "var(--ink)", label: "complete" }, { value: totals.interrupted, color: "var(--caution)", label: "interrupted" }]} />
			</div>}
			{totals.arms > 0 && <div className="space-y-2">
				<Stat label="Attacks with a run arm" value={of(totals.attacksRun, totals.attacks)} sub="of attacks any run recorded" tone={totals.attacksRun < totals.attacks ? "caution" : undefined} />
				<Meter total={totals.attacks} label={`${totals.attacksRun} of ${totals.attacks} attacks`} parts={[{ value: totals.attacksRun, color: "var(--ink)", label: "run" }]} />
			</div>}
			{totals.ran > 0 && <div className="space-y-2">
				<Stat label="Attribution complete" value={of(totals.attributed, totals.ran)} sub="arms whose intervention times are known" tone={totals.attributed < totals.ran ? "caution" : undefined} />
				<Meter total={totals.ran} label={`${totals.attributed} of ${totals.ran} arms`} parts={[{ value: totals.attributed, color: "var(--ink)", label: "complete" }]} />
			</div>}
			{utility.length > 0 && <Stat label="Benign completion" value={completion.length ? `${format(Math.min(...completion), "fraction")}${completion.length > 1 ? ` to ${format(Math.max(...completion), "fraction")}` : ""}` : <Value value={null} unknown="not recorded" />} sub={completion.length ? `lowest and highest of ${completion.length} arms` : undefined} />}
			<div className="space-y-2">
				<Stat label="Evidence recorded" value={of(present, cells.length)} sub={`${items.length} gap${items.length === 1 ? "" : "s"} listed below`} tone={present < cells.length / 2 ? "caution" : undefined} />
				<Meter total={cells.length} label={`${present} of ${cells.length} evidence kinds recorded`} parts={[{ value: cells.filter(c => c.state === "present").length, color: "var(--pass)", label: "recorded" }, { value: cells.filter(c => c.state === "partial").length, color: "var(--caution)", label: "partial" }]} />
			</div>
		</div>
		<CoverageStrip cells={cells} />
		<RootWarnings list={list} />
		<div className="space-y-3">
			<h4 className="text-sm font-semibold">What this run is missing</h4>
			<GapList items={items} />
		</div>
	</div>;
}
