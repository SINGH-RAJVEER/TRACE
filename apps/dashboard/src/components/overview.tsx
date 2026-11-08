import { TriangleAlert } from "lucide-react";
import { COVERAGE_GROUPS, compact, coverage, format, gaps, outcomeGrid, sortAttacks, type CoverageCell, type CoverageState, type OutcomeRow } from "../data";
import type { RunList } from "../types";
import { cn } from "../lib";
import { ArmSwatch, CoverageMark, Legend, Tip, TipRows } from "./charts";
import type { Loaded } from "./run";

function campaignItems(l: Loaded) {
	return l.detail && "items" in l.detail.campaigns ? l.detail.campaigns.items : [];
}

/** Every attack named by any run, so a run that skipped an attack shows the gap. */
export function attackUniverse(loaded: Loaded[]): string[] {
	return sortAttacks(loaded.flatMap(l => campaignItems(l).map(row => row.attack ?? "unknown")));
}

/** The combined state of several coverage cells: present only when all are, missing only when none are. */
export function combined(all: CoverageCell[]): CoverageState {
	const cells = all.filter(c => c.state !== "na");
	if (!cells.length) return all.length ? "na" : "missing";
	if (cells.some(c => c.state === "error")) return "error";
	if (cells.every(c => c.state === "present")) return "present";
	if (cells.every(c => c.state === "missing")) return "missing";
	return "partial";
}

export function RootWarnings({ list }: { list: RunList }) {
	const missing = list.roots.filter(root => !root.exists);
	if (!missing.length && list.index.state === "ok") return null;
	return <div className="flex items-start gap-2 text-sm text-caution">
		<TriangleAlert size={16} className="mt-0.5 shrink-0" aria-hidden />
		<div>
			{missing.map(root => <p key={root.name}>Root <span className="num">{root.name}</span> does not exist; its artifacts show as missing.</p>)}
			{list.index.state !== "ok" && <p>Index <span className="num">{list.index.path}</span> is {list.index.state}.</p>}
		</div>
	</div>;
}

function CoverageTable({ loaded, attacks, onSelect }: { loaded: Loaded[]; attacks: string[]; onSelect: (id: string) => void }) {
	const attached = loaded.filter(l => l.run.kind === "study" || l.run.run_id);
	const loose = loaded.filter(l => !(l.run.kind === "study" || l.run.run_id));
	const row = (l: Loaded) => {
		const cells = l.detail ? coverage(l.detail) : null;
		const count = l.detail ? gaps(l.detail, attacks).length : null;
		return <tr key={l.run.id} className="border-t border-rule/70">
			<th className="py-2 pr-6 text-left font-normal">
				<button type="button" onClick={() => onSelect(l.run.id)} title={l.run.label} className="num text-left text-ink underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-accent">{l.run.kind === "study" ? l.run.label : l.run.label.split("/").slice(-2).join("/")}</button>
			</th>
			{cells ? COVERAGE_GROUPS.map(group => {
				const groupCells = cells.filter(c => c.group === group);
				return <td key={group} className="px-3 py-2">
					<Tip content={<TipRows title={group} rows={groupCells.map(c => [c.label, c.state === "present" ? c.detail : c.state])} />}><CoverageMark state={combined(groupCells)} /></Tip>
				</td>;
			}) : <td colSpan={COVERAGE_GROUPS.length} className="px-3 py-2 text-fail">{l.error ?? "loading"}</td>}
			<td className="px-3 py-2 text-right">
				{count !== null && <button type="button" onClick={() => onSelect(l.run.id)} title="Open the run to see what is missing" className={cn("num rounded px-1.5 py-0.5", count ? "bg-caution/12 text-caution" : "text-faint")}>{count}</button>}
			</td>
		</tr>;
	};
	return <div className="overflow-x-auto">
		<table className="w-full border-collapse text-sm">
			<thead><tr className="text-left text-xs text-muted">
				<th className="pb-2 font-medium">Run</th>
				{COVERAGE_GROUPS.map(g => <th key={g} className="px-3 pb-2 font-medium">{g}</th>)}
				<th className="px-3 pb-2 text-right font-medium">Gaps</th>
			</tr></thead>
			<tbody>
				{attached.map(row)}
				{loose.length > 0 && <tr><td colSpan={COVERAGE_GROUPS.length + 2} className="pt-5 pb-1 text-xs text-muted">Student plans not attached to any run</td></tr>}
				{loose.map(row)}
			</tbody>
		</table>
	</div>;
}

function CellMark({ cell }: { cell: OutcomeRow | undefined }) {
	const state = cell?.state ?? "absent";
	const tip = cell && <TipRows title={`${cell.attack} / ${cell.arm}`} rows={[["Status", cell.status ?? "no row"], ["Answered", format(cell.received, "int") ?? "not recorded"], ["Released tokens", format(cell.releasedTokens, "int") ?? (cell.assumed ? `≈${format(cell.assumed.tokens, "int")} assumed` : "not recorded")], ...(cell.assumed ? [["Basis", cell.assumed.basis] as [string, string]] : [])]} />;
	const body = state === "absent"
		? <span className="block h-6 w-full rounded-[3px] border border-dashed border-faint/50" />
		: state === "gate-refused" || state === "not-run"
			? cell?.assumed
				? <span className="num flex h-6 w-full items-center justify-center rounded-[3px] border border-dashed border-faint/60 text-[0.68rem] text-faint">≈{compact(cell.assumed.tokens)}</span>
				: <span className="hatched block h-6 w-full rounded-[3px]" />
			: <span className={cn("num flex h-6 w-full items-center justify-center rounded-[3px] text-[0.68rem]", state === "complete" ? "bg-wash text-ink" : "bg-caution/15 text-caution")}>{compact(cell?.releasedTokens) ?? "?"}</span>;
	return tip ? <Tip content={tip} className="w-full">{body}</Tip> : body;
}

function ArmMatrix({ loaded, attacks }: { loaded: Loaded[]; attacks: string[] }) {
	const runs = loaded.filter(l => campaignItems(l).length).map(l => ({ ...l, grid: outcomeGrid(campaignItems(l), attacks) }));
	if (!runs.length) return null;
	return <section className="space-y-3">
		<h2 className="text-sm font-semibold">Released tokens by attack and arm</h2>
		<Legend items={[{ label: "ran", color: "var(--wash)" }, { label: "interrupted", color: "color-mix(in srgb, var(--caution) 15%, transparent)" }, { label: "refused by the gate, assumed let through (≈)", assumed: true }, { label: "refused, no basis for an estimate", hatched: true }, { label: "no row", outline: true }]} />
		<div className="overflow-x-auto">
			<table className="border-separate border-spacing-[3px] text-xs">
				<thead>
					<tr>
						<th />
						{runs.map(r => <th key={r.run.id} colSpan={r.grid.arms.length} className="num border-b border-rule px-1 pb-1 text-left font-medium">{r.run.label}</th>)}
					</tr>
					<tr>
						<th />
						{runs.flatMap(r => r.grid.arms.map(arm => <th key={`${r.run.id}-${arm}`} className="px-1 pt-1 text-left font-normal text-muted" title={arm}>
							<span className="flex items-center gap-1 whitespace-nowrap"><ArmSwatch arm={arm} className="size-2" />{arm.slice(0, 5)}</span>
						</th>))}
					</tr>
				</thead>
				<tbody>{attacks.map(attack => <tr key={attack}>
					<th className="num pr-2 text-left font-semibold">{attack}</th>
					{runs.flatMap(r => r.grid.arms.map(arm => <td key={`${r.run.id}-${arm}`} className="w-14 min-w-14 p-0">
						<CellMark cell={r.grid.cells.find(c => c.attack === attack && c.arm === arm)} />
					</td>))}
				</tr>)}</tbody>
			</table>
		</div>
	</section>;
}

export function Overview({ list, loaded, onSelect }: { list: RunList; loaded: Loaded[]; onSelect: (id: string) => void }) {
	const attacks = attackUniverse(loaded);
	return <div className="space-y-12">
		<RootWarnings list={list} />
		<section className="space-y-3">
			<h2 className="text-sm font-semibold">What each run recorded</h2>
			<CoverageTable loaded={loaded} attacks={attacks} onSelect={onSelect} />
		</section>
		<ArmMatrix loaded={loaded} attacks={attacks} />
	</div>;
}
