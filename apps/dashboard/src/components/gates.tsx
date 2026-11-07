import type { GateArmRow, GateReport, IndexRow, RunDetail } from "../types";
import { Facts, Hash, Reasons, Section, Source, Unavailable, Value, Verdict } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

function Gain({ gain }: { gain: GateArmRow["pooled"] | undefined }) {
	if (!gain) return <Value value={null} unknown="missing" />;
	return <span className="num">
		{gain.gain_pp >= 0 ? "+" : ""}{gain.gain_pp.toFixed(1)} [{gain.ci95_pp[0].toFixed(1)}, {gain.ci95_pp[1].toFixed(1)}]
		{!gain.positive_excluding_zero && <span className="ml-1 text-fail" title="Not positive with an interval excluding zero">x</span>}
	</span>;
}

function ArmTable({ title, rows, seeds }: { title: string; rows: GateArmRow[]; seeds: number[] }) {
	if (!rows.length) return <Unavailable reason={`No ${title} scores were read by the gate.`} />;
	return <Table>
		<TableHeader><TableRow><TableHead>{title}</TableHead>{seeds.map(s => <TableHead key={s} className="text-right">seed {s}</TableHead>)}<TableHead className="text-right">pooled</TableHead></TableRow></TableHeader>
		<TableBody>{rows.map(row => <TableRow key={row.n}>
			<TableCell className="num">n = {row.n}</TableCell>
			{seeds.map(s => <TableCell key={s} className="text-right"><Gain gain={row.seeds[String(s)]} /></TableCell>)}
			<TableCell className="text-right"><Gain gain={row.pooled} /></TableCell>
		</TableRow>)}</TableBody>
	</Table>;
}

function Report({ gate }: { gate: GateReport }) {
	const seeds = gate.declared_seeds ?? [];
	return <div className="space-y-3">
		<div className="flex flex-wrap items-center gap-2">
			<Badge variant="info">{gate.gate ?? "unknown gate"}</Badge>
			<Verdict passed={gate.passed} status={gate.status} />
			<Badge variant={gate.integrity === "intact" ? "outline" : "fail"}>{gate.integrity ?? "not checked"}</Badge>
		</div>
		<Source place={gate.place} file={gate.file} />
		<Facts items={[
			["Task", <Value value={gate.task} />],
			["Base", <Value value={gate.base ? `${gate.base.model} @ ${gate.base.revision}` : null} />],
			["Student recipe", <Hash value={gate.student_revision ?? null} />],
			["Declared seeds", <Value value={seeds.join(", ") || null} />],
			["Suite", <Hash value={gate.suite_sha256} />],
			["Inputs", <Hash value={gate.inputs_sha256} />],
		]} />
		<div>
			<h4 className="mb-1 text-sm font-medium">Reasons</h4>
			<Reasons reasons={gate.reasons} />
			{gate.action && <p className="mt-2 text-sm text-muted">{gate.action}</p>}
		</div>
		{gate.gate === "phase1-clean" && <div className="space-y-4">
			<p className="text-xs text-muted">Gain over the untrained base in percentage points with a 95% family-resampled interval; x marks a gain that is not positive excluding zero.</p>
			<ArmTable title="clean (teacher data)" rows={gate.results?.clean ?? []} seeds={seeds} />
			<ArmTable title="gold (diagnostic)" rows={gate.results?.gold ?? []} seeds={seeds} />
			{gate.gold_diagnostic && <p className="text-sm text-muted">Gold arm learned: <Value value={gate.gold_diagnostic.learned} />. {gate.gold_diagnostic.note}</p>}
		</div>}
	</div>;
}

function IndexGates({ rows }: { rows: IndexRow[] }) {
	return <Table>
		<TableHeader><TableRow><TableHead>Gate</TableHead><TableHead>Task</TableHead><TableHead>Status</TableHead><TableHead>Passed</TableHead><TableHead>Reason</TableHead><TableHead className="text-right">Seeds</TableHead><TableHead>Report</TableHead></TableRow></TableHeader>
		<TableBody>{rows.map(r => <TableRow key={String(r.run_id)}>
			<TableCell>{r.attack}</TableCell><TableCell><Value value={r.student_task} /></TableCell><TableCell><Value value={r.status} /></TableCell>
			<TableCell><Verdict passed={r.gate_passed as boolean | null} /></TableCell>
			<TableCell className="max-w-md text-sm"><Value value={r.gate_reason} unknown="none recorded" /></TableCell>
			<TableCell className="text-right"><Value value={r.student_seeds} /></TableCell>
			<TableCell><Hash value={r.gate_report_sha256 as string | null} /></TableCell>
		</TableRow>)}</TableBody>
	</Table>;
}

export function GatesView({ data }: { data: RunDetail["gates"] }) {
	if ("state" in data) return <Unavailable reason={data.reason} tone="error" />;
	return <div className="space-y-8">
		<Section title="Baseline gates" kicker="Headroom and clean-student gates, failing ones included. A gate that did not pass means the learning experiment needs fixing before any defense comparison.">
			{data.reports.state === "available" ? <div className="space-y-10">{data.reports.items.map(g => <Report key={g.place.path} gate={g} />)}</div> : <Unavailable reason={data.reports.reason} />}
		</Section>
		<Section title="Gate rows in the index">
			{data.index.state === "available" ? <IndexGates rows={data.index.items} /> : <Unavailable reason={data.index.reason} />}
		</Section>
		<Section title="Detectability gate refusals" kicker="Detector arms that collect declined to run because the attacker's schedule did not escalate beyond held-out benign accounts. No responses were collected for them.">
			{data.detectability.state === "available" ? <div className="flex flex-wrap gap-2">{data.detectability.items.map(r => <Badge key={`${r.attack}/${r.arm}`} variant="caution">{r.attack} / {r.arm}{r.accumulator ? ` (${r.accumulator})` : ""}</Badge>)}</div> : <Unavailable reason={data.detectability.reason} />}
		</Section>
	</div>;
}
