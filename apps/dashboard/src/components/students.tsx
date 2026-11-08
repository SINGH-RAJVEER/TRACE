import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, XAxis, YAxis } from "recharts";
import { curvePlot, isNumber } from "../data";
import type { Curve, IndexRow, RunDetail, Transfer, TransferSeed } from "../types";
import { Facts, Source, Unavailable, Value } from "./evidence";
import { ChartContainer, ChartLegend, ChartTooltip, type ChartConfig } from "./ui/chart";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

const ARMS: ChartConfig = {
	teacher: { label: "Teacher-data student", color: "var(--accent)" },
	gold: { label: "Gold-answer student", color: "var(--gold)" },
};

type TipEntry = { dataKey?: unknown; value?: unknown; payload?: Record<string, unknown> };

function CurveTip({ active, payload, label }: { active?: boolean; payload?: TipEntry[]; label?: unknown }) {
	if (!active || !payload?.length) return null;
	const point = payload[0].payload ?? {};
	return <div className="chart-tooltip space-y-1">
		<div className="font-medium">n = {String(label)}</div>
		{(["teacher", "gold"] as const).filter(arm => isNumber(point[`${arm}_mean`])).map(arm => {
			const band = point[`${arm}_band`] as [number, number];
			return <div key={arm} className="flex items-center gap-2">
				<span className="inline-block h-0.5 w-3" style={{ background: ARMS[arm].color }} />
				<span>{ARMS[arm].label}</span>
				<span className="num ml-auto">{(point[`${arm}_mean`] as number).toFixed(1)} pp [{band[0].toFixed(1)}, {band[1].toFixed(1)}]</span>
			</div>;
		})}
	</div>;
}

export function CurveChart({ curve }: { curve: Curve }) {
	const plot = curvePlot(curve);
	if (plot.state === "unavailable") return <Unavailable reason={plot.reason} />;
	const sizes = [...new Set(plot.series.flatMap(s => s.points.map(p => p.n)))].sort((a, b) => a - b);
	const data = sizes.map(n => {
		const row: Record<string, unknown> = { n };
		for (const series of plot.series) {
			const point = series.points.find(p => p.n === n);
			if (point) {
				row[`${series.arm}_mean`] = point.mean;
				row[`${series.arm}_band`] = point.band;
			}
		}
		return row;
	});
	const config = Object.fromEntries(plot.series.map(s => [s.arm, ARMS[s.arm]]));
	return <figure className="space-y-2">
		<ChartLegend config={config} />
		<ChartContainer config={config}>
			<ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
				<CartesianGrid vertical={false} strokeDasharray="2 4" />
				<XAxis dataKey="n" type="number" scale="log" domain={["dataMin", "dataMax"]} ticks={sizes} tickLine={false} label={{ value: "training examples (log scale)", position: "insideBottom", offset: -12 }} />
				<YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => `${v} pp`} />
				<ReferenceLine y={0} strokeDasharray="3 3" />
				{isNumber(plot.target) && <ReferenceLine y={plot.target} strokeDasharray="6 3" label={{ value: "declared target", position: "insideTopRight" }} />}
				<ChartTooltip content={<CurveTip />} />
				{plot.series.map(s => <Area key={`${s.arm}-band`} dataKey={`${s.arm}_band`} stroke="none" fill={ARMS[s.arm].color} fillOpacity={0.14} isAnimationActive={false} connectNulls={false} />)}
				{plot.series.map(s => <Line key={s.arm} dataKey={`${s.arm}_mean`} stroke={ARMS[s.arm].color} strokeWidth={2} dot={{ r: 4, strokeWidth: 2, fill: "var(--paper)" }} isAnimationActive={false} connectNulls={false} />)}
			</ComposedChart>
		</ChartContainer>
	</figure>;
}

const ROLES = ["teacher", "base", "clean", "defended"] as const;

function TransferSeedView({ label, report }: { label: string; report: TransferSeed }) {
	const raw = report.raw;
	const decision = report.decision;
	return <div className="space-y-3">
		<h4 className="text-sm font-medium">{label === "single" ? "Single training seed" : `Training seed ${label}`}</h4>
		<Table>
			<TableHeader><TableRow>{ROLES.map(role => <TableHead key={role} className="text-right">{role}</TableHead>)}</TableRow></TableHeader>
			<TableBody><TableRow>{ROLES.map(role => <TableCell key={role} className="text-right">
				<Value value={raw?.overall?.[role]} as="fraction" />
				{raw?.overall_interval?.[role] && <div className="num text-xs text-muted">[{(raw.overall_interval[role][0] * 100).toFixed(1)}, {(raw.overall_interval[role][1] * 100).toFixed(1)}]</div>}
			</TableCell>)}</TableRow></TableBody>
			<TableCaption>Accuracy on {report.items ?? "?"} paired items in {report.families ?? "?"} task families, with 95% family-resampled intervals.</TableCaption>
		</Table>
		<Facts items={[
			["Teacher - base headroom", report.headroom ? <span><Value value={report.headroom.estimate} as="fraction" /> {!report.headroom.usable && <span className="text-xs text-caution">{report.headroom.reason}</span>}</span> : <Value value={null} />],
			["Clean gain", report.clean_gain ? <span><Value value={report.clean_gain.estimate} as="fraction" /> {!report.clean_gain.usable && <span className="text-xs text-caution">{report.clean_gain.reason}</span>}</span> : <Value value={null} />],
			["Transfer, clean", <Value value={report.normalized_transfer?.clean.value} unknown={report.normalized_transfer?.clean.reason ?? "undefined"} />],
			["Transfer, defended", <Value value={report.normalized_transfer?.defended.value} unknown={report.normalized_transfer?.defended.reason ?? "undefined"} />],
			["Suppression", <Value value={report.suppression?.value} unknown={report.suppression?.reason ?? "withheld"} />],
			["Clean - defended", <Value value={decision?.clean_minus_defended?.value} as="fraction" />],
			["Meets proposed target", <Value value={decision?.meets_target} unknown="withheld" />],
			["At interval lower bound", <Value value={decision?.meets_target_at_interval_lower} unknown="withheld" />],
		]} />
		{report.suppression_gate && !report.suppression_gate.covered && <p className="text-xs text-caution">Suppression withheld: {report.suppression_gate.reason}</p>}
	</div>;
}

function TransferBlock({ item }: { item: Transfer }) {
	return <div className="space-y-4">
		<Source place={item.place} file={item.file} />
		{item.evidence && <p className="text-sm text-muted">{item.evidence}</p>}
		{Object.keys(item.seeds).length === 0 ? <Unavailable reason="The file did not contain a transfer report." /> : Object.entries(item.seeds).map(([label, report]) => <TransferSeedView key={label} label={label} report={report} />)}
	</div>;
}

function StudentRows({ rows }: { rows: IndexRow[] }) {
	return <Table>
		<TableHeader><TableRow><TableHead>Curve</TableHead><TableHead>Arm</TableHead><TableHead>Task</TableHead><TableHead className="text-right">n</TableHead><TableHead className="text-right">Seeds</TableHead><TableHead className="text-right">Gain</TableHead><TableHead className="text-right">95% interval</TableHead><TableHead>Target reached</TableHead><TableHead>Training runs</TableHead></TableRow></TableHeader>
		<TableBody>{rows.map(r => <TableRow key={`${r.run_id}/${r.arm}/${r.stratum}`}>
			<TableCell className="num text-xs">{String(r.run_id)}</TableCell><TableCell>{r.arm}</TableCell><TableCell><Value value={r.student_task} /></TableCell>
			<TableCell className="text-right"><Value value={r.student_n} as="int" /></TableCell><TableCell className="text-right"><Value value={r.student_seeds} /></TableCell>
			<TableCell className="text-right"><Value value={r.student_gain_pp} as="pp" /></TableCell>
			<TableCell className="num text-right">{isNumber(r.student_gain_lower) && isNumber(r.student_gain_upper) ? `[${r.student_gain_lower.toFixed(1)}, ${r.student_gain_upper.toFixed(1)}]` : <Value value={null} />}</TableCell>
			<TableCell><Value value={r.student_target_reached} unknown="no target" /></TableCell>
			<TableCell className="text-xs"><Value value={r.training_runs} unknown={r.training_ledger_sha256 ? "ledger not in index" : "not recorded"} /></TableCell>
		</TableRow>)}</TableBody>
	</Table>;
}

/** Transfer reports and index rows, shown only where a run recorded them. */
export function StudentDetails({ data }: { data: RunDetail["students"] }) {
	if ("state" in data) return null;
	if (!data.transfers.items.length && !data.index.items.length) return null;
	return <div className="space-y-8">
		{data.transfers.items.map(t => <TransferBlock key={t.place.path} item={t} />)}
		{data.index.items.length > 0 && <StudentRows rows={data.index.items} />}
	</div>;
}
