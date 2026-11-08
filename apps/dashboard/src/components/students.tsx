import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, XAxis, YAxis } from "recharts";
import { checkpointPlot, curvePlot, format, isNumber, shortLabel } from "../data";
import { HBar, Legend, Panel, TipRows } from "./charts";
import type { CheckpointRow, Checkpoints, Curve, IndexRow, RunDetail, Transfer, TransferSeed } from "../types";
import { Facts, Section, Source, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
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

function CurveChart({ curve }: { curve: Curve }) {
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
		<figcaption className="text-xs text-muted">Mean gain over the paired untrained base, in percentage points, with the 95% interval from resampling training seeds and test items. Seeds measure optimization variability on fixed data, not independent campaigns.</figcaption>
	</figure>;
}

function CurveBlock({ curve }: { curve: Curve }) {
	const rows = [...(curve.summary?.results ?? []), ...(curve.summary?.gold_results ?? [])];
	return <div className="space-y-4">
		<div className="flex flex-wrap items-center gap-2">
			<Badge variant="info">{curve.task}</Badge>
			<Badge>{curve.version ?? "version not recorded"}</Badge>
			<Badge variant={curve.integrity === "intact" ? "outline" : "fail"}>{curve.integrity ?? "plan not read"}</Badge>
			{curve.include_pending && <Badge variant="caution">includes unreviewed pending responses</Badge>}
			{curve.seeds && curve.seeds.length < 2 && <Badge variant="caution">single seed: development evidence</Badge>}
		</div>
		<Source place={curve.place} file={curve.summary_file} />
		<Facts items={[
			["Student base", <Value value={curve.recipe.base_model ? `${curve.recipe.base_model} @ ${curve.recipe.revision ?? "?"}` : null} />],
			["Tokenizer", <Value value={curve.recipe.tokenizer_model ? `${curve.recipe.tokenizer_model} @ ${curve.recipe.tokenizer_revision ?? "?"}` : null} />],
			["Sizes", <Value value={curve.sizes?.join(", ")} />],
			["Training seeds", <Value value={curve.seeds?.join(", ")} />],
			["Reference gain", <Value value={curve.summary?.reference_gain_pp} as="pp" />],
			["Declared target", <Value value={curve.summary?.target_gain_pp} as="pp" unknown="undefined" />],
			["Smallest n at target", <Value value={curve.summary?.smallest_n} as="int" unknown="not reached" />],
			["Bootstrap", <Value value={curve.summary?.bootstrap ? `${curve.summary.bootstrap.replicates} replicates, seed ${curve.summary.bootstrap.seed}` : null} />],
		]} />
		<CurveChart curve={curve} />
		{rows.length > 0 && <Table>
			<TableHeader><TableRow><TableHead>Arm</TableHead><TableHead className="text-right">n</TableHead><TableHead className="text-right">Mean gain</TableHead><TableHead className="text-right">95% interval</TableHead><TableHead>Per-seed gains</TableHead></TableRow></TableHeader>
			<TableBody>{rows.map(r => <TableRow key={`${r.arm ?? "teacher"}-${r.n}`}>
				<TableCell>{r.arm === "gold" ? "gold" : "teacher"}</TableCell>
				<TableCell className="text-right"><Value value={r.n} as="int" /></TableCell>
				<TableCell className="text-right"><Value value={r.mean_gain_pp} as="pp" /></TableCell>
				<TableCell className="num text-right">[{r.ci95_pp[0].toFixed(1)}, {r.ci95_pp[1].toFixed(1)}]</TableCell>
				<TableCell className="num">{r.seed_gains_pp.map(g => g.toFixed(1)).join(", ")}</TableCell>
			</TableRow>)}</TableBody>
		</Table>}
		{curve.summary?.warning && <p className="text-xs text-muted">{curve.summary.warning}</p>}
	</div>;
}

function CheckpointBlock({ item }: { item: Checkpoints }) {
	const summary = item.summary;
	if (!summary) return <div className="space-y-2"><Source place={item.place} file={item.plan_file} /><Unavailable reason="The checkpoint plan has no summary.json yet; run the checkpoint summary." /></div>;
	const plot = checkpointPlot(summary.checkpoints);
	const config: ChartConfig = { mean: { label: "Mean accuracy across seeds", color: "var(--accent)" } };
	return <div className="space-y-4">
		<Source place={item.place} file={item.summary_file} />
		<Facts items={[
			["Task", <Value value={item.task} />],
			["Student base", <Value value={item.recipe.base_model ? `${item.recipe.base_model} @ ${item.recipe.revision ?? "?"}` : null} />],
			["Seeds", <Value value={summary.seeds?.join(", ")} />],
			["Base accuracy", <Value value={summary.base_accuracy} as="fraction" />],
			["Target accuracy", <Value value={summary.target_accuracy} as="fraction" />],
			["Base already meets target", <Value value={summary.base_meets_target} />],
			["Target status", <Value value={summary.target_status} />],
			["First at target", <Value value={summary.first_reaching_target?.id} unknown="none among reached checkpoints" />],
		]} />
		{plot.state === "available" ? <figure className="space-y-2">
			<ChartContainer config={config}>
				<ComposedChart data={plot.points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
					<CartesianGrid vertical={false} strokeDasharray="2 4" />
					<XAxis dataKey="id" type="category" tickLine={false} />
					<YAxis tickLine={false} axisLine={false} width={48} domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
					{isNumber(summary.target_accuracy) && <ReferenceLine y={summary.target_accuracy} strokeDasharray="6 3" label={{ value: "target", position: "insideTopRight" }} />}
					{isNumber(summary.base_accuracy) && <ReferenceLine y={summary.base_accuracy} strokeDasharray="2 3" label={{ value: "base", position: "insideBottomRight" }} />}
					<ChartTooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ background: "var(--paper)", border: "1px solid var(--rule)", fontSize: 12 }} />
					<Line dataKey="mean" name="Mean accuracy" stroke="var(--accent)" strokeWidth={2} dot={{ r: 4, strokeWidth: 2, fill: "var(--paper)" }} isAnimationActive={false} />
				</ComposedChart>
			</ChartContainer>
			<figcaption className="text-xs text-muted">Reached checkpoints in plan order. {plot.censored.length ? `Censored and not drawn: ${plot.censored.join(", ")}.` : ""}</figcaption>
		</figure> : <p className="text-xs text-muted">No trend line: a trend needs at least two reached checkpoints, and each reached checkpoint is drawn in the accuracy chart above.{plot.censored.length ? ` Censored: ${plot.censored.join(", ")}.` : ""}</p>}
		<Table>
			<TableHeader><TableRow><TableHead>Checkpoint</TableHead><TableHead>Kind</TableHead><TableHead>Status</TableHead><TableHead className="text-right">Cutoff</TableHead><TableHead className="text-right">Acquired</TableHead><TableHead className="text-right">Trainable</TableHead><TableHead className="text-right">Mean accuracy</TableHead><TableHead className="text-right">Gain</TableHead><TableHead className="text-right">Seeds at target</TableHead></TableRow></TableHeader>
			<TableBody>{(summary.checkpoints ?? []).map(r => <TableRow key={r.id}>
				<TableCell className="num">{r.id}</TableCell><TableCell>{r.kind}</TableCell><TableCell><Badge variant={r.status === "censored" ? "caution" : "outline"}>{r.status}</Badge></TableCell>
				<TableCell className="text-right"><Value value={r.cutoff_seconds} as="seconds" /></TableCell>
				<TableCell className="text-right"><Value value={r.acquired_count} as="int" /></TableCell>
				<TableCell className="text-right"><Value value={r.trainable_count} as="int" /></TableCell>
				<TableCell className="text-right"><Value value={r.mean_accuracy} as="fraction" unknown="censored" /></TableCell>
				<TableCell className="text-right"><Value value={r.mean_gain_pp} as="pp" /></TableCell>
				<TableCell className="text-right"><Value value={r.seeds_reaching_target} /></TableCell>
			</TableRow>)}</TableBody>
		</Table>
		{summary.claim_boundary && <p className="text-xs text-muted">{summary.claim_boundary}</p>}
	</div>;
}

/**
 * Accuracy of every checkpoint of every plan on one 0 to 100% scale, with the base and the target
 * marked. One point is drawn as a bar, not a trend; censored checkpoints and plans without a
 * summary keep a row with the reason.
 */
function CheckpointAccuracy({ items }: { items: Checkpoints[] }) {
	const rows = items.flatMap(item => {
		const plan = shortLabel(item.place.path);
		if (!item.summary) return [{ key: plan, plan, id: null as string | null, value: null as number | null, reason: "no summary.json yet", base: null as number | null, target: item.target_accuracy, row: null as CheckpointRow | null }];
		return (item.summary.checkpoints ?? []).map(row => ({ key: `${plan}/${row.id}`, plan, id: row.id, value: row.status === "censored" ? null : row.mean_accuracy ?? null, reason: row.status === "censored" ? "censored, not reached" : "accuracy not recorded", base: item.summary?.base_accuracy ?? null, target: item.summary?.target_accuracy ?? item.target_accuracy, row }));
	});
	if (!rows.length) return null;
	const bases = [...new Set(rows.map(r => r.base).filter(isNumber))];
	const targets = [...new Set(rows.map(r => r.target).filter(isNumber))];
	const at = (v: number) => `${v * 100}%`;
	return <Panel title="Student accuracy by checkpoint" note="Mean accuracy across training seeds on the frozen suite, on a 0 to 100% scale. Plans are listed side by side, not pooled.">
		<Legend items={[{ label: "Mean accuracy", color: "var(--accent)" }, ...(bases.length ? [{ label: "Untrained base", color: "var(--faint)" }] : []), ...(targets.length ? [{ label: "Declared target", color: "var(--gold)" }] : []), { label: "Not recorded or censored", hatched: true }]} />
		<div className="space-y-2">
			{rows.map(r => <div key={r.key} className="grid grid-cols-[14rem_1fr] items-center gap-3 text-xs">
				<span className="num truncate text-muted" title={r.key}>{r.plan}{r.id ? <span className="text-faint"> / {r.id}</span> : null}</span>
				<span className="relative flex h-5 items-center">
					{isNumber(r.base) && <span aria-hidden className="absolute top-0 h-full border-l-2 border-faint" style={{ left: at(r.base) }} />}
					{isNumber(r.target) && <span aria-hidden className="absolute top-0 h-full border-l-2 border-dashed border-gold" style={{ left: at(r.target) }} />}
					<HBar value={r.value} max={1} color="var(--accent)" label={format(r.value, "fraction")} missing={r.reason}
						tip={<TipRows title={r.key} rows={[["Mean accuracy", format(r.value, "fraction") ?? r.reason], ["Seed range", r.row && isNumber(r.row.min_accuracy) && isNumber(r.row.max_accuracy) ? `${format(r.row.min_accuracy, "fraction")} to ${format(r.row.max_accuracy, "fraction")}` : "not recorded"], ["Gain over base", format(r.row?.mean_gain_pp, "pp") ?? "not recorded"], ["Base", format(r.base, "fraction") ?? "not recorded"], ["Target", format(r.target, "fraction") ?? "not recorded"], ["Trainable examples", format(r.row?.trainable_count, "int") ?? "not recorded"]]} />} />
				</span>
			</div>)}
		</div>
	</Panel>;
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

export function StudentsView({ data }: { data: RunDetail["students"] }) {
	if ("state" in data) return <Unavailable reason={data.reason} tone="error" />;
	return <div className="space-y-8">
		<Section title="Learning curves" kicker="Students fine-tuned from the pinned base on teacher responses (and on gold answers where planned), scored on a frozen suite. Subsets are hash-ordered, so these curves say nothing about intervention timing.">
			{data.curves.state === "available" ? <div className="space-y-10">{data.curves.items.map(c => <CurveBlock key={c.place.path} curve={c} />)}</div> : <Unavailable reason={data.curves.reason} />}
		</Section>
		<Section title="Chronological checkpoints" kicker="Students trained on what a campaign had actually received by each cutoff. Unreached checkpoints are censored, not zero.">
			{data.checkpoints.state === "available" ? <div className="space-y-10"><CheckpointAccuracy items={data.checkpoints.items} />{data.checkpoints.items.map(c => <CheckpointBlock key={c.place.path} item={c} />)}</div> : <Unavailable reason={data.checkpoints.reason} />}
		</Section>
		<Section title="Base, clean and defended students" kicker="Paired correctness of the teacher, base, clean-data student and defended-data student. Suppression is reported only when a passing clean-student gate covers the same task, base and recipe.">
			{data.transfers.state === "available" ? <div className="space-y-10">{data.transfers.items.map(t => <TransferBlock key={t.place.path} item={t} />)}</div> : <Unavailable reason={data.transfers.reason} />}
		</Section>
		<Section title="Student rows in the index">
			{data.index.state === "available" ? <StudentRows rows={data.index.items} /> : <Unavailable reason={data.index.reason} />}
		</Section>
	</div>;
}

