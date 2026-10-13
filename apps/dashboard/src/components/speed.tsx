import { Bar, BarChart, CartesianGrid, Line, LineChart, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts";
import { ChartTooltip, ChartTooltipContent } from "./ui/chart";
import { Plot, SlotSwatch, Tip, barCursor, barStyle, lineCursor, runLegend, tick, useLog, valueAxis, xTitle, yTitle } from "./plot";
import { format, histogram, seconds, title } from "../data";
import { colors, isBefore, labels, statusLabels, type Report, type Slot } from "../types";

const rate = (value: number) => value >= 100 ? `${Math.round(value)}` : `${+value.toFixed(1)}`;

export function Speed({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	return <div className="plots-grid">
		<Latency report={report} benchmarks={benchmarks} available={available} />
		<Distribution report={report} benchmarks={benchmarks} available={available} />
		<Tradeoff report={report} benchmarks={benchmarks} available={available} />
		<Throughput report={report} benchmarks={benchmarks} available={available} />
		<Details report={report} benchmarks={benchmarks} available={available} />
	</div>;
}

function Latency({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const scale = useLog();
	const rows = benchmarks.map(b => Object.assign({ benchmark: b, name: title(b) }, ...available.map(slot => ({ [slot]: report.scores[slot]?.[b]?.latency_p50 ?? null }))));
	const values = rows.flatMap(row => available.map(slot => row[slot] as number | null));
	return <Plot name="Typical response time" legend={runLegend(available)} scale={scale}
		note="Median seconds to generate one answer. Lower is faster. Hover a bar for the slowest 5% (p95). Use log scale when one model is many times slower than the other.">
		<BarChart key={String(scale.log)} data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2} barCategoryGap="22%">
			<CartesianGrid vertical={false} />
			<XAxis dataKey="name" tick={tick} tickLine={false} />
			<YAxis {...valueAxis(scale.log, values, seconds)} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Seconds per answer", scale.log)} />
			<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={(v, entry) => <>{seconds(v)} <small>p95 {format(report.scores[entry.dataKey as Slot]?.[entry.payload?.benchmark as string]?.latency_p95, "s")}</small></>} />} />
			{available.map(slot => <Bar key={slot} name={labels[slot]} dataKey={slot} {...barStyle(slot)} maxBarSize={28} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
		</BarChart>
	</Plot>;
}

function Distribution({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const scale = useLog();
	const { rows, domain } = histogram(report, benchmarks, available, scale.log);
	const axis = valueAxis(scale.log, domain, seconds);
	return <Plot name="Spread of response times" legend={runLegend(available, true)} scale={scale}
		note="How many answers took each amount of time. A tall, narrow peak means consistent speed; a long tail to the right means occasional slow answers. Excludes API errors.">
		<LineChart key={String(scale.log)} data={rows} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
			<CartesianGrid vertical={false} />
			<XAxis dataKey="x" {...axis} tick={tick} tickLine={false} label={xTitle("Seconds per answer", scale.log)} />
			<YAxis allowDecimals={false} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Answers")} />
			<ChartTooltip cursor={lineCursor} content={props => <Tip {...props} line heading={(_, payload) => {
				const row = payload[0]?.payload as { low: number; high: number } | undefined;
				return row ? `${seconds(row.low)} to ${seconds(row.high)}` : "";
			}} value={v => `${v} answers`} />} />
			{available.map(slot => <Line key={slot} name={labels[slot]} dataKey={slot} type="monotone" stroke={colors[slot]} strokeWidth={2} strokeDasharray={isBefore(slot) ? "5 4" : undefined} dot={false} activeDot={{ r: 4, strokeWidth: 0 }} isAnimationActive={false} />)}
		</LineChart>
	</Plot>;
}

function Tradeoff({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const scale = useLog();
	const series = available.map(slot => ({ slot, points: benchmarks.map(b => ({ benchmark: title(b), latency: report.scores[slot]?.[b]?.latency_p50, score: report.scores[slot]?.[b]?.score })).filter(p => p.latency != null && p.latency > 0 && p.score != null) }));
	const values = series.flatMap(s => s.points.map(p => p.latency));
	return <Plot name="Accuracy vs. speed" legend={runLegend(available)} scale={scale}
		note="One dot per model run and benchmark. Up is more accurate, left is faster, so the best trade-off sits top-left. Hollow dots are before distillation, filled dots after.">
		<ScatterChart key={String(scale.log)} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
			<CartesianGrid />
			<XAxis dataKey="latency" name="Median time" {...valueAxis(scale.log, values, seconds)} tick={tick} tickLine={false} label={xTitle("Median seconds per answer", scale.log)} />
			<YAxis dataKey="score" name="Accuracy" type="number" domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={v => `${v}%`} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Accuracy (%)")} />
			<ZAxis range={[70, 70]} />
			<ChartTooltip cursor={false} content={({ active, payload }) => {
				const point = payload?.[0]?.payload as { benchmark: string; latency: number; score: number; slot: Slot } | undefined;
				if (!active || !point) return null;
				return <ChartTooltipContent active payload={[{ name: "Accuracy", value: point.score, dataKey: "score" }, { name: "Median time", value: point.latency, dataKey: "latency" }]}
					label={point.benchmark} heading={() => <><SlotSwatch slot={point.slot} />{labels[point.slot]} · {point.benchmark}</>}
					value={(v, entry) => entry.dataKey === "score" ? format(v, "%") : seconds(v)} swatch={() => null} />;
			}} />
			{series.map(({ slot, points }) => <Scatter key={slot} name={labels[slot]} data={points.map(p => ({ ...p, slot }))} fill={isBefore(slot) ? "var(--background)" : colors[slot]} stroke={colors[slot]} strokeWidth={2} isAnimationActive={false} />)}
		</ScatterChart>
	</Plot>;
}

function Throughput({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const scale = useLog();
	const rows = benchmarks.map(b => Object.assign({ benchmark: b, name: title(b) }, ...available.map(slot => {
		const value = report.scores[slot]?.[b]?.tokens_per_second;
		return { [slot]: value != null && value > 0 ? value : null };
	})));
	const values = rows.flatMap(row => available.map(slot => row[slot] as number | null));
	if (!values.some(v => v != null)) return <Plot name="Generation speed" note="No run reported output token counts, so tokens per second is unavailable." />;
	return <Plot name="Generation speed" legend={runLegend(available)} scale={scale}
		note="Output tokens per second, averaged over each benchmark. Higher is faster. Includes prompt processing and network time, so it understates raw decoding speed.">
		<BarChart key={String(scale.log)} data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2} barCategoryGap="22%">
			<CartesianGrid vertical={false} />
			<XAxis dataKey="name" tick={tick} tickLine={false} />
			<YAxis {...valueAxis(scale.log, values, rate)} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Tokens per second", scale.log)} />
			<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={v => `${rate(v)} tok/s`} />} />
			{available.map(slot => <Bar key={slot} name={labels[slot]} dataKey={slot} {...barStyle(slot)} maxBarSize={28} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
		</BarChart>
	</Plot>;
}

function Details({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const problems = (statuses: Record<string, number>) => Object.entries(statuses).filter(([key, count]) => key !== "ok" && count > 0).map(([key, count]) => `${count} ${(statusLabels[key] ?? key).toLowerCase()}`).join(", ");
	return <section className="plot wide">
		<div className="plot-header"><h2>Timing and reliability</h2></div>
		<p className="plot-note">Exact numbers behind the charts. Problems lists answers that could not be scored normally; each one counts as wrong. Rerun API errors before comparing model quality.</p>
		<div className="table-scroll"><table>
			<thead><tr><th>Benchmark</th><th>Run</th><th className="num">Median</th><th className="num">p95</th><th className="num">Tokens / s</th><th className="num">Token counts known</th><th>Problems</th></tr></thead>
			<tbody>{benchmarks.flatMap(b => available.map((slot, i) => {
				const s = report.scores[slot]?.[b];
				if (!s) return null;
				return <tr key={`${b}:${slot}`} className={i === 0 ? "group-start" : ""}>
					<th>{i === 0 ? title(b) : ""}</th>
					<td><span className="th-run"><SlotSwatch slot={slot} />{labels[slot]}</span></td>
					<td className="num">{format(s.latency_p50, "s", 2)}</td>
					<td className="num">{format(s.latency_p95, "s", 2)}</td>
					<td className="num">{format(s.tokens_per_second, "", 1)}</td>
					<td className="num">{format(s.token_coverage * 100, "%", 0)}</td>
					<td className={problems(s.statuses) ? "" : "muted"}>{problems(s.statuses) || "None"}</td>
				</tr>;
			}))}</tbody>
		</table></div>
	</section>;
}
