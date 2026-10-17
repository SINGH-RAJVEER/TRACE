import { Bar, BarChart, CartesianGrid, ErrorBar, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartTooltip } from "./ui/chart";
import { Plot, Swatch, SlotSwatch, Tip, barCursor, barStyle, ink, runLegend, tick, xTitle, yTitle } from "./plot";
import { format, heat, heatRamp, scoreRows, signed, title, taskScore, taskInterval, taskDelta, taskDeltaInterval } from "../data";
import { labels, modelColors, type Report, type Slot } from "../types";

const percent = (value: number) => `${Math.round(value * 10) / 10}%`;
const outcomes = [
	{ key: "regressed", label: "Lost (right → wrong)", color: "#e66767" },
	{ key: "both_wrong", label: "Wrong both times", color: "#3d3d3a" },
	{ key: "both_correct", label: "Right both times", color: "#8a8984" },
	{ key: "improved", label: "Gained (wrong → right)", color: "#199e70" },
] as const;

export function Accuracy({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const rows = scoreRows(report, benchmarks);
	return <div className="plots-grid">
		<Plot wide tall name="Task capability" legend={runLegend(available)}
			note="AG News label accuracy, SQuAD token F1, GSM8K numeric exact match, and coding pass@1. Whiskers show descriptive 95% item intervals.">
			<BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2} barCategoryGap="22%">
				<CartesianGrid vertical={false} />
				<XAxis dataKey="name" tick={tick} tickLine={false} />
				<YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={percent} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Task score (%)")} />
				<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={(v, entry) => {
					const summary = report.scores[entry.dataKey as Slot]?.[entry.payload?.benchmark as string];
					return <>{format(v, "%")} <small>{entry.payload?.benchmark === "squad" ? `${summary?.n} questions` : `${summary?.correct}/${summary?.n}`}</small></>;
				}} />} />
				{available.map(slot => <Bar key={slot} name={labels[slot]} dataKey={slot} {...barStyle(slot)} maxBarSize={28} radius={[4, 4, 0, 0]} isAnimationActive={false}>
					<ErrorBar dataKey={`${slot}_error`} width={4} strokeWidth={1.5} stroke={ink.secondary} direction="y" />
				</Bar>)}
			</BarChart>
		</Plot>
		<Change report={report} benchmarks={benchmarks} />
		<Outcomes report={report} benchmarks={benchmarks} />
		<Agreement report={report} benchmarks={benchmarks} />
		<Subjects report={report} benchmarks={benchmarks} available={available} />
	</div>;
}

function Change({ report, benchmarks }: { report: Report; benchmarks: string[] }) {
	const student = report.comparisons.student_gain, teacher = report.comparisons.teacher_drift;
	if (!student && !teacher) return <Plot name="Change after distillation" note="Appears once before and after runs are recorded." />;
	const rows = benchmarks.map(b => {
		const s = taskDelta(student?.[b], b), t = taskDelta(teacher?.[b], b);
		const si = taskDeltaInterval(student?.[b], b), ti = taskDeltaInterval(teacher?.[b], b);
		return { name: title(b), student: s, teacher: t,
			student_error: s == null || !si ? null : [s - si[0], si[1] - s],
			teacher_error: t == null || !ti ? null : [t - ti[0], ti[1] - t] };
	});
	return <Plot name="Student gain and teacher drift" legend={[
		{ key: "student", label: "Student gain", swatch: <Swatch color={modelColors.student} /> },
		{ key: "teacher", label: "Teacher drift", swatch: <Swatch color={modelColors.teacher} /> },
	]} note="After minus before, in percentage points, on the same questions. Whiskers show descriptive 95% paired item intervals, using token F1 for SQuAD. The teacher's second run measures drift without training the teacher.">
		<BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 20, left: 8 }} barGap={2} barCategoryGap="24%">
			<CartesianGrid horizontal={false} />
			<XAxis type="number" tick={tick} tickLine={false} tickFormatter={v => signed(v, "", 0)} label={xTitle("Change (percentage points)")} />
			<YAxis type="category" dataKey="name" tick={tick} tickLine={false} axisLine={false} width={96} />
			<ReferenceLine x={0} stroke={ink.baseline} />
			<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={v => signed(v, " pp")} />} />
			<Bar name="Student gain" dataKey="student" fill={modelColors.student} maxBarSize={16} radius={2} isAnimationActive={false}><ErrorBar dataKey="student_error" width={4} strokeWidth={1.5} stroke={ink.secondary} direction="x" /></Bar>
			<Bar name="Teacher drift" dataKey="teacher" fill={modelColors.teacher} maxBarSize={16} radius={2} isAnimationActive={false}><ErrorBar dataKey="teacher_error" width={4} strokeWidth={1.5} stroke={ink.secondary} direction="x" /></Bar>
		</BarChart>
	</Plot>;
}

function Outcomes({ report, benchmarks }: { report: Report; benchmarks: string[] }) {
	const comparison = report.comparisons.student_gain;
	if (!comparison) return <Plot name="Student answers, question by question" note="Appears once both student runs are recorded." />;
	const rows = benchmarks.filter(b => comparison[b]).map(b => ({ name: title(b), ...comparison[b] }));
	return <Plot name="Student answers, question by question" legend={outcomes.map(o => ({ key: o.key, label: o.label, swatch: <Swatch color={o.color} /> }))}
		note="Each bar counts exact-match outcomes for every question, including secondary SQuAD exact match. It shows how many answers the student flipped after distillation, not just the net score.">
		<BarChart data={rows} layout="vertical" stackOffset="expand" margin={{ top: 4, right: 16, bottom: 20, left: 8 }} barCategoryGap="28%">
			<XAxis type="number" tick={tick} tickLine={false} tickFormatter={v => `${Math.round(v * 100)}%`} ticks={[0, .25, .5, .75, 1]} label={xTitle("Share of questions")} />
			<YAxis type="category" dataKey="name" tick={tick} tickLine={false} axisLine={false} width={96} />
			<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={(v, entry) => `${v} of ${entry.payload?.n}`} />} />
			{outcomes.map(o => <Bar key={o.key} name={o.label} dataKey={o.key} stackId="outcomes" fill={o.color} stroke="var(--background)" strokeWidth={2} isAnimationActive={false} />)}
		</BarChart>
	</Plot>;
}

function Agreement({ report, benchmarks }: { report: Report; benchmarks: string[] }) {
	const rows = benchmarks.map(b => ({
		name: title(b),
		student_before: report.comparisons.teacher_student_before?.[b]?.answer_agreement,
		student_after: report.comparisons.teacher_student_after?.[b]?.answer_agreement,
	}));
	const shown = (["student_before", "student_after"] as const).filter(slot => rows.some(row => row[slot] != null));
	if (!rows.length || !shown.length) return <Plot name="How often the student agrees with the teacher" note="Needs a victim and student run from the same phase." />;
	return <Plot name="How often the student agrees with the teacher" legend={runLegend(shown)}
		note="Share of questions where the student gave the same final answer as the teacher. Rising agreement means the student is imitating the teacher, which includes copying its mistakes. The denominator contains pairs with parseable predictions; inspect coverage in the report. Coding functional agreement is not measured.">
		<BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2} barCategoryGap="28%">
			<CartesianGrid vertical={false} />
			<XAxis dataKey="name" tick={tick} tickLine={false} />
			<YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={percent} tick={tick} tickLine={false} axisLine={false} width={56} label={yTitle("Same answer (%)")} />
			<ChartTooltip cursor={barCursor} content={props => <Tip {...props} value={v => format(v, "%")} />} />
			{shown.map(slot => <Bar key={slot} name={labels[slot]} dataKey={slot} {...barStyle(slot)} maxBarSize={28} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
		</BarChart>
	</Plot>;
}

function Subjects({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const entries = Object.entries(report.slices).filter(([key]) => benchmarks.includes(key.split("/")[0]));
	const name = (key: string) => {
		const [benchmark, category] = key.split("/");
		return category === benchmark ? title(benchmark) : `${title(benchmark)} · ${category.replaceAll("_", " ")}`;
	};
	return <section className="plot wide">
		<div className="plot-header"><h2>Task score by category</h2></div>
		<p className="plot-note">Same task metric as above, split by category. Brighter cells score higher. Subjects with few questions (n) swing a lot from chance alone.</p>
		<div className="heat-legend" aria-label="Color scale"><span>0%</span><span className="heat-bar" style={{ background: `linear-gradient(to right, ${heatRamp.join(", ")})` }} /><span>100%</span></div>
		<div className="table-scroll"><table className="heatmap">
			<thead><tr><th>Subject</th><th className="num">n</th>{available.map(slot => <th key={slot}><span className="th-run"><SlotSwatch slot={slot} />{labels[slot]}</span></th>)}</tr></thead>
			<tbody>{entries.map(([key, values]) => <tr key={key}>
				<th>{name(key)}</th>
				<td className="num muted">{Object.values(values)[0]?.n ?? "n/a"}</td>
				{available.map(slot => {
					const value = values[slot];
					return <td key={slot} style={heat(taskScore(value, key.split("/")[0]))} title={value ? `${value.n} questions · 95% CI ${taskInterval(value, key.split("/")[0])?.map(v => v.toFixed(1)).join("–") ?? "n/a"}%` : "Not recorded"}>{format(taskScore(value, key.split("/")[0]), "%")}</td>;
				})}
			</tr>)}</tbody>
		</table></div>
	</section>;
}
