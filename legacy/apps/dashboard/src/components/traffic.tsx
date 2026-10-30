import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { Choice } from "./choice";
import { Button } from "./ui/button";
import { ChartTooltip } from "./ui/chart";
import { Plot, Swatch, Tip, tick, xTitle, yTitle } from "./plot";
import { format, signed } from "../data";

type Phase = "before" | "after";
type Label = "genuine" | "distillation";
type Summary = {
	n: number; succeeded: number; success_percent: number; blocked: number;
	throttled: number; rate_limited: number; decision_coverage: number;
	false_positive_percent?: number | null; statuses: Record<string, number>;
	successful_requests_per_second: number; successful_output_tokens_per_second: number | null;
	latency_p50: number | null; latency_p95: number | null; schedule_delay_p95: number;
};
type Actor = Summary & {
	id: string; label: Label; scenario: string; first_throttle_seconds: number | null;
	first_block_seconds: number | null; terminal_blocked_requests: number; observed_stopped: boolean;
	observed_stop_seconds: number | null;
};
type Window = { start_seconds: number; genuine: Summary; distillation: Summary };
type Score = { genuine: Summary; distillation: Summary; actors: Record<string, Actor>; windows: Window[] };
type TrafficReport = {
	id: string; experiment: string; demo: boolean; warnings: string[];
	suite: { sha256: string; requests: unknown[]; duration_seconds: number };
	scores: Partial<Record<Phase, Score>>;
	comparison: Partial<Record<Label, { success_delta_pp: number; throughput_reduction_percent: number | null; latency_p95_delta_seconds: number | null }>>;
};
type Experiment = { id: string; name: string; demo: boolean; phases: Phase[] };
const phases: Phase[] = ["before", "after"];
const labels: Label[] = ["genuine", "distillation"];
const phaseName = { before: "Before harness", after: "After harness" };
const labelName = { genuine: "Genuine requests", distillation: "Distillation attempts" };

async function read<T>(url: string, signal: AbortSignal): Promise<T> {
	const response = await fetch(url, { signal });
	if (!response.ok) throw new Error((await response.json()).detail ?? `HTTP ${response.status}`);
	return response.json();
}

export function Traffic({ reload }: { reload: number }) {
	const [experiments, setExperiments] = useState<Experiment[]>([]);
	const [selection, setSelection] = useState("");
	const [report, setReport] = useState<TrafficReport | null>(null);
	const [error, setError] = useState("");
	const [loading, setLoading] = useState(true);
	useEffect(() => {
		const controller = new AbortController();
		setLoading(true);
		setError("");
		read<{ experiments: Experiment[]; errors: string[] }>("/api/traffic/experiments", controller.signal).then(data => {
			setExperiments(data.experiments);
			setSelection(current => data.experiments.some(e => e.id === current) ? current : data.experiments[0]?.id ?? "");
			setError(data.errors.join(". "));
			if (!data.experiments.length) { setReport(null); setLoading(false); }
		}).catch(reason => { if (!controller.signal.aborted) { setError(reason.message); setLoading(false); } });
		return () => controller.abort();
	}, [reload]);
	useEffect(() => {
		if (!selection) return;
		const controller = new AbortController();
		setLoading(true);
		setReport(null);
		read<TrafficReport>(`/api/traffic/reports/${selection}`, controller.signal).then(value => {
			setReport(value); setLoading(false);
		}).catch(reason => { if (!controller.signal.aborted) { setError(reason.message); setLoading(false); } });
		return () => controller.abort();
	}, [selection, reload]);
	return <>
		{error && <div role="alert" className="notice error">{error}</div>}
		<div className="toolbar">
			<h2>Genuine access and extraction resistance</h2>
			{experiments.length > 0 && <Choice id="traffic-experiment" label="Traffic experiment" value={selection} onChange={setSelection} options={experiments.map(e => ({ value: e.id, label: `${e.name}${e.demo ? " (simulated)" : ""}` }))} />}
			{report && <Button variant="outline" onClick={() => {
				const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, "\t")], { type: "application/json" }));
				const link = document.createElement("a"); link.href = url; link.download = `traffic-${report.id}.json`; link.click(); URL.revokeObjectURL(url);
			}}>Export traffic JSON</Button>}
		</div>
		{loading && <p role="status">Loading traffic results</p>}
		{!loading && !report && !error && <div className="empty"><h2>No harness traffic runs yet</h2><p>Replay the same simulated genuine users and extraction attempts before and after deploying the harness. Each phase measures actual request outcomes.</p><pre>uv run distill benchmark traffic freeze --output data/benchmarks/traffic.json{"\n"}uv run distill benchmark traffic run --config configs/traffic.example.toml --phase before</pre></div>}
		{!loading && report && <>
			{report.demo && <div className="notice demo" role="status"><strong>Simulated outcomes.</strong> Preview fixtures only. No harness or model was measured.</div>}
			{report.warnings.map(warning => <div className="notice" key={warning}>{warning}</div>)}
			{phases.some(phase => !report.scores[phase]) && <p className="plot-note">Waiting for {phases.filter(phase => !report.scores[phase]).map(phase => phaseName[phase].toLowerCase()).join(" and ")} results. The baseline is measured, never assumed to be 100%.</p>}
			<section className="plot">
				<h2>Request outcomes</h2>
				<p className="plot-note">Success means a complete, nonempty API response. False positives are confirmed harness blocks of genuine requests. Throttling and other failures remain separate; this view does not score answer correctness.</p>
				<div className="table-scroll"><table><thead><tr><th>Traffic</th><th>Phase</th><th className="num">Succeeded</th><th className="num">Success</th><th className="num">Blocked</th><th className="num">False positives</th><th className="num">Throttled</th><th className="num">HTTP 429</th><th>Other failures</th></tr></thead>
					<tbody>{labels.flatMap(label => phases.map(phase => {
						const s = report.scores[phase]?.[label];
						if (!s) return null;
						return <tr key={`${phase}-${label}`}><th>{labelName[label]}</th><td>{phaseName[phase]}</td><td className="num">{s.succeeded}/{s.n}</td><td className="num">{format(s.success_percent, "%")}</td><td className="num">{s.blocked}<small>{format(s.decision_coverage * 100, "%", 0)} decisions known</small></td><td className="num">{label === "genuine" ? format(s.false_positive_percent, "%") : "N/A"}</td><td className="num">{s.throttled}</td><td className="num">{s.rate_limited}</td><td>{Object.entries(s.statuses).filter(([status]) => !["ok", "blocked", "rate_limited"].includes(status)).map(([status, n]) => `${n} ${status}`).join(", ") || "None"}</td></tr>;
					}))}</tbody></table></div>
				<p className="small-note">Blocked counts are confirmed observations. An unavailable false-positive percentage means decision telemetry is incomplete. A throttled request may still succeed or receive HTTP 429, so these columns can overlap.</p>
			</section>
			<div className="plots-grid">
				{labels.map(label => <Timeline key={label} report={report} label={label} />)}
			</div>
			<section className="plot"><h2>Delay and delivered throughput</h2><p className="plot-note">Latency covers successful responses. Throughput includes the full replay and all outstanding responses. Client scheduling delay is reported separately.</p>
				<div className="table-scroll"><table><thead><tr><th>Traffic</th><th>Phase</th><th className="num">Median</th><th className="num">p95</th><th className="num">Responses / s</th><th className="num">Output tokens / s</th><th className="num">Client delay p95</th></tr></thead><tbody>{labels.flatMap(label => phases.map(phase => {
					const s = report.scores[phase]?.[label];
					return s && <tr key={`${phase}-${label}`}><th>{labelName[label]}</th><td>{phaseName[phase]}</td><td className="num">{format(s.latency_p50, "s", 3)}</td><td className="num">{format(s.latency_p95, "s", 3)}</td><td className="num">{format(s.successful_requests_per_second, "", 3)}</td><td className="num">{format(s.successful_output_tokens_per_second, "", 2)}</td><td className="num">{format(s.schedule_delay_p95, "s", 3)}</td></tr>;
				}))}</tbody></table></div>
				{labels.map(label => {
					const delta = report.comparison[label];
					return delta && <p className="plot-note" key={label}>{labelName[label]}: {signed(delta.success_delta_pp, " pp")} success change; {format(delta.throughput_reduction_percent, "%")} delivered-throughput reduction; {signed(delta.latency_p95_delta_seconds, "s")} p95 latency change.</p>;
				})}
			</section>
			<section className="plot"><h2>Extraction progression by actor</h2><p className="plot-note">Times are seconds from replay start to the first confirmed decision response. An observed stop requires at least two final blocked requests after all successful in-flight completions have drained. It describes this finite replay.</p>
				<div className="table-scroll"><table><thead><tr><th>Actor / scenario</th><th>Phase</th><th className="num">First throttle</th><th className="num">First block</th><th className="num">Final blocked requests</th><th>Observed stopped</th></tr></thead><tbody>{phases.flatMap(phase => Object.values(report.scores[phase]?.actors ?? {}).filter(a => a.label === "distillation").map(a => <tr key={`${phase}-${a.id}`}><th>{a.id}<small>{a.scenario.replaceAll("_", " ")}</small></th><td>{phaseName[phase]}</td><td className="num">{format(a.first_throttle_seconds, "s", 2)}</td><td className="num">{format(a.first_block_seconds, "s", 2)}</td><td className="num">{a.terminal_blocked_requests}</td><td>{a.observed_stopped ? `Yes, from ${format(a.observed_stop_seconds, "s", 2)}` : "No"}</td></tr>))}</tbody></table></div>
			</section>
			<footer><span>{report.experiment} · traffic {report.suite.sha256.slice(0, 12)} · {report.suite.requests.length} requests · {report.suite.duration_seconds}s schedule</span><span>{report.demo ? "Simulated outcomes" : "Measured replay of synthetic traffic"}</span></footer>
		</>}
	</>;
}

function Timeline({ report, label }: { report: TrafficReport; label: Label }) {
	const rows = new Map<number, Record<string, number>>();
	for (const phase of phases) for (const window of report.scores[phase]?.windows ?? []) {
		const row = rows.get(window.start_seconds) ?? { seconds: window.start_seconds };
		row[phase] = window[label].successful_requests_per_second;
		rows.set(window.start_seconds, row);
	}
	const colors = { before: "#8f8e87", after: label === "genuine" ? "#3fbf8a" : "#e66767" };
	return <Plot name={`${labelName[label]} delivered over time`} note="Successful responses per second, grouped by completion time. A zero records no delivered response in that window." legend={phases.filter(phase => report.scores[phase]).map(phase => ({ key: phase, label: phaseName[phase], swatch: <Swatch color={colors[phase]} line dashed={phase === "before"} /> }))}>
		<LineChart data={[...rows.values()].sort((a, b) => a.seconds - b.seconds)} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
			<CartesianGrid vertical={false} /><XAxis dataKey="seconds" tick={tick} label={xTitle("Seconds from start")} /><YAxis tick={tick} width={56} label={yTitle("Responses / s")} domain={[0, "auto"]} />
			<ChartTooltip content={props => <Tip {...props} value={v => `${format(v, "", 3)} responses/s`} />} />
			{phases.filter(phase => report.scores[phase]).map(phase => <Line key={phase} dataKey={phase} name={phaseName[phase]} stroke={colors[phase]} strokeDasharray={phase === "before" ? "5 4" : undefined} dot={false} type="stepAfter" isAnimationActive={false} connectNulls={false} />)}
		</LineChart>
	</Plot>;
}
