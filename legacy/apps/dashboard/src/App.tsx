import { useEffect, useState } from "react";
import { ArrowDownToLine, LoaderCircle, RefreshCw } from "lucide-react";
import { Button } from "./components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Choice } from "./components/choice";
import { SlotSwatch } from "./components/plot";
import { Accuracy } from "./components/accuracy";
import { Speed } from "./components/speed";
import { Answers } from "./components/answers";
import { Method } from "./components/method";
import { Traffic } from "./components/traffic";
import { format, signed, title, taskScore, metricLabel } from "./data";
import { labels, slots, type Experiment, type Report, type Slot } from "./types";

function exportReport(report: Report) {
	const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, "\t")], { type: "application/json" }));
	const link = document.createElement("a");
	link.href = url;
	link.download = `distillation-${report.id}.json`;
	link.click();
	URL.revokeObjectURL(url);
}

export default function App() {
	const [view, setView] = useState<"quality" | "traffic">("quality");
	const [experiments, setExperiments] = useState<Experiment[]>([]);
	const [selection, setSelection] = useState("");
	const [report, setReport] = useState<Report | null>(null);
	const [error, setError] = useState("");
	const [loading, setLoading] = useState(true);
	const [reload, setReload] = useState(0);
	const [scope, setScope] = useState("all");
	const [active, setActive] = useState<Slot[]>([...slots]);
	useEffect(() => {
		const controller = new AbortController();
		setLoading(true);
		setError("");
		fetch("/api/experiments", { signal: controller.signal }).then(async response => {
			if (!response.ok) throw new Error(`Cannot load experiments (HTTP ${response.status}).`);
			return response.json() as Promise<{ experiments: Experiment[]; errors: string[] }>;
		}).then(data => {
			setExperiments(data.experiments);
			setSelection(current => data.experiments.some(e => e.id === current) ? current : data.experiments[0]?.id ?? "");
			if (data.errors.length) setError(data.errors.join(". "));
			if (!data.experiments.length) setLoading(false);
		}).catch(reason => {
			if (!controller.signal.aborted) { setError(String(reason.message)); setLoading(false); }
		});
		return () => controller.abort();
	}, [reload]);
	useEffect(() => {
		if (!selection) { setReport(null); return; }
		const controller = new AbortController();
		setLoading(true);
		fetch(`/api/reports/${selection}`, { signal: controller.signal }).then(async response => {
			if (!response.ok) { const body = await response.json(); throw new Error(body.detail ?? "Cannot load this experiment."); }
			return response.json() as Promise<Report>;
		}).then(data => { setReport(data); setLoading(false); }).catch(reason => {
			if (!controller.signal.aborted) { setError(String(reason.message)); setReport(null); setLoading(false); }
		});
		return () => controller.abort();
	}, [selection, reload]);
	const benchmarks = report?.benchmarks.filter(b => scope === "all" || scope === b) ?? [];
	const available = active.filter(slot => report?.runs[slot]);
	return <div className="app-shell">
		<header className="topbar">
			<div><h1>Evaluating an account-level harness against multi-domain model distillation</h1><p>Task capability, legitimate access, and resistance to extraction.</p></div>
			<div className="actions">
				<Button variant={view === "quality" ? "outline" : "ghost"} aria-pressed={view === "quality"} onClick={() => setView("quality")}>Task capability</Button>
				<Button variant={view === "traffic" ? "outline" : "ghost"} aria-pressed={view === "traffic"} onClick={() => setView("traffic")}>Harness traffic</Button>
				{view === "quality" && experiments.length > 0 && <div className="field"><label htmlFor="experiment">Experiment</label><Choice id="experiment" label="Experiment" value={selection} onChange={setSelection} options={experiments.map(e => ({ value: e.id, label: `${e.name}${e.demo ? " (simulated)" : ""} · ${new Date(e.updated_at).toLocaleDateString()}` }))} /></div>}
				<Button variant="outline" onClick={() => setReload(v => v + 1)} aria-label="Reload results"><RefreshCw size={15} />Reload</Button>
				{view === "quality" && report && <Button variant="outline" onClick={() => exportReport(report)}><ArrowDownToLine size={15} />Export JSON</Button>}
			</div>
		</header>
		<main>
			{view === "traffic" && <Traffic reload={reload} />}
			{view === "quality" && <>
				{error && <div role="alert" className="notice error">{error} Start the results API with <code>uv run distill benchmark dashboard</code>, then reload.</div>}
				{loading && <div className="empty" role="status"><LoaderCircle className="loading-icon" />Loading results</div>}
				{!loading && !report && !error && <div className="empty">
					<h2>No benchmark runs yet</h2>
					<p>Build a task and evaluate the victim and student to fill this dashboard.</p>
					<pre>uv run --extra eval distill tasks build --task ag_news{"\n"}uv run distill benchmark run --config configs/eval.ag_news.toml --stage before</pre>
					<p>To preview the dashboard with simulated data:</p>
					<pre>uv run distill benchmark demo{"\n"}uv run distill benchmark dashboard --results-dir outputs/tasks-demo/ag_news</pre>
				</div>}
				{!loading && report && <>
					{report.demo && <div role="status" className="notice demo"><strong>Simulated data.</strong> These numbers are fixtures for previewing the dashboard. No model was evaluated.</div>}
					{slots.some(slot => !report.runs[slot]) && <div className="notice">Waiting for: {slots.filter(slot => !report.runs[slot]).map(slot => labels[slot]).join(", ")}. Before/after comparisons appear once both student runs exist.</div>}
					<Summary report={report} />
					<Tabs defaultValue="accuracy">
						<div className="toolbar">
							<TabsList aria-label="Views">
								<TabsTrigger value="accuracy">Task scores</TabsTrigger>
								<TabsTrigger value="speed">Speed</TabsTrigger>
								<TabsTrigger value="answers">Answers</TabsTrigger>
								<TabsTrigger value="method">Method</TabsTrigger>
							</TabsList>
							<div className="filters">
								<div className="field inline"><label htmlFor="scope">Tasks</label><Choice id="scope" label="Tasks" value={scope} onChange={setScope} options={[
									{ value: "all", label: "All" },
									...report.benchmarks.map(b => ({ value: b, label: title(b) })),
								]} /></div>
								<div className="run-toggles" role="group" aria-label="Runs shown in charts">
									{slots.map(slot => <Button key={slot} size="sm" variant="ghost" aria-pressed={active.includes(slot)} disabled={!report.runs[slot]}
										onClick={() => setActive(current => current.includes(slot) ? current.filter(s => s !== slot) : slots.filter(s => s === slot || current.includes(s)))}
										className={active.includes(slot) ? "run-on" : "run-off"}><SlotSwatch slot={slot} />{labels[slot]}</Button>)}
								</div>
							</div>
						</div>
						<TabsContent value="accuracy"><Accuracy report={report} benchmarks={benchmarks} available={available} /></TabsContent>
						<TabsContent value="speed"><Speed report={report} benchmarks={benchmarks} available={available} /></TabsContent>
						<TabsContent value="answers"><Answers report={report} benchmarks={benchmarks} available={available} /></TabsContent>
						<TabsContent value="method"><Method report={report} /></TabsContent>
					</Tabs>
					<footer><span>{report.experiment} · suite {report.suite.sha256.slice(0, 12)} · {report.suite.items.length} questions · scorer {report.suite.scorer_version}</span><span>{report.demo ? "Simulated data" : "Measured runs"}</span></footer>
				</>}
			</>}
		</main>
	</div>;
}

function Summary({ report }: { report: Report }) {
	const task = report.benchmarks[0];
	const score = (slot: Slot) => taskScore(report.scores[slot]?.[task], task);
	const before = score("student_before"), after = score("student_after");
	const change = before != null && after != null ? after - before : null;
	const tiles: { label: string; value: string; caption: string; slot?: Slot; tone?: "up" | "down" }[] = [
		{ label: labels.student_before, value: format(before, "%"), caption: metricLabel(task), slot: "student_before" },
		{ label: labels.student_after, value: format(after, "%"), caption: metricLabel(task), slot: "student_after" },
		{ label: "Student change", value: signed(change, " pp"), caption: "percentage points", tone: change == null || change === 0 ? undefined : change > 0 ? "up" : "down" },
		{ label: labels.teacher, value: format(score("teacher"), "%"), caption: metricLabel(task), slot: "teacher" },
		{ label: labels.teacher_after, value: format(score("teacher_after"), "%"), caption: metricLabel(task), slot: "teacher_after" },
	];
	return <section className="summary" aria-label="Summary">
		<p className="headline">{before == null || after == null
			? `${title(task)}. The student comparison appears once both runs are recorded.`
			: `${title(task)} ${metricLabel(task)} after distillation is ${after.toFixed(1)}%, a change of ${signed(change, " percentage points")} from ${before.toFixed(1)}%.`}</p>
		<dl className="tiles">{tiles.map(tile => <div key={tile.label}>
			<dt>{tile.slot && <SlotSwatch slot={tile.slot} />}{tile.label}</dt>
			<dd className={tile.tone}>{tile.value}<span>{tile.caption}</span></dd>
		</div>)}</dl>
		<p className="small-note">Scores describe this task and adapter. Defended and undefended conditions are separate experiments.</p>
	</section>;
}
