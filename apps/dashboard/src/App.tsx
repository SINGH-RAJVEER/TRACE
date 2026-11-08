import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { coverage, shortLabel, type CoverageCell, type CoverageState } from "./data";
import type { RunDetail, RunList } from "./types";
import { cn } from "./lib";
import { CampaignsView } from "./components/campaigns";
import { CoverageMark } from "./components/charts";
import { Hash, Unavailable, Value } from "./components/evidence";
import { GatesView } from "./components/gates";
import { Overview, RunSummaryView, attackUniverse, type Loaded } from "./components/overview";
import { ProvenanceView } from "./components/provenance";
import { StudentsView } from "./components/students";
import { TeacherView } from "./components/teacher";
import { UtilityView } from "./components/utility";
import { Badge } from "./components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";

type Load<T> = { state: "loading" } | { state: "ready"; data: T } | { state: "error"; message: string };

async function getJson<T>(url: string): Promise<T> {
	const response = await fetch(url);
	if (!response.ok) throw new Error(`${url} answered ${response.status}`);
	return response.json() as Promise<T>;
}

const OVERVIEW = "overview";

function selectedFromHash(): string {
	const match = window.location.hash.match(/run=([0-9a-f]{16})/);
	return match ? match[1] : OVERVIEW;
}

/** The combined state of several coverage cells: present only when all are, missing only when none are. */
function combined(all: CoverageCell[]): CoverageState {
	const cells = all.filter(c => c.state !== "na");
	if (!cells.length) return all.length ? "na" : "missing";
	if (cells.some(c => c.state === "error")) return "error";
	if (cells.every(c => c.state === "present")) return "present";
	if (cells.every(c => c.state === "missing")) return "missing";
	return "partial";
}

const SECTIONS = [
	{ id: "summary", label: "Summary", groups: [] as string[] },
	{ id: "acquisition", label: "Acquisition", groups: ["Acquisition"] },
	{ id: "utility", label: "Legitimate utility", groups: ["Utility"] },
	{ id: "teacher", label: "Teacher", groups: ["Teacher"] },
	{ id: "students", label: "Students", groups: ["Students"] },
	{ id: "gates", label: "Gates", groups: ["Gates"] },
	{ id: "provenance", label: "Provenance", groups: [] as string[] },
] as const;

function RunSection({ id, title, kicker, state, children }: { id: string; title: string; kicker?: React.ReactNode; state?: CoverageState; children: React.ReactNode }) {
	return <section id={id} className="scroll-mt-20 space-y-6 border-t border-rule pt-8 first:border-t-0 first:pt-0">
		<header className="space-y-1">
			<div className="flex items-center gap-3">
				<h3 className="display text-2xl">{title}</h3>
				{state && <CoverageMark state={state} />}
			</div>
			{kicker && <p className="max-w-3xl text-sm text-muted">{kicker}</p>}
		</header>
		{children}
	</section>;
}

/** A group whose every artifact is missing collapses to one list of reasons. */
function MissingGroup({ cells }: { cells: CoverageCell[] }) {
	return <ul className="space-y-2">
		{cells.map(c => <li key={c.key} className="grid gap-x-4 gap-y-0.5 text-sm sm:grid-cols-[12rem_1fr]">
			<span className="flex items-center gap-2"><CoverageMark state={c.state} compact /><span className="font-medium">{c.label}</span></span>
			<span className="text-muted">{c.detail}</span>
		</li>)}
	</ul>;
}

function RunPage({ entry, attacks, list }: { entry: Loaded; attacks: string[]; list: RunList }) {
	const { run, detail } = entry;
	const cells = useMemo(() => detail ? coverage(detail) : [], [detail]);
	const groupCells = (group: string) => cells.filter(c => c.group === group);
	const ready = !!detail;
	useEffect(() => {
		const section = window.location.hash.match(/[&#]s=([a-z]+)/)?.[1];
		if (ready && section) document.getElementById(section)?.scrollIntoView({ block: "start" });
	}, [ready]);
	const stateOf = (groups: readonly string[]) => groups.length ? combined(cells.filter(c => groups.includes(c.group))) : undefined;

	return <div className="lg:grid lg:grid-cols-[11rem_1fr] lg:gap-10">
		<nav aria-label="Sections" className="hidden lg:block">
			<ul className="sticky top-20 space-y-0.5 text-sm">
				{SECTIONS.filter(s => stateOf(s.groups) !== "na").map(s => {
					const state = stateOf(s.groups);
					return <li key={s.id}><a href={`#run=${run.id}&s=${s.id}`} onClick={e => { e.preventDefault(); window.history.replaceState(null, "", `#run=${run.id}&s=${s.id}`); document.getElementById(s.id)?.scrollIntoView({ behavior: "smooth", block: "start" }); }} className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-muted hover:bg-wash hover:text-ink">
						<span>{s.label}</span>{state && <CoverageMark state={state} compact />}
					</a></li>;
				})}
			</ul>
		</nav>
		<div className="min-w-0 space-y-10">
			<header className="space-y-3">
				<div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
					<h2 className="display num text-3xl" style={{ fontFamily: "var(--font-sans)" }}>{run.label}</h2>
					<Badge>{run.kind}</Badge>
					{run.studies.map(study => <Badge key={study} variant={study === "confirmatory" ? "info" : "caution"}>{study}</Badge>)}
					{!run.studies.length && <Value value={null} unknown="study not recorded" />}
				</div>
				<div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted">
					<span>Readings <Value value={run.readings.join(", ") || null} /></span>
					<span>Commit {run.commits.length ? run.commits.map(c => <Hash key={c} value={c} />) : <Value value={null} />}</span>
					<span>Directories <Value value={run.places.map(p => `${p.root}/${p.path}`).join(", ") || null} unknown="none found" /></span>
				</div>
			</header>
			{!detail && <Unavailable tone={entry.error ? "error" : "neutral"} reason={entry.error ?? "Reading this run's artifacts."} />}
			{detail && <>
				<RunSection id="summary" title="Summary">
					<RunSummaryView detail={detail} attacks={attacks} list={list} />
				</RunSection>
				{stateOf(["Acquisition"]) !== "na" && <RunSection id="acquisition" title="Acquisition" state={stateOf(["Acquisition"])} kicker="What each campaign obtained from the teacher and when the policy intervened. These are acquisition counts, not student or model accuracy.">
					<CampaignsView data={detail.campaigns} attacks={attacks} />
				</RunSection>}
				{stateOf(["Utility"]) !== "na" && <RunSection id="utility" title="Legitimate utility" state={stateOf(["Utility"])} kicker="How well each arm still served the held-out benign workload.">
					<UtilityView data={detail.utility} />
				</RunSection>}
				{stateOf(["Teacher"]) !== "na" && <RunSection id="teacher" title="Teacher" state={stateOf(["Teacher"])} kicker="The teacher as configured, its headroom over the student base and whether it could serve the workload.">
					<TeacherView data={detail.teacher} provenance={detail.provenance} hideMissing={combined(groupCells("Teacher")) === "missing"} />
					{combined(groupCells("Teacher")) === "missing" && <MissingGroup cells={groupCells("Teacher")} />}
				</RunSection>}
				{stateOf(["Students"]) !== "na" && <RunSection id="students" title="Students" state={stateOf(["Students"])} kicker="Students trained on what the teacher released, scored on a frozen suite.">
					{combined(groupCells("Students")) === "missing" ? <MissingGroup cells={groupCells("Students")} /> : <StudentsView data={detail.students} />}
				</RunSection>}
				{stateOf(["Gates"]) !== "na" && <RunSection id="gates" title="Gates" state={stateOf(["Gates"])} kicker="Baseline gates, failing ones included, and campaign arms the detectability gate refused.">
					<GatesView data={detail.gates} />
				</RunSection>}
				<RunSection id="provenance" title="Provenance" kicker="Index labels and the files the runner wrote beside its results.">
					<ProvenanceView data={detail.provenance} />
				</RunSection>
			</>}
		</div>
	</div>;
}

export default function App() {
	const [list, setList] = useState<Load<RunList>>({ state: "loading" });
	const [details, setDetails] = useState<Record<string, { detail?: RunDetail; error?: string }>>({});
	const [selected, setSelected] = useState<string>(selectedFromHash);

	useEffect(() => {
		getJson<RunList>("/api/runs").then(data => {
			setList({ state: "ready", data });
			for (const run of data.runs) {
				getJson<RunDetail>(`/api/runs/${run.id}`).then(
					detail => setDetails(prev => ({ ...prev, [run.id]: { detail } })),
					error => setDetails(prev => ({ ...prev, [run.id]: { error: String(error) } })),
				);
			}
		}, error => setList({ state: "error", message: String(error) }));
	}, []);

	useEffect(() => {
		const onHash = () => setSelected(selectedFromHash());
		window.addEventListener("hashchange", onHash);
		return () => window.removeEventListener("hashchange", onHash);
	}, []);

	const select = (id: string) => {
		setSelected(id);
		window.history.replaceState(null, "", id === OVERVIEW ? window.location.pathname : `#run=${id}`);
		window.scrollTo({ top: 0 });
	};

	const runs = list.state === "ready" ? list.data.runs : [];
	const loaded: Loaded[] = runs.map(run => ({ run, detail: details[run.id]?.detail ?? null, error: details[run.id]?.error }));
	const current = loaded.find(l => l.run.id === selected) ?? null;
	const attacks = useMemo(() => attackUniverse(loaded), [details, runs.length]);

	return <div className="min-h-screen">
		<header className="sticky top-0 z-40 border-b border-rule bg-paper/90 backdrop-blur">
			<div className="mx-auto flex max-w-[110rem] flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-8">
				<button type="button" onClick={() => select(OVERVIEW)} className="display text-xl leading-none">TRACE results</button>
				{runs.length > 0 && <>
					<nav aria-label="Runs" className="hidden flex-wrap items-center gap-1 md:flex">
						{[{ id: OVERVIEW, label: "All runs" }, ...runs.map(r => ({ id: r.id, label: shortLabel(r.label) }))].map(item => <button key={item.id} type="button" title={runs.find(r => r.id === item.id)?.label} onClick={() => select(item.id)} aria-current={(current?.run.id ?? OVERVIEW) === item.id ? "page" : undefined}
							className={cn("rounded-md px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-accent", (current?.run.id ?? OVERVIEW) === item.id ? "bg-ink text-paper" : "text-muted hover:bg-wash hover:text-ink", item.id !== OVERVIEW && "num")}>
							{item.label}
						</button>)}
					</nav>
					<div className="w-full md:hidden">
						<Select value={current?.run.id ?? OVERVIEW} onValueChange={select}>
							<SelectTrigger aria-label="Run"><SelectValue /></SelectTrigger>
							<SelectContent>
								<SelectItem value={OVERVIEW}>All runs</SelectItem>
								{runs.map(run => <SelectItem key={run.id} value={run.id}>{run.label}</SelectItem>)}
							</SelectContent>
						</Select>
					</div>
				</>}
				{list.state === "ready" && <span className="ml-auto hidden text-xs text-faint xl:inline">
					index {list.data.index.state === "ok" ? `${list.data.index.rows} rows` : list.data.index.state} · roots {list.data.roots.map(root => <span key={root.name} className={cn("num ml-1", !root.exists && "text-caution line-through")} title={root.exists ? undefined : "directory does not exist"}>{root.name}</span>)}
				</span>}
			</div>
		</header>
		<main className="mx-auto max-w-[110rem] px-4 py-8 sm:px-8">
			{list.state === "loading" && <p className="text-sm text-muted">Reading the configured roots.</p>}
			{list.state === "error" && <Unavailable tone="error" reason={`The dashboard API did not answer: ${list.message}. Start it with just dashboard.`} />}
			{list.state === "ready" && runs.length === 0 && <Unavailable reason="No indexed runs and no recognized artifacts under the configured roots. Set DASHBOARD_ROOTS or export a reading." />}
			{list.state === "ready" && runs.length > 0 && (current ? <RunPage key={current.run.id} entry={current} attacks={attacks} list={list.data} /> : <Overview list={list.data} loaded={loaded} onSelect={select} />)}
		</main>
		<footer className="mx-auto max-w-[110rem] px-4 pb-10 text-xs text-faint sm:px-8">Read-only view of recorded study artifacts. Nothing here runs a model or contacts a teacher.</footer>
	</div>;
}
