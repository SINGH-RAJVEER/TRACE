import { useEffect, useMemo, useState } from "react";
import { studyLabel } from "./data";
import type { RunDetail, RunList, RunSummary } from "./types";
import { cn } from "./lib";
import { CampaignsView } from "./components/campaigns";
import { Hash, Unavailable, Value } from "./components/evidence";
import { GatesView } from "./components/gates";
import { ProvenanceView } from "./components/provenance";
import { StudentsView } from "./components/students";
import { TeacherView } from "./components/teacher";
import { UtilityView } from "./components/utility";
import { Badge } from "./components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";

type Load<T> = { state: "loading" } | { state: "ready"; data: T } | { state: "error"; message: string };

async function getJson<T>(url: string): Promise<T> {
	const response = await fetch(url);
	if (!response.ok) throw new Error(`${url} answered ${response.status}`);
	return response.json() as Promise<T>;
}

function selectedFromHash(): string | null {
	const match = window.location.hash.match(/run=([0-9a-f]{16})/);
	return match ? match[1] : null;
}

/** Whether a tab has anything to show, from the section states the API returned. */
function available(value: unknown): boolean {
	if (!value || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	if (record.state === "available") return true;
	if (record.state === "error" || record.state === "unavailable") return false;
	if ("sources" in record) return true;
	return Object.values(record).some(available);
}

const TABS = [
	["provenance", "Provenance"],
	["campaigns", "Acquisition"],
	["utility", "Legitimate utility"],
	["teacher", "Teacher"],
	["students", "Students"],
	["gates", "Gates"],
] as const;

function RunEntry({ run, active, onSelect }: { run: RunSummary; active: boolean; onSelect: () => void }) {
	const rows = Object.entries(run.rows).map(([kind, count]) => `${count} ${kind}`).join(" · ");
	const artifacts = Object.entries(run.artifacts).map(([kind, count]) => `${count} ${kind}`).join(" · ");
	return <button type="button" onClick={onSelect} aria-current={active ? "true" : undefined} className={cn("w-full border-l-2 px-3 py-2.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-accent", active ? "border-ink bg-wash" : "border-transparent hover:bg-wash/60")}>
		<div className="flex items-baseline justify-between gap-2">
			<span className="num truncate text-ink">{run.label}</span>
			<span className="text-xs text-muted">{run.kind}</span>
		</div>
		<div className="mt-0.5 text-xs text-muted">{studyLabel(run.studies) ?? "study not recorded"}{run.statuses.includes("gate-refused") ? " · gate refusals" : ""}</div>
		<div className="mt-0.5 truncate text-xs text-muted">{[rows, artifacts].filter(Boolean).join(" · ") || "no rows or artifacts"}</div>
	</button>;
}

export default function App() {
	const [list, setList] = useState<Load<RunList>>({ state: "loading" });
	const [selected, setSelected] = useState<string | null>(selectedFromHash);
	const [detail, setDetail] = useState<Load<RunDetail> | null>(null);
	const [tab, setTab] = useState<string>("provenance");

	useEffect(() => {
		getJson<RunList>("/api/runs").then(data => setList({ state: "ready", data }), error => setList({ state: "error", message: String(error) }));
	}, []);

	const runs = list.state === "ready" ? list.data.runs : [];
	const current = runs.find(run => run.id === selected) ?? runs[0] ?? null;

	useEffect(() => {
		if (!current) return;
		window.history.replaceState(null, "", `#run=${current.id}`);
		setDetail({ state: "loading" });
		let live = true;
		getJson<RunDetail>(`/api/runs/${current.id}`).then(data => live && setDetail({ state: "ready", data }), error => live && setDetail({ state: "error", message: String(error) }));
		return () => { live = false; };
	}, [current?.id]);

	const ready = detail?.state === "ready" ? detail.data : null;
	const marks = useMemo(() => Object.fromEntries(TABS.map(([key]) => [key, ready ? available(ready[key]) : false])), [ready]);

	return <div className="min-h-screen lg:grid lg:grid-cols-[19rem_1fr]">
		<aside className="border-b border-rule lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto lg:border-r lg:border-b-0">
			<div className="space-y-1 px-4 pt-5 pb-4">
				<h1 className="display text-2xl leading-none">TRACE results</h1>
				<p className="text-xs text-muted">Read-only view of recorded study artifacts. Nothing here runs a model or contacts a teacher.</p>
			</div>
			{list.state === "ready" && <div className="space-y-1 border-t border-rule px-4 py-3 text-xs text-muted">
				<div>Index <span className="num">{list.data.index.path}</span>: {list.data.index.state === "ok" ? `${list.data.index.rows} rows` : `${list.data.index.state}${list.data.index.error ? `, ${list.data.index.error}` : ""}`}</div>
				<div>Roots: {list.data.roots.map(root => <span key={root.name} className={cn("num mr-2", !root.exists && "line-through")} title={root.exists ? undefined : "directory does not exist"}>{root.name}</span>)}</div>
			</div>}
			<div className="px-4 pb-3 lg:hidden">
				{runs.length > 0 && <Select value={current?.id} onValueChange={setSelected}>
					<SelectTrigger aria-label="Run"><SelectValue placeholder="Select a run" /></SelectTrigger>
					<SelectContent>{runs.map(run => <SelectItem key={run.id} value={run.id}>{run.label}</SelectItem>)}</SelectContent>
				</Select>}
			</div>
			<nav aria-label="Runs" className="hidden border-t border-rule py-2 lg:block">
				{runs.map(run => <RunEntry key={run.id} run={run} active={run.id === current?.id} onSelect={() => setSelected(run.id)} />)}
			</nav>
		</aside>
		<main className="min-w-0 px-4 py-6 sm:px-8">
			{list.state === "loading" && <p className="text-sm text-muted">Reading the configured roots.</p>}
			{list.state === "error" && <Unavailable tone="error" reason={`The dashboard API did not answer: ${list.message}. Start it with just dashboard.`} />}
			{list.state === "ready" && runs.length === 0 && <Unavailable reason="No indexed runs and no recognized artifacts under the configured roots. Set DASHBOARD_ROOTS or export a reading." />}
			{current && <div className="space-y-6">
				<header className="space-y-3">
					<div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
						<h2 className="display num text-3xl" style={{ fontFamily: "var(--font-sans)" }}>{current.label}</h2>
						<Badge>{current.kind}</Badge>
						{current.studies.map(study => <Badge key={study} variant={study === "confirmatory" ? "info" : "caution"}>{study}</Badge>)}
						{!current.studies.length && <Value value={null} unknown="study not recorded" />}
					</div>
					<div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted">
						<span>Readings <Value value={current.readings.join(", ") || null} /></span>
						<span>Commit {current.commits.length ? current.commits.map(c => <Hash key={c} value={c} />) : <Value value={null} />}</span>
						<span>Directories <Value value={current.places.map(p => `${p.root}/${p.path}`).join(", ") || null} unknown="none found" /></span>
					</div>
				</header>
				{detail?.state === "loading" && <p className="text-sm text-muted">Reading this run's artifacts.</p>}
				{detail?.state === "error" && <Unavailable tone="error" reason={detail.message} />}
				{ready && <Tabs value={tab} onValueChange={setTab}>
					<TabsList>{TABS.map(([key, label]) => <TabsTrigger key={key} value={key}>
						<span className={cn("mr-1.5 inline-block size-1.5 rounded-full align-middle", marks[key] ? "bg-accent" : "border border-muted")} aria-hidden />{label}
						<span className="sr-only">{marks[key] ? " (has data)" : " (unavailable)"}</span>
					</TabsTrigger>)}</TabsList>
					<TabsContent value="provenance"><ProvenanceView data={ready.provenance} /></TabsContent>
					<TabsContent value="campaigns"><CampaignsView data={ready.campaigns} /></TabsContent>
					<TabsContent value="utility"><UtilityView data={ready.utility} /></TabsContent>
					<TabsContent value="teacher"><TeacherView data={ready.teacher} provenance={ready.provenance} /></TabsContent>
					<TabsContent value="students"><StudentsView data={ready.students} /></TabsContent>
					<TabsContent value="gates"><GatesView data={ready.gates} /></TabsContent>
				</Tabs>}
			</div>}
		</main>
	</div>;
}
