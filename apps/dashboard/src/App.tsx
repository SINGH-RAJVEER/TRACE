import { useEffect, useMemo, useState } from "react";
import { shortLabel } from "./data";
import type { RunDetail, RunList } from "./types";
import { cn } from "./lib";
import { Unavailable } from "./components/evidence";
import { Overview, RootWarnings, attackUniverse } from "./components/overview";
import { RunPage, type Loaded } from "./components/run";
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
			<div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 sm:px-8">
				<button type="button" onClick={() => select(OVERVIEW)} className="text-base font-semibold leading-none">TRACE results</button>
				{runs.length > 0 && <>
					<nav aria-label="Runs" className="hidden flex-wrap items-center gap-1 md:flex">
						{[{ id: OVERVIEW, label: "All runs", loose: false }, ...runs.map(r => ({ id: r.id, label: shortLabel(r.label), loose: r.kind !== "study" && !r.run_id }))].map((item, i, all) => <span key={item.id} className="flex items-center gap-1">
							{item.loose && !all[i - 1]?.loose && <span aria-hidden className="mx-2 h-4 border-l border-rule" />}
							<button type="button" title={runs.find(r => r.id === item.id)?.label} onClick={() => select(item.id)} aria-current={(current?.run.id ?? OVERVIEW) === item.id ? "page" : undefined}
								className={cn("rounded-md px-2 py-1 text-[0.8rem] transition-colors focus-visible:outline-2 focus-visible:outline-accent", (current?.run.id ?? OVERVIEW) === item.id ? "bg-ink text-paper" : "text-muted hover:bg-wash hover:text-ink", item.id !== OVERVIEW && "num", item.loose && (current?.run.id ?? OVERVIEW) !== item.id && "text-faint")}>
								{item.label}
							</button>
						</span>)}
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
			</div>
		</header>
		<main className="mx-auto max-w-6xl px-4 py-8 sm:px-8">
			{list.state === "loading" && <p className="text-sm text-muted">Reading the configured roots.</p>}
			{list.state === "error" && <Unavailable tone="error" reason={`The dashboard API did not answer: ${list.message}. Start it with just dashboard.`} />}
			{list.state === "ready" && runs.length === 0 && <Unavailable reason="No indexed runs and no recognized artifacts under the configured roots. Set DASHBOARD_ROOTS or export a reading." />}
			{list.state === "ready" && runs.length > 0 && (current ? <div className="space-y-6"><RootWarnings list={list.data} /><RunPage key={current.run.id} entry={current} attacks={attacks} /></div> : <Overview list={list.data} loaded={loaded} onSelect={select} />)}
		</main>
	</div>;
}
