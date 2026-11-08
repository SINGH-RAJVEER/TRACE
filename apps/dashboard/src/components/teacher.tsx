import type { Capacity, Failed, GateReport, Provenance, RunDetail, ScoreFile } from "../types";
import { Facts, Hash, Reasons, Section, Source, Unavailable, Value, Verdict } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

function Headroom({ gate }: { gate: GateReport }) {
	const results = gate.results;
	const headroom = results?.headroom;
	return <div className="space-y-3">
		<div className="flex flex-wrap items-center gap-2">
			<Verdict passed={gate.passed} status={gate.status} />
			<Badge variant={gate.integrity === "intact" ? "outline" : "fail"}>{gate.integrity ?? "not checked"}</Badge>
			<Badge>{gate.task ?? "task not recorded"}</Badge>
		</div>
		<Source place={gate.place} file={gate.file} />
		<Facts items={[
			["Teacher", <Value value={gate.teacher ? `${gate.teacher.model} @ ${gate.teacher.revision}` : null} />],
			["Base", <Value value={gate.base ? `${gate.base.model} @ ${gate.base.revision}` : null} />],
			["Teacher accuracy", <Value value={results?.teacher_accuracy} as="fraction" />],
			["Base accuracy", <Value value={results?.base_accuracy} as="fraction" />],
			["Headroom", <span><Value value={headroom?.gain_pp} as="pp" /> {headroom && <span className="num text-muted">95% [{headroom.ci95_pp[0].toFixed(1)}, {headroom.ci95_pp[1].toFixed(1)}]</span>}</span>],
			["Declared margin", <Value value={gate.parameters?.margin_points as number | undefined} as="pp" />],
			["Problems / families", <Value value={gate.items !== null && gate.families !== null ? `${gate.items} / ${gate.families}` : null} />],
			["Suite", <Hash value={gate.suite_sha256} />],
		]} />
		<Reasons reasons={gate.reasons} />
		{gate.action && <p className="text-sm text-muted">{gate.action}</p>}
	</div>;
}

function CapacityReport({ report }: { report: Capacity }) {
	return <div className="space-y-3">
		<div className="flex flex-wrap items-center gap-2"><Verdict passed={report.passed} /><span className="text-sm text-muted">against the declared limits</span></div>
		<Source place={report.place} file={report.file} />
		<Facts items={[
			["Offered / completed", <Value value={report.offered !== null && report.completed !== null ? `${report.offered} / ${report.completed}` : null} />],
			["Errors / timeouts", <Value value={report.errors !== null && report.timeouts !== null ? `${report.errors} / ${report.timeouts}` : null} />],
			["Offered rate", <Value value={report.offered_rps} as="rate" />],
			["Completed rate", <Value value={report.completed_rps} as="rate" />],
			["p50 / p95 / p99", <span className="num">{[report.latency_p50_seconds, report.latency_p95_seconds, report.latency_p99_seconds].map(v => v === null ? "?" : v.toFixed(1)).join(" / ")} s</span>],
			["Queue mid / end", <Value value={report.queued_mid !== null && report.queued_at_end !== null ? `${report.queued_mid} / ${report.queued_at_end}` : null} />],
			["Concurrency", <Value value={report.concurrency} />],
			["Timeout", <Value value={report.timeout_seconds} as="seconds" />],
			["Defended responses", <Value value={report.defended_responses} as="int" />],
			["Schedule", <Hash value={report.schedule_sha256} />],
		]} />
		{report.checks && <div className="flex flex-wrap gap-2">{Object.entries(report.checks).map(([name, ok]) => <Badge key={name} variant={ok ? "outline" : "fail"}>{name}: {ok ? "met" : "not met"}</Badge>)}</div>}
	</div>;
}

function Scores({ items }: { items: ScoreFile[] }) {
	return <Table>
		<TableHeader><TableRow><TableHead>Role</TableHead><TableHead>Model</TableHead><TableHead>Revision</TableHead><TableHead>Task</TableHead><TableHead className="text-right">Problems</TableHead><TableHead className="text-right">Share correct</TableHead><TableHead>Suite</TableHead><TableHead>File</TableHead></TableRow></TableHeader>
		<TableBody>{items.map(s => <TableRow key={s.place.path}>
			<TableCell><Value value={s.role} /></TableCell><TableCell><Value value={s.model} /></TableCell><TableCell><Hash value={s.revision} /></TableCell>
			<TableCell><Value value={s.task} /></TableCell><TableCell className="text-right"><Value value={s.items} as="int" /></TableCell>
			<TableCell className="text-right"><Value value={s.accuracy} as="fraction" /></TableCell><TableCell><Hash value={s.suite_sha256} /></TableCell>
			<TableCell className="num text-xs text-muted">{s.place.path}{s.file.state !== "ok" && ` (${s.file.state})`}</TableCell>
		</TableRow>)}</TableBody>
	</Table>;
}

export function TeacherView({ data, provenance, hideMissing = false }: { data: RunDetail["teacher"]; provenance: Provenance | Failed; hideMissing?: boolean }) {
	if ("state" in data) return <Unavailable reason={data.reason} tone="error" />;
	const configured = "state" in provenance ? [] : provenance.sources.filter(s => s.teacher.TEACHER_MODEL || s.teacher.TEACHER_UPSTREAM_MODEL);
	return <div className="space-y-8">
		<Section title="Teacher model" kicker="The teacher as each run configured it. Revisions are pinned hashes; a quantized or GGUF teacher is a separate model whose results are not pooled with others.">
			{configured.length ? <Table>
				<TableHeader><TableRow><TableHead>Source</TableHead><TableHead>Model</TableHead><TableHead>Revision</TableHead><TableHead>Upstream model</TableHead><TableHead>GGUF file</TableHead></TableRow></TableHeader>
				<TableBody>{configured.map(s => <TableRow key={`${s.root}/${s.path}`}>
					<TableCell className="num text-xs">{s.root}/{s.path}</TableCell>
					<TableCell><Value value={s.teacher.TEACHER_MODEL} /></TableCell>
					<TableCell><Hash value={s.teacher.TEACHER_MODEL_REVISION} /></TableCell>
					<TableCell><Value value={s.teacher.TEACHER_UPSTREAM_MODEL} /></TableCell>
					<TableCell><Value value={s.teacher.TEACHER_GGUF} /></TableCell>
				</TableRow>)}</TableBody>
			</Table> : <Unavailable reason="No run configuration in this run's directories names a teacher model." />}
		</Section>
		{!hideMissing && <Section title="Task headroom over the student base" kicker="Teacher and pinned base scored on the same held-out problems. The gate passes only when the gain meets the declared margin and its interval excludes zero. This is task capability.">
			{data.headroom.state === "available" ? <div className="space-y-6">{data.headroom.items.map(g => <Headroom key={g.place.path} gate={g} />)}</div> : <Unavailable reason={data.headroom.reason} />}
		</Section>}
		{!hideMissing && <Section title="Serving capacity preflight" kicker="Throughput, latency, errors and queueing of the undefended teacher on the frozen benign schedule. It says whether the teacher could serve the workload, not whether it answers correctly.">
			{data.capacity.state === "available" ? <div className="space-y-6">{data.capacity.items.map(c => <CapacityReport key={c.place.path} report={c} />)}</div> : <Unavailable reason={data.capacity.reason} />}
		</Section>}
		{!hideMissing && <Section title="Score files" kicker="Per-problem correctness files that feed the headroom gate.">
			{data.scores.state === "available" ? <Scores items={data.scores.items} /> : <Unavailable reason={data.scores.reason} />}
		</Section>}
	</div>;
}
