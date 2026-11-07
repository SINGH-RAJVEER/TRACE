import { studyLabel } from "../data";
import type { Failed, Provenance } from "../types";
import { Facts, Hash, Section, Source, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

export function ProvenanceView({ data }: { data: Provenance | Failed }) {
	if ("state" in data) return <Unavailable reason={data.reason} tone="error" />;
	return <div className="space-y-8">
		<Section title="Index labels" kicker="What readings/index.tsv records for this run. Study labels come from the campaign rows; utility rows written before study labels existed show none.">
			<Facts items={[
				["Run ID", <Value value={data.run_id} unknown="not in the index" />],
				["Readings", <Value value={data.readings.join(", ") || null} />],
				["Study", <Value value={studyLabel(data.studies)} />],
				["Commits", data.commits.length ? <span className="flex flex-wrap gap-2">{data.commits.map(c => <Hash key={c} value={c} />)}</span> : <Value value={null} unknown="no commit recorded" />],
				["Seeds", <Value value={data.seeds.join(", ") || null} />],
				["Attacker partition", <Value value={data.partitions.join(", ") || null} />],
			]} />
			{Object.keys(data.statuses).length > 0 && <div className="flex flex-wrap gap-2">
				{Object.entries(data.statuses).map(([key, count]) => <Badge key={key}>{key} · {count}</Badge>)}
			</div>}
		</Section>
		{data.sources.length === 0 && <Section title="Run directories"><Unavailable reason="No directory under the configured roots matches this run's ID or reading name." /></Section>}
		{data.sources.map(source => <Section key={`${source.root}/${source.path}`} title={`${source.root}/${source.path}`} kicker="Files the runner wrote beside its results. Settings come from the most recent provenance attempt; values whose names suggest credentials are redacted by the API.">
			<Facts items={[
				["Exit", <Value value={source.exit} />],
				["Exported", <Value value={source.export?.exported_at as string | undefined} unknown={source.export_file.state === "missing" ? "no export.json" : "not recorded"} />],
				["Export commit", <Hash value={source.export?.commit as string | undefined} />],
				["Checksummed files", <Value value={source.checksums} as="int" unknown="no SHA256SUMS" />],
				["Provenance attempts", <Value value={source.attempts.join(", ") || null} />],
				["Settings from", <Value value={source.settings_attempt} />],
			]} />
			<div className="grid gap-6 lg:grid-cols-2">
				<div>
					<h4 className="mb-2 text-sm font-medium">Teacher identity as configured</h4>
					<Facts items={Object.entries(source.teacher).map(([key, value]) => [key, <Value value={value} />])} />
					<p className="mt-2 text-xs text-muted">Quantization and serving stack are shown only as the configuration wrote them. A GGUF teacher is a different model from the pinned vLLM teacher and is not directly comparable.</p>
				</div>
				<div>
					<h4 className="mb-2 text-sm font-medium">Study settings</h4>
					<Facts items={Object.entries(source.study_settings).map(([key, value]) => [key, <Value value={value} />])} />
				</div>
			</div>
			{source.settings.length > 0 && <details className="text-sm">
				<summary className="cursor-pointer text-muted">All {source.settings.length} configuration values</summary>
				<Table className="mt-2"><TableHeader><TableRow><TableHead>Key</TableHead><TableHead>Value</TableHead></TableRow></TableHeader>
					<TableBody>{source.settings.map(pair => <TableRow key={pair.key}><TableCell className="num">{pair.key}</TableCell><TableCell><Value value={pair.value || null} unknown="empty" /></TableCell></TableRow>)}</TableBody>
				</Table>
			</details>}
			{Object.entries(source.texts).map(([name, text]) => text && <details key={name} className="text-sm">
				<summary className="cursor-pointer text-muted">{name}.txt</summary>
				<pre className="prose-pre mt-2 border-l-2 border-rule pl-3">{text}</pre>
			</details>)}
			{source.readme ? <details className="text-sm" open>
				<summary className="cursor-pointer text-muted">README.md{source.readme_truncated ? ` (${source.readme_truncated})` : ""}</summary>
				<pre className="prose-pre mt-2 max-h-[32rem] overflow-y-auto border-l-2 border-rule pl-3">{source.readme}</pre>
			</details> : <Source place={source} />}
		</Section>)}
	</div>;
}
