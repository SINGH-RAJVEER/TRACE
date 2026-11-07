import { STRATUM_LABELS, strataByGroup } from "../data";
import type { Failed, Section as SectionData, UtilityArm } from "../types";
import { Facts, Hash, Section, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

function Validity({ arm }: { arm: UtilityArm }) {
	const variant = arm.validity === "valid" ? "pass" : arm.validity === "invalid" ? "fail" : "outline";
	return <span className="inline-flex flex-col gap-1">
		<Badge variant={variant}>{arm.validity ?? "unknown"} baseline</Badge>
		{arm.validity_reason && <span className="max-w-72 text-xs text-muted">{arm.validity_reason}</span>}
	</span>;
}

export function UtilityView({ data }: { data: SectionData<UtilityArm> | Failed }) {
	if (data.state === "error") return <Unavailable reason={data.reason} tone="error" />;
	if (data.state === "unavailable") return <Section title="Legitimate-user utility"><Unavailable reason={data.reason} /></Section>;
	const arms = data.items;
	const groups = [...new Set(arms.flatMap(arm => strataByGroup(arm).map(g => g.group)))];
	return <div className="space-y-8">
		<Section
			title="Legitimate-user utility, pooled"
			kicker="The held-out benign workload replayed through each arm. Completion is served over submitted; served-only quality scores answered requests; unconditional utility counts unanswered requests as failures. Latency covers served requests only."
		>
			<Table>
				<TableHeader><TableRow>
					<TableHead>Arm</TableHead><TableHead>Policy</TableHead><TableHead>Comparison validity</TableHead>
					<TableHead className="text-right">Submitted</TableHead><TableHead className="text-right">Served</TableHead><TableHead className="text-right">Completion</TableHead>
					<TableHead className="text-right">Served-only quality</TableHead><TableHead className="text-right">Unconditional utility</TableHead>
					<TableHead className="text-right">p50</TableHead><TableHead className="text-right">p95</TableHead><TableHead className="text-right">p99</TableHead>
				</TableRow></TableHeader>
				<TableBody>{arms.map(arm => <TableRow key={arm.arm}>
					<TableCell className="font-medium">{arm.arm}</TableCell>
					<TableCell><Value value={[arm.policy, arm.accumulator].filter(Boolean).join(" / ") || null} /></TableCell>
					<TableCell><Validity arm={arm} /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.submitted} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.served} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.completion_rate} as="fraction" /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.served_only_quality} as="fraction" /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.unconditional_utility} as="fraction" /></TableCell>
					<TableCell className="text-right"><Value value={arm.detail?.pooled.latency_p50} as="seconds" /></TableCell>
					<TableCell className="text-right"><Value value={arm.pooled?.latency_p95} as="seconds" /></TableCell>
					<TableCell className="text-right"><Value value={arm.detail?.pooled.latency_p99} as="seconds" /></TableCell>
				</TableRow>)}</TableBody>
				<TableCaption>
					{arms.some(a => a.source === "directory") ? "Read from utility.json in the run directory; the index does not hold these rows yet. " : "From the results index. "}
					p50 and p99 come from utility.json and are not recorded where only the index is available. Unknown validity means no teacher capacity preflight was bound to the run; it is not evidence of overload.
				</TableCaption>
			</Table>
		</Section>
		{groups.map(group => {
			const values = [...new Set(arms.flatMap(arm => arm.strata.filter(s => s.group === group).map(s => s.value)))].sort();
			return <Section key={group} title={`By ${STRATUM_LABELS[group]?.toLowerCase() ?? group}`} kicker="Unconditional utility, completion and p95 latency per arm. A group can lose service while the pooled value looks acceptable.">
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead rowSpan={2}>{STRATUM_LABELS[group] ?? group}</TableHead>
							{arms.map(arm => <TableHead key={arm.arm} colSpan={4} className="border-l border-rule text-center">{arm.arm}</TableHead>)}
						</TableRow>
						<TableRow>{arms.flatMap(arm => [
							<TableHead key={`${arm.arm}-n`} className="border-l border-rule text-right">Submitted</TableHead>,
							<TableHead key={`${arm.arm}-c`} className="text-right">Completion</TableHead>,
							<TableHead key={`${arm.arm}-u`} className="text-right">Utility</TableHead>,
							<TableHead key={`${arm.arm}-p`} className="text-right">p95</TableHead>,
						])}</TableRow>
					</TableHeader>
					<TableBody>{values.map(value => <TableRow key={value}>
						<TableCell className="num">{value}</TableCell>
						{arms.flatMap(arm => {
							const row = arm.strata.find(s => s.group === group && s.value === value);
							return [
								<TableCell key={`${arm.arm}-n`} className="border-l border-rule text-right"><Value value={row?.submitted} as="int" /></TableCell>,
								<TableCell key={`${arm.arm}-c`} className="text-right"><Value value={row?.completion_rate} as="fraction" /></TableCell>,
								<TableCell key={`${arm.arm}-u`} className="text-right"><Value value={row?.unconditional_utility} as="fraction" /></TableCell>,
								<TableCell key={`${arm.arm}-p`} className="text-right"><Value value={row?.latency_p95} as="seconds" /></TableCell>,
							];
						})}
					</TableRow>)}</TableBody>
				</Table>
			</Section>;
		})}
		<Section title="Measurement details" kicker="Load settings and bindings recorded with each arm's utility file.">
			<div className="grid gap-6 lg:grid-cols-3">
				{arms.map(arm => <div key={arm.arm} className="space-y-2">
					<h4 className="text-sm font-medium">{arm.arm}</h4>
					<Facts items={[
						["Measurement", <Value value={arm.detail?.measurement_version} />],
						["Concurrency", <Value value={arm.detail?.load?.concurrency} />],
						["Deadline", <Value value={arm.detail?.load?.deadline_seconds} as="seconds" />],
						["Throughput", <Value value={arm.detail?.pooled.throughput} as="rate" />],
						["Preflight", <Hash value={arm.preflight_sha256} />],
						["Completion floor", <Value value={arm.min_completion_rate} as="fraction" />],
						["TTFT", <Value value={null} unknown={arm.detail?.ttft_reason ?? "not recorded"} />],
						["File", <Value value={arm.file ? `${arm.file.path} (${arm.file.state})` : null} unknown="utility.json not found" />],
					]} />
				</div>)}
			</div>
		</Section>
	</div>;
}
