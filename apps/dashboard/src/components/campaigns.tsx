import type { CampaignItem, Failed, Section as SectionData } from "../types";
import { Hash, Section, Status, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

const LEVELS = [1, 2, 3, 4];

export function CampaignsView({ data }: { data: SectionData<CampaignItem> | Failed }) {
	if (data.state === "error") return <Unavailable reason={data.reason} tone="error" />;
	const rows = data.items;
	return <div className="space-y-8">
		<Section
			title="What each campaign acquired"
			kicker="Requests the teacher answered for the attacker's accounts and when the policy intervened. These are acquisition counts, not student or model accuracy, and they do not show that distillation was prevented."
			aside={rows.some(r => r.source === "directory") ? <Badge variant="caution">read from run directories; not yet in the index</Badge> : undefined}
		>
			{data.state === "unavailable" ? <Unavailable reason={data.reason} /> : <Table>
				<TableHeader><TableRow>
					<TableHead>Attack</TableHead><TableHead>Arm</TableHead><TableHead>Policy</TableHead><TableHead>Status</TableHead>
					<TableHead className="text-right">Attempts</TableHead><TableHead className="text-right">Received</TableHead><TableHead className="text-right">Blocked</TableHead>
					<TableHead className="text-right">Challenged</TableHead><TableHead className="text-right">Failed</TableHead><TableHead className="text-right">Released tokens</TableHead>
					<TableHead className="text-right">Distinct examples</TableHead><TableHead className="text-right">First alert</TableHead><TableHead className="text-right">First restriction</TableHead>
					<TableHead className="text-right">First refusal</TableHead><TableHead>Attribution</TableHead>
				</TableRow></TableHeader>
				<TableBody>{rows.map(row => <TableRow key={`${row.attack}/${row.arm}/${row.stratum}`}>
					<TableCell className="num">{row.attack}</TableCell>
					<TableCell>{row.arm}</TableCell>
					<TableCell><Value value={[row.policy, row.accumulator].filter(Boolean).join(" / ") || null} /></TableCell>
					<TableCell>
						<Status status={row.status} />
						{row.status !== "complete" && row.outcome && <p className="mt-1 max-w-56 text-xs text-muted">{row.outcome}</p>}
					</TableCell>
					<TableCell className="text-right"><Value value={row.attempts} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={row.received} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={row.blocked} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={row.challenge} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={row.failed} as="int" /></TableCell>
					<TableCell className="text-right">
						<Value value={row.released_tokens} as="int" />
						{typeof row.released_tokens_unknown === "number" && row.released_tokens_unknown > 0 && <p className="text-xs text-caution">{row.released_tokens_unknown} usages unknown</p>}
					</TableCell>
					<TableCell className="text-right"><Value value={row.distinct_examples} as="int" /></TableCell>
					<TableCell className="text-right"><Value value={row.first_alert_seconds} as="seconds" unknown="none observed" /></TableCell>
					<TableCell className="text-right"><Value value={row.first_restriction_seconds} as="seconds" unknown="none observed" /></TableCell>
					<TableCell className="text-right"><Value value={row.first_refusal_seconds} as="seconds" unknown="none observed" /></TableCell>
					<TableCell>
						{row.attribution_complete === false ? <Badge variant="caution">incomplete · {row.unattributed_queries ?? "?"} unattributed</Badge> : <Value value={row.attribution_complete === true ? "complete" : null} />}
					</TableCell>
				</TableRow>)}</TableBody>
				<TableCaption>Empty intervention cells mean none was observed only when attribution is complete; otherwise the time is unknown. Gate-refused arms sent no requests and keep their row.</TableCaption>
			</Table>}
			{data.errors && data.errors.length > 0 && <div className="space-y-1">{data.errors.map(e => <Unavailable key={`${e.attack}/${e.arm}`} tone="error" reason={`${e.attack}/${e.arm}: ${e.error}`} />)}</div>}
		</Section>
		{rows.length > 0 && <Section title="Escalation levels and identifiers" kicker="J is the first attempt index whose account reached the level; Q is received responses completed strictly before that time. A level with no value is unreached only when attribution is complete; otherwise it is unknown.">
			<Table>
				<TableHeader><TableRow>
					<TableHead>Attack</TableHead><TableHead>Arm</TableHead>
					{LEVELS.map(level => <TableHead key={level} className="text-right">L{level} s / J / Q</TableHead>)}
					<TableHead>Seed</TableHead><TableHead>Budget</TableHead><TableHead>Partition</TableHead><TableHead>Study</TableHead><TableHead>Reading label</TableHead><TableHead>Contract</TableHead><TableHead>Ledger</TableHead>
				</TableRow></TableHeader>
				<TableBody>{rows.map(row => <TableRow key={`${row.attack}/${row.arm}`}>
					<TableCell className="num">{row.attack}</TableCell><TableCell>{row.arm}</TableCell>
					{LEVELS.map(level => <TableCell key={level} className="text-right whitespace-nowrap">
						{row[`L${level}_J`] === null ? <Value value={null} unknown={row.attribution_complete === true ? "unreached" : "unknown"} /> : <span className="num"><Value value={row[`L${level}_seconds`]} as="seconds" /> · {String(row[`L${level}_J`])} · {String(row[`L${level}_Q`] ?? "?")}</span>}
					</TableCell>)}
					<TableCell><Value value={row.seed} /></TableCell>
					<TableCell><Value value={row.attempt_budget} as="int" /></TableCell>
					<TableCell><Value value={row.partition} /></TableCell>
					<TableCell><Value value={row.study} /></TableCell>
					<TableCell><Value value={row.reading_label} /></TableCell>
					<TableCell><Hash value={row.contract_sha256 as string | null} /></TableCell>
					<TableCell><Hash value={row.ledger_sha256 as string | null} /></TableCell>
				</TableRow>)}</TableBody>
			</Table>
		</Section>}
	</div>;
}
