import { useState } from "react";
import { STRATUM_LABELS, armColor, format, isNumber, seqStep, sortArms, strataByGroup } from "../data";
import type { Failed, Maybe, Section as SectionData, UtilityArm, UtilityMetrics } from "../types";
import { cn } from "../lib";
import { ArmLegend, ArmName, HBar, Legend, More, Panel, Tip, TipRows } from "./charts";
import { Facts, Hash, Unavailable, Value } from "./evidence";
import { Badge } from "./ui/badge";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";

const RATES = [
	{ key: "completion_rate", label: "Completion", note: "served / submitted" },
	{ key: "served_only_quality", label: "Served-only quality", note: "score of answered requests" },
	{ key: "unconditional_utility", label: "Unconditional utility", note: "unanswered count as failures" },
] as const;

type RateKey = (typeof RATES)[number]["key"];

function Validity({ arm }: { arm: UtilityArm }) {
	const variant = arm.validity === "valid" ? "pass" : arm.validity === "invalid" ? "fail" : "outline";
	return <span className="inline-flex flex-col gap-1">
		<Badge variant={variant}>{arm.validity ?? "unknown"}</Badge>
		{arm.validity_reason && <span className="max-w-72 text-xs text-muted">{arm.validity_reason}</span>}
	</span>;
}

function PooledRates({ arms }: { arms: UtilityArm[] }) {
	return <div className="grid gap-x-10 gap-y-6 md:grid-cols-3">
		{RATES.map(rate => <div key={rate.key} className="min-w-0 space-y-2">
			<div className="text-xs"><span className="font-semibold text-ink">{rate.label}</span> <span className="text-faint">{rate.note}</span></div>
			{arms.map(arm => {
				const value = arm.pooled?.[rate.key] ?? null;
				return <div key={arm.arm} className="grid grid-cols-[6.5rem_1fr] items-center gap-3 text-xs">
					<span className="truncate text-muted"><ArmName arm={arm.arm} /></span>
					<HBar value={value} max={1} color={armColor(arm.arm)} label={format(value, "fraction")}
						tip={<TipRows title={`${arm.arm}: ${rate.label}`} rows={[[rate.label, format(value, "fraction") ?? "not recorded"], ["Submitted", format(arm.pooled?.submitted, "int") ?? "not recorded"], ["Served", format(arm.pooled?.served, "int") ?? "not recorded"]]} />} />
				</div>;
			})}
			<div className="grid grid-cols-[6.5rem_1fr] gap-3 text-[0.65rem] text-faint"><span /><span className="flex justify-between"><span>0%</span><span>100%</span></span></div>
		</div>)}
	</div>;
}

function Latency({ arms }: { arms: UtilityArm[] }) {
	const values = arms.flatMap(arm => [arm.detail?.pooled.latency_p50, arm.pooled?.latency_p95, arm.detail?.pooled.latency_p99]).filter(isNumber);
	const deadlines = arms.map(arm => arm.detail?.load?.deadline_seconds).filter(isNumber);
	const max = Math.max(1, ...values, ...deadlines) * 1.03;
	const at = (v: number) => `${(v / max) * 100}%`;
	return <div className="space-y-2">
		{arms.map(arm => {
			const p50 = arm.detail?.pooled.latency_p50 ?? null;
			const p95 = arm.pooled?.latency_p95 ?? null;
			const p99 = arm.detail?.pooled.latency_p99 ?? null;
			const deadline = arm.detail?.load?.deadline_seconds;
			const known = [p50, p95, p99].filter(isNumber);
			const tip = <TipRows title={`${arm.arm}: served-request latency`} rows={[["p50", format(p50, "seconds") ?? "not recorded"], ["p95", format(p95, "seconds") ?? "not recorded"], ["p99", format(p99, "seconds") ?? "not recorded"], ["Deadline", format(deadline, "seconds") ?? "not recorded"]]} />;
			return <div key={arm.arm} className="grid grid-cols-[6.5rem_1fr_9rem] items-center gap-3 text-xs">
				<span className="truncate text-muted"><ArmName arm={arm.arm} /></span>
				<Tip content={tip} className="w-full">
					<span className="relative flex h-5 w-full items-center">
						<span className="absolute inset-x-0 top-1/2 h-px bg-rule" />
						{isNumber(deadline) && <span className="absolute top-0 h-full border-l border-dashed border-faint" style={{ left: at(deadline) }} />}
						{known.length >= 2 && <span className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full opacity-40" style={{ left: at(Math.min(...known)), width: `${((Math.max(...known) - Math.min(...known)) / max) * 100}%`, background: armColor(arm.arm) }} />}
						{([["p50", p50, "size-2.5"], ["p95", p95, "size-3"], ["p99", p99, "size-2.5"]] as const).map(([name, v, size]) => isNumber(v) && <span key={name} aria-hidden className={cn("absolute -translate-x-1/2 rounded-full border-2 border-paper", size)} style={{ left: at(v), background: name === "p95" ? armColor(arm.arm) : "var(--ink)" }} />)}
					</span>
				</Tip>
				<span className="num text-right">
					{format(p95, "seconds") ?? "?"} p95
					{(p50 === null || p99 === null) && <span className="unrecorded ml-1">p50/p99 missing</span>}
				</span>
			</div>;
		})}
		<div className="grid grid-cols-[6.5rem_1fr_9rem] gap-3 text-[0.65rem] text-faint"><span /><span className="flex justify-between"><span>0 s</span><span>{format(max, "seconds")}</span></span><span /></div>
	</div>;
}

function Heatmap({ arms }: { arms: UtilityArm[] }) {
	const [metric, setMetric] = useState<RateKey>("completion_rate");
	const groups = [...new Set(arms.flatMap(arm => strataByGroup(arm).map(g => g.group)))];
	if (!groups.length) return <Unavailable reason="No stratum rows were recorded for any arm." />;
	const label = RATES.find(r => r.key === metric)!.label;
	return <div className="space-y-3">
		<div className="flex flex-wrap items-center justify-between gap-3">
			<Tabs value={metric} onValueChange={v => setMetric(v as RateKey)}>
				<TabsList className="border-b-0">{RATES.map(r => <TabsTrigger key={r.key} value={r.key} className="py-1 text-xs">{r.label}</TabsTrigger>)}</TabsList>
			</Tabs>
			<div className="flex items-center gap-2 text-[0.65rem] text-faint">
				<span>0%</span>
				<span className="flex h-2.5 w-32 overflow-hidden rounded-[3px]">{Array.from({ length: 8 }, (_, i) => <span key={i} className="h-full flex-1" style={{ background: `var(--seq-${i})` }} />)}</span>
				<span>100%</span>
				<span className="ml-2 inline-block h-2.5 w-5 rounded-[3px] hatched" /> <span>not recorded</span>
			</div>
		</div>
		<div className="overflow-x-auto">
			<table className="border-separate border-spacing-[2px] text-xs">
				<thead><tr>
					<th className="w-40" />
					{arms.map(arm => <th key={arm.arm} className="px-2 pb-1 text-left font-medium text-muted"><ArmName arm={arm.arm} /></th>)}
				</tr></thead>
				{groups.map(group => {
					const values = [...new Set(arms.flatMap(arm => arm.strata.filter(s => s.group === group).map(s => s.value)))].sort();
					return <tbody key={group}>
						<tr><th colSpan={arms.length + 1} className="pt-3 pb-1 text-left text-[0.68rem] font-semibold tracking-wide text-faint uppercase">{STRATUM_LABELS[group] ?? group}</th></tr>
						{values.map(value => <tr key={value}>
							<th className="num pr-3 text-left font-normal text-muted">{value}</th>
							{arms.map(arm => {
								const row = arm.strata.find(s => s.group === group && s.value === value) as (UtilityMetrics & { value: string }) | undefined;
								const v: Maybe<number> = row ? row[metric] : null;
								const step = seqStep(v);
								const tip = <TipRows title={`${arm.arm} · ${value}`} rows={[["Submitted", format(row?.submitted, "int") ?? "not recorded"], ["Served", format(row?.served, "int") ?? "not recorded"], ...RATES.map(r => [r.label, format(row?.[r.key], "fraction") ?? "not recorded"] as [string, string]), ["p95 latency", format(row?.latency_p95, "seconds") ?? "not recorded"]]} />;
								return <td key={arm.arm} className="p-0">
									<Tip content={tip} className="w-full">
										{step === null
											? <span className="hatched flex h-8 w-full min-w-28 items-center rounded-[3px] px-2 text-[0.68rem] text-muted">not recorded</span>
											: <span className="num flex h-8 w-full min-w-28 items-center rounded-[3px] px-2" style={{ background: `var(--seq-${step})`, color: step >= 5 ? "var(--on-seq-high)" : step === 4 ? "var(--on-seq-4)" : "var(--on-seq-low)" }}>{format(v, "fraction")}</span>}
									</Tip>
								</td>;
							})}
						</tr>)}
					</tbody>;
				})}
			</table>
		</div>
		<p className="text-xs text-muted">{label} per stratum. A group can lose service while the pooled value looks acceptable; darker cells are higher.</p>
	</div>;
}

export function UtilityView({ data }: { data: SectionData<UtilityArm> | Failed }) {
	if (data.state === "error") return <Unavailable reason={data.reason} tone="error" />;
	if (data.state === "unavailable") return <Unavailable reason={data.reason} />;
	const order = sortArms(data.items.map(a => a.arm));
	const arms = [...data.items].sort((a, b) => order.indexOf(a.arm) - order.indexOf(b.arm));
	const fromDirectory = arms.some(a => a.source === "directory");
	return <div className="space-y-10">
		<div className="flex flex-wrap items-center justify-between gap-3">
			<ArmLegend arms={order} />
			<div className="flex flex-wrap gap-2">
				{fromDirectory && <Badge variant="caution">read from utility.json; not yet in the index</Badge>}
				{arms.every(a => a.validity !== "valid" && a.validity !== "invalid") && <Badge>comparison validity unknown for every arm</Badge>}
			</div>
		</div>

		<Panel title="Pooled service for legitimate users" note="The held-out benign workload replayed through each arm. All three rates share a 0 to 100% scale.">
			<PooledRates arms={arms} />
		</Panel>

		<Panel title="Latency of served requests" note="p50, p95 and p99 in seconds; the larger dot is p95 and the dashed line is the client deadline. p50 and p99 come only from utility.json.">
			<Legend items={[{ label: "p95, in the arm color", color: "var(--faint)" }, { label: "p50 and p99", color: "var(--ink)" }, { label: "deadline", outline: true }]} />
			<Latency arms={arms} />
		</Panel>

		<Panel title="By domain, profile and client type">
			<Heatmap arms={arms} />
		</Panel>

		<div className="space-y-3">
			<More label="Pooled values and validity">
				<Table>
					<TableHeader><TableRow>
						<TableHead>Arm</TableHead><TableHead>Policy</TableHead><TableHead>Comparison validity</TableHead>
						<TableHead className="text-right">Submitted</TableHead><TableHead className="text-right">Served</TableHead><TableHead className="text-right">Completion</TableHead>
						<TableHead className="text-right">Served-only quality</TableHead><TableHead className="text-right">Unconditional utility</TableHead>
						<TableHead className="text-right">p50</TableHead><TableHead className="text-right">p95</TableHead><TableHead className="text-right">p99</TableHead>
					</TableRow></TableHeader>
					<TableBody>{arms.map(arm => <TableRow key={arm.arm}>
						<TableCell className="font-medium"><ArmName arm={arm.arm} /></TableCell>
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
						{fromDirectory ? "Read from utility.json in the run directory; the index does not hold these rows yet. " : "From the results index. "}
						Unknown validity means no teacher capacity preflight was bound to the run; it is not evidence of overload.
					</TableCaption>
				</Table>
			</More>
			<More label="Measurement settings">
				<div className="grid gap-8 lg:grid-cols-3">
					{arms.map(arm => <div key={arm.arm} className="space-y-2">
						<h4 className="text-sm font-medium"><ArmName arm={arm.arm} /></h4>
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
			</More>
		</div>
	</div>;
}
