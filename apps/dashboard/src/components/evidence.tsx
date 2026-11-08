import * as React from "react";
import { Circle, CircleDot, OctagonX, TriangleAlert } from "lucide-react";
import { format, shortHash, verdict } from "../data";
import type { FileState, Maybe, Place } from "../types";
import { cn } from "../lib";
import { Badge } from "./ui/badge";

type As = Parameters<typeof format>[1];

/** A recorded value in mono, or the hatched "not recorded" mark. Zero and false are values. */
export function Value({ value, as, unknown = "not recorded", className }: { value: unknown; as?: As; unknown?: string; className?: string }) {
	const text = format(value, as);
	if (text === null) return <span className="unrecorded" title="No value was recorded for this field">{unknown}</span>;
	return <span className={cn("num", className)}>{text}</span>;
}

export function Hash({ value }: { value: Maybe<string> | undefined }) {
	const text = shortHash(value);
	return text ? <span className="num text-muted" title={value ?? undefined}>{text}</span> : <Value value={null} />;
}

/** Pass, fail, development-only or unknown, always with a word beside the color. */
export function Verdict({ passed, status }: { passed: Maybe<boolean> | undefined; status?: Maybe<string> }) {
	const result = verdict(passed, status);
	if (result === "pass") return <Badge variant="pass"><CircleDot size={11} />{status ?? "passed"}</Badge>;
	if (result === "fail") return <Badge variant="fail"><OctagonX size={11} />{status ?? "failed"}</Badge>;
	if (result === "development") return <Badge variant="caution"><TriangleAlert size={11} />development only</Badge>;
	return <Badge><Circle size={11} />{status ?? "unknown"}</Badge>;
}

export function Status({ status }: { status: Maybe<string> | undefined }) {
	if (!status) return <Value value={null} />;
	const variant = status === "complete" ? "outline" : status === "gate-refused" || status === "interrupted" || status === "not-run" ? "caution" : "outline";
	return <Badge variant={variant}>{status}</Badge>;
}

export function Section({ title, kicker, children, aside }: { title: string; kicker?: React.ReactNode; aside?: React.ReactNode; children: React.ReactNode }) {
	return <section className="space-y-3 border-t border-rule pt-5 first:border-t-0 first:pt-0">
		<header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
			<div>
				<h3 className="display text-lg">{title}</h3>
				{kicker && <p className="max-w-3xl text-sm text-muted">{kicker}</p>}
			</div>
			{aside}
		</header>
		{children}
	</section>;
}

/** An explicit unavailable state with the reason the API gave. */
export function Unavailable({ reason, tone = "neutral" }: { reason: Maybe<string> | undefined; tone?: "neutral" | "error" }) {
	return <p className={cn("flex items-start gap-2 text-sm", tone === "error" ? "text-fail" : "text-muted")}>
		<span className="unrecorded mt-0.5">{tone === "error" ? "error" : "unavailable"}</span>
		<span>{reason ?? "No artifact was found."}</span>
	</p>;
}

export function Source({ place, file }: { place: Place; file?: Maybe<FileState> }) {
	return <p className="text-xs text-muted">
		<span className="num">{place.root}/{place.path}</span>
		{file?.sha256 && <> · sha256 <Hash value={file.sha256} /></>}
		{file && file.state !== "ok" && <> · <span className="text-caution">{file.state}{file.error ? `: ${file.error}` : ""}</span></>}
	</p>;
}

export function Facts({ items }: { items: [string, React.ReactNode][] }) {
	return <div className="@container">
		<dl className="grid grid-cols-[minmax(8rem,max-content)_1fr] gap-x-6 gap-y-1.5 text-sm @3xl:grid-cols-[minmax(8rem,max-content)_1fr_minmax(8rem,max-content)_1fr]">
			{items.map(([label, value]) => <React.Fragment key={label}><dt className="text-muted">{label}</dt><dd className="min-w-0 break-words">{value}</dd></React.Fragment>)}
		</dl>
	</div>;
}

export function Reasons({ reasons }: { reasons: Maybe<string[]> | undefined }) {
	if (!reasons) return <Value value={null} unknown="no reasons recorded" />;
	if (!reasons.length) return <span className="text-sm text-muted">No failing reasons.</span>;
	return <ul className="list-disc space-y-1 pl-5 text-sm">{reasons.map(reason => <li key={reason}>{reason}</li>)}</ul>;
}
