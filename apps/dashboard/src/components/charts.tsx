// Small hand-drawn chart pieces. They draw a missing value as its own hatched mark, so an absent
// bar can never be read as a zero-length one.

import * as React from "react";
import { CircleAlert, CircleCheck, CircleDashed, CircleDot, CircleMinus } from "lucide-react";
import { ARM_LABELS, armColor, type CoverageState } from "../data";
import { cn } from "../lib";

/**
 * Hover and focus tooltip for a mark. It is placed with fixed coordinates so scrolling tables
 * cannot clip it, and flips below the mark near the top of the window.
 */
export function Tip({ content, children, className }: { content: React.ReactNode; children: React.ReactNode; className?: string }) {
	const [at, setAt] = React.useState<{ x: number; y: number; below: boolean } | null>(null);
	const show = (event: React.SyntheticEvent<HTMLSpanElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		const below = rect.top < 160;
		setAt({ x: Math.min(Math.max(rect.left + rect.width / 2, 120), window.innerWidth - 120), y: below ? rect.bottom + 6 : rect.top - 6, below });
	};
	return <span tabIndex={0} onMouseEnter={show} onFocus={show} onMouseLeave={() => setAt(null)} onBlur={() => setAt(null)} className={cn("inline-flex outline-none focus-visible:ring-2 focus-visible:ring-accent", className)}>
		{children}
		{at && <span role="tooltip" className="chart-tooltip pointer-events-none fixed z-50 max-w-sm whitespace-nowrap" style={{ left: at.x, top: at.y, transform: `translate(-50%, ${at.below ? "0" : "-100%"})` }}>{content}</span>}
	</span>;
}

export function TipRows({ title, rows }: { title?: React.ReactNode; rows: [React.ReactNode, React.ReactNode][] }) {
	return <span className="block space-y-0.5 text-left">
		{title && <span className="block font-medium text-ink">{title}</span>}
		{rows.map(([label, value], i) => <span key={i} className="flex gap-4 text-muted"><span>{label}</span><span className="num ml-auto text-ink">{value}</span></span>)}
	</span>;
}

export function ArmSwatch({ arm, className }: { arm: string; className?: string }) {
	return <span aria-hidden className={cn("inline-block size-2.5 shrink-0 rounded-[3px]", className)} style={{ background: armColor(arm) }} />;
}

export function ArmName({ arm }: { arm: string }) {
	return <span className="inline-flex items-center gap-1.5 whitespace-nowrap"><ArmSwatch arm={arm} /><span>{arm}</span></span>;
}

export function ArmLegend({ arms }: { arms: string[] }) {
	return <ul className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
		{arms.map(arm => <li key={arm} className="flex items-center gap-1.5"><ArmSwatch arm={arm} />{ARM_LABELS[arm] ?? arm}<span className="num text-faint">{arm}</span></li>)}
	</ul>;
}

export function Legend({ items }: { items: { label: string; color?: string; hatched?: boolean; outline?: boolean }[] }) {
	return <ul className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
		{items.map(item => <li key={item.label} className="flex items-center gap-1.5">
			<span aria-hidden className={cn("inline-block h-2.5 w-3.5 rounded-[3px]", item.hatched && "hatched", item.outline && "border border-dashed border-faint")} style={item.color ? { background: item.color } : undefined} />
			{item.label}
		</li>)}
	</ul>;
}

/** A large figure with a label, without a box around it. */
export function Stat({ label, value, sub, tone }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "caution" | "fail" | "pass" }) {
	return <div className="min-w-0 space-y-1">
		<div className="text-xs font-medium tracking-wide text-muted uppercase">{label}</div>
		<div className={cn("figure text-3xl leading-none", tone === "caution" && "text-caution", tone === "fail" && "text-fail", tone === "pass" && "text-pass")}>{value}</div>
		{sub && <div className="text-xs text-muted">{sub}</div>}
	</div>;
}

/** A ratio as a thin track, with the parts named in the label rather than by color alone. */
export function Meter({ parts, total, label }: { parts: { value: number; color: string; label: string }[]; total: number; label: string }) {
	return <div role="img" aria-label={label} className="flex h-1.5 w-full gap-[2px] overflow-hidden rounded-full bg-track">
		{parts.filter(p => p.value > 0).map(p => <span key={p.label} className="h-full first:rounded-l-full last:rounded-r-full" style={{ width: `${(p.value / Math.max(total, 1)) * 100}%`, background: p.color }} />)}
	</div>;
}

const COVERAGE: Record<CoverageState, { label: string; icon: typeof CircleCheck; className: string }> = {
	present: { label: "recorded", icon: CircleCheck, className: "text-pass" },
	partial: { label: "partial", icon: CircleDot, className: "text-caution" },
	missing: { label: "missing", icon: CircleDashed, className: "text-faint" },
	error: { label: "error", icon: CircleAlert, className: "text-fail" },
	na: { label: "n/a", icon: CircleMinus, className: "text-faint/60" },
};

export function CoverageMark({ state, compact = false }: { state: CoverageState; compact?: boolean }) {
	const spec = COVERAGE[state];
	const Icon = spec.icon;
	return <span className={cn("inline-flex items-center gap-1 text-xs font-medium", spec.className)}>
		<Icon size={14} strokeWidth={2.2} aria-hidden />
		<span className={cn(compact && "sr-only")}>{spec.label}</span>
	</span>;
}

/** Horizontal bar against a shared scale; a null value draws the hatched "not recorded" bar. */
export function HBar({ value, max, color, label, missing, tip, height = "h-3.5" }: { value: number | null; max: number; color: string; label?: React.ReactNode; missing?: string; tip?: React.ReactNode; height?: string }) {
	const width = value === null ? 0 : Math.max((value / Math.max(max, 1e-12)) * 100, value > 0 ? 0.6 : 0);
	const bar = value === null
		? <span className="flex min-w-0 items-center gap-2"><span className={cn("hatched inline-block w-16 rounded-[3px]", height)} /><span className="unrecorded">{missing ?? "not recorded"}</span></span>
		: <span className="flex min-w-0 flex-1 items-center gap-2">
			<span className={cn("inline-block rounded-r-[4px]", height)} style={{ width: `${width}%`, background: color, minWidth: value > 0 ? 2 : 0 }} />
			{label !== undefined && <span className="num shrink-0 text-xs text-ink">{label}</span>}
		</span>;
	return tip ? <Tip content={tip} className="w-full">{bar}</Tip> : bar;
}

/** A labeled block of charts, sitting on the page without borders. */
export function Panel({ title, note, children, className, aside }: { title: string; note?: React.ReactNode; children: React.ReactNode; className?: string; aside?: React.ReactNode }) {
	return <div className={cn("min-w-0 space-y-3", className)}>
		<div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
			<div>
				<h4 className="text-sm font-semibold">{title}</h4>
				{note && <p className="max-w-2xl text-xs text-muted">{note}</p>}
			</div>
			{aside}
		</div>
		{children}
	</div>;
}

/** A collapsed table or list behind a quiet toggle. */
export function More({ label, children }: { label: string; children: React.ReactNode }) {
	return <details className="group text-sm">
		<summary className="inline-flex cursor-pointer items-center gap-1.5 text-xs font-medium text-muted hover:text-ink">
			<span className="inline-block transition-transform group-open:rotate-90" aria-hidden>&#9656;</span>{label}
		</summary>
		<div className="mt-3">{children}</div>
	</details>;
}
