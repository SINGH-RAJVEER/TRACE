// Small hand-drawn chart pieces. A missing value is drawn as its own hatched mark, so an absent
// bar can never be read as a zero-length one.

import * as React from "react";
import { CircleAlert, CircleCheck, CircleDashed, CircleDot, CircleMinus } from "lucide-react";
import { armColor, type CoverageState } from "../data";
import { cn } from "../lib";

/**
 * Hover and focus tooltip for a mark. It is placed with fixed coordinates so scrolling containers
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

export type LegendItem = { label: string; color?: string; hatched?: boolean; outline?: boolean; assumed?: boolean; line?: "solid" | "dashed" };

export function Legend({ items }: { items: LegendItem[] }) {
	return <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
		{items.map(item => <li key={item.label} className="flex items-center gap-1.5">
			{item.line
				? <span aria-hidden className={cn("inline-block h-3 border-l-2", item.line === "dashed" && "border-dashed")} style={{ borderColor: item.color }} />
				: <span aria-hidden className={cn("inline-block h-2.5 w-2.5 rounded-[3px]", item.hatched && "hatched", item.outline && "border border-dashed border-faint", item.assumed && "border border-dashed border-faint/80")} style={item.assumed ? { background: "color-mix(in srgb, var(--faint) 28%, transparent)" } : item.color ? { background: item.color } : undefined} />}
			{item.label}
		</li>)}
	</ul>;
}

/** A large figure with a label, without a box around it. */
export function Stat({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "caution" | "fail" | "pass" }) {
	return <div className="min-w-0 space-y-1">
		<div className="text-xs text-muted">{label}</div>
		<div className={cn("figure text-2xl leading-none", tone === "caution" && "text-caution", tone === "fail" && "text-fail", tone === "pass" && "text-pass")}>{value}</div>
	</div>;
}

const COVERAGE: Record<CoverageState, { label: string; icon: typeof CircleCheck; className: string }> = {
	present: { label: "recorded", icon: CircleCheck, className: "text-pass" },
	partial: { label: "partial", icon: CircleDot, className: "text-caution" },
	missing: { label: "missing", icon: CircleDashed, className: "text-faint" },
	error: { label: "error", icon: CircleAlert, className: "text-fail" },
	na: { label: "n/a", icon: CircleMinus, className: "text-faint/50" },
};

export function CoverageMark({ state, compact = false }: { state: CoverageState; compact?: boolean }) {
	const spec = COVERAGE[state];
	const Icon = spec.icon;
	return <span className={cn("inline-flex items-center gap-1 text-xs font-medium", spec.className)}>
		<Icon size={14} strokeWidth={2.2} aria-hidden />
		<span className={cn(compact && "sr-only")}>{spec.label}</span>
	</span>;
}

/** A chart with a title and nothing else around it. */
export function Figure({ title, legend, children, className }: { title: string; legend?: LegendItem[]; children: React.ReactNode; className?: string }) {
	return <figure className={cn("min-w-0 space-y-3", className)}>
		<figcaption className="space-y-1.5">
			<h3 className="text-sm font-semibold">{title}</h3>
			{legend && <Legend items={legend} />}
		</figcaption>
		{children}
	</figure>;
}

/** A collapsed block behind a quiet toggle. */
export function More({ label, children }: { label: string; children: React.ReactNode }) {
	return <details className="group text-sm">
		<summary className="inline-flex cursor-pointer items-center gap-1.5 text-xs font-medium text-muted hover:text-ink">
			<span className="inline-block transition-transform group-open:rotate-90" aria-hidden>&#9656;</span>{label}
		</summary>
		<div className="mt-4">{children}</div>
	</details>;
}

// ---------------------------------------------------------------------------------------------
// Grouped bar chart. One group per category, one bar per arm, every bar on the same axis.

export type BarState = "value" | "assumed" | "no-value" | "absent" | "none";

export type Bar = {
	key: string;
	color: string;
	value: number | null;
	/** value: drawn to scale; no-value: hatched stub; absent: dashed stub (no row); none: ran, nothing happened. */
	state: BarState;
	label?: string;
	flag?: string;
	tip: React.ReactNode;
};

export type Scale = { kind: "log" | "linear"; min: number; max: number; ticks: number[]; tick: (value: number) => string };

/** A log axis over whole decades, so every recorded value is visible however small. */
export function logScale(values: number[], tick: (value: number) => string): Scale {
	const positive = values.filter(v => v > 0);
	if (!positive.length) return { kind: "log", min: 1, max: 10, ticks: [1, 10], tick };
	const top = Math.max(...positive);
	const lo = Math.floor(Math.log10(Math.min(...positive)));
	const decade = 10 ** Math.floor(Math.log10(top));
	const max = Math.max([1, 2, 5, 10].map(m => m * decade).find(v => v >= top) ?? 10 * decade, 10 ** (lo + 1));
	const ticks: number[] = [];
	for (let e = lo; 10 ** e <= max; e++) ticks.push(10 ** e);
	if (max / ticks[ticks.length - 1] >= 3) ticks.push(max);
	return { kind: "log", min: 10 ** lo, max, ticks, tick };
}

/** A linear axis from zero to a rounded maximum. */
export function linearScale(values: number[], tick: (value: number) => string, fixedMax?: number): Scale {
	const top = fixedMax ?? Math.max(...values, 0);
	if (fixedMax !== undefined) return { kind: "linear", min: 0, max: fixedMax, ticks: [0, 0.25, 0.5, 0.75, 1].map(f => f * fixedMax), tick };
	const raw = top > 0 ? top / 4 : 1;
	const magnitude = 10 ** Math.floor(Math.log10(raw));
	const step = [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(s => s >= raw) ?? raw;
	const max = Math.max(step * Math.ceil(top / step), step);
	const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, i) => i * step);
	return { kind: "linear", min: 0, max, ticks, tick };
}

function position(scale: Scale, value: number): number {
	if (scale.kind === "log") {
		if (value <= scale.min) return 0;
		return Math.min((Math.log10(value) - Math.log10(scale.min)) / (Math.log10(scale.max) - Math.log10(scale.min)), 1);
	}
	return Math.min(Math.max(value / scale.max, 0), 1);
}

function BarMark({ bar, scale, label = "own" }: { bar: Bar; scale: Scale; label?: "own" | "span" | "none" }) {
	let body: React.ReactNode;
	if ((bar.state === "value" || bar.state === "assumed") && bar.value !== null) {
		const height = Math.max(position(scale, bar.value) * 100, 1.5);
		const assumed = bar.state === "assumed";
		body = <span className="relative flex h-full w-full flex-col justify-end">
			{label !== "none" && <span className={cn("absolute -translate-x-1/2 text-[0.6rem] tabular-nums tracking-tight whitespace-nowrap", label === "span" ? "left-[calc(100%+1px)]" : "left-1/2", assumed ? "text-faint" : "text-muted")} style={{ bottom: `calc(${height}% + 2px)` }}>{bar.flag && <span className="text-caution">{bar.flag} </span>}{assumed && "≈"}{bar.label}</span>}
			<span className={cn("block w-full rounded-t-[4px]", assumed && "border border-b-0 border-dashed border-faint/80")} style={{ height: `${height}%`, background: assumed ? "color-mix(in srgb, var(--faint) 28%, transparent)" : bar.color }} />
		</span>;
	} else {
		const style = bar.state === "absent" ? "border border-dashed border-faint/70" : bar.state === "none" ? "border border-faint/70" : "hatched";
		body = <span className="relative flex h-full w-full flex-col justify-end">
			{bar.state === "none" && <span className="absolute bottom-4 left-1/2 -translate-x-1/2 text-[0.62rem] text-faint">none</span>}
			<span className={cn("block h-3 w-full rounded-t-[3px]", style)} />
		</span>;
	}
	return <Tip content={bar.tip} className="h-full max-w-7 min-w-1.5 flex-1">{body}</Tip>;
}

export function GroupedBars({ groups, scale, height = 180 }: { groups: { key: string; label: string; bars: Bar[] }[]; scale: Scale; height?: number }) {
	return <div className="flex gap-2 pt-4">
		<div className="relative w-9 shrink-0" style={{ height }}>
			{scale.ticks.map(t => <span key={t} className="num absolute right-0 translate-y-1/2 text-[0.62rem] text-faint" style={{ bottom: `${position(scale, t) * 100}%` }}>{scale.tick(t)}</span>)}
		</div>
		<div className="min-w-0 flex-1">
			<div className="relative" style={{ height }}>
				{scale.ticks.map(t => <span key={t} aria-hidden className={cn("absolute inset-x-0 border-t", position(scale, t) === 0 ? "border-faint/60" : "border-rule/70")} style={{ bottom: `${position(scale, t) * 100}%` }} />)}
				<div className="relative flex h-full gap-3 px-1 sm:gap-5">
					{groups.map(g => <div key={g.key} className="flex h-full flex-1 items-end justify-center gap-[2px]">
						{g.bars.map((bar, i) => {
							const twin = (other: Bar | undefined) => other?.state === "assumed" && bar.state === "assumed" && other.label === bar.label;
							return <BarMark key={bar.key} bar={bar} scale={scale} label={twin(g.bars[i - 1]) ? "none" : twin(g.bars[i + 1]) ? "span" : "own"} />;
						})}
					</div>)}
				</div>
			</div>
			<div className="flex gap-3 px-1 pt-1.5 sm:gap-5">
				{groups.map(g => <span key={g.key} className="flex-1 truncate text-center text-xs text-muted" title={g.label}>{g.label}</span>)}
			</div>
		</div>
	</div>;
}
