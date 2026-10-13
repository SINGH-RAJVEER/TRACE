import * as React from "react";
import { ResponsiveContainer, Tooltip, Legend } from "recharts";
import { cn } from "../../lib";

export type ChartConfig = Record<string, { label: string; color: string }>;
export function ChartContainer({ config, children, className }: { config: ChartConfig; children: React.ComponentProps<typeof ResponsiveContainer>["children"]; className?: string }) {
	const style = Object.fromEntries(Object.entries(config).map(([key, value]) => [`--color-${key}`, value.color])) as React.CSSProperties;
	return <div data-slot="chart" style={style} className={cn("h-80 min-w-0 w-full text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line]:stroke-border [&_.recharts-cartesian-axis-line]:stroke-input [&_.recharts-label]:fill-muted-foreground [&_.recharts-surface]:outline-none", className)}><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div>;
}
export const ChartTooltip = Tooltip;
export const ChartLegend = Legend;

type TooltipEntry = { name?: string | number; value?: unknown; color?: string; dataKey?: unknown; payload?: Record<string, unknown> };
export function ChartTooltipContent({ active, payload, label, heading, value, swatch }: {
	active?: boolean; payload?: TooltipEntry[]; label?: unknown;
	heading?: (label: unknown, payload: TooltipEntry[]) => React.ReactNode;
	value: (value: number, entry: TooltipEntry) => React.ReactNode;
	swatch?: (entry: TooltipEntry) => React.ReactNode;
}) {
	if (!active || !payload?.length) return null;
	return <div className="chart-tooltip">
		<div className="chart-tooltip-heading">{heading ? heading(label, payload) : String(label ?? "")}</div>
		{payload.filter(entry => typeof entry.value === "number").map(entry => <div className="chart-tooltip-row" key={String(entry.dataKey ?? entry.name)}>
			{swatch ? swatch(entry) : <span className="swatch" style={{ background: entry.color }} />}
			<span>{entry.name}</span>
			<strong>{value(entry.value as number, entry)}</strong>
		</div>)}
	</div>;
}
