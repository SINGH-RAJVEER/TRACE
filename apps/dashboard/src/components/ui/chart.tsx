import * as React from "react";
import { ResponsiveContainer, Tooltip } from "recharts";
import { cn } from "../../lib";

export type ChartConfig = Record<string, { label: string; color: string }>;

export function ChartContainer({ config, children, className }: { config: ChartConfig; children: React.ComponentProps<typeof ResponsiveContainer>["children"]; className?: string }) {
	const style = Object.fromEntries(Object.entries(config).map(([key, value]) => [`--color-${key}`, value.color])) as React.CSSProperties;
	return <div data-slot="chart" style={style} className={cn("h-72 w-full min-w-0 text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted [&_.recharts-cartesian-grid_line]:stroke-rule [&_.recharts-cartesian-axis-line]:stroke-rule [&_.recharts-reference-line_line]:stroke-muted [&_.recharts-label]:fill-muted [&_.recharts-surface]:outline-none", className)}><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div>;
}

export const ChartTooltip = Tooltip;

export function ChartLegend({ config }: { config: ChartConfig }) {
	return <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">{Object.entries(config).map(([key, item]) => <li key={key} className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded" style={{ background: item.color }} />{item.label}</li>)}</ul>;
}
