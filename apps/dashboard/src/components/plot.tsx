import { useId, useState, type ReactNode } from "react";
import { ChartContainer, ChartTooltipContent, type ChartConfig } from "./ui/chart";
import { Switch } from "./ui/switch";
import { linearTicks, logDomain, logTicks } from "../data";
import { colors, isBefore, labels, slots, type Slot } from "../types";

export const chartConfig: ChartConfig = Object.fromEntries(slots.map(slot => [slot, { label: labels[slot], color: colors[slot] }]));
export const tick = { fontSize: 11 };
export const barCursor = { fill: "rgba(255, 255, 255, 0.04)" };
export const lineCursor = { stroke: "#383835" };
export const ink = { secondary: "#c3c2b7", baseline: "#56554f" };

// Before = translucent fill with a solid outline, after = solid fill. Same hue per model.
export const barStyle = (slot: Slot) => isBefore(slot)
	? { fill: colors[slot], fillOpacity: .28, stroke: colors[slot], strokeWidth: 1.5 }
	: { fill: colors[slot], fillOpacity: 1, stroke: "none", strokeWidth: 0 };

export function Swatch({ color, faded = false, line = false, dashed = false }: { color: string; faded?: boolean; line?: boolean; dashed?: boolean }) {
	if (line) return <span className="swatch-line" style={{ borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}` }} />;
	return <span className="swatch" style={{ background: faded ? `${color}47` : color, boxShadow: `inset 0 0 0 1.5px ${color}` }} />;
}
export const SlotSwatch = ({ slot, line = false }: { slot: Slot; line?: boolean }) => <Swatch color={colors[slot]} faded={isBefore(slot)} line={line} dashed={line && isBefore(slot)} />;

export type LegendItem = { key: string; label: string; swatch: ReactNode };
export function Legend({ items }: { items: LegendItem[] }) {
	return <ul className="legend" aria-label="Legend">{items.map(item => <li key={item.key}>{item.swatch}{item.label}</li>)}</ul>;
}
export const runLegend = (runs: Slot[], line = false): LegendItem[] => runs.map(slot => ({ key: slot, label: labels[slot], swatch: <SlotSwatch slot={slot} line={line} /> }));

export function ScaleSwitch({ log, onChange }: { log: boolean; onChange: (log: boolean) => void }) {
	const id = useId();
	return <div className="scale-switch"><Switch id={id} checked={log} onCheckedChange={onChange} /><label htmlFor={id}>Log scale</label></div>;
}

export function Plot({ name, note, legend, scale, wide = false, tall = false, children, after }: {
	name: string; note: ReactNode; legend?: LegendItem[]; scale?: { log: boolean; onChange: (log: boolean) => void };
	wide?: boolean; tall?: boolean; children?: React.ComponentProps<typeof ChartContainer>["children"]; after?: ReactNode;
}) {
	return <section className={`plot ${wide ? "wide" : ""}`}>
		<div className="plot-header"><h2>{name}</h2>{scale && <ScaleSwitch {...scale} />}</div>
		<p className="plot-note">{note}</p>
		{legend && <Legend items={legend} />}
		{children && <ChartContainer config={chartConfig} className={tall ? "h-96" : "h-72"}>{children}</ChartContainer>}
		{after}
	</section>;
}

export const useLog = () => {
	const [log, setLog] = useState(false);
	return { log, onChange: setLog };
};

// Axis props for a positive-valued measure that can switch between linear and log scales.
export function valueAxis(log: boolean, values: (number | null | undefined)[], format: (v: number) => string) {
	if (log) {
		const domain = logDomain(values);
		return { type: "number" as const, scale: "log" as const, domain, ticks: logTicks(domain), allowDataOverflow: true, tickFormatter: format };
	}
	const ticks = linearTicks(values);
	return { type: "number" as const, scale: "linear" as const, domain: [0, ticks[ticks.length - 1]] as [number, number], ticks, tickFormatter: format };
}
const scaleNote = (text: string, log = false) => log ? `${text} · log scale` : text;
export const yTitle = (text: string, log = false) => ({ value: scaleNote(text, log), angle: -90, position: "insideLeft" as const, offset: 4, style: { textAnchor: "middle" as const } });
export const xTitle = (text: string, log = false) => ({ value: scaleNote(text, log), position: "insideBottom" as const, offset: -14 });

export function Tip({ value, heading, line = false, ...props }: Omit<React.ComponentProps<typeof ChartTooltipContent>, "swatch"> & { line?: boolean }) {
	return <ChartTooltipContent {...props} heading={heading} value={value} swatch={entry => slots.includes(entry.dataKey as Slot) ? <SlotSwatch slot={entry.dataKey as Slot} line={line} /> : <Swatch color={String(entry.color ?? entry.payload?.fill ?? "#898781")} />} />;
}
