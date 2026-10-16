import { benchmarkLabels, slots, type Report, type Slot, type Summary, type Paired } from "./types";

export const title = (name: string) => benchmarkLabels[name] ?? name;
export const format = (value: number | null | undefined, suffix = "", digits = 1) => value == null || !Number.isFinite(value) ? "n/a" : `${value.toFixed(digits)}${suffix}`;
export const signed = (value: number | null | undefined, suffix = "", digits = 1) => value == null ? "n/a" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}${suffix}`;
export const seconds = (value: number) => value < 1 ? `${+value.toFixed(2)}s` : value < 10 ? `${+value.toFixed(1)}s` : `${Math.round(value)}s`;

export const taskScore = (summary: Summary | undefined, benchmark: string) => benchmark === "squad" ? summary?.f1 ?? null : summary?.score ?? null;
export const taskInterval = (summary: Summary | undefined, benchmark: string) => benchmark === "squad" ? summary?.f1_ci95 : summary?.ci95;
export const taskDelta = (comparison: Paired | undefined, benchmark: string) => benchmark === "squad" ? comparison?.f1_delta_pp : comparison?.delta_pp;
export const taskDeltaInterval = (comparison: Paired | undefined, benchmark: string) => benchmark === "squad" ? comparison?.f1_ci95 : comparison?.ci95;
export const metricLabel = (benchmark: string) => benchmark === "squad" ? "token F1" : benchmark === "gsm8k" ? "numeric exact match" : "label accuracy";

export function scoreRows(report: Report, benchmarks: string[]) {
	return benchmarks.map(benchmark => Object.assign({ benchmark, name: title(benchmark) }, ...slots.map(slot => {
		const summary = report.scores[slot]?.[benchmark];
		const score = taskScore(summary, benchmark), interval = taskInterval(summary, benchmark);
		return { [slot]: score, [`${slot}_error`]: score == null || !interval ? null : [score - interval[0], interval[1] - score] };
	})));
}

const below = (value: number) => {
	const exponent = Math.floor(Math.log10(value)), mantissa = value / 10 ** exponent;
	return (mantissa >= 5 ? 5 : mantissa >= 2 ? 2 : 1) * 10 ** exponent;
};
const above = (value: number) => {
	const exponent = Math.floor(Math.log10(value)), mantissa = value / 10 ** exponent;
	return (mantissa <= 1 ? 1 : mantissa <= 2 ? 2 : mantissa <= 5 ? 5 : 10) * 10 ** exponent;
};
// Log axes need a strictly positive domain snapped to readable 1-2-5 values.
export function logDomain(values: (number | null | undefined)[]): [number, number] {
	const positive = values.filter((v): v is number => v != null && v > 0);
	if (!positive.length) return [0.1, 1];
	const low = below(Math.min(...positive)), high = above(Math.max(...positive));
	return high > low ? [low, high] : [low / 2, high * 2];
}
export function logTicks([low, high]: [number, number]) {
	const pick = (steps: number[]) => {
		const ticks: number[] = [];
		for (let exponent = Math.floor(Math.log10(low)); exponent <= Math.ceil(Math.log10(high)); exponent++)
			for (const step of steps) {
				const tick = +(step * 10 ** exponent).toPrecision(3);
				if (tick >= low * .999 && tick <= high * 1.001) ticks.push(tick);
			}
		return ticks;
	};
	const ticks = pick([1, 2, 5]);
	return ticks.length >= 3 ? ticks : pick([1, 1.5, 2, 3, 5, 7]);
}

// Linear axes from zero with about four round 1-2-5 steps.
export function linearTicks(values: (number | null | undefined)[]) {
	const max = Math.max(0, ...values.filter((v): v is number => v != null && Number.isFinite(v)));
	if (max <= 0) return [0, 1];
	const step = above(max / 4);
	return Array.from({ length: Math.ceil(max / step - 1e-9) + 1 }, (_, i) => +(i * step).toPrecision(6));
}

// Response-time frequency per run. Log mode uses geometric bins so short and long tails
// both get resolution; each point sits at its bin center on the matching axis scale.
export function histogram(report: Report, benchmarks: string[], active: Slot[], log: boolean, bins = 16) {
	const included = report.samples.filter(s => benchmarks.includes(s.item.benchmark));
	const latency = (slot: Slot, sample: typeof included[number]) => {
		const r = sample.results[slot];
		return r && r.status !== "error" ? r.latency_seconds : null;
	};
	const values = included.flatMap(s => active.map(slot => latency(slot, s))).filter((v): v is number => v != null && (!log || v > 0));
	if (!values.length) return { rows: [], domain: [0, 1] as [number, number] };
	const min = Math.min(...values), max = Math.max(...values);
	const domain: [number, number] = log ? logDomain([min, max]) : [0, max > 0 ? max : 1];
	const edge = (i: number) => log ? domain[0] * (domain[1] / domain[0]) ** (i / bins) : domain[0] + (domain[1] - domain[0]) * i / bins;
	const index = (v: number) => {
		const position = log ? Math.log(v / domain[0]) / Math.log(domain[1] / domain[0]) : (v - domain[0]) / (domain[1] - domain[0]);
		return Math.max(0, Math.min(bins - 1, Math.floor(position * bins)));
	};
	const rows = Array.from({ length: bins }, (_, i) => {
		const low = edge(i), high = edge(i + 1);
		const row: Record<string, number> = { low, high, x: log ? Math.sqrt(low * high) : (low + high) / 2 };
		for (const slot of active) row[slot] = 0;
		return row;
	});
	for (const sample of included) for (const slot of active) {
		const v = latency(slot, sample);
		if (v != null && (!log || v > 0)) rows[index(v)][slot] += 1;
	}
	return { rows, domain };
}

// Sequential single-hue ramp (blue, dark surface): low values recede toward the background.
const ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"];
export function heat(score: number | null | undefined) {
	if (score == null) return { background: "transparent", color: "var(--muted-foreground)" };
	const step = Math.round(Math.max(0, Math.min(100, score)) / 10);
	return { background: ramp[step], color: step >= 6 ? "#0b0b0b" : "#ffffff" };
}
export const heatRamp = ramp;
