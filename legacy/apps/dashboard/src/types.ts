// The initial victim is the reference; the second run measures serving drift.
export const slots = ["student_before", "student_after", "teacher", "teacher_after"] as const;
export type Slot = typeof slots[number];
export const labels: Record<Slot, string> = {
	student_before: "Student before", student_after: "Student after",
	teacher: "Teacher before", teacher_after: "Teacher after",
};
// Hue identifies the model. The student's phase is a second channel (faded outline = before,
// solid = after), so before/after never depends on telling two shades apart.
export const modelColors = { teacher: "#3987e5", student: "#d95926" } as const;
const modelOf = (slot: Slot) => slot.startsWith("teacher") ? "teacher" : "student";
export const isBefore = (slot: Slot) => slot === "student_before" || slot === "teacher";
export const colors: Record<Slot, string> = Object.fromEntries(slots.map(slot => [slot, modelColors[modelOf(slot)]])) as Record<Slot, string>;
export const benchmarkLabels: Record<string, string> = {
	gsm8k: "GSM8K", coding: "Coding (pass@1)",
	ag_news: "AG News", squad: "SQuAD 1.1",
};
export const statusLabels: Record<string, string> = {
	ok: "Answered", invalid: "Unparseable", truncated: "Truncated", error: "API error", timeout: "Execution timed out",
};
export type Summary = {
	n: number; correct: number; score: number | null; ci95: [number, number];
	f1?: number; f1_ci95?: [number, number];
	statuses: Record<string, number>; latency_p50: number | null; latency_p95: number | null;
	tokens_per_second: number | null; token_coverage: number;
};
export type Paired = {
	n: number; delta_pp: number; ci95: [number, number]; improved: number; regressed: number;
	f1_delta_pp?: number; f1_ci95?: [number, number];
	both_correct: number; both_wrong: number; answer_agreement: number | null; agreement_n: number;
};
type Result = {
	item_id: string; benchmark: string; category: string; response: string; prediction: string | null;
	correct: boolean; status: string; latency_seconds: number;
};
type Example = {
	item: { id: string; benchmark: string; question: string; category: string; answer: string; choices: string[]; metric: string };
	results: Partial<Record<Slot, Result>>;
};
export type Report = {
	id: string; experiment: string; demo: boolean; benchmarks: string[]; warnings: string[];
	suite: { sha256: string; items: unknown[]; seed: number; scorer_version: string; sources: { benchmark: string; repository: string; revision: string; configuration: string; split: string; license: string; url: string; selected: number; available: number }[] };
	runs: Partial<Record<Slot, { id: string; completed_at: string; model: { model: string; revision: string; backend: string; adapter_path: string | null }; generation: { max_output_tokens: number; seed: number }; code_revision: string; training_sha256: Record<string, string> }>>;
	scores: Partial<Record<Slot, Record<string, Summary>>>;
	slices: Record<string, Partial<Record<Slot, Summary>>>;
	comparisons: Record<string, Record<string, Paired>>;
	samples: Example[];
};
export type Experiment = { id: string; name: string; demo: boolean; slots: Slot[]; updated_at: string; suite_sha256: string };
