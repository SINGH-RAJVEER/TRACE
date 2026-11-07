// Shapes returned by the read-only dashboard API (apps/dashboard-api). A null value is unknown or
// not recorded; it is never a zero.

export type Maybe<T> = T | null;

export type FileState = {
	state: "ok" | "missing" | "malformed" | "lfs-pointer" | "too-large" | "refused";
	path: string;
	error: Maybe<string>;
	sha256: Maybe<string>;
};

export type Place = { root: string; path: string };

export type Section<T> = {
	state: "available" | "unavailable" | "error";
	reason: Maybe<string>;
	items: T[];
	errors?: { attack: string; arm: string; error: string }[];
};

export type RunSummary = {
	id: string;
	label: string;
	kind: string;
	run_id: Maybe<string>;
	readings: string[];
	studies: string[];
	commits: string[];
	statuses: string[];
	rows: Record<string, number>;
	artifacts: Record<string, number>;
	places: Place[];
};

export type RunList = {
	roots: { name: string; exists: boolean }[];
	index: { state: string; path: string; error: Maybe<string>; rows: number };
	runs: RunSummary[];
};

export type IndexRow = Record<string, string | number | boolean | null> & {
	kind: string;
	attack: Maybe<string>;
	arm: Maybe<string>;
	status: Maybe<string>;
};

export type CampaignItem = IndexRow & { source: "index" | "directory"; outcome: Maybe<string> };

export type UtilityMetrics = {
	submitted: Maybe<number>;
	served: Maybe<number>;
	completion_rate: Maybe<number>;
	served_only_quality: Maybe<number>;
	unconditional_utility: Maybe<number>;
	latency_p95: Maybe<number>;
};

export type Latency = { n: Maybe<number>; p50: Maybe<number>; p95: Maybe<number>; p99: Maybe<number> };

export type UtilityArm = {
	arm: string;
	policy: Maybe<string>;
	accumulator: Maybe<string>;
	study: Maybe<string>;
	status: Maybe<string>;
	validity: Maybe<string>;
	validity_reason: Maybe<string>;
	preflight_sha256: Maybe<string>;
	min_completion_rate: Maybe<number>;
	source: "index" | "directory";
	pooled: Maybe<UtilityMetrics>;
	strata: (UtilityMetrics & { group: string; value: string })[];
	file: Maybe<FileState>;
	detail: Maybe<{
		pooled: Record<string, Maybe<number>>;
		measurement_version: Maybe<string>;
		load: Maybe<Record<string, unknown>>;
		ttft_reason: Maybe<string>;
		baseline: unknown;
		strata_latency: Record<string, Record<string, Latency>>;
	}>;
};

export type Gain = { gain_pp: number; ci95_pp: [number, number]; positive_excluding_zero: boolean };

export type GateReport = {
	place: Place;
	file: FileState;
	integrity: Maybe<string>;
	gate: Maybe<"phase0-headroom" | "phase1-clean" | string>;
	task: Maybe<string>;
	base: Maybe<{ model: string; revision: string }>;
	teacher?: Maybe<{ model: string; revision: string }>;
	student_revision?: Maybe<string>;
	suite_sha256: Maybe<string>;
	items: Maybe<number>;
	families: Maybe<number>;
	declared_seeds?: Maybe<number[]>;
	parameters: Maybe<Record<string, unknown>>;
	results: Maybe<{
		base_accuracy?: number;
		teacher_accuracy?: number;
		headroom?: Gain;
		clean?: GateArmRow[];
		gold?: GateArmRow[];
	}>;
	gold_diagnostic?: Maybe<{ learned: boolean; note: string }>;
	status: Maybe<string>;
	passed: Maybe<boolean>;
	reasons: Maybe<string[]>;
	action: Maybe<string>;
	inputs_sha256: Maybe<string>;
};

export type GateArmRow = { n: number; seeds: Record<string, Maybe<Gain>>; pooled: Maybe<Gain> };

export type Capacity = {
	place: Place;
	file: FileState;
	schedule_sha256: Maybe<string>;
	slo: Maybe<Record<string, number>>;
	offered: Maybe<number>;
	completed: Maybe<number>;
	errors: Maybe<number>;
	timeouts: Maybe<number>;
	defended_responses: Maybe<number>;
	offered_rps: Maybe<number>;
	completed_rps: Maybe<number>;
	latency_p50_seconds: Maybe<number>;
	latency_p95_seconds: Maybe<number>;
	latency_p99_seconds: Maybe<number>;
	queued_mid: Maybe<number>;
	queued_at_end: Maybe<number>;
	concurrency: Maybe<number>;
	timeout_seconds: Maybe<number>;
	checks: Maybe<Record<string, boolean>>;
	passed: Maybe<boolean>;
};

export type ScoreFile = {
	place: Place;
	file: FileState;
	role: Maybe<string>;
	model: Maybe<string>;
	revision: Maybe<string>;
	task: Maybe<string>;
	suite_sha256: Maybe<string>;
	evaluation_max_new_tokens: Maybe<number>;
	items: Maybe<number>;
	accuracy: Maybe<number>;
};

export type CurveRow = { n: number; arm?: string; seed_gains_pp: number[]; mean_gain_pp: number; ci95_pp: [number, number] };

export type Curve = {
	place: Place;
	plan_file: FileState;
	summary_file: FileState;
	integrity: Maybe<string>;
	version: Maybe<string>;
	task: string;
	scope: Maybe<string>;
	sizes: Maybe<number[]>;
	seeds: Maybe<number[]>;
	include_pending: Maybe<boolean>;
	target_fraction: Maybe<number>;
	training_ledgers: Maybe<number>;
	cells: number;
	recipe: Record<string, Maybe<string>>;
	summary: Maybe<{
		results: Maybe<CurveRow[]>;
		gold_results: Maybe<CurveRow[]>;
		reference_gain_pp: Maybe<number>;
		target_gain_pp: Maybe<number>;
		smallest_n: Maybe<number>;
		bootstrap: Maybe<{ replicates: number; seed: number; method: string }>;
		warning: Maybe<string>;
	}>;
};

export type CheckpointRow = {
	id: string;
	kind: string;
	status: string;
	cutoff_seconds: Maybe<number>;
	acquired_count: Maybe<number>;
	trainable_count: Maybe<number>;
	mean_accuracy: Maybe<number>;
	min_accuracy?: Maybe<number>;
	max_accuracy?: Maybe<number>;
	seed_sd?: Maybe<number>;
	mean_gain_pp?: Maybe<number>;
	seeds_reaching_target?: Maybe<number>;
	mean_reaches_target?: Maybe<boolean>;
	seeds: Maybe<{ seed: number; accuracy: number; gain_pp: number }[]>;
	note?: string;
};

export type Checkpoints = {
	place: Place;
	plan_file: FileState;
	summary_file: FileState;
	task: Maybe<string>;
	seeds: Maybe<number[]>;
	target_accuracy: Maybe<number>;
	recipe: Record<string, Maybe<string>>;
	summary: Maybe<{
		version: Maybe<string>;
		scope: Maybe<string>;
		target_accuracy: Maybe<number>;
		base_accuracy: Maybe<number>;
		base_meets_target: Maybe<boolean>;
		seeds: Maybe<number[]>;
		checkpoints: Maybe<CheckpointRow[]>;
		first_reaching_target: Maybe<{ id: string; mean_accuracy: number; seeds_reaching_target: number; seed_count: number }>;
		target_status: Maybe<string>;
		censored_checkpoints: Maybe<string[]>;
		limitations: unknown;
		claim_boundary: Maybe<string>;
	}>;
};

export type Quantity = { value: Maybe<number>; interval: Maybe<[number, number]>; reason?: Maybe<string> };

export type Denominator = { estimate: number; interval: [number, number]; minimum: number; usable: boolean; reason: Maybe<string> };

export type TransferSeed = {
	evidence: Maybe<string>;
	families: Maybe<number>;
	items: Maybe<number>;
	raw: Maybe<{ overall: Record<string, number>; overall_interval: Record<string, [number, number]> }>;
	headroom: Maybe<Denominator>;
	clean_gain: Maybe<Denominator>;
	normalized_transfer: Maybe<{ clean: Quantity; defended: Quantity }>;
	suppression: Maybe<Quantity>;
	suppression_gate: Maybe<{ covered: boolean; reason: Maybe<string> }>;
	decision: Maybe<{
		clean_minus_defended: Quantity;
		meets_target: Maybe<boolean>;
		meets_target_at_interval_lower: Maybe<boolean>;
		withheld_reason: Maybe<string>;
	}>;
};

export type Transfer = {
	place: Place;
	file: FileState;
	evidence: Maybe<string>;
	seeds: Record<string, TransferSeed>;
	seed_variability: Maybe<Record<string, unknown>>;
	seeds_meeting_target: Maybe<number>;
};

export type SourceProvenance = Place & {
	exit: Maybe<string>;
	export: Maybe<Record<string, unknown>>;
	export_file: FileState;
	readme: Maybe<string>;
	readme_truncated: Maybe<string>;
	checksums: Maybe<number>;
	attempts: string[];
	settings_attempt: Maybe<string>;
	settings: { key: string; value: string }[];
	study_settings: Record<string, Maybe<string>>;
	teacher: Record<string, Maybe<string>>;
	texts: Record<string, Maybe<string>>;
};

export type Provenance = {
	run_id: Maybe<string>;
	readings: string[];
	commits: string[];
	commit_recorded: boolean;
	studies: string[];
	statuses: Record<string, number>;
	seeds: number[];
	partitions: string[];
	sources: SourceProvenance[];
};

export type Failed = { state: "error"; reason: string };

export type RunDetail = {
	run: RunSummary;
	provenance: Provenance | Failed;
	campaigns: Section<CampaignItem> | Failed;
	utility: Section<UtilityArm> | Failed;
	teacher: { capacity: Section<Capacity>; headroom: Section<GateReport>; scores: Section<ScoreFile> } | Failed;
	students: { curves: Section<Curve>; checkpoints: Section<Checkpoints>; transfers: Section<Transfer>; index: Section<IndexRow> } | Failed;
	gates: { reports: Section<GateReport>; index: Section<IndexRow>; detectability: Section<IndexRow> } | Failed;
};
