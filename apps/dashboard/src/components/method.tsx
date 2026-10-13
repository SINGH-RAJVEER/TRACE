import { SlotSwatch } from "./plot";
import { title } from "../data";
import { labels, slots, type Report } from "../types";

export function Method({ report }: { report: Report }) {
	return <div className="method">
		<section>
			<h2>How scores are measured</h2>
			<ul>
				<li>Every model answers every question once, greedily (temperature 0), with the same output budget and thinking disabled.</li>
				<li>GSM8K needs the exact final number. ARC-Challenge and MMLU need a single answer letter. TruthfulQA uses the two-option format. HumanEval and MBPP run the official Python tests (pass@1) in a network-disabled container.</li>
				<li>Blank, unparseable, truncated, failed, and timed-out answers count as wrong. API errors also count as wrong, so rerun them before judging model quality.</li>
				<li>Headline numbers average the benchmarks with equal weight, regardless of how many questions each has.</li>
				<li>The teacher is evaluated before and after student training. Its initial score stays the reference for gap closure; its change measures serving drift.</li>
				<li>The teacher is never trained, so it is evaluated once and serves as the fixed reference for both student runs.</li>
				<li>Teacher gap closed = (student after − student before) ÷ (teacher − student before). It is unavailable when the teacher did not beat the base student, and can exceed 100% if the student overtakes the teacher.</li>
				<li>Accuracy whiskers are 95% Wilson intervals. Change whiskers are 95% paired bootstrap intervals (2,000 resamples, fixed seed) on the same questions. Both describe this question sample, not variation between training runs.</li>
				<li>Runs are only compared when they share the suite hash, scorer version, generation settings, and coding sandbox. The latest completed run per slot is shown.</li>
			</ul>
		</section>
		{report.warnings.length > 0 && <section><h2>Run notes</h2><ul>{report.warnings.map(note => <li key={note}>{note}</li>)}</ul></section>}
		<section>
			<h2>Datasets</h2>
			<div className="table-scroll"><table>
				<thead><tr><th>Benchmark</th><th>Dataset</th><th>Split</th><th className="num">Questions used</th><th>License</th><th>Revision</th></tr></thead>
				<tbody>{report.suite.sources.map(source => <tr key={source.benchmark}>
					<th>{title(source.benchmark)}</th>
					<td>{report.demo ? source.repository : <a href={source.url} target="_blank" rel="noreferrer">{source.repository}</a>}<small>{source.configuration}</small></td>
					<td>{source.split}</td>
					<td className="num">{source.selected} of {source.available}</td>
					<td>{source.license}</td>
					<td><code title={source.revision}>{source.revision.slice(0, 12)}</code></td>
				</tr>)}</tbody>
			</table></div>
			<p className="small-note">Questions were sampled with seed {report.suite.seed}, spread across categories. Use full test splits for final reporting. This cannot rule out benchmark contamination in pretraining.</p>
		</section>
		<section>
			<h2>Models and runs</h2>
			{slots.map(slot => {
				const run = report.runs[slot];
				return <div className="run-provenance" key={slot}>
					<h3><SlotSwatch slot={slot} />{labels[slot]}</h3>
					{run ? <dl>
						<dt>Model</dt><dd>{run.model.model} <code>{run.model.revision}</code></dd>
						<dt>Adapter</dt><dd>{run.model.adapter_path ?? "None"}</dd>
						<dt>Run</dt><dd><code>{run.id}</code> · finished {new Date(run.completed_at).toLocaleString()}</dd>
						<dt>Generation</dt><dd>Temperature 0 · up to {run.generation.max_output_tokens} output tokens · seed {run.generation.seed}</dd>
						<dt>Code sandbox</dt><dd><code>{run.code_image_id ?? "None recorded"}</code></dd>
						<dt>Evaluation code</dt><dd><code>{run.code_revision || "Not recorded"}</code></dd>
						<dt>Training files checked</dt><dd>{Object.keys(run.training_sha256).join(", ") || "None supplied"}</dd>
					</dl> : <p className="muted">Not recorded yet.</p>}
				</div>;
			})}
		</section>
	</div>;
}
