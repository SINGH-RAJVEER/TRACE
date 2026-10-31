import { SlotSwatch } from "./plot";
import { title } from "../data";
import { labels, slots, type Report } from "../types";

export function Method({ report }: { report: Report }) {
	return <div className="method">
		<section>
			<h2>How scores are measured</h2>
			<ul>
				<li>Every model answers every question once, greedily (temperature 0), with the same output budget and thinking disabled.</li>
				<li>AG News uses label accuracy, SQuAD uses best-gold token F1 with secondary exact match, GSM8K uses numeric exact match, and coding uses sandboxed single-completion pass@1.</li>
				<li>Blank, unparseable, truncated, failed, and timed-out answers count as wrong. API errors are retained and reported separately. Coding sandbox infrastructure errors abort evaluation.</li>
				<li>The teacher is evaluated before and after student training. Its change measures serving drift.</li>
				<li>Binary score whiskers use Wilson intervals and binary changes use 2,000 paired item bootstrap draws. SQuAD F1 and changes use normal-approximation mean intervals. These are descriptive item intervals; source-document and campaign uncertainty require separate study analysis.</li>
				<li>Runs are only compared when they share the suite hash, scorer version, generation settings, and simulation flag. The latest completed run per slot is shown.</li>
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
			<p className="small-note">Task splits and development sampling use seed {report.suite.seed}. Final task suites use the official held-out split. This cannot rule out benchmark contamination in pretraining.</p>
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
						<dt>Evaluation code</dt><dd><code>{run.code_revision || "Not recorded"}</code></dd>
						<dt>Training files checked</dt><dd>{Object.keys(run.training_sha256).join(", ") || "None supplied"}</dd>
					</dl> : <p className="muted">Not recorded yet.</p>}
				</div>;
			})}
		</section>
	</div>;
}
