import { useState } from "react";
import { ArrowLeft, ArrowRight, CircleCheck, CircleX, Search } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Choice } from "./choice";
import { SlotSwatch } from "./plot";
import { format, title } from "../data";
import { labels, statusLabels, type Report, type Slot } from "../types";

const pageSize = 10;

export function Answers({ report, benchmarks, available }: { report: Report; benchmarks: string[]; available: Slot[] }) {
	const [search, setSearch] = useState("");
	const [outcome, setOutcome] = useState("all");
	const [page, setPage] = useState(0);
	const query = search.toLowerCase();
	const examples = report.samples.filter(example => {
		if (!benchmarks.includes(example.item.benchmark)) return false;
		if (!`${example.item.question} ${example.item.id} ${example.item.category}`.toLowerCase().includes(query)) return false;
		const before = example.results.student_before, after = example.results.student_after;
		if (outcome === "improved") return !!before && !!after && !before.correct && after.correct;
		if (outcome === "regressed") return !!before && !!after && before.correct && !after.correct;
		if (outcome === "failed") return available.some(slot => example.results[slot] && !example.results[slot]!.correct);
		return true;
	});
	const maxPage = Math.max(0, Math.ceil(examples.length / pageSize) - 1);
	const current = Math.min(page, maxPage);
	return <section>
		<div className="example-controls">
			<div className="search"><Search size={16} /><Input aria-label="Search questions" placeholder="Search question, ID, or subject" value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} /></div>
			<Choice label="Filter by outcome" value={outcome} onChange={v => { setOutcome(v); setPage(0); }} options={[
				{ value: "all", label: "All questions" },
				{ value: "improved", label: "Student gained (wrong → right)" },
				{ value: "regressed", label: "Student lost (right → wrong)" },
				{ value: "failed", label: "Any run wrong" },
			]} />
			<span>{examples.length} of {report.samples.length} questions</span>
		</div>
		{examples.length === 0 && <div className="empty">No questions match this filter.</div>}
		{examples.slice(current * pageSize, current * pageSize + pageSize).map(example => <article className="example" key={example.item.id}>
			<div className="example-heading"><span>{title(example.item.benchmark)} · {example.item.category.replaceAll("_", " ")}</span><code>{example.item.id}</code></div>
			<pre className="question">{example.item.question}</pre>
			{example.item.choices.length > 0 && <ul className="choices">{example.item.choices.map((choice, i) => <li key={i}>{choice}</li>)}</ul>}
			<p className="reference"><span>Expected answer</span>{example.item.answer}</p>
			<div className="answers-grid">{available.map(slot => {
				const result = example.results[slot];
				return <div className="answer" key={slot}>
					<div className="answer-heading">
						<SlotSwatch slot={slot} />{labels[slot]}
						{result?.correct
							? <span className="verdict pass"><CircleCheck size={14} />Correct</span>
							: <span className="verdict fail"><CircleX size={14} />{result ? result.status === "ok" ? "Wrong" : statusLabels[result.status] ?? result.status : "Not recorded"}</span>}
					</div>
					<pre>{result?.response || "No response recorded."}</pre>
					<small>Parsed answer: {result?.prediction ?? "none"} · {format(result?.latency_seconds, "s", 2)}</small>
				</div>;
			})}</div>
		</article>)}
		{examples.length > pageSize && <div className="pagination">
			<Button variant="outline" size="sm" disabled={current === 0} onClick={() => setPage(current - 1)}><ArrowLeft size={14} />Previous</Button>
			<span>Page {current + 1} of {maxPage + 1}</span>
			<Button variant="outline" size="sm" disabled={current >= maxPage} onClick={() => setPage(current + 1)}>Next<ArrowRight size={14} /></Button>
		</div>}
	</section>;
}
