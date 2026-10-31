"""Build the self-contained research report from Markdown and source registers."""

# Embedded document paragraphs and HTML/CSS remain unwrapped.
# ruff: noqa: E501

import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
NOTES = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MD = REPORTS / "Frontier distillation defense design.md"
raw = []
for name in ["frontier-sources.json", "detection-sources.json", "output-sources.json"]:
	data = json.loads((NOTES / name).read_text())
	for record in data["sources"]:
		raw.append(dict(record, research_register=name))


def canonical(url):
	match = re.search(r"arxiv.org/(?:abs|html|pdf)/(\d+\.\d+)", url)
	if match:
		return "https://arxiv.org/abs/" + match[1]
	parts = urlsplit(url)
	return urlunsplit(
		("https", parts.netloc.lower().removeprefix("www."), parts.path.rstrip("/"), "", "")
	)


sources = []
by_canonical = {}
by_title = {}
for record in raw:
	key = canonical(record["url"])
	title = re.sub(r"\W+", "", record["title"].lower())
	entry = by_canonical.get(key) or (by_title.get(title) if len(title) > 35 else None)
	if entry is None:
		entry = {
			"id": record["id"],
			"aliases": [],
			"title": record["title"],
			"canonical_url": key,
			"url": record["url"],
			"records": [],
		}
		sources.append(entry)
	entry["aliases"].append(record["id"])
	entry["records"].append(record)
	by_canonical[key] = entry
	by_title[title] = entry


def category(entry):
	kinds = " ".join(r["kind"] for r in entry["records"]).lower()
	if "anonymous" in kinds or "reddit" in entry["url"]:
		return "Anonymous, unverified account"
	if any(
		x in kinds
		for x in [
			"provider",
			"official documentation",
			"official api documentation",
			"support",
			"transparency",
		]
	):
		return "Provider disclosure or official documentation"
	if "submission" in kinds:
		return "Submission with partial verification"
	if "preprint" in kinds or "arxiv record" in kinds:
		return "Preprint or mixed preprint record"
	if "peer" in kinds or "conference" in kinds or "journal" in kinds or "proceedings" in kinds:
		return "Peer-reviewed or proceedings record"
	return "Code, practitioner writing or other primary material"


for entry in sources:
	entry["category"] = category(entry)
counts = Counter(s["category"] for s in sources)
register = {
	"title": "Frontier distillation defense source register",
	"retrieved_at": "2026-10-31",
	"raw_record_count": len(raw),
	"deduplicated_source_count": len(sources),
	"category_counts": dict(counts),
	"deduplication": "Exact normalized URL, arXiv work identifier across versions/formats, and exact distinctive normalized title longer than 35 characters; generic titles such as Rate limits remain separate. Every original ID and record retained. Different inspected versions remain visible in records.",
	"sources": sources,
}
(REPORTS / "Frontier distillation defense sources.json").write_text(
	json.dumps(register, indent="\t", ensure_ascii=False) + "\n"
)
lookup = {r["id"]: r for r in raw}
text = MD.read_text()
text = text.split("\n<!-- GENERATED SOURCES -->")[0].rstrip() + "\n"
text += "\n<!-- GENERATED SOURCES -->\n\n"
text += f"The register contains **{len(sources)} deduplicated sources from {len(raw)} original research records**. "
text += (
	"Categories are editorial evidence labels, not quality scores: "
	+ "; ".join(f"{n} {kind.lower()}" for kind, n in counts.items())
	+ ".\n\n"
)
for entry in sources:
	text += f"### {', '.join(entry['aliases'])}: {entry['title']}\n\n"
	text += f"[{entry['title']}]({entry['url']}). Evidence category: {entry['category']}.\n\n"
	for r in entry["records"]:
		limit = r.get("limitations", [])
		if isinstance(limit, list):
			limit = "; ".join(limit)
		text += f"{r['id']}. {r.get('authors_or_org', 'Author metadata unavailable')}. Date/version label: {r.get('date', 'Not established')}. Type: {r['kind']}. Verification: {r['verification']}. Limitation: {limit}.\n\n"
MD.write_text(text)


# Small deterministic renderer for this report's Markdown subset; no external packages.
def inline(value):
	placeholders = []

	def keep(fragment):
		placeholders.append(fragment)
		return f"\x00{len(placeholders) - 1}\x00"

	value = re.sub(r"`([^`]+)`", lambda m: keep("<code>" + html.escape(m[1]) + "</code>"), value)
	value = re.sub(
		r"\[([^\]]+)\]\(([^)]+)\)",
		lambda m: keep(
			'<a href="' + html.escape(m[2], quote=True) + '">' + html.escape(m[1]) + "</a>"
		),
		value,
	)
	value = re.sub(
		r"\[([FDO]\d{2})\]",
		lambda m: keep(
			'<a class="citation" href="'
			+ html.escape(lookup[m[1]]["url"], quote=True)
			+ '" title="'
			+ html.escape(lookup[m[1]]["title"], quote=True)
			+ '">'
			+ m[1]
			+ "</a>"
		),
		value,
	)
	value = html.escape(value)
	value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
	value = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m[1])], value)
	return value


slug_counts = Counter()


def slug(value):
	base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
	slug_counts[base] += 1
	return base if slug_counts[base] == 1 else f"{base}-{slug_counts[base]}"


lines = text.splitlines()
body = []
toc = []
i = 0
in_bib = False
source_open = False
while i < len(lines):
	line = lines[i]
	if not line.strip() or line.startswith("<!--"):
		i += 1
		continue
	if line.startswith("```"):
		code = []
		i += 1
		while i < len(lines) and not lines[i].startswith("```"):
			code.append(lines[i])
			i += 1
		body.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
		i += 1
		continue
	if line.startswith("#"):
		level = len(line) - len(line.lstrip("#"))
		title = line[level:].strip()
		anchor = slug(title)
		if title == "Search scope and annotated source register":
			in_bib = True
		if in_bib and level == 3:
			if source_open:
				body.append("</section>")
			aliases = title.split(":", 1)[0].split(", ")
			entry = next(s for s in sources if aliases[0] in s["aliases"])
			body.append(
				'<section class="source-entry" data-category="'
				+ html.escape(entry["category"], quote=True)
				+ '">'
			)
			source_open = True
		body.append(
			f'<h{level} id="{anchor}">{inline(title)}<a class="anchor" href="#{anchor}" aria-label="Link to this section">#</a></h{level}>'
		)
		if level == 2:
			toc.append((anchor, title))
		if title == "Search scope and annotated source register":
			body.append(
				'<div class="source-tools" hidden><label for="source-search">Search the source register</label><input id="source-search" type="search" placeholder="Title, author, source ID, evidence or limitation"><label for="source-category">Evidence category</label><select id="source-category"><option value="">All categories</option>'
				+ "".join("<option>" + html.escape(k) + "</option>" for k in counts)
				+ '</select><p id="source-count" role="status"></p></div>'
			)
		i += 1
		continue
	if line.startswith("|"):
		rows = []
		while i < len(lines) and lines[i].startswith("|"):
			cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
			if not all(re.fullmatch(r":?-+:?", x) for x in cells):
				rows.append(cells)
			i += 1
		body.append(
			'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable comparison table"><table><thead><tr>'
			+ "".join('<th scope="col">' + inline(x) + "</th>" for x in rows[0])
			+ "</tr></thead><tbody>"
			+ "".join(
				"<tr>" + "".join("<td>" + inline(x) + "</td>" for x in row) + "</tr>"
				for row in rows[1:]
			)
			+ "</tbody></table></div>"
		)
		continue
	paragraph = [line]
	i += 1
	while (
		i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", "<!--"))
	):
		paragraph.append(lines[i])
		i += 1
	body.append("<p>" + inline(" ".join(paragraph)) + "</p>")
if source_open:
	body.append("</section>")

css = """
:root { color-scheme: light dark; --ink:#182839; --muted:#4e6376; --accent:#1d607a; --line:#b6c7d5; --wash:#eaf1f7; --paper:#f8fbfd; --focus:#9a4a02; }
* { box-sizing:border-box; }
html,body { background:transparent; margin:0; padding:0; }
body { color:var(--ink); font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:17px; line-height:1.7; }
a { color:var(--accent); text-underline-offset:3px; text-decoration-thickness:1px; overflow-wrap:anywhere; }
a:hover { text-decoration-thickness:2px; }
a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible { outline:3px solid var(--focus); outline-offset:4px; }
.report { max-width:1460px; margin:0 auto; background:transparent; }
.masthead { padding:2.5rem 1.2rem 1.5rem; border-bottom:4px solid var(--accent); }
.eyebrow { margin:0; font:600 .76rem/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.11em; text-transform:uppercase; color:var(--muted); }
.masthead h1 { font-size:clamp(2.25rem,5.4vw,4.9rem); font-weight:650; letter-spacing:-.055em; line-height:1.02; max-width:1040px; margin:.8rem 0 1rem; }
.deck { max-width:850px; margin:0; color:var(--muted); font-size:1.1rem; }
.layout { display:grid; grid-template-columns:244px minmax(0,1fr); gap:2.5rem; padding:0 1.2rem; }
nav { align-self:start; position:sticky; top:1rem; margin-top:2rem; font-size:.83rem; line-height:1.45; }
nav summary { font-weight:700; cursor:pointer; padding:.5rem 0; }
nav ol { padding-left:1.25rem; margin:.7rem 0 1.2rem; }
nav li { padding:.32rem 0; }
nav a { color:var(--muted); text-decoration:none; }
nav a:hover { color:var(--accent); text-decoration:underline; }
nav .meta { color:var(--muted); font-size:.75rem; border-top:1px solid var(--line); padding-top:1rem; }
main { min-width:0; max-width:1000px; padding:1rem 0 4rem; }
main>p:first-child { font-size:1.17rem; line-height:1.7; }
p { margin:1rem 0; max-width:86ch; }
h2 { font-size:clamp(1.65rem,2.7vw,2.35rem); line-height:1.2; letter-spacing:-.025em; margin:3.5rem 0 1.5rem; border-top:1px solid var(--line); padding-top:1.25rem; scroll-margin-top:1rem; }
h3 { font-size:1.18rem; line-height:1.4; margin:2rem 0 .6rem; scroll-margin-top:1rem; }
.anchor { font-size:.7em; margin-left:.5rem; text-decoration:none; opacity:.35; }
.anchor:hover,.anchor:focus-visible { opacity:1; }
strong { font-weight:700; }
.citation { font:600 .75em/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:nowrap; margin-inline:.08rem; }
code { font: .86em/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; background:var(--wash); padding:.08em .25em; overflow-wrap:anywhere; }
pre { background:var(--wash); padding:1rem; overflow:auto; line-height:1.5; }
pre code { padding:0; background:none; white-space:pre; }
.table-scroll { overflow:auto; margin:1.7rem 0; border-block:1px solid var(--line); }
table { border-collapse:collapse; width:100%; min-width:760px; font-size:.84rem; line-height:1.55; }
th,td { text-align:left; vertical-align:top; padding:.8rem .75rem; border-bottom:1px solid var(--line); }
th { background:var(--wash); font-size:.76rem; letter-spacing:.025em; }
tr:last-child td { border-bottom:0; }
td:first-child { font-weight:600; min-width:145px; }
.source-entry { padding:.6rem 0 1rem; border-bottom:1px solid var(--line); }
.source-entry h3 { margin:1rem 0 .6rem; }
.source-entry p { font-size:.87rem; line-height:1.6; color:var(--muted); }
.source-tools { margin:1.5rem 0; padding:1rem 0; border-block:2px solid var(--accent); }
.source-tools label { display:block; font-size:.85rem; font-weight:650; margin:.6rem 0 .25rem; }
input,select { font:inherit; color:var(--ink); background:var(--paper); border:1px solid var(--line); border-radius:0; padding:.65rem; width:100%; max-width:640px; }
#source-count { font-size:.8rem; }
.skip { position:absolute; left:1rem; top:-10rem; z-index:10; background:var(--paper); padding:.6rem; }
.skip:focus { top:1rem; }
.flow { margin:2rem 0; padding:1rem 0; border-block:2px solid var(--accent); }
.flow figcaption { font-size:.82rem; color:var(--muted); margin-bottom:.8rem; }
.flow ol { display:grid; grid-template-columns:repeat(4,1fr); list-style:none; margin:0; padding:0; gap:1rem; }
.flow li { border-top:4px solid var(--line); padding-top:.6rem; font-size:.85rem; }
.flow strong { display:block; color:var(--accent); }
.flow small { display:block; margin-top:.25rem; }
footer { border-top:1px solid var(--line); margin-top:2rem; padding:1rem 0; color:var(--muted); font-size:.8rem; }
@media(prefers-color-scheme:dark) { :root { --ink:#dce7ef; --muted:#a5b9c9; --accent:#85c6db; --line:#40586c; --wash:#1c2f40; --paper:#142536; --focus:#f0ba84; } }
@media(max-width:1000px) { .layout { grid-template-columns:1fr; gap:0; } nav { position:static; margin-top:1rem; } nav ol { columns:2; column-gap:2rem; } nav li { break-inside:avoid; } nav .meta { display:none; } main { max-width:none; } }
@media(max-width:560px) { body { font-size:16px; } .masthead { padding:1.5rem .85rem 1rem; } .layout { padding:0 .85rem; } nav ol { columns:1; } h2 { margin-top:2.5rem; } .flow ol { grid-template-columns:1fr 1fr; } }
@media print { :root { color-scheme:light; --ink:#000; --muted:#333; --line:#bbb; --accent:#164f65; --wash:#f2f2f2; } body { font-size:10pt; } .masthead { padding:0 0 1rem; } .layout { display:block; padding:0; } nav,.source-tools,.anchor,.skip { display:none!important; } main { max-width:none; } h2,h3 { break-after:avoid; } .source-entry { break-inside:avoid; } .table-scroll { overflow:visible; } table { min-width:0; font-size:8pt; } th,td { padding:.4rem; } a { color:inherit; } pre { white-space:pre-wrap; } h2 { margin-top:2rem; } [hidden] { display:block!important; } }
"""
flow = '<figure class="flow"><figcaption>Proposed release-control sequence. Conceptual architecture, not measured effectiveness.</figcaption><ol><li><strong>Resolve identity</strong><small>Account, verified organization, uncertain links</small></li><li><strong>Reserve allowance</strong><small>Atomic request, output and concurrency grants</small></li><li><strong>Release under policy</strong><small>Useful output tier, optional checks, metered prefixes</small></li><li><strong>Settle and investigate</strong><small>Durable usage, conditional evidence, review and appeal</small></li></ol></figure>'
content = "\n".join(body)
# Main title is in the masthead; insert one conceptual diagram at the architecture boundary.
content = re.sub(r"^<h1.*?</h1>\n?", "", content, count=1)
content = content.replace(
	'<h2 id="an-evolutionary-trace-architecture-can-make-acquisition-auditable">',
	flow + '<h2 id="an-evolutionary-trace-architecture-can-make-acquisition-auditable">',
)
script = """
const tools=document.querySelector('.source-tools');
if(tools){
	tools.hidden=false;
	const query=document.querySelector('#source-search');
	const category=document.querySelector('#source-category');
	const entries=[...document.querySelectorAll('.source-entry')];
	const update=()=>{
		let visible=0;
		for(const entry of entries){
			const matches=entry.textContent.toLowerCase().includes(query.value.trim().toLowerCase())&&(!category.value||entry.dataset.category===category.value);
			entry.hidden=!matches;
			if(matches)visible++;
		}
		document.querySelector('#source-count').textContent=`${visible} of ${entries.length} sources shown. Filters affect only the bibliography.`;
	};
	query.addEventListener('input',update);category.addEventListener('change',update);update();
	window.addEventListener('beforeprint',()=>{for(const entry of entries)entry.hidden=false;});
	window.addEventListener('afterprint',update);
}
"""
html_doc = (
	'<!doctype html>\n<html lang="en">\n<head>\n\t<meta charset="utf-8">\n\t<meta name="viewport" content="width=device-width, initial-scale=1">\n\t<meta name="description" content="An evidence-driven design review of frontier anti-distillation defenses and a practical TRACE research roadmap, October 2026.">\n\t<title>Build defenses around acquired model capability</title>\n\t<style>'
	+ css
	+ '</style>\n</head>\n<body>\n\t<a class="skip" href="#report-content">Skip to report</a>\n\t<div class="report">\n\t\t<header class="masthead"><p class="eyebrow">TRACE research dossier / 31 October 2026</p><h1>Build defenses around acquired model capability</h1><p class="deck">A practical frontier-provider design, a bounded TRACE roadmap and an originality test. Proposed architectures have no measured TRACE effectiveness gain.</p></header>\n\t\t<div class="layout">\n\t\t\t<nav aria-label="Report contents"><details open><summary>Read the report</summary><ol>'
	+ "".join('<li><a href="#' + a + '">' + html.escape(t) + "</a></li>" for a, t in toc)
	+ '</ol></details><p class="meta">'
	+ str(len(sources))
	+ " distinct sources / "
	+ str(len(raw))
	+ ' research records<br>Offline report, live citation links<br>Evidence checked to 2026-10-31</p></nav>\n\t\t\t<main id="report-content">\n'
	+ content
	+ '\n<footer>Research synthesis only. No production implementation, live-service change, new teacher campaign or student experiment was performed. <a href="Frontier%20distillation%20defense%20sources.json">Machine-readable source register</a> · <a href="Frontier%20distillation%20defense%20design.md">Markdown report</a></footer>\n\t\t\t</main>\n\t\t</div>\n\t</div>\n\t<script>'
	+ script
	+ "</script>\n</body>\n</html>\n"
)
# Tabs are the indentation convention. Generated text nodes stay on whole lines.
(REPORTS / "Frontier distillation defense design.html").write_text(html_doc)
print(
	json.dumps(
		{
			"narrative_words": len(text.split("<!-- GENERATED SOURCES -->")[0].split()),
			"total_markdown_words": len(text.split()),
			"raw_records": len(raw),
			"deduplicated_sources": len(sources),
			"categories": dict(counts),
			"html_bytes": len(html_doc.encode()),
		},
		indent=2,
	)
)
