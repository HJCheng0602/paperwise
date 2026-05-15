"""
research-helper CLI

Commands:
  rh read   --pdf paper.pdf           # mode 2: local PDF
  rh read   --arxiv 2310.01234        # mode 2: arxiv ID (downloads PDF)
  rh survey --query "RAG" --max 20   # mode 1: domain survey
  rh kb list                          # list KB papers
  rh kb search "contrastive learning" # semantic search in KB
  rh kb stats                         # KB statistics
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from research_helper import config

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_dir(name: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", name)[:80]
    return config.OUTPUTS_DIR / safe


def _ensure_api_key() -> None:
    if config.LLM_PROVIDER == "anthropic" and not config.ANTHROPIC_API_KEY:
        console.print("[red]Error:[/] ANTHROPIC_API_KEY is not set. Add it to .env or environment.")
        sys.exit(1)
    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        console.print("[red]Error:[/] OPENAI_API_KEY is not set. Add it to .env or environment.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def main():
    """Research Helper — paper reading and domain survey tool."""
    pass


# ---------------------------------------------------------------------------
# rh read
# ---------------------------------------------------------------------------

@main.command()
@click.option("--pdf", "pdf_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to a local PDF file.")
@click.option("--arxiv", "arxiv_id", default=None,
              help="Arxiv paper ID (e.g. 2310.01234) or full URL.")
@click.option("--force", is_flag=True, default=False,
              help="Regenerate report even if it already exists.")
@click.option("--no-kb", is_flag=True, default=False,
              help="Skip knowledge base indexing after report generation.")
@click.option("--external-kb", "external_kb_path", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="Path to a JSON file of external KB entries (e.g. from Zotero MCP). "
                   "Replaces ChromaDB KB for this report.")
def read(pdf_path: Path | None, arxiv_id: str | None, force: bool, no_kb: bool,
         external_kb_path: Path | None):
    """Generate a deep-reading report for a single paper."""
    if not pdf_path and not arxiv_id:
        console.print("[red]Error:[/] Provide --pdf or --arxiv.")
        sys.exit(1)

    _ensure_api_key()

    external_entries = None
    if external_kb_path:
        from research_helper.kb.external import load_external_kb
        external_entries = load_external_kb(external_kb_path)
        if not external_entries:
            console.print("[yellow]Warning:[/] External KB file loaded but contains no valid entries.")

    from research_helper.readers import arxiv_reader, pdf_reader
    from research_helper.reports import single_paper
    import json

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:

        # --- Fetch or load metadata ---
        if arxiv_id:
            task = prog.add_task("Fetching metadata from Arxiv…")
            meta = arxiv_reader.fetch_meta(arxiv_id)
            prog.update(task, description=f"[green]Fetched:[/] {meta.title[:60]}")
            paper_dir = _paper_dir(meta.title)

            cached_pdf = paper_dir / f"{re.sub(r'[^\\w\\-]', '_', meta.arxiv_id)}.pdf"
            if not cached_pdf.exists() or force:
                prog.update(task, description="Downloading PDF…")
                pdf_path = arxiv_reader.download_pdf(meta, paper_dir)
            else:
                pdf_path = cached_pdf
            prog.update(task, description="[green]PDF ready[/]")
        else:
            meta = _meta_from_pdf(pdf_path)
            paper_dir = _paper_dir(pdf_path.stem)

        # Save meta.json
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": meta.arxiv_id,
                    "title": meta.title,
                    "authors": meta.authors,
                    "published": meta.published,
                    "abstract": meta.abstract,
                    "categories": meta.categories,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # --- Extract text ---
        task2 = prog.add_task("Extracting text from PDF…")
        full_text = pdf_reader.extract_text(pdf_path)
        prog.update(task2, description=f"[green]Extracted[/] {len(full_text):,} chars")

        # --- Generate report ---
        task3 = prog.add_task("Generating report with LLM…")
        report_path = single_paper.generate(paper_dir, meta, full_text, force=force,
                                             external_kb_entries=external_entries)
        prog.update(task3, description=f"[green]Report saved →[/] {report_path}")

        # --- Index into KB ---
        if not no_kb:
            task4 = prog.add_task("Indexing into knowledge base…")
            single_paper.add_to_kb(meta, report_path)
            prog.update(task4, description="[green]Added to KB[/]")

    from research_helper.utils import cost_tracker
    cost_tracker.flush_to_log(meta.title[:60])
    s = cost_tracker.session_summary()

    console.rule()
    console.print(f"[bold green]Done![/] Report: [cyan]{report_path}[/]")
    console.print(f"Directory: [cyan]{paper_dir}[/]")
    console.print(
        f"[dim]本次费用：[/][yellow]${s['cost_usd']:.4f}[/]"
        f"[dim]  (in {s['input_tokens']:,} / out {s['output_tokens']:,} / embed {s['embed_tokens']:,} tokens,"
        f" {s['calls']} calls)[/]"
    )


def _meta_from_pdf(pdf_path: Path):
    from research_helper.readers.arxiv_reader import PaperMeta
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        info = doc.metadata
        doc.close()
        title = info.get("title") or pdf_path.stem
        author = info.get("author") or "Unknown"
        authors = [a.strip() for a in author.split(";") if a.strip()] or ["Unknown"]
    except Exception:
        title = pdf_path.stem
        authors = ["Unknown"]

    return PaperMeta(
        arxiv_id="local",
        title=title,
        authors=authors,
        abstract="",
        published="",
        pdf_url="",
    )


# ---------------------------------------------------------------------------
# rh survey
# ---------------------------------------------------------------------------

@main.command()
@click.option("--query", "-q", required=True, help="Research keyword / topic to survey.")
@click.option("--max", "max_papers", default=config.SURVEY_MAX_PAPERS, show_default=True,
              help="Maximum number of papers to retrieve.")
@click.option("--force", is_flag=True, default=False,
              help="Regenerate survey even if it already exists.")
def survey(query: str, max_papers: int, force: bool):
    """Search Arxiv and generate a domain survey report."""
    _ensure_api_key()

    from research_helper.readers import arxiv_reader
    from research_helper.reports import survey as survey_report

    output_dir = _paper_dir(f"survey_{query}")
    report_path = output_dir / "survey.md"

    if report_path.exists() and not force:
        console.print(f"[yellow]Survey already exists:[/] {report_path}")
        console.print("Use --force to regenerate.")
        return

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        task = prog.add_task(f'Searching Arxiv for "{query}" (max {max_papers})…')
        papers = arxiv_reader.search_papers(query, max_results=max_papers)
        prog.update(task, description=f"[green]Found {len(papers)} papers[/]")

        task2 = prog.add_task("Generating survey report with LLM…")
        path = survey_report.generate(query, papers, output_dir)
        prog.update(task2, description=f"[green]Survey saved →[/] {path}")

    from research_helper.utils import cost_tracker
    cost_tracker.flush_to_log(f"survey:{query[:40]}")
    s = cost_tracker.session_summary()

    console.rule()
    console.print(f"[bold green]Done![/] Survey: [cyan]{path}[/]")
    console.print(f"Papers list: [cyan]{output_dir / 'papers.json'}[/]")
    console.print(
        f"[dim]本次费用：[/][yellow]${s['cost_usd']:.4f}[/]"
        f"[dim]  (in {s['input_tokens']:,} / out {s['output_tokens']:,} tokens)[/]"
    )


# ---------------------------------------------------------------------------
# rh kb
# ---------------------------------------------------------------------------

@main.group()
def kb():
    """Manage the local knowledge base."""
    pass


@kb.command("list")
def kb_list():
    """List all papers indexed in the knowledge base."""
    from research_helper.kb import store
    papers = store.list_papers()
    if not papers:
        console.print("[yellow]Knowledge base is empty.[/] Run [cyan]rh read[/] to add papers.")
        return

    table = Table(title=f"Knowledge Base ({len(papers)} papers)", show_lines=False)
    table.add_column("Arxiv ID", style="cyan", no_wrap=True)
    table.add_column("Published", style="dim", no_wrap=True)
    table.add_column("Title")
    for p in papers:
        table.add_row(p["arxiv_id"], p["published"], p["title"])
    console.print(table)


@kb.command("search")
@click.argument("query")
@click.option("--top", default=5, show_default=True, help="Number of results to return.")
def kb_search(query: str, top: int):
    """Semantic search over the knowledge base."""
    from research_helper.kb import store
    with Progress(SpinnerColumn(), TextColumn("Searching…"), console=console) as prog:
        prog.add_task("")
        results = store.query(query, top_k=top)

    if not results:
        console.print("[yellow]No results found.[/] Is the knowledge base empty?")
        return

    for i, r in enumerate(results, 1):
        sim = 1 - r.distance  # cosine distance → similarity
        console.print(f"\n[bold]{i}. {r.title}[/] [dim]({r.published}, {r.arxiv_id})[/]")
        console.print(f"   [cyan]相似度：{sim:.2f}[/]  来源：{r.source}")
        console.print(f"   {r.text[:200].strip()}…")


@kb.command("stats")
def kb_stats():
    """Show knowledge base statistics."""
    from research_helper.kb import store
    n = store.count()
    console.print(f"[bold]Knowledge base:[/] {n} papers indexed")
    db_path = Path("outputs/.kb")
    if db_path.exists():
        size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
        console.print(f"Storage: [cyan]{db_path}[/] ({size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# rh graph
# ---------------------------------------------------------------------------

@main.command()
@click.option("--out", "out_dir", default="outputs/graph", show_default=True,
              help="Output directory for graph files.")
@click.option("--threshold", default=0.55, show_default=True,
              help="Cosine similarity threshold for paper-paper edges.")
@click.option("--no-cache", is_flag=True, default=False,
              help="Re-extract concepts/relations even if cached.")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the HTML graph in the default browser after generation.")
def graph(out_dir: str, threshold: float, no_cache: bool, open_browser: bool):
    """Build a knowledge graph from all papers in outputs/."""
    from research_helper.kb.graph import build, export_json, export_html
    import shutil

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if no_cache:
        for d in config.OUTPUTS_DIR.iterdir():
            cache = d / "graph_info.json"
            if cache.exists():
                cache.unlink()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        task = prog.add_task("Building knowledge graph…")

        def _update(msg: str):
            prog.update(task, description=msg)

        nodes, edges = build(similarity_threshold=threshold, progress_cb=_update)

        prog.update(task, description="Exporting JSON…")
        json_path = out_path / "graph.json"
        export_json(nodes, edges, json_path)

        prog.update(task, description="Rendering HTML…")
        html_path = out_path / "graph.html"
        export_html(nodes, edges, html_path)

        prog.update(task, description="[green]Done![/]")

    paper_count = sum(1 for n in nodes if n.type == "paper")
    concept_count = sum(1 for n in nodes if n.type == "concept")
    console.rule()
    console.print(
        f"[bold green]Knowledge graph built![/]  "
        f"{paper_count} 篇论文 · {concept_count} 个概念 · {len(edges)} 条边"
    )
    console.print(f"HTML:  [cyan]{html_path}[/]")
    console.print(f"JSON:  [cyan]{json_path}[/]")

    from research_helper.utils import cost_tracker
    cost_tracker.flush_to_log("graph")
    s = cost_tracker.session_summary()
    if s["calls"]:
        console.print(
            f"[dim]本次费用：[/][yellow]${s['cost_usd']:.4f}[/]"
            f"[dim]  ({s['input_tokens']:,} in / {s['output_tokens']:,} out tokens, {s['calls']} calls)[/]"
        )

    if open_browser:
        import webbrowser
        webbrowser.open(html_path.resolve().as_uri())


# ---------------------------------------------------------------------------
# rh cost
# ---------------------------------------------------------------------------

@main.command()
@click.option("--last", default=10, show_default=True, help="显示最近 N 条记录")
def cost(last: int):
    """Show API cost history."""
    from research_helper.utils import cost_tracker
    from rich.table import Table

    entries = cost_tracker.read_log()
    if not entries:
        console.print("[yellow]暂无消费记录。[/] 运行 rh read 或 rh survey 后会自动记录。")
        return

    table = Table(title="API 消费记录", show_lines=False)
    table.add_column("时间", style="dim", no_wrap=True)
    table.add_column("任务")
    table.add_column("模型", style="dim")
    table.add_column("Input tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("Embed tok", justify="right")
    table.add_column("费用 (USD)", justify="right", style="yellow")

    for e in entries[-last:]:
        table.add_row(
            e["ts"],
            e["label"],
            e.get("model", ""),
            f"{e.get('input_tokens', 0):,}",
            f"{e.get('output_tokens', 0):,}",
            f"{e.get('embed_tokens', 0):,}",
            f"${e['cost_usd']:.4f}",
        )

    console.print(table)
    total = cost_tracker.total_cost()
    console.print(f"\n[bold]累计总费用：[/][yellow]${total:.4f}[/]  (~¥{total * 7.2:.2f})")


if __name__ == "__main__":
    main()
