"""Mode 2: Generate a deep-reading report for a single paper."""
from __future__ import annotations
from pathlib import Path

from research_helper import config
from research_helper.llm import client as llm
from research_helper.readers.arxiv_reader import PaperMeta
from research_helper.utils.cache import load_cache, save_cache

SYSTEM_PROMPT = """\
你是一位资深 AI 研究员，正在为自己撰写论文精读笔记。
写作要求：
- 语言为学术中文，表达精确，不啰嗦
- 每个 section 至少 200 字，重要内容可更长
- 用具体数字、公式、方法名支撑论点，不写空话
- 遇到方法细节时，解释清楚其设计动机，不只是罗列
- 知识库中有相关论文时，做有实质内容的横向对比，指出异同
- 不引用知识库和论文原文之外未出现的论文
- 所有数学公式必须使用 Markdown 格式：行内公式用 $...$，独立公式用 $$...$$，不得使用 \\( \\) 或 \\[ \\]\
"""

REPORT_TEMPLATE = """\
# {title}

**作者**：{authors}
**发表时间**：{published}
**Arxiv ID**：{arxiv_id}
**领域**：{categories}

---

## 1. 研究问题与动机

{q1}

## 2. 核心方法

{q2}

## 3. 实验设计与结果

{q3}

## 4. 与相关工作的比较

{q4}

## 5. 局限性与未来工作

{q5}

## 6. 我的评价与启发

{q6}
"""

# Each section is a separate LLM call for depth and focus
SECTION_PROMPTS = [
    # q1
    """\
请深入分析这篇论文的研究问题与动机，至少 200 字：
- 领域背景：该问题属于哪个研究方向，当前主流方法是什么
- 核心痛点：现有方法存在哪些根本性缺陷或局限
- 研究动机：作者为什么认为这个问题值得解决，重要性体现在哪里
- 论文目标：作者的核心 claim 是什么

{kb_section}论文标题：{title}

论文内容：
{content}""",

    # q2
    """\
请深入分析这篇论文的核心方法，至少 300 字：
- 整体架构：方法的总体设计思路和模块划分
- 关键创新：与此前方法相比，最核心的技术创新是什么，解决了哪个具体问题
- 重要细节：关键模块的设计（可引用公式、超参数、算法步骤），并解释每个设计选择背后的动机
- 实现要点：训练策略、目标函数、推理方式中有哪些值得注意的地方

{kb_section}论文标题：{title}

论文内容：
{content}""",

    # q3
    """\
请深入分析这篇论文的实验部分，至少 200 字：
- 任务与数据集：评估了哪些任务，使用了哪些数据集，规模如何
- 基线选取：与哪些方法进行了比较，这些基线的选取是否合理
- 主要结果：关键指标上的具体数字，提升幅度是否显著
- 消融实验：哪些组件被单独验证，结论是什么
- 结果可信度：实验设计有无明显缺陷或遗漏

{kb_section}论文标题：{title}

论文内容：
{content}""",

    # q4
    """\
请将这篇论文与相关工作进行深入比较，至少 200 字：
- 优势：本文方法在哪些方面明显优于已有工作，技术层面的原因是什么
- 不足：相比相关工作，本文在哪些场景或指标上仍有差距
- 差异化：本文与最相近的工作的本质区别是什么

【重要】若知识库中有相关论文，请优先与其进行具体对比，引用时注明论文标题和 arxiv ID。
不要编造知识库和论文原文中未出现的引用。

{kb_section}论文标题：{title}

论文内容：
{content}""",

    # q5
    """\
请分析这篇论文的局限性与未来方向，至少 150 字：
- 作者承认的局限：论文中明确提到的不足或适用范围限制
- 未被承认的潜在问题：你认为该方法可能存在但作者未讨论的问题
- 未来工作：论文提出或你认为值得探索的后续研究方向，尽量具体

{kb_section}论文标题：{title}

论文内容：
{content}""",

    # q6
    """\
请写出你对这篇论文的个人评价与研究启发，至少 150 字：
- 论文价值：这篇论文在领域内的贡献和地位如何
- 方法迁移：核心思路是否可以迁移到其他问题，如何迁移
- 对自己研究的启发：这篇论文给你带来了哪些具体的想法或新的研究问题

{kb_section}论文标题：{title}

论文内容：
{content}""",
]

KB_SECTION_TEMPLATE = """\
## 知识库上下文（已读论文，可用于对比）

{entries}

---
"""


def _build_kb_section(query_text: str, current_arxiv_id: str = "",
                      external_kb_entries: list | None = None) -> str:
    if external_kb_entries is not None:
        entries = [e for e in external_kb_entries if e.arxiv_id != current_arxiv_id][:5]
        if not entries:
            import sys
            print("[KB] 外部知识库为空或无相关论文", file=sys.stderr)
            return ""
        import sys
        print(f"[KB] 外部知识库提供 {len(entries)} 篇相关论文：", file=sys.stderr)
        for e in entries:
            print(f"  - {e.title} ({e.arxiv_id})", file=sys.stderr)
    else:
        try:
            from research_helper.kb import store
            entries = store.query(query_text, top_k=6)
        except Exception:
            return ""
        entries = [e for e in entries if e.arxiv_id != current_arxiv_id][:5]
        if not entries:
            import sys
            print("[KB] 知识库为空或无相关论文", file=sys.stderr)
            return ""
        import sys
        print(f"[KB] 检索到 {len(entries)} 篇相关论文：", file=sys.stderr)
        for e in entries:
            print(f"  - {e.title} ({e.arxiv_id})", file=sys.stderr)
    lines = []
    for e in entries:
        lines.append(
            f"- **{e.title}** ({e.published}, arxiv:{e.arxiv_id})\n"
            f"  {e.text[:600].strip()}"
        )
    return KB_SECTION_TEMPLATE.format(entries="\n\n".join(lines))


def _chunk_text(text: str) -> list[str]:
    size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _summarize_chunks(title: str, chunks: list[str]) -> str:
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = (
            f"这是论文《{title}》的第 {i+1}/{len(chunks)} 段，"
            "请提取方法、实验、结论等关键信息，输出 400 字以内的中文摘要：\n\n" + chunk
        )
        summaries.append(llm.complete(SYSTEM_PROMPT, prompt, max_tokens=1000))
    return "\n\n".join(f"[第{i+1}段摘要]\n{s}" for i, s in enumerate(summaries))


def _prepare_content(title: str, full_text: str) -> str:
    if len(full_text) <= config.CHUNK_SIZE * 2:
        return full_text
    return _summarize_chunks(title, _chunk_text(full_text))


def generate(
    paper_dir: Path,
    meta: PaperMeta,
    full_text: str,
    force: bool = False,
    external_kb_entries: list | None = None,
) -> Path:
    report_path = paper_dir / "report.md"
    if report_path.exists() and not force and external_kb_entries is None:
        return report_path

    cached = load_cache(paper_dir, "analysis")
    if cached and not force and external_kb_entries is None:
        answers = cached
    else:
        content = _prepare_content(meta.title, full_text)
        query_text = f"{meta.title}\n{meta.abstract}"
        kb_section = _build_kb_section(query_text, current_arxiv_id=meta.arxiv_id,
                                        external_kb_entries=external_kb_entries)

        answers = []
        for prompt_tpl in SECTION_PROMPTS:
            prompt = prompt_tpl.format(
                title=meta.title,
                content=content,
                kb_section=kb_section,
            )
            answers.append(llm.complete(SYSTEM_PROMPT, prompt, max_tokens=10000))

        save_cache(paper_dir, "analysis", answers)

    report = REPORT_TEMPLATE.format(
        title=meta.title,
        authors="，".join(meta.authors[:6]) + ("等" if len(meta.authors) > 6 else ""),
        published=meta.published,
        arxiv_id=meta.arxiv_id,
        categories="、".join(meta.categories) if meta.categories else "N/A",
        q1=answers[0],
        q2=answers[1],
        q3=answers[2],
        q4=answers[3],
        q5=answers[4],
        q6=answers[5],
    )

    paper_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def add_to_kb(meta: PaperMeta, report_path: Path) -> None:
    try:
        from research_helper.kb import store
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        store.add(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            published=meta.published,
            abstract=meta.abstract,
            report_text=report_text,
        )
    except Exception as exc:
        import sys
        print(f"[KB] Warning: failed to index paper — {exc}", file=sys.stderr)
