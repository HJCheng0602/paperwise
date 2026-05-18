"""Build a knowledge graph from all papers in the KB / outputs directory."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from research_helper import config
from research_helper.llm import client as llm

# ── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = "你是论文信息提取助手。只输出合法 JSON，不加 markdown 代码块或任何其他内容。"

_EXTRACT_PROMPT = (
    "从以下论文精读报告中提取结构化信息。\n\n"
    "输出严格 JSON，包含两个字段：\n"
    "1. concepts: 列表，3-8 个核心概念/方法标签，每个不超过 5 词，优先英文专有名词。"
    '例如 ["KV Cache", "RoPE", "RAG", "Speculative Decoding"]\n'
    "2. relations: 列表，该论文与其他论文的显式关系。每项包含："
    " target_title（被比较论文标题）, target_arxiv（arxiv ID 或 null）,"
    ' relation（builds_on / compares_to / extends / contradicts）, description（一句话中文说明）。\n\n'
    "报告（前 6000 字）：\n{report}"
)

# ── Data classes ──────────────────────────────────────────────────────────────

EdgeType = Literal["similar_to", "builds_on", "compares_to", "extends", "contradicts", "uses"]


@dataclass
class GNode:
    id: str
    type: Literal["paper", "concept"]
    label: str
    arxiv_id: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)


@dataclass
class GEdge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    description: str = ""


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_info(paper_dir: Path, report: str) -> dict:
    cache = paper_dir / "graph_info.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    raw = llm.complete(_SYSTEM, _EXTRACT_PROMPT.format(report=report[:6000]), max_tokens=3000)
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        info = json.loads(raw)
    except Exception:
        # Last-resort: extract first {...} block
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        try:
            info = json.loads(m.group()) if m else {}
        except Exception:
            info = {}

    info.setdefault("concepts", [])
    info.setdefault("relations", [])

    cache.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


# ── Graph builder ─────────────────────────────────────────────────────────────

def build(
    outputs_dir: Path | None = None,
    similarity_threshold: float = 0.55,
    progress_cb=None,
    skip_similarity: bool = False,
) -> tuple[list[GNode], list[GEdge]]:
    """Build graph from all papers in outputs_dir. Returns (nodes, edges)."""
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR

    paper_dirs = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )

    nodes: list[GNode] = []
    edges: list[GEdge] = []
    paper_nodes: dict[str, GNode] = {}       # arxiv_id → GNode
    concept_nodes: dict[str, GNode] = {}     # label → GNode
    title_to_arxiv: dict[str, str] = {}      # lower full title → arxiv_id (for substring match)

    # ── Pass 1: paper nodes ───────────────────────────────────────────────────
    paper_infos: list[tuple[str, dict]] = []

    for i, paper_dir in enumerate(paper_dirs):
        if progress_cb:
            progress_cb(f"Processing {paper_dir.name} ({i+1}/{len(paper_dirs)})…")

        meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
        arxiv_id = meta["arxiv_id"]
        title = meta["title"]

        node = GNode(
            id=f"paper:{arxiv_id}",
            type="paper",
            label=title,
            arxiv_id=arxiv_id,
            published=meta.get("published", ""),
            categories=meta.get("categories", []),
        )
        nodes.append(node)
        paper_nodes[arxiv_id] = node
        title_to_arxiv[title.lower()] = arxiv_id

        report_path = paper_dir / "report.md"
        if report_path.exists():
            info = _extract_info(paper_dir, report_path.read_text(encoding="utf-8"))
            paper_infos.append((arxiv_id, info))

    # ── Pass 2: concept nodes + paper→concept edges ───────────────────────────
    for arxiv_id, info in paper_infos:
        for concept in info.get("concepts", []):
            concept = concept.strip()
            if not concept:
                continue
            if concept not in concept_nodes:
                cnode = GNode(id=f"concept:{concept}", type="concept", label=concept)
                concept_nodes[concept] = cnode
                nodes.append(cnode)
            edges.append(GEdge(
                source=f"paper:{arxiv_id}",
                target=f"concept:{concept}",
                type="uses",
            ))

    # ── Pass 3: explicit relation edges ──────────────────────────────────────
    for arxiv_id, info in paper_infos:
        for rel in info.get("relations", []):
            target = rel.get("target_arxiv") if rel.get("target_arxiv") in paper_nodes else None
            if not target:
                query = rel.get("target_title", "").lower().strip()
                # exact match first, then substring match (handles LLM short-form titles)
                target = title_to_arxiv.get(query) or next(
                    (aid for full, aid in title_to_arxiv.items()
                     if query and (query in full or full.startswith(query[:20]))),
                    None,
                )
            if not target or target not in paper_nodes or target == arxiv_id:
                continue
            edge_type = rel.get("relation", "compares_to")
            if edge_type not in ("builds_on", "compares_to", "extends", "contradicts"):
                edge_type = "compares_to"
            edges.append(GEdge(
                source=f"paper:{arxiv_id}",
                target=f"paper:{target}",
                type=edge_type,
                description=rel.get("description", ""),
            ))

    # ── Pass 4: similarity edges from vector store ───────────────────────────
    if not skip_similarity and len(paper_nodes) >= 2:
        if progress_cb:
            progress_cb("Computing similarity edges…")
        _add_similarity_edges(paper_nodes, edges, similarity_threshold)

    return nodes, edges


def _add_similarity_edges(
    paper_nodes: dict[str, GNode],
    edges: list[GEdge],
    threshold: float,
) -> None:
    try:
        from research_helper.kb import store
    except Exception:
        return

    existing_sim: set[tuple[str, str]] = set()

    for arxiv_id, node in paper_nodes.items():
        try:
            results = store.query(node.label, top_k=len(paper_nodes) * 3)
        except Exception:
            continue
        for r in results:
            if r.arxiv_id == arxiv_id or r.arxiv_id not in paper_nodes:
                continue
            sim = 1 - r.distance
            if sim < threshold:
                continue
            key = tuple(sorted([arxiv_id, r.arxiv_id]))
            if key in existing_sim:
                continue
            existing_sim.add(key)
            a, b = sorted([f"paper:{arxiv_id}", f"paper:{r.arxiv_id}"])
            edges.append(GEdge(source=a, target=b, type="similar_to", weight=round(sim, 3)))


# ── Export ────────────────────────────────────────────────────────────────────

def export_json(nodes: list[GNode], edges: list[GEdge], out_path: Path) -> None:
    data = {
        "nodes": [vars(n) for n in nodes],
        "edges": [vars(e) for e in edges],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_EDGE_META: dict[str, dict] = {
    "similar_to":  {"color": "#5b8dee", "label": "向量相似",    "dashes": True,  "arrows": False},
    "builds_on":   {"color": "#00a884", "label": "基于",        "dashes": False, "arrows": True},
    "extends":     {"color": "#00a884", "label": "扩展",        "dashes": False, "arrows": True},
    "compares_to": {"color": "#e07820", "label": "对比",        "dashes": False, "arrows": True},
    "contradicts": {"color": "#d02848", "label": "矛盾",        "dashes": False, "arrows": True},
    "uses":        {"color": "#9090bb", "label": "使用概念",    "dashes": False, "arrows": True},
}

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"/>
<title>Research Knowledge Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link  href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet"/>
<style>
  *{{ box-sizing:border-box; margin:0; padding:0 }}
  body{{ background:#fdf0f5; color:#2a1520; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden }}

  /* ── Top bar ── */
  #topbar{{
    position:fixed; top:0; left:0; right:0; height:52px; z-index:100;
    background:rgba(255,245,249,0.96); border-bottom:1px solid #f0c8d8;
    display:flex; align-items:center; gap:12px; padding:0 18px;
    backdrop-filter:blur(6px);
  }}
  #topbar h1{{ font-size:15px; font-weight:600; color:#c0407a; letter-spacing:.5px; white-space:nowrap }}
  #stats-line{{ font-size:12px; color:#b090a0; flex:1 }}
  #search-wrap{{ position:relative }}
  #search{{
    background:#fff0f4; border:1px solid #f0c8d8; border-radius:6px;
    color:#2a1520; font-size:13px; padding:6px 10px 6px 30px; width:200px; outline:none;
    transition:border-color .2s;
  }}
  #search:focus{{ border-color:#e0507a }}
  #search-wrap::before{{
    content:"⌕"; position:absolute; left:9px; top:50%; transform:translateY(-50%);
    color:#c8a0b8; font-size:16px; pointer-events:none;
  }}

  /* ── Toolbar buttons ── */
  .btn{{
    background:#fff0f4; border:1px solid #f0c8d8; border-radius:6px;
    color:#7a4060; font-size:12px; cursor:pointer; padding:5px 10px;
    transition:all .15s; white-space:nowrap;
  }}
  .btn:hover{{ background:#ffe0ea; border-color:#e0507a; color:#c0304a }}
  .btn.active{{ background:#ffd0e4; border-color:#e0507a; color:#c0304a }}

  /* ── Canvas area ── */
  #network-wrap{{
    position:fixed; top:52px; left:0; right:320px; bottom:0;
  }}
  #network{{ width:100%; height:100% }}

  /* ── Right sidebar ── */
  #sidebar{{
    position:fixed; top:52px; right:0; width:320px; bottom:0;
    background:#fff8fb; border-left:1px solid #f0c8d8;
    display:flex; flex-direction:column; overflow:hidden;
  }}

  /* legend panel */
  #legend{{
    padding:16px; border-bottom:1px solid #f0c8d8; flex-shrink:0;
  }}
  #legend h2{{ font-size:12px; font-weight:600; color:#b090a0; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px }}
  .leg-row{{ display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; color:#5a3a48 }}
  .leg-dot{{ width:12px; height:12px; border-radius:50%; flex-shrink:0 }}
  .leg-diamond{{ width:10px; height:10px; transform:rotate(45deg); flex-shrink:0 }}
  .leg-line{{ width:22px; height:3px; flex-shrink:0; border-radius:2px }}
  .leg-dashed{{ background:repeating-linear-gradient(90deg,#5b8dee 0,#5b8dee 4px,transparent 4px,transparent 8px) }}

  /* filter panel */
  #filters{{
    padding:12px 16px; border-bottom:1px solid #f0c8d8; flex-shrink:0;
  }}
  #filters h2{{ font-size:12px; font-weight:600; color:#b090a0; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px }}
  .filter-row{{ display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:12px }}
  .filter-row input[type=checkbox]{{ accent-color:#e0507a; cursor:pointer }}
  .filter-row label{{ color:#5a3a48; cursor:pointer; flex:1 }}
  .filter-count{{ font-size:11px; color:#c8a0b8; margin-left:auto }}

  /* info panel */
  #info-panel{{
    flex:1; overflow-y:auto; padding:16px;
    scrollbar-width:thin; scrollbar-color:#f0c8d8 transparent;
  }}
  #info-panel::-webkit-scrollbar{{ width:4px }}
  #info-panel::-webkit-scrollbar-thumb{{ background:#f0c8d8; border-radius:2px }}
  #info-placeholder{{ color:#c8a0b8; font-size:13px; text-align:center; margin-top:40px; line-height:1.8 }}
  #info-content{{ display:none }}
  #info-type-badge{{
    display:inline-block; font-size:10px; font-weight:700; letter-spacing:1px;
    padding:2px 8px; border-radius:12px; text-transform:uppercase; margin-bottom:10px;
  }}
  #info-title{{ font-size:14px; font-weight:600; color:#1a2a4a; line-height:1.5; margin-bottom:8px }}
  #info-meta{{ font-size:12px; color:#8a6878; margin-bottom:12px; line-height:1.8 }}
  #info-neighbors h3{{ font-size:11px; color:#b090a0; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px }}
  .nb-item{{
    background:#fff0f4; border:1px solid #f0c8d8; border-radius:6px;
    padding:8px 10px; margin-bottom:6px; font-size:12px; cursor:pointer;
    transition:border-color .15s;
  }}
  .nb-item:hover{{ border-color:#e0507a }}
  .nb-item .nb-label{{ color:#2a1520; line-height:1.4 }}
  .nb-item .nb-rel{{ font-size:11px; margin-top:3px }}

  /* loading overlay */
  #loading{{
    position:fixed; inset:0; background:#fdf0f5;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    z-index:999; transition:opacity .4s;
  }}
  #loading-spinner{{
    width:36px; height:36px; border:3px solid #f0c8d8;
    border-top-color:#e0507a; border-radius:50%; animation:spin .7s linear infinite;
    margin-bottom:16px;
  }}
  @keyframes spin{{ to{{ transform:rotate(360deg) }} }}
  #loading p{{ color:#b090a0; font-size:13px }}
</style>
</head>
<body>

<div id="loading">
  <div id="loading-spinner"></div>
  <p>正在布局知识图谱…</p>
</div>

<!-- Top bar -->
<div id="topbar">
  <h1>📄 Research Knowledge Graph</h1>
  <span id="stats-line"></span>
  <div id="search-wrap">
    <input id="search" type="text" placeholder="搜索节点…" autocomplete="off"/>
  </div>
  <button class="btn" id="btn-fit" title="重置视图">⊡ 重置</button>
  <button class="btn active" id="btn-physics" title="切换物理引擎">⚛ 物理</button>
  <button class="btn" id="btn-concepts" title="隐藏/显示概念节点">◆ 概念</button>
</div>

<!-- Graph canvas -->
<div id="network-wrap"><div id="network"></div></div>

<!-- Right sidebar -->
<div id="sidebar">
  <div id="legend">
    <h2>节点类型</h2>
    <div class="leg-row"><div class="leg-dot" style="background:#5b8dee;border:2px solid #2a5ccc"></div> 论文</div>
    <div class="leg-row"><div class="leg-diamond" style="background:#f07030"></div> 概念</div>
    <h2 style="margin-top:12px">关系类型</h2>
    <div class="leg-row"><div class="leg-line leg-dashed"></div> 向量相似</div>
    <div class="leg-row"><div class="leg-line" style="background:#00c49a"></div> 基于 / 扩展</div>
    <div class="leg-row"><div class="leg-line" style="background:#f0882a"></div> 对比</div>
    <div class="leg-row"><div class="leg-line" style="background:#e03050"></div> 矛盾</div>
    <div class="leg-row"><div class="leg-line" style="background:#9090bb"></div> 使用概念</div>
  </div>

  <div id="filters">
    <h2>边过滤</h2>
    {filter_rows}
  </div>

  <div id="info-panel">
    <div id="info-placeholder">点击节点<br/>查看详情</div>
    <div id="info-content">
      <span id="info-type-badge"></span>
      <div id="info-title"></div>
      <div id="info-meta"></div>
      <div id="info-neighbors"></div>
    </div>
  </div>
</div>

<script>
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};

// ── Build vis datasets ────────────────────────────────────────────────────────
const EDGE_META = {edge_meta_json};

function nodeVisObj(n) {{
  const isPaper = n.type === "paper";
  const shortLabel = isPaper ? (n.label.length > 36 ? n.label.slice(0,34)+"…" : n.label)
                              : n.label;
  return {{
    id: n.id,
    label: shortLabel,
    fullLabel: n.label,
    type: n.type,
    arxiv_id: n.arxiv_id || "",
    published: n.published || "",
    categories: n.categories || [],
    shape: isPaper ? "dot" : "diamond",
    size: isPaper ? 20 : 12,
    color: isPaper
      ? {{ background:"#5b8dee", border:"#2a5ccc",
           highlight:{{ background:"#3a6cdd", border:"#1a4ccc" }},
           hover:{{ background:"#4a7cde", border:"#2a5ccc" }} }}
      : {{ background:"#f07030", border:"#c05010",
           highlight:{{ background:"#e06020", border:"#c05010" }},
           hover:{{ background:"#e06828", border:"#c05010" }} }},
    font: isPaper
      ? {{ size:13, color:"#ffffff", strokeWidth:3, strokeColor:"#2a4aaa" }}
      : {{ size:10, color:"#ffffff", strokeWidth:2, strokeColor:"#a04010" }},
  }};
}}

function edgeVisObj(e, idx) {{
  const m = EDGE_META[e.type] || {{ color:"#888", dashes:false, arrows:true }};
  const isSim = e.type === "similar_to";
  return {{
    id: idx,
    from: e.source,
    to: e.target,
    edgeType: e.type,
    description: e.description || "",
    weight: e.weight || 1,
    color: {{ color: m.color, highlight: m.color, hover: m.color, opacity: 0.75 }},
    width: isSim ? Math.max(1, e.weight * 3.5) : 1.8,
    dashes: m.dashes,
    arrows: m.arrows ? {{ to:{{ enabled:true, scaleFactor:0.6 }} }} : {{ to:{{ enabled:false }} }},
    title: isSim ? `相似度: ${{(e.weight||0).toFixed(3)}}` : (e.description || m.label),
    smooth: {{ type: "dynamic" }},
  }};
}}

const visNodes = new vis.DataSet(RAW_NODES.map(nodeVisObj));
const visEdges = new vis.DataSet(RAW_EDGES.map(edgeVisObj));

// ── Network ───────────────────────────────────────────────────────────────────
const container = document.getElementById("network");
const network = new vis.Network(container, {{ nodes: visNodes, edges: visEdges }}, {{
  physics: {{
    enabled: true,
    forceAtlas2Based: {{
      gravitationalConstant: -80,
      centralGravity: 0.006,
      springLength: 200,
      springConstant: 0.08,
      damping: 0.9,
    }},
    solver: "forceAtlas2Based",
    stabilization: {{ iterations: 400, updateInterval: 30 }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 80,
    navigationButtons: false,
    keyboard: {{ enabled: true, speed: {{ x:10,y:10,zoom:.05 }} }},
  }},
  edges: {{ smooth: {{ type:"dynamic" }} }},
}});

network.on("stabilizationIterationsDone", () => {{
  network.setOptions({{ physics:{{ enabled:false }} }});
  document.getElementById("btn-physics").classList.remove("active");
  document.getElementById("loading").style.opacity = "0";
  setTimeout(() => document.getElementById("loading").style.display="none", 400);
}});

// ── Stats ─────────────────────────────────────────────────────────────────────
const paperCount   = RAW_NODES.filter(n=>n.type==="paper").length;
const conceptCount = RAW_NODES.filter(n=>n.type==="concept").length;
document.getElementById("stats-line").textContent =
  `${{paperCount}} 篇论文  ·  ${{conceptCount}} 个概念  ·  ${{RAW_EDGES.length}} 条边`;

// ── Sidebar: node info ────────────────────────────────────────────────────────
function showNodeInfo(nodeId) {{
  const n = RAW_NODES.find(x => x.id === nodeId);
  if (!n) return;

  document.getElementById("info-placeholder").style.display = "none";
  document.getElementById("info-content").style.display = "block";

  const badge = document.getElementById("info-type-badge");
  badge.textContent = n.type === "paper" ? "论文" : "概念";
  badge.style.background = n.type === "paper" ? "#ddeaff" : "#ffe8d8";
  badge.style.color       = n.type === "paper" ? "#2a5ccc" : "#c05010";

  document.getElementById("info-title").textContent = n.label;

  let meta = "";
  if (n.arxiv_id) meta += `<b>Arxiv:</b> ${{n.arxiv_id}}<br>`;
  if (n.published) meta += `<b>发表:</b> ${{n.published}}<br>`;
  if (n.categories && n.categories.length) meta += `<b>分类:</b> ${{n.categories.join("、")}}<br>`;
  document.getElementById("info-meta").innerHTML = meta;

  // Neighbours
  const connEdges = RAW_EDGES.filter(e => e.source===nodeId || e.target===nodeId);
  let nbHtml = "";
  if (connEdges.length) {{
    nbHtml += `<h3>${{connEdges.length}} 条连接</h3>`;
    connEdges.slice(0,20).forEach(e => {{
      const otherId = e.source===nodeId ? e.target : e.source;
      const other   = RAW_NODES.find(x=>x.id===otherId);
      if (!other) return;
      const m = EDGE_META[e.type] || {{ color:"#888", label:e.type }};
      const dir = e.source===nodeId ? "→" : "←";
      nbHtml += `<div class="nb-item" onclick="focusNode('${{otherId}}')">
        <div class="nb-label">${{other.label.length>50?other.label.slice(0,48)+"…":other.label}}</div>
        <div class="nb-rel" style="color:${{m.color}}">${{dir}} ${{m.label}}${{e.description?" · "+e.description:""}}</div>
      </div>`;
    }});
  }}
  document.getElementById("info-neighbors").innerHTML = nbHtml;
}}

function focusNode(nodeId) {{
  network.selectNodes([nodeId]);
  network.focus(nodeId, {{ scale:1.2, animation:{{ duration:400, easingFunction:"easeInOutQuad" }} }});
  showNodeInfo(nodeId);
}}

network.on("selectNode", e => showNodeInfo(e.nodes[0]));
network.on("deselectNode", () => {{
  document.getElementById("info-placeholder").style.display = "block";
  document.getElementById("info-content").style.display = "none";
}});

// ── Controls ──────────────────────────────────────────────────────────────────
let physicsOn = false;
document.getElementById("btn-physics").addEventListener("click", () => {{
  physicsOn = !physicsOn;
  network.setOptions({{ physics:{{ enabled:physicsOn }} }});
  document.getElementById("btn-physics").classList.toggle("active", physicsOn);
}});

document.getElementById("btn-fit").addEventListener("click", () => {{
  network.fit({{ animation:{{ duration:500, easingFunction:"easeInOutQuad" }} }});
}});

let conceptsVisible = true;
document.getElementById("btn-concepts").addEventListener("click", () => {{
  conceptsVisible = !conceptsVisible;
  const ids = RAW_NODES.filter(n=>n.type==="concept").map(n=>n.id);
  visNodes.update(ids.map(id => ({{ id, hidden:!conceptsVisible }})));
  document.getElementById("btn-concepts").classList.toggle("active", conceptsVisible);
}});

// ── Search ────────────────────────────────────────────────────────────────────
document.getElementById("search").addEventListener("input", function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{
    visNodes.update(RAW_NODES.map(n => ({{ id:n.id, hidden:false, opacity:1 }})));
    return;
  }}
  const updates = RAW_NODES.map(n => {{
    const match = n.label.toLowerCase().includes(q) || (n.arxiv_id||"").includes(q);
    return {{ id:n.id, hidden:false, opacity: match ? 1 : 0.15 }};
  }});
  visNodes.update(updates);
  const hits = RAW_NODES.filter(n => n.label.toLowerCase().includes(q) || (n.arxiv_id||"").includes(q));
  if (hits.length === 1) focusNode(hits[0].id);
}});

// ── Edge type filter ──────────────────────────────────────────────────────────
function applyEdgeFilter() {{
  const hidden = {{}};
  document.querySelectorAll(".ef-check").forEach(cb => {{
    if (!cb.checked) hidden[cb.dataset.type] = true;
  }});
  visEdges.update(RAW_EDGES.map((e,i) => ({{ id:i, hidden:!!hidden[e.type] }})));
}}
document.querySelectorAll(".ef-check").forEach(cb => {{
  cb.addEventListener("change", applyEdgeFilter);
}});
</script>
</body>
</html>
"""


def export_html(nodes: list[GNode], edges: list[GEdge], out_path: Path) -> None:
    import json as _json

    nodes_json = _json.dumps([vars(n) for n in nodes], ensure_ascii=False)
    edges_json = _json.dumps([vars(e) for e in edges], ensure_ascii=False)
    edge_meta_json = _json.dumps(
        {k: {"color": v["color"], "label": v["label"], "dashes": v["dashes"]}
         for k, v in _EDGE_META.items()},
        ensure_ascii=False,
    )

    # Count edges per type for filter badges
    type_counts: dict[str, int] = {}
    for e in edges:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    filter_rows = "\n".join(
        f'    <div class="filter-row">'
        f'<input type="checkbox" class="ef-check" id="ef-{etype}" data-type="{etype}" checked/>'
        f'<label for="ef-{etype}" style="color:{meta["color"]}">{meta["label"]}</label>'
        f'<span class="filter-count">{type_counts.get(etype, 0)}</span>'
        f'</div>'
        for etype, meta in _EDGE_META.items()
        if type_counts.get(etype, 0) > 0
    )

    html = _HTML_TEMPLATE.format(
        nodes_json=nodes_json,
        edges_json=edges_json,
        edge_meta_json=edge_meta_json,
        filter_rows=filter_rows,
    )
    out_path.write_text(html, encoding="utf-8")
