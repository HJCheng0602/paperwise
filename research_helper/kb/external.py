"""Load external knowledge base entries from a JSON file.

Format: a JSON array of objects, each with:
    title      (required) — paper title
    arxiv_id   (required) — arxiv identifier
    text       (required) — abstract or content snippet
    published  (optional) — publication date, default ""
    doc_id     (optional) — unique identifier, default same as arxiv_id
    source     (optional) — source label, default "external"
    distance   (optional) — semantic distance, default 0.0
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from research_helper.kb._types import KBEntry


def load_external_kb(path: Path) -> list[KBEntry]:
    entries: list[KBEntry] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[KB] Failed to read external KB file {path}: {exc}", file=sys.stderr)
        return entries

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[KB] Invalid JSON in external KB file {path}: {exc}", file=sys.stderr)
        return entries

    if not isinstance(data, list):
        print(f"[KB] External KB file must contain a JSON array, got {type(data).__name__}", file=sys.stderr)
        return entries

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            print(f"[KB] Warning: entry {i} is not a dict, skipping", file=sys.stderr)
            continue
        title = obj.get("title", "").strip()
        arxiv_id = obj.get("arxiv_id", "").strip()
        text = obj.get("text", "").strip()
        if not title or not arxiv_id or not text:
            print(f"[KB] Warning: entry {i} missing required field (title/arxiv_id/text), skipping", file=sys.stderr)
            continue
        entries.append(KBEntry(
            doc_id=obj.get("doc_id", arxiv_id),
            text=text,
            title=title,
            arxiv_id=arxiv_id,
            published=obj.get("published", ""),
            source=obj.get("source", "external"),
            distance=float(obj.get("distance", 0.0)),
        ))

    if not entries:
        if data:
            print("[KB] Warning: external KB file contained no valid entries", file=sys.stderr)
    else:
        print(f"[KB] Loaded {len(entries)} external KB entries from {path}", file=sys.stderr)

    return entries
