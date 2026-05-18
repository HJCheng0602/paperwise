from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KBEntry:
    doc_id: str
    text: str
    title: str
    arxiv_id: str
    published: str
    source: str          # "abstract" | "report" | "external"
    distance: float = 0.0
