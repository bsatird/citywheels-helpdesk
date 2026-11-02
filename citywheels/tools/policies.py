"""``search_policies`` -- keyword search (BM25) over ``data/policies/*.md``.

The documents are tiny (three files, about twenty sections), so a hand-written
BM25 ranker is enough: no extra dependency, no network, no vector store, and --
most importantly for the assignment -- it is deterministic and easy to unit test.

Each ``##`` heading becomes one searchable section. The heading is indexed with a
higher weight than the body, because the headings are the topic labels
("Late return", "Are there age limits?" ...).
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import config

POLICY_DIR_REF = "data/policies"

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have how i if in into is it its me my of on or our should so than that
    the their them then there these they this to too was we were what when where
    which who why will with would you your about any also just much many
    """.split()
)

#: Small domain vocabulary so "cost" also matches "price", "kid" matches "kids" ...
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "opening": ("open", "hour"),
    "hours": ("open", "hour"),
    "hour": ("opening",),
    "time": ("open", "hours"),
    "cost": ("price",),
    "costs": ("price",),
    "pricing": ("price",),
    "expensive": ("price",),
    "cheap": ("price",),
    "child": ("kids",),
    "children": ("kids", "child"),
    "kid": ("kids",),
    "cancel": ("cancellation",),
    "cancelling": ("cancel", "cancellation"),
    "cancellation": ("cancel",),
    "refund": ("cancel",),
    "refunds": ("refund",),
    "money": ("refund", "deposit"),
    "electric": ("e-bike", "ebike"),
    "ebike": ("e-bike",),
    "bike": ("rental",),
    "rental": ("bike",),
    "late": ("delay", "return"),
    "delay": ("late",),
    "helmet": ("helmets",),
    "age": ("years", "old"),
    "older": ("age", "years"),
}


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, *, expand: bool = True) -> list[str]:
    """Lower-case word tokens without stopwords, lightly singularised."""
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        # "e-bike" contributes e-bike, ebike, e and bike
        variants = [raw]
        if "-" in raw:
            variants.append(raw.replace("-", ""))
            variants.extend(raw.split("-"))
        for variant in variants:
            if len(variant) < 2:
                continue
            variant = _singular(variant)
            if variant in _STOPWORDS:
                continue
            tokens.append(variant)
            if expand:
                tokens.extend(_SYNONYMS.get(variant, ()))
    return tokens


@dataclass(frozen=True)
class Section:
    """One ``##`` block of one policy document."""

    doc: str
    doc_title: str
    heading: str
    anchor: str
    text: str

    @property
    def source(self) -> str:
        return f"{POLICY_DIR_REF}/{self.doc}#{self.anchor}"

    def to_dict(self, score: float = 0.0, matched: list[str] | None = None) -> dict:
        return {
            "source": self.source,
            "doc": self.doc,
            "section": self.heading,
            "score": round(score, 3),
            "matched_terms": matched or [],
            "text": self.text,
        }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def split_markdown(text: str, doc_name: str) -> list[Section]:
    """Split one Markdown document into :class:`Section` objects on ``##`` headings."""
    sections: list[Section] = []
    doc_title = doc_name
    heading: str | None = None
    body: list[str] = []
    intro: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if heading is not None and content:
            sections.append(Section(doc_name, doc_title, heading, _slug(heading), content))

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            body = []
            continue
        if line.startswith("# ") and heading is None:
            doc_title = line[2:].strip()
            continue
        if heading is None:
            intro.append(line)
        else:
            body.append(line)
    flush()

    preamble = "\n".join(intro).strip()
    if preamble:
        sections.insert(
            0, Section(doc_name, doc_title, doc_title, _slug(doc_title), preamble)
        )
    return sections


class _BM25:
    """Plain Okapi BM25 over a list of token lists."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.total = len(documents)
        self.avg_length = (sum(self.lengths) / self.total) if self.total else 0.0
        self.doc_freq: Counter[str] = Counter()
        for counter in self.counts:
            self.doc_freq.update(counter.keys())

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.total - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], index: int) -> float:
        counts = self.counts[index]
        length = self.lengths[index] or 1
        average = self.avg_length or 1
        norm = self.k1 * (1 - self.b + self.b * length / average)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            score += self.idf(term) * (frequency * (self.k1 + 1)) / (frequency + norm)
        return score


@dataclass(frozen=True)
class PolicyIndex:
    sections: tuple[Section, ...]
    bm25: _BM25

    def search(self, query: str, top_k: int = 3) -> list[tuple[Section, float, list[str]]]:
        terms = tokenize(query)
        if not terms:
            return []
        unique: list[str] = []
        for term in terms:
            if term not in unique:
                unique.append(term)

        scored: list[tuple[Section, float, list[str]]] = []
        for position, section in enumerate(self.sections):
            score = self.bm25.score(unique, position)
            if score <= 0:
                continue
            matched = [term for term in unique if self.bm25.counts[position].get(term)]
            scored.append((section, score, matched))

        scored.sort(key=lambda item: (-item[1], item[0].source))
        return scored[: max(1, top_k)]


@functools.lru_cache(maxsize=8)
def load_index(directory: str | Path | None = None) -> PolicyIndex:
    """Load and index all policy documents (cached per directory)."""
    policies = Path(directory) if directory is not None else config.POLICIES_DIR
    sections: list[Section] = []
    for path in sorted(policies.glob("*.md")):
        sections.extend(split_markdown(path.read_text(encoding="utf-8"), path.name))
    if not sections:
        raise FileNotFoundError(f"No policy documents found in {policies}")

    documents = [
        tokenize(section.heading, expand=False) * 3
        + tokenize(section.text, expand=False)
        for section in sections
    ]
    return PolicyIndex(tuple(sections), _BM25(documents))


def search_policies(
    query: str, top_k: int = 3, *, directory: str | Path | None = None
) -> dict:
    """Tool implementation: the policy passages that match ``query``.

    ``found: false`` means nothing in the shop's documents matches, which the
    model must report instead of answering from general knowledge.
    """
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        return {
            "found": False,
            "query": "",
            "results": [],
            "message": (
                "No search query was given. Ask the customer what exactly they want "
                "to know about CityWheels."
            ),
        }

    try:
        top_k = max(1, min(int(top_k), 5))
    except (TypeError, ValueError):
        top_k = 3

    index = load_index(str(directory) if directory is not None else None)
    hits = index.search(cleaned, top_k)
    if not hits:
        return {
            "found": False,
            "query": cleaned,
            "results": [],
            "message": (
                "Nothing in the CityWheels price list, cancellation rules or FAQ "
                "matches this question. Say that you could not find it in the "
                "CityWheels documents and that the customer should contact the staff "
                f"at {config.SUPPORT_EMAIL} / {config.SUPPORT_PHONE}."
            ),
        }

    return {
        "found": True,
        "query": cleaned,
        "results": [section.to_dict(score, matched) for section, score, matched in hits],
        "message": (
            "Use only the facts in these passages. Mention the document section you "
            "took the answer from."
        ),
    }

