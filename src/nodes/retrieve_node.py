"""AgentCore Platform v1.0"""

# Domain node 2 — deterministic keyword retrieval over the plan/service catalog.
#
# Corpus selection:
#   - when the caller supplied catalog_documents, THAT is the corpus for this
#     invocation. This is the path a real deployment uses: plan catalogs are
#     operator-specific and change constantly, so the calling system passes the
#     slice it wants answered from;
#   - otherwise the bundled sample catalog at the configured kb_path is used,
#     which keeps the template runnable and its tests deterministic.
#
# Retrieval is fully deterministic: term overlap scored per field, no embedding
# model and no vector store. The output contract (retrieved_documents as JSON)
# is store-agnostic, so swapping in a vector store changes only this node.
#
# Config: top_k and kb_path come from retrieval_config, the runtime block the
# inner graph seeds into State. Both are bounds-checked here rather than
# trusted; module defaults mirror config/config.yaml and apply when the field
# is unseeded, which happens when this node is exercised on its own.
#
# Wired by the inner graph. Returns only changed state keys (partial dict).

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.caller_contract import TOP_K_MAX, TOP_K_MIN, CallerDataError, finite_in_range
from src.schemas.state import from_json, to_json
from src.text_safety import render_safe_text
from framework.schemas.agent_status import AgentStatus

logger = logging.getLogger(__name__)

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.70,
    "kb_path": "config/kb/telecom_kb.json",
}

# Repository root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation — deterministic, no NLP dependency.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query term found in the title counts for more than
# one found only in the body.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and one/two-character noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_top_k(state: AgentState) -> int:
    """Effective retrieval depth: seeded runtime config, else the module default.

    The value is bounds-checked like any other: a malformed runtime block falls
    back to the default rather than propagating a nonsensical depth.
    """
    effective = dict(_DEFAULT_RETRIEVAL)
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    try:
        return int(finite_in_range(effective.get("top_k"), "retrieval.top_k", TOP_K_MIN, TOP_K_MAX, integer=True))
    except CallerDataError:
        logger.warning("RetrieveNode: configured retrieval.top_k is out of contract; using the default")
        return int(_DEFAULT_RETRIEVAL["top_k"])


def _resolve_kb_path(state: AgentState) -> str:
    """Configured catalog path, falling back to the bundled sample corpus."""
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        candidate = from_state.get("kb_path")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return str(_DEFAULT_RETRIEVAL["kb_path"])


def _load_bundled_catalog(kb_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the bundled sample catalog. A missing or malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append("The bundled catalog could not be read; no passages were available.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("The bundled catalog must be a list of entries; none were used.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field weight per query term, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the catalog corpus against the query and emit ranked candidates.

    Input state keys:
        search_query:     normalised query
        query_filters:    JSON dict, optional category filter
        caller_catalog:   JSON list of caller-supplied entries (when supplied)
        retrieval_config: seeded runtime retrieval block (JSON)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates, score descending
        intake_notes:        JSON list[str], only on a corpus anomaly
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Searching the knowledge base...")
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}
        top_k = _resolve_top_k(state)

        caller_catalog = from_json(state.get("caller_catalog"), None)
        if isinstance(caller_catalog, list) and caller_catalog:
            entries: List[Dict[str, Any]] = [e for e in caller_catalog if isinstance(e, dict)]
            notes: List[str] = []
            corpus = "caller"
        else:
            entries, notes = _load_bundled_catalog(_resolve_kb_path(state))
            corpus = "bundled"

        category = filters.get("category")
        if category:
            entries = [e for e in entries if str(e.get("category", "")).lower() == str(category).lower()]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": render_safe_text(str(entry.get("title", "")), 200),
                    "category": str(entry.get("category", "")),
                    "source": render_safe_text(str(entry.get("source", "")), 200),
                    "score": score,
                    "excerpt": render_safe_text(str(entry.get("content", "")), _EXCERPT_CHARS),
                }
            )

        # Deterministic ordering: score descending, then id ascending for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a pool wider than top_k — RerankFilterNode makes the final cut
        # after the category boost and the relevance floor.
        candidates = candidates[: max(top_k * 3, 10)]

        emit_trace_event(
            "retrieve_complete",
            {
                "corpus": corpus,
                "candidates": len(candidates),
                "catalog_entries": len(entries),
                "query_tokens": len(query_tokens),
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to, never clobber, the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
