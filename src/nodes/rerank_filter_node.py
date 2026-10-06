"""AgentCore Platform v1.0"""

# Domain node 3 — rerank the candidates and enforce the relevance floor.
#
# Deterministic: a small category-match boost on top of the retrieval score,
# then drop everything below score_threshold and cap the survivors at top_k.
#
# Config precedence for both knobs: a caller override (already validated
# against the contract by InputValidateNode) wins over the runtime block
# seeded into State, which in turn wins over the module defaults that mirror
# config/config.yaml. Every value is re-checked here — a threshold that
# arrived as NaN would compare False against every score and silently pass the
# whole candidate set through the filter this node exists to apply.
#
# Wired by the inner graph. Returns only changed state keys (partial dict).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.caller_contract import (
    SCORE_THRESHOLD_MAX,
    SCORE_THRESHOLD_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    CallerDataError,
    finite_in_range,
)
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

logger = logging.getLogger(__name__)

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_TOP_K = 5
_DEFAULT_SCORE_THRESHOLD = 0.70

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_number(
    candidates: List[Any],
    field: str,
    minimum: float,
    maximum: float,
    default: float,
    integer: bool = False,
) -> float:
    """First in-contract value from the precedence chain, else the default.

    A value that is absent, non-numeric, non-finite or out of range is skipped
    rather than clamped: clamping a nonsensical threshold would silently
    substitute a filter the operator never configured.
    """
    for value in candidates:
        if value is None:
            continue
        try:
            return finite_in_range(value, field, minimum, maximum, integer=integer)
        except CallerDataError:
            logger.warning("RerankFilterNode: %s is out of contract; falling back", field)
    return default


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the relevance floor, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates
        query_filters:       JSON dict, optional category / top_k / threshold overrides
        retrieval_config:    seeded runtime retrieval block (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages, score descending
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

        emit_progress("Ranking the results...")
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        runtime = from_json(state.get("retrieval_config"), {}) or {}
        if not isinstance(runtime, dict):
            runtime = {}

        top_k = int(
            _resolve_number(
                [filters.get("top_k"), runtime.get("top_k")],
                "retrieval.top_k",
                TOP_K_MIN,
                TOP_K_MAX,
                float(_DEFAULT_TOP_K),
                integer=True,
            )
        )
        score_threshold = _resolve_number(
            [filters.get("score_threshold"), runtime.get("score_threshold")],
            "retrieval.score_threshold",
            SCORE_THRESHOLD_MIN,
            SCORE_THRESHOLD_MAX,
            _DEFAULT_SCORE_THRESHOLD,
        )

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy — inputs stay immutable
            score = _resolve_number([entry.get("score")], "candidate.score", 0.0, 1.0, 0.0)
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score descending, then id ascending for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]

        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": len(reranked) - len(kept),
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
