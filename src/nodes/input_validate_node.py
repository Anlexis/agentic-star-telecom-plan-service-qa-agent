"""AgentCore Platform v1.0"""

# Domain node 1 — parse and bound the incoming plan/service request.
#
# Two channels reach this node, and both are caller-controlled:
#
#   the question    the outer graph hands over the normalised, identifier-
#                   stripped question. It may be plain text, or a JSON
#                   envelope {"query": ..., "category": ..., "top_k": ...}
#                   carrying structured parameters inline.
#
#   input_context   the caller-data object, bridged in from the outer graph
#                   (src/graph/context_bridge.py). It may carry retrieval
#                   overrides and a catalog slice that REPLACES the bundled
#                   sample corpus for this invocation.
#
# input_context wins where both channels set the same parameter — it is the
# explicit, typed channel; the envelope is the convenience one.
#
# Every field from either channel goes through the shared contract in
# src/caller_contract.py: bounded, finite, inert, and failing CLOSED with an
# error that names the field and never repeats its value. Re-validating here
# rather than trusting the outer node keeps the guarantee attached to the node
# that owns the contract, so it holds however this graph is driven.
#
# Wired by the inner graph. Returns only changed state keys (partial dict).

import json
import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INPUT_REJECTED
from src.caller_contract import (
    MAX_QUESTION_CHARS,
    TOP_K_MAX,
    TOP_K_MIN,
    CallerDataError,
    finite_in_range,
    inert_identifier,
    validate_input_context,
)
from src.schemas.state import to_json
from src.text_safety import collapse_whitespace

logger = logging.getLogger(__name__)


# Marker for a run that completes without an answer because a caller-supplied
# value fell outside its documented contract. The caller can correct the
# value and send the request again.
_CODE_INVALID_REQUEST = "INVALID_REQUEST"


class InputValidateNode(FunctionNode):
    """Turn the request into a normalised query plus validated filters.

    Input state keys:
        validated_input | user_input: the request payload
        input_context:                caller data bridged from the outer graph

    Output state keys (partial dict):
        search_query:   normalised free-text query
        query_filters:  JSON dict {"category", "top_k", "score_threshold"}
        caller_catalog: JSON list of caller-supplied catalog entries (when given)
        intake_notes:   JSON list[str], only when something worth noting happened
        status/error_log: set only when the request is refused
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def _refuse(self, state: AgentState, message: str) -> Dict[str, Any]:
        """Decline a caller-supplied value and COMPLETE the run.

        Both call sites are values the caller can correct: a structured
        parameter outside its documented contract. The retrieval path still
        fails closed - no search runs, no answer is assembled - and the audit
        event is recorded exactly as before. What changed is the reporting:
        terminating would end the caller's turn and surface only an exception
        type, leaving the field name reachable solely from the audit trail.
        Completing with the reason lets the caller correct the value and send
        the request again on the same conversation.

        The message names a field, never a value.
        """
        logger.warning("InputValidateNode: request declined")
        emit_trace_event("input_validate_declined", {"reason": _CODE_INVALID_REQUEST}, state)
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": _CODE_INVALID_REQUEST,
            "error_log": [f"InputValidateNode: {message}"],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        context, context_error = validate_input_context(state.get("input_context"))
        if context_error:
            return self._refuse(state, context_error)

        query = ""
        category: Optional[str] = context.get("category")
        top_k: Optional[int] = context.get("top_k")
        score_threshold: Optional[float] = context.get("score_threshold")

        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            payload: Any = None
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append("The request looked like JSON but did not parse; it was read as plain text.")
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                try:
                    if category is None and payload.get("category") is not None:
                        category = inert_identifier(payload["category"], "query.category")
                    if top_k is None and payload.get("top_k") is not None:
                        top_k = int(
                            finite_in_range(payload["top_k"], "query.top_k", TOP_K_MIN, TOP_K_MAX, integer=True)
                        )
                except CallerDataError as error:
                    return self._refuse(state, str(error))
            else:
                query = text
        else:
            notes.append("The request carried no question to search for.")

        query = collapse_whitespace(query)
        if len(query) > MAX_QUESTION_CHARS:
            query = query[:MAX_QUESTION_CHARS]
            notes.append(f"The question was truncated to {MAX_QUESTION_CHARS} characters.")

        filters = {"category": category, "top_k": top_k, "score_threshold": score_threshold}
        catalog = context.get("catalog_documents")

        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
                "has_score_threshold_override": score_threshold is not None,
                "caller_documents": len(catalog) if catalog else 0,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if catalog is not None:
            out["caller_catalog"] = to_json(catalog)
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
