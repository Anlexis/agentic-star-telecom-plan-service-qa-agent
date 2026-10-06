"""AgentCore Platform v1.0"""

# State is a flat TypedDict — never a Pydantic model. Checkpoints are written
# with msgpack serialization, and a model instance in a State field corrupts
# silently rather than failing loudly.
#
# For the same reason, structured fields (a dict, or a list of dicts) are
# stored as JSON STRINGS: producers call to_json() on write, consumers call
# from_json() on read. A bare container in a checkpointed field is a defect,
# not a shortcut.
#
# The fields below cover both layers of the nested graph: the outer backbone
# and the inner domain workflow.
#
# Confidentiality note: account numbers, phone numbers and e-mail addresses in
# the question are removed by the input boundary before any field here is
# written, and again at the output boundary before anything is published. Only
# the normalised query, catalog passage summaries and the assembled answer are
# persisted — never a raw subscriber identifier.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string.

    None passes through unchanged so an unset field stays distinguishable from
    an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict or list.

    None, empty or malformed input yields ``default``, so a missing or corrupt
    field is not fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for the telecom plan and service Q&A agent.

    Shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, ...) are inherited. Domain fields are
    NotRequired so the TypedDict is valid at graph initialisation, before any
    node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by PreProcessNode / PlanServiceQAGraphNode
    # ------------------------------------------------------------------

    # Normalised, identifier-stripped question. The raw input is not persisted
    # beyond the input boundary.
    validated_input: NotRequired[str]

    # Final answer, mapped from the inner graph's formatted_answer.
    plan_service_answer: NotRequired[str]

    # ------------------------------------------------------------------
    # Inner layer — the domain workflow
    # ------------------------------------------------------------------

    # InputValidateNode outputs.
    # Normalised free-text query (whitespace-collapsed, length-capped).
    search_query: NotRequired[str]

    # JSON STRING of the validated query parameters. Deserialised shape:
    # {"category": str | None, "top_k": int | None, "score_threshold": float | None}.
    query_filters: NotRequired[Optional[str]]

    # JSON STRING of the caller-supplied catalog slice, when one was given.
    # Deserialised shape: list[dict] with the entry fields validated in
    # src/caller_contract.py. When present it replaces the bundled corpus.
    caller_catalog: NotRequired[Optional[str]]

    # JSON STRING of the runtime `retrieval` block forwarded by the outer graph.
    # Deserialised shape: {"top_k": int, "score_threshold": float,
    # "kb_path": str, "hybrid_search": bool}.
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output. JSON STRING of scored candidates; each entry is
    # {"id", "title", "category", "source", "score", "excerpt"}.
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output. JSON STRING of the passages that cleared the
    # relevance floor, capped at top_k. Same entry shape as above.
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs.
    # Answer body carrying numbered citation markers.
    grounded_answer: NotRequired[str]

    # JSON STRING of citations; each entry is {"ref", "id", "title", "source"}.
    citations: NotRequired[Optional[str]]

    # OutputFormatNode output — the rendered answer document (body, sources,
    # disclaimer), surfaced to the outer graph.
    formatted_answer: NotRequired[str]

    # Intake notes accumulated while parsing the request. JSON STRING of
    # list[str]; carries no caller values.
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry (an empty question, one that is too long,
    # an out-of-contract structured parameter). The run still completes: no
    # retrieval is performed, no answer is assembled, and the domain audit
    # event for the rejection is still emitted. Carrying this as a completion
    # marker rather than a terminal error is what lets the caller see the
    # reason and send a corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing — framework-managed; not written by node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history is inherited; listed here only for orientation.
