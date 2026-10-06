"""AgentCore Platform v1.0"""

# Domain node 5 (terminal) — compose the answer document: the grounded body,
# the Sources list, and the standing informational disclaimer.
#
# The disclaimer belongs to this node's output contract, not to the outer
# post_process slot: post_process gates the document, it does not compose it.
# The published shape both nodes work to is declared once in
# src/output_schema.py.
#
# Wired by the inner graph. The inner get_output() surfaces formatted_answer
# and status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.output_schema import (
    ANSWER_HEADING,
    INFORMATIONAL_DISCLAIMER,
    NO_SOURCES_LINE,
    SOURCES_HEADING,
)
from src.schemas.state import from_json


class OutputFormatNode(FunctionNode):
    """Compose the answer document: body, sources, disclaimer.

    Input state keys:
        grounded_answer: answer body carrying [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: the rendered document
        status:           the success value as a plain string — State never
                          carries a bare enum
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

        emit_progress("Formatting the response...")
        grounded_answer = state.get("grounded_answer") or "No answer is available for this request."
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = [ANSWER_HEADING, "", grounded_answer, "", SOURCES_HEADING]
        listed = 0
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            ref = citation.get("ref", "?")
            title = str(citation.get("title", "")).strip()
            source = str(citation.get("source", "")).strip()
            suffix = f" ({source})" if source else ""
            lines.append(f"- [{ref}] {title}{suffix}")
            listed += 1
        if not listed:
            lines.append(NO_SOURCES_LINE)
        lines.extend(["", "---", "", f"*{INFORMATIONAL_DISCLAIMER}*"])

        formatted_answer = "\n".join(lines)

        emit_trace_event(
            "output_format_complete",
            {"answer_chars": len(formatted_answer), "citation_count": listed},
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
