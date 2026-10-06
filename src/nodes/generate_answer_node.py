"""AgentCore Platform v1.0"""

# Domain node 4 — assemble the grounded answer from the ranked passages.
#
# The bundled answer builder is deterministic: a lead sentence plus one cited
# point per surviving passage, each carrying a numbered marker [n]. Nothing
# outside ranked_documents reaches the answer body, so the answer is grounded
# by construction and cannot over-commit on pricing or coverage the catalog
# does not support.
#
# The caller's question is deliberately NOT echoed into the answer. The body is
# assembled from catalog passages only; a verbatim echo of the question would
# put caller-controlled text on the external surface, where it could forge
# document structure, and the output boundary treats such an echo as a leak.
#
# config/prompts/answer_synthesis_prompt.md describes the seam for a
# model-backed builder: it would consume the same ranked_documents input and
# emit the same grounded_answer / citations contract, so no other node changes.
#
# Wired by the inner graph. Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Answer body used when no passage cleared the relevance floor.
NO_COVERAGE_ANSWER = (
    "The plan and service catalog does not contain enough coverage to answer "
    "this question. Rephrase the query with more specific plan, pricing, or "
    "service terms, or contact customer support for a manual lookup."
)

# Lead sentence for an answer that does have supporting passages.
_LEAD_SENTENCE = "The following passages from the plan and service catalog address this question:"

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap it."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Assemble the grounded answer body and its numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages

    Output state keys (partial dict):
        grounded_answer: answer body carrying [n] citation markers
        citations:       JSON list [{ref, id, title, source}]
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

        emit_progress("Composing the answer...")
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []

        citations: List[Dict[str, Any]] = []

        if not ranked:
            grounded_answer = NO_COVERAGE_ANSWER
        else:
            lines: List[str] = [_LEAD_SENTENCE, ""]
            for doc in ranked:
                if not isinstance(doc, dict):
                    continue
                ref = len(citations) + 1
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines) if citations else NO_COVERAGE_ANSWER

        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not citations,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
