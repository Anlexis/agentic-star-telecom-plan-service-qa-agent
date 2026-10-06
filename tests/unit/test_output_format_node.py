# Rendering the answer document.
#
# Invoked as node(state) with an anonymous caller. formatted_answer is a
# domain field, not a scanned input field, and the standing informational
# disclaimer is part of THIS node's output contract.
#
# Deterministic — no model call, no network.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json

_DISCLAIMER_FRAGMENT = "for informational purposes only"


def _make_state(grounded_answer, citations, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_body_sources_disclaimer(self):
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-001",
                    "title": "Unlimited Plus plan pricing and monthly data allowance",
                    "source": "TEL Consumer Plan Catalog",
                }
            ]
        )
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Telecom Plan & Service Answer")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] Unlimited Plus plan pricing and monthly data allowance (TEL Consumer Plan Catalog)" in answer
        assert _DISCLAIMER_FRAGMENT in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the enum.
        assert not isinstance(result["status"], AgentStatus)

    def test_fmt_02_source_suffix_omitted_when_blank(self):
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-001",
                    "title": "Unlimited Plus plan pricing and monthly data allowance",
                    "source": "",
                }
            ]
        )
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] Unlimited Plus plan pricing and monthly data allowance\n" in answer + "\n"
        assert "()" not in answer

    def test_fmt_03_disclaimer_present_on_every_answer(self):
        # The disclaimer must ride WITH the substance, never separately.
        for grounded in ("a body.", ""):
            answer = OutputFormatNode()(_make_state(grounded, to_json([])))["formatted_answer"]
            assert _DISCLAIMER_FRAGMENT in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no catalog passage cleared the relevance threshold)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value
