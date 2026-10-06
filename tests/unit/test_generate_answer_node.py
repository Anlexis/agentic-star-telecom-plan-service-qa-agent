# Assembling the grounded answer.
#
# Invoked as node(state) with an anonymous caller. The grounded answer and its
# citations are domain fields, not scanned input fields, so Title-Case catalog
# titles inside them are safe to assert on.
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "plan_pricing",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="unlimited plus plan pricing", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc("kb-001", "Unlimited Plus plan pricing and monthly data allowance", "costs ¥6,980 per month."),
            _doc("kb-002", "5G access and mobile hotspot tethering across Plus-tier plans", "no additional surcharge."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Unlimited Plus plan pricing and monthly data allowance:" in answer
        assert "[2] 5G access and mobile hotspot tethering across Plus-tier plans:" in answer

    def test_the_caller_question_is_never_echoed_into_the_answer(self):
        """The body is assembled from catalog passages only.

        An echo of the question would put caller-controlled text on the
        external surface, where it could forge document structure — and the
        output boundary treats such an echo as a leak.
        """
        question = "unlimited plus plan pricing"
        ranked = _ranked(_doc("kb-001", "Unlimited Plus plan pricing and monthly data allowance", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query=question))
        assert question not in result["grounded_answer"]
        assert result["grounded_answer"].startswith("The following passages from the plan and service catalog")

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "Unlimited Plus plan pricing and monthly data allowance",
                "a.",
                source="TEL Consumer Plan Catalog",
            ),
            _doc("kb-002", "5G access and mobile hotspot tethering across Plus-tier plans", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-001", "kb-002"]
        assert citations[0]["source"] == "TEL Consumer Plan Catalog"

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        ranked = _ranked(_doc("kb-001", "Unlimited Plus plan pricing and monthly data allowance", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "Unlimited Plus plan pricing and monthly data allowance",
                "costs ¥6,980 per month before the autopay discount.",
            )
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "costs ¥6,980 per month before the autopay discount." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    def test_gen_05_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain enough coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain enough coverage" in result["grounded_answer"]
