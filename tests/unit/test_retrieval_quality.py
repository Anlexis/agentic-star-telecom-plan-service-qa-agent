# TEL-C2-001 — Unit Tests: retrieval quality over the seeded KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode → RetrieveNode → RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against config/kb/telecom_kb.json and pins
# the expected top hit per domain query. The scorer is deterministic
# (keyword field-weights, stable tie-break), so exact top-1 assertions are
# safe and catch KB / scorer / threshold regressions. Every query below is
# verified (against the real SDK) to be the SOLE survivor of the 0.70
# relevance floor for its expected entry — one query per seeded KB category.
#
# Mirrors docs/03_test_spec.md §2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"] for entry in json.loads((_ROOT / "config" / "kb" / "telecom_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.70  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-ranked entry id) — one per catalog category, verified
# against the deterministic scorer. Payloads are lowercase and identifier-free
# so the framework's input mask leaves them intact (validated_input is a
# scanned field name).
_GOLDEN_QUERIES = [
    ("what is the monthly price and data allowance for the unlimited plus plan", "kb-001"),
    ("does 5g network access include an extra surcharge on plus-tier hotspot tethering plans", "kb-002"),
    ("what is the minimum contract term and early termination fee", "kb-003"),
    ("what is the billing cycle, due dates, and autopay discount", "kb-004"),
    ("what is the sim activation and number porting procedure", "kb-005"),
    ("what are the international roaming rates and how do i enable roaming before travel", "kb-006"),
    ("what are the device installment and trade-in credit terms", "kb-007"),
    ("what is the 5g and lte network coverage commitment in my area", "kb-008"),
    ("how do i report a network outage to technical support", "kb-009"),
    ("what happens to my final bill when i cancel my plan", "kb-010"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_pricing_query_keeps_only_the_unlimited_plus_entry(self):
        # Off-topic passages (e.g. the 5G/hotspot entry) score below the
        # floor and are cut — precision, not just recall.
        kept = _search(_GOLDEN_QUERIES[0][0])
        assert [d["id"] for d in kept] == ["kb-001"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        payload = json.dumps({"query": "what is the billing cycle and due dates", "category": "billing"})
        kept = _search(payload)
        assert kept, "billing category carries seeded entries"
        assert {d["category"] for d in kept} == {"billing"}
        assert kept[0]["id"] == "kb-004"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain enough coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
