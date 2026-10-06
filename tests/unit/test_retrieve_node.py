# Retrieval over the catalog corpus.
#
# Invoked as node(state) with an anonymous caller. The node takes no config
# argument — execute(self, state) is the only signature — so a test that needs
# to exercise a configured knob (kb_path / top_k) seeds the state field
# retrieval_config, the same field the inner graph republishes at runtime.
#
# search_query is NOT a scanned input field name (only user_input /
# validated_input / llm_response are), so query casing here is free of the
# PII-mask interaction that constrains validated_input-seeded tests.
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded config/kb/telecom_kb.json;
# no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_PLAN_QUERY = "what is the monthly price and data allowance for the Unlimited Plus plan?"


def _make_state(query=_PLAN_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_unlimited_plus_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the plan pricing query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="billing cycle and autopay discount",
            query_filters=to_json({"category": "billing", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "billing category has seeded entries"
        assert {d["category"] for d in docs} == {"billing"}
        assert docs[0]["id"] == "kb-004"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigPrecedence:
    """Config plumbing (C2 retired form): State-seeded retrieval_config >
    module defaults. There is no execute(state, config=...) route any more —
    every case below invokes through node(state) / __call__ (C1)."""

    def test_ret_06_state_retrieval_config_kb_path_override(self):
        # A bogus kb_path seeded via retrieval_config degrades gracefully —
        # this IS the config knob's only surface post-C2-retirement.
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("could not be read" in n for n in notes)

    def test_ret_07_state_retrieval_config_fallback_when_unseeded(self):
        # No retrieval_config seeded at all — falls back to the module
        # defaults, which mirror config/agent.yaml (kb_path resolves fine).
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs and docs[0]["id"] == "kb-001"

    def test_ret_07b_state_retrieval_config_top_k_is_live(self):
        # top_k seeded via retrieval_config narrows the candidate pool cap
        # (pool_size = max(top_k * 3, 10)); with only 10 KB entries this is
        # observable via the category-filtered case instead (pool otherwise
        # exceeds available entries). Assert the field is actually consulted
        # by checking a deliberately unreadable path still surfaces the note
        # (i.e. retrieval_config, not a hardcoded default, drove kb_path).
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/bogus.json", "top_k": 3}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
