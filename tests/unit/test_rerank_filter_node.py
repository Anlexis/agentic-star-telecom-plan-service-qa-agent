# Reranking and the relevance floor.
#
# Invoked as node(state) with an anonymous caller. The node takes no config
# argument — execute(self, state) is the only signature — so a test that needs
# to exercise a configured knob (score_threshold / top_k) seeds the state field
# retrieval_config.
#
# Mirrors docs/03_test_spec.md §2.4 (RRF-01..RRF-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="plan_pricing"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9), _doc("kb-b", 0.5)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]  # 0.5 < default 0.70 floor

    def test_rrf_02_state_score_threshold_override(self):
        # C2 retired — config knob travels via State, not a direct execute() arg.
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.75)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]

    def test_rrf_03_state_top_k_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.75)],
            retrieval_config=to_json({"top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestCategoryBoost:
    def test_rrf_04_matching_category_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("kb-a", 0.74, category="plan_features"), _doc("kb-b", 0.68, category="billing")],
            query_filters=to_json({"category": "billing", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-b", "kb-a"]
        assert kept[0]["score"] == 0.78  # 0.68 + 0.1 category boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("kb-a", 0.95, category="billing")],
            query_filters=to_json({"category": "billing", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0


class TestCallerTopK:
    def test_rrf_06_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.75)],
            query_filters=to_json({"category": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_a_caller_override_wins_over_the_deployment_default(self):
        """The caller gets what it asked for, bounded by the contract — never
        a silently different depth."""
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.75)],
            query_filters=to_json({"category": None, "top_k": 3}),
            retrieval_config=to_json({"top_k": 2, "score_threshold": 0.70}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b", "kb-c"]

    def test_a_caller_score_threshold_override_is_honoured(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.4)],
            query_filters=to_json({"category": None, "top_k": None, "score_threshold": 0.3}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]

    def test_an_out_of_contract_configured_value_falls_back_rather_than_clamping(self):
        """A nonsensical configured threshold must not become a silently
        different filter — the documented default applies instead."""
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.5)],
            retrieval_config=to_json({"score_threshold": float("nan")}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]


class TestRobustness:
    def test_rrf_07_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "kb-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("kb-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_08_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("kb-b", 0.9), _doc("kb-a", 0.9)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]
