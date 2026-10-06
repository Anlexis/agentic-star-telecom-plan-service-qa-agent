# The inner request parser: two caller channels, one contract.
#
# Invoked as node(state) with an anonymous caller — the external trust gate
# lives on the outer boundary. Payloads are lowercase and identifier-free so
# the framework's mask leaves the fixture intact (validated_input is a scanned
# field name).

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_plain_text_becomes_the_query(self):
        result = InputValidateNode()(_make_state("unlimited plus plan pricing"))
        assert result["search_query"] == "unlimited plus plan pricing"
        assert from_json(result["query_filters"]) == {"category": None, "top_k": None, "score_threshold": None}

    def test_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  unlimited   plus\n plan pricing "))
        assert result["search_query"] == "unlimited plus plan pricing"

    def test_structured_fields_travel_as_json_strings(self):
        result = InputValidateNode()(_make_state("unlimited plus plan pricing"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_an_envelope_supplies_query_category_and_depth(self):
        payload = json.dumps({"query": "billing cycle and due dates", "category": "billing", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "billing cycle and due dates"
        filters = from_json(result["query_filters"])
        assert filters["category"] == "billing"
        assert filters["top_k"] == 2

    def test_the_question_alias_is_accepted(self):
        result = InputValidateNode()(_make_state(json.dumps({"question": "what plans include 5g access?"})))
        assert result["search_query"] == "what plans include 5g access?"

    def test_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        assert any("did not parse" in note for note in from_json(result.get("intake_notes"), []))

    @pytest.mark.parametrize("value", [99, -5, 0, "many", float("nan"), "Infinity", True, 3.5])
    def test_an_out_of_contract_envelope_depth_fails_closed(self, value):
        """The envelope is a convenience channel, not a looser one."""
        payload = json.dumps({"query": "unlimited plus plan pricing", "top_k": value})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "INVALID_REQUEST"
        assert "search_query" not in result
        assert "query.top_k" in " ".join(result["error_log"])

    def test_an_out_of_contract_envelope_category_fails_closed(self):
        payload = json.dumps({"query": "roaming charges", "category": "  ROAMING "})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "INVALID_REQUEST"
        assert "query.category" in " ".join(result["error_log"])


class TestCallerDataChannel:
    def test_caller_data_supplies_the_filters(self):
        result = InputValidateNode()(
            _make_state(
                "roaming charges",
                input_context={"category": "roaming", "top_k": 4, "score_threshold": 0.25},
            )
        )
        filters = from_json(result["query_filters"])
        assert filters == {"category": "roaming", "top_k": 4, "score_threshold": 0.25}

    def test_caller_data_wins_over_the_envelope(self):
        payload = json.dumps({"query": "roaming charges", "category": "billing", "top_k": 2})
        result = InputValidateNode()(_make_state(payload, input_context={"category": "roaming", "top_k": 7}))
        filters = from_json(result["query_filters"])
        assert filters["category"] == "roaming"
        assert filters["top_k"] == 7

    def test_a_caller_catalog_is_carried_forward(self):
        result = InputValidateNode()(
            _make_state(
                "fibre bundle",
                input_context={
                    "catalog_documents": [{"id": "op-1", "title": "Fibre bundle", "content": "1Gbps home internet."}]
                },
            )
        )
        catalog = from_json(result["caller_catalog"])
        assert isinstance(result["caller_catalog"], str)
        assert catalog[0]["id"] == "op-1"

    def test_no_caller_catalog_key_when_none_was_supplied(self):
        result = InputValidateNode()(_make_state("roaming charges"))
        assert "caller_catalog" not in result

    @pytest.mark.parametrize(
        "context",
        [
            {"top_k": float("nan")},
            {"score_threshold": "Infinity"},
            {"category": "Plan Pricing"},
            {"catalog_documents": [{"id": "bad id"}]},
        ],
    )
    def test_out_of_contract_caller_data_fails_closed(self, context):
        result = InputValidateNode()(_make_state("roaming charges", input_context=context))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "INVALID_REQUEST"
        assert "search_query" not in result


class TestSizeAndEmptyGuards:
    def test_an_oversize_query_is_truncated(self):
        result = InputValidateNode()(_make_state("unlimited data plan " * 150))
        assert len(result["search_query"]) == 2000
        assert any("truncated" in note for note in from_json(result.get("intake_notes"), []))

    def test_an_empty_request_is_noted_rather_than_refused(self):
        """An empty inner request degrades to a no-coverage answer, not a crash."""
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        assert any("no question" in note for note in from_json(result.get("intake_notes"), []))
