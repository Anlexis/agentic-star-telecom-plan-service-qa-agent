# Outer graph composition, and the agent end to end.
#
# Drives the REAL outer agent through invoke(). The end-to-end context is an
# authenticated caller — the level the manifest declares. The internal
# shortcut context is never used here: it would over-privilege the run and
# hide regressions on the outer trust gate.
#
# Deterministic — no model call, no network.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    Graph,
    PlanServiceQAGraphNode,
    TelecomPlanServiceQAAgent,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_PLAN_QUERY = "what is the monthly price and data allowance for the unlimited plus plan"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(TelecomPlanServiceQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is TelecomPlanServiceQAAgent

    def test_state_schema_is_state(self):
        assert TelecomPlanServiceQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = TelecomPlanServiceQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], PlanServiceQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in TelecomPlanServiceQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = PlanServiceQAGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = PlanServiceQAGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = PlanServiceQAGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH plan_service_answer and
        # result (the output boundary reads state["result"]).
        assert delta == {
            "plan_service_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # Empty when neither layer declined: the key is always present so a
            # reason can never be dropped at the boundary.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert PlanServiceQAGraphNode.error_strategy == "propagate"
        assert PlanServiceQAGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_manifest(self, monkeypatch):
        # Even with an unreadable runtime file the forwarded config carries the
        # fallback retrieval/llm blocks — never {}.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = PlanServiceQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/telecom_kb.json"
        assert cfg["configurable"]["llm"]


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_PLAN_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_PLAN_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Telecom Plan & Service Answer")
        assert "[1]" in output
        assert "for informational purposes only" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_PLAN_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "PlanServiceQAGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain enough coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """The trust gate at graph level: an anonymous invoke is refused by
        the pre_process slot. The error status short-circuits the main slot —
        the framework skips a node whose incoming state already carries an
        error — and routing goes straight to finalize, so no domain answer is
        ever produced."""
        result = _run(_PLAN_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """Serialization helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.97, "title": "unlimited plus plan pricing"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "plan_pricing", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
