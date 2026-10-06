"""AgentCore Platform v1.0"""

# The inner graph of the nested two-layer architecture. It encapsulates the
# whole telecom plan and service Q&A workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Built by PlanServiceQAGraphNode.get_subgraph() in src/graph/graph.py.
# get_output() shapes the dict that merge_output() consumes there.
#
# Structural rules this file honours:
#   - inherits BaseGraph (fully custom topology, no forced backbone)
#   - implements all seven abstract members
#   - register_nodes() does not call super() — it is abstract in BaseGraph
#   - every domain node is constructed with no arguments
#   - initialize / finalize are outer-backbone concerns and are not registered
#   - get_output() is designed together with the outer merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow: validate, retrieve, rerank, answer, format.

    Constructed by PlanServiceQAGraphNode.get_subgraph(), which passes the
    runtime config read from config/config.yaml.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  parse and bound the request
          -> retrieve        (RetrieveNode)       score the catalog corpus
          -> rerank_filter   (RerankFilterNode)   boost, threshold, cut to top_k
          -> generate_answer (GenerateAnswerNode) grounded answer + citations
          -> output_format   (OutputFormatNode)   final document + disclaimer
          -> END

    Every node is a FunctionNode returning partial-dict state updates.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "tel_c2_001_plan_service_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across the inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the inner graph's config before compilation.

        The forwarded `retrieval` block is read per call by the domain nodes,
        which bound every value they read and fall back to their own defaults,
        so an absent block is not fatal and nothing is raised here.
        """
        pass

    # -- Config and caller data forwarded into state ---------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner initial state with runtime config and caller data.

        Two things have to cross into the inner graph:

        `retrieval_config` — the forwarded runtime `retrieval` block, stored as
        a JSON string (structured State fields travel as JSON strings so
        checkpoint serialization stays safe). RetrieveNode and RerankFilterNode
        read their top_k / score_threshold from it, so the declared values are
        live at runtime rather than dead declarations.

        `input_context` — the caller's own data. The framework's GraphNode does
        not forward it into the inner graph, so the outer node stashes it and
        this hook reads it back (see src/graph/context_bridge.py).
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "input_context": get_caller_input_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all five domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. initialize
        and finalize are not registered here; they are outer-backbone
        concerns. Every node is constructed with no arguments: config reaches
        them through State, never through a per-call argument.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear Q&A topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear — there is no conditional branching
        between domain nodes, so route() exists to satisfy the base class but
        add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the base class.

        Never called at runtime for this linear topology. It returns END on an
        error status so an unexpected call cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the dict the outer graph receives as sub_result.

        Designed together with PlanServiceQAGraphNode.merge_output() so the
        field names agree:

            this get_output()   emits: formatted_answer, citations, status, ...
            outer merge_output() reads: formatted_answer, citations, status

        The remaining fields are surfaced for observability and for future
        outer-merge extensions without an inner-graph change.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
