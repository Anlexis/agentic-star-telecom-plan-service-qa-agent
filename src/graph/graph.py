"""AgentCore Platform v1.0"""

# Outer graph for the telecom plan and service Q&A agent.
#
# Architecture (two layers):
#
#   Outer backbone (fixed by the framework — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (retry, max_retry)
#                                             -> pre_process
#
#   The `main` slot holds PlanServiceQAGraphNode, a GraphNode subclass that
#   delegates the whole domain workflow to DomainWorkflowGraph — an inner
#   graph running input_validate -> retrieve -> rerank_filter ->
#   generate_answer -> output_format.
#
#   Domain complexity lives entirely inside the inner graph; the outer
#   backbone stays a thin, uniform shell.
#
# Files:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- carries input_context across the boundary
#
# The class name is a three-way contract:
#   this file's class:            TelecomPlanServiceQAAgent
#   config/agent.yaml `class:`    "src.graph.graph.TelecomPlanServiceQAAgent"
#   src/api/server.py import      from src.graph.graph import TelecomPlanServiceQAAgent

from pathlib import Path
from typing import Any, ClassVar, Dict

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

# Runtime parameters: src/graph/graph.py -> parents[2] = repository root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Fallbacks mirror config/config.yaml so the forwarded blocks are never empty,
# even in a deployment layout where the file cannot be read.
_FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.70,
    "kb_path": "config/kb/telecom_kb.json",
    "hybrid_search": False,
}
_FALLBACK_LLM: Dict[str, Any] = {
    "temperature": 0.1,
    "max_tokens": 2500,
}


def load_runtime_config() -> Dict[str, Any]:
    """Read config/config.yaml — the agent's runtime parameters.

    This is the file the registry loads and passes to the graph constructor as
    `config=`; src/api/server.py reads it through this function so a standalone
    process honours the same declared values (max_retry, timeout_s, retrieval
    tuning) instead of silently running on framework defaults.

    An unreadable or malformed file yields {} — callers apply their own
    fallbacks — rather than preventing the agent from starting.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class PlanServiceQAGraphNode(GraphNode):
    """The `main` slot: delegates to the inner domain workflow graph.

    Contracts:
      get_subgraph()   build the inner graph with the forwarded runtime config
      extract_input()  choose the string handed to the inner invoke(), and
                       stash the caller's input_context for the inner graph
      merge_output()   map the inner result back into outer state (changed keys only)
      error_strategy   "propagate": inner failures re-raise as SubgraphError
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    # "propagate" re-raises inner graph exceptions as SubgraphError (fail fast).
    # "handle" would route them to on_subgraph_error() for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: human-in-the-loop interrupts stay contained inside the inner graph.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the runtime `retrieval` and `llm` blocks to the inner graph.

        Reads config/config.yaml and returns the two tuning blocks under
        config["configurable"] — never an empty dict. The inner graph
        republishes the `retrieval` block into its initial state so
        RetrieveNode and RerankFilterNode read live top_k / score_threshold
        values rather than dead declarations. The `llm` block is forwarded for
        a model-backed answer node; the bundled deterministic one ignores it.
        """
        runtime = self._runtime_config
        retrieval = runtime.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = runtime.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        return {"configurable": {"retrieval": retrieval, "llm": llm}}

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        Imported inside the method to keep module import order simple; the
        inner graph receives the forwarded runtime config through its
        constructor, while its domain nodes take no constructor arguments and
        read every knob from State.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to search on,
        so running the inner graph would only produce a second, vaguer reason
        for the same rejection - and overwrite the specific one already
        settled. Completing here keeps the original reason intact.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: Dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string handed to the inner graph, and bridge caller data.

        PreProcessNode validates the question and writes the normalised,
        identifier-stripped form to validated_input; prefer that and fall back
        to user_input when a node is exercised outside the backbone.

        The caller's input_context is stashed here because the framework does
        not forward it into the inner graph (see src/graph/context_bridge.py).
        """
        set_caller_input_context(state.get("input_context") or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner result into the outer state delta (changed keys only).

        Designed together with DomainWorkflowGraph.get_output(), which emits
        formatted_answer / citations / status. The rendered answer is written
        to `result` as well as to plan_service_answer, because the outer
        post_process slot reads `result` — without that mapping the finalized
        output would always be empty.
        """
        return {
            "plan_service_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "status": sub_result.get("status"),
            # Outer reason wins. A reason already settled before the inner run
            # is the real one; the inner graph only ever sees the downstream
            # consequence of it ("there was no query to search"), so taking the
            # inner value first would replace a specific reason with a generic
            # one - and a plain sub_result.get() would erase the outer reason
            # entirely whenever the inner run did not set its own.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }


class TelecomPlanServiceQAAgent(AgentBaseGraph):
    """Telecom plan and service Q&A agent (retrieval pipeline, nested).

    Inherits AgentBaseGraph directly. Domain logic is encapsulated in
    PlanServiceQAGraphNode (the `main` slot), which delegates to
    DomainWorkflowGraph.

    Backbone:
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the only override. add_edges() is not overridden —
    backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier used at registration."""
        return "TelecomPlanServiceQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all five backbone slots.

        super().register_nodes() must be called first: it injects the
        framework's default initialize node (schema_version, session_id,
        trust_level) and finalize node (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = PlanServiceQAGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Back-compat alias: config/agent.yaml declares the dotted path to the class
# above and src/api/server.py imports it directly; both names point at the agent.
Graph = TelecomPlanServiceQAAgent
