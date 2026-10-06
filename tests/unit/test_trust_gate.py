# The trust gate: which callers reach which nodes.
#
# Every invocation goes through node(state), i.e. the framework's __call__,
# which runs the trust gate before execute(). A denial RETURNS an error dict
# rather than raising, and execute() never runs — so an execute-only output key
# must be absent from the returned dict. That absence is the assertion that
# matters: it proves nothing was carried forward.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode


def _make_state(trust_value: str, user_input: str = "which plan gives the most data under 5000 yen?", **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """All invocations go through node(state) / __call__."""

    def test_an_anonymous_caller_reaches_an_inner_node(self):
        result = InputValidateNode()(_make_state(TrustLevel.ANONYMOUS.value))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query"), "an inner node should produce a normalised query"

    def test_an_anonymous_caller_is_denied_at_the_input_boundary(self):
        result = PreProcessNode()(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "validated_input" not in result, "execute() must not run on a denial"

    def test_an_authenticated_caller_clears_the_input_boundary(self):
        result = PreProcessNode()(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_the_nodes_own_checks_still_run_after_the_gate(self):
        result = PreProcessNode()(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        # The gate passes, the node declines the request, and the run completes
        # carrying the reason so the caller can send a question and try again.
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("error_code") == "EMPTY_INPUT"
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_an_anonymous_caller_is_denied_at_the_output_boundary(self):
        result = PostProcessNode()(
            _make_state(TrustLevel.ANONYMOUS.value, result="a clean catalog-grounded plan answer")
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a denial"

    def test_an_authenticated_caller_clears_the_output_boundary(self):
        result = PostProcessNode()(
            _make_state(TrustLevel.VERIFIED_EXTERNAL.value, result="a clean catalog-grounded plan answer")
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestTrustLevelMatrix:
    """The declared trust matrix (docs/02_design.md).

    The outer boundary slots require an authenticated caller; the inner domain
    nodes admit anonymous callers, because the external gate lives on the outer
    backbone and a stricter inner level would deny a real authenticated invoke
    at runtime.
    """

    def test_the_boundary_nodes_require_an_authenticated_caller(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_the_inner_domain_nodes_admit_anonymous_callers(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert (
                node_cls.required_trust_level is TrustLevel.ANONYMOUS
            ), f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS"
