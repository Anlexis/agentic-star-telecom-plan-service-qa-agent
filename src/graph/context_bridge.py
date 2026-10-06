"""AgentCore Platform v1.0"""

# Carries the caller's input_context across the outer -> inner graph boundary.
#
# Why this exists: the framework's GraphNode runs the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward
# the outer state's input_context. Without a bridge, an inner node reading
# state["input_context"] would always see {} — so the caller's catalog slice
# and retrieval overrides would never reach the nodes that use them.
#
# The two sanctioned subclass hooks hand the value over:
#
#   PlanServiceQAGraphNode.extract_input(state)   runs BEFORE subgraph.invoke()
#       -> set_caller_input_context(...)
#   DomainWorkflowGraph._extra_initial_state()    runs INSIDE subgraph.invoke()
#       -> returns {"input_context": get_caller_input_context()}
#
# A ContextVar keeps the hand-off per-thread and per-task, so concurrent
# invocations in one process cannot observe each other's caller data.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_INPUT_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "tel_c2_001_caller_input_context",
    default=None,
)


def set_caller_input_context(input_context: Optional[Dict[str, Any]]) -> None:
    """Stash the outer graph's input_context for the imminent inner-graph invoke."""
    _CALLER_INPUT_CONTEXT.set(dict(input_context) if input_context else {})


def get_caller_input_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed input_context; {} when none was set."""
    return _CALLER_INPUT_CONTEXT.get() or {}
