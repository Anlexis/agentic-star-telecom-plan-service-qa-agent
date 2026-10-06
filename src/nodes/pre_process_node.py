"""AgentCore Platform v1.0"""

# The outer backbone's pre_process slot: the agent's input boundary.
#
# Responsibilities, in order:
#   1. trust gate — required_trust_level is VERIFIED_EXTERNAL, so an
#      unauthenticated caller is refused here and never reaches the domain
#      pipeline (src/api/server.py raises an authenticated bearer caller to
#      that level);
#   2. shape checks — a missing, non-string, empty or over-long question is
#      refused before anything downstream runs;
#   3. instruction-injection screen — a question that tries to redirect the
#      agent is refused by the agent itself, not left to whatever gate happens
#      to sit in front of it. The screen runs on the question as received,
#      before any rewriting, so no earlier transformation can erase the shape
#      it looks for;
#   4. caller-data contract — every field of input_context is bounds-checked;
#      an out-of-contract field aborts the run and the error names the field
#      only, never the rejected value;
#   5. identifier screen — account numbers, phone numbers and e-mail addresses
#      are removed from the question before validated_input is written, so a
#      raw subscriber identifier never reaches a domain node or a checkpoint.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INPUT_REJECTED
from src.caller_contract import MAX_QUESTION_CHARS, screen_question, validate_input_context
from src.text_safety import collapse_whitespace, strip_control_chars, strip_subscriber_identifiers

logger = logging.getLogger(__name__)


class PreProcessNode(FunctionNode):
    """Validate and screen the caller's question before the pipeline runs.

    Input state keys:
        user_input:    caller-supplied plan or service question
        input_context: optional caller data (see src/caller_contract.py)

    Output state keys (partial dict):
        validated_input:  normalised, identifier-stripped question
        enriched_context: channel metadata
        status:           SUCCESS, or ERROR when the request is refused
        error_log:        set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _refuse(self, state: AgentState, reason: str, message: str) -> Dict[str, Any]:
        """Terminal refusal: the agent will not process this content at all.

        Reserved for content a reworded retry should not get past. A request
        the caller can correct goes through _decline() instead, so a refusal is
        never presented as something a corrected request would be accepted.
        """
        logger.warning("PreProcessNode: request refused (%s)", reason)
        emit_trace_event("input_validation_failed", {"reason": reason}, state)
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
        }

    def _decline(self, state: AgentState, code: str, message: str) -> Dict[str, Any]:
        """Decline a request the caller can correct, and COMPLETE the run.

        No question is processed and the audit event is still recorded, exactly
        as with a refusal - what differs is the reporting. Terminating here
        would end the caller's turn and surface only an exception type, leaving
        the reason reachable solely from the audit trail; completing with the
        reason lets the caller fix the value and send the request again on the
        same conversation.
        """
        logger.warning("PreProcessNode: request declined (%s)", code)
        emit_trace_event("input_validation_declined", {"reason": code}, state)
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": code,
            "error_log": [f"PreProcessNode: {message}"],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context") or {}

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._decline(state, "EMPTY_INPUT", "user_input is empty or missing")

        if len(user_input) > MAX_QUESTION_CHARS:
            return self._decline(
                state,
                "QUESTION_TOO_LONG",
                f"the question exceeds {MAX_QUESTION_CHARS} characters",
            )

        # Screen before rewriting: the pattern scan must see the question as
        # the caller sent it.
        family = screen_question(user_input)
        if family:
            return self._refuse(
                state,
                f"injection_{family}",
                "the question contains an instruction the agent does not accept",
            )

        context, context_error = validate_input_context(input_context)
        if context_error:
            return self._decline(state, "INVALID_REQUEST", context_error)

        cleaned = collapse_whitespace(strip_control_chars(user_input))
        validated_input = strip_subscriber_identifiers(cleaned)

        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "caller_documents": len(context.get("catalog_documents", [])),
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "TelecomPlanServiceQAAgent",
                "channel": context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }
