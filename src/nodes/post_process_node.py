"""AgentCore Platform v1.0"""

# The outer backbone's post_process slot: the agent's output boundary.
#
# It enforces the published invariant declared in src/output_schema.py, in
# four independent layers. The ORDER matters and is part of the design:
#
#   (1) credential scan — runs FIRST, on the document exactly as assembled.
#       A scan that looks for a pattern must see the text before anything
#       rewrites it; a redaction applied earlier could break an API key or a
#       token into pieces the scanner no longer recognises. A hit withholds
#       the whole document — there is no partial release.
#   (2) caller-text echo — the answer is assembled from catalog passages and
#       their citations only, so a verbatim embedding of the caller's own
#       question means caller-controlled text reached the external surface.
#       Any such embedding is replaced.
#   (3) subscriber identifiers — mobile numbers, e-mail addresses and long
#       account or device numbers are removed. This screen is deliberately
#       narrower than the one on the input side: a catalog publishes support
#       hotlines and short numeric codes on purpose, and deleting those would
#       silently remove the answer the caller asked for.
#   (4) grounding and disclaimer — every [n] marker in the body must resolve
#       to a listed source, or the answer makes a claim nothing backs and is
#       withheld; and the standing disclaimer must be present, restored by the
#       gate if it is not.
#
# Each layer records its own audit event, so an operator can tell which one
# fired.
#
# The gate is the module-level function _security_gate_output(), called from
# inside execute(). It is deliberately not an instance method on the node: the
# framework auto-wraps node gate hooks, and defining one here would collide
# with that machinery.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG
from src.output_schema import INFORMATIONAL_DISCLAIMER, cited_refs, listed_refs
from src.text_safety import REDACTION, redact_published_identifiers

logger = logging.getLogger(__name__)

# Credential shapes that must never appear in a published answer.
# Ordered most specific first; the first hit names the violation.
_CREDENTIAL_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE)),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

# State fields that must never be embedded verbatim in the published answer.
# search_query carries the weight here: the framework masks user_input and
# validated_input at every node boundary, so by the time this node runs those
# two hold a masked copy that no longer matches the document. search_query is
# the unmasked normalised question, and it is the field an answer builder would
# actually echo. The other two stay in the list as a cheap second check.
_BLOCKED_FIELDS = ("search_query", "validated_input", "user_input")

# A caller-text fragment shorter than this is not evidence of an echo — short
# strings collide with ordinary catalog wording.
_MIN_ECHO_CHARS = 24

_WITHHELD_DOCUMENT = (
    "[This answer was withheld at the output boundary. Review the generated " "plan and service answer and retry.]"
)


def _iter_nested_strings(value: Any) -> Iterator[str]:
    """Yield every string reachable inside value (dict / list / tuple recursion).

    Today's answer is a plain string, but the walk stays recursive so a future
    structured field cannot smuggle content past a top-level-only scan.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_nested_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_nested_strings(child)


def _scan_credentials(content: Any) -> Optional[str]:
    """Return the name of the first credential shape found, or None."""
    for text in _iter_nested_strings(content):
        for name, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(text):
                return name
    return None


def _redact_caller_echo(document: str, state: AgentState) -> Tuple[str, int]:
    """Replace any verbatim embedding of caller-supplied request text."""
    redactions = 0
    for field in _BLOCKED_FIELDS:
        value = state.get(field)
        if not isinstance(value, str):
            continue
        fragment = value.strip()
        if len(fragment) < _MIN_ECHO_CHARS or fragment not in document:
            continue
        document = document.replace(fragment, REDACTION)
        redactions += 1
    return document, redactions


def _security_gate_output(document: str, state: AgentState) -> Tuple[Optional[str], str, Dict[str, int]]:
    """Run the four boundary layers over the assembled answer.

    Returns ``(violation, document, counters)``. A non-None violation means the
    answer must be withheld; otherwise ``document`` is the cleared text, which
    may differ from the input where a redaction or a restoration was applied.
    """
    counters = {"caller_echo": 0, "identifiers": 0, "disclaimer_restored": 0}

    # (1) Credentials — scanned on the document as assembled, before any rewrite.
    violation = _scan_credentials(document)
    if violation:
        return violation, document, counters

    # (2) Verbatim caller text.
    document, counters["caller_echo"] = _redact_caller_echo(document, state)

    # (3) Subscriber identifiers.
    document, counters["identifiers"] = redact_published_identifiers(document)

    # (4) Grounding, then the disclaimer.
    dangling = cited_refs(document) - listed_refs(document)
    if dangling:
        return "ungrounded_citation", document, counters

    if INFORMATIONAL_DISCLAIMER not in document:
        document = f"{document}\n\n---\n\n*{INFORMATIONAL_DISCLAIMER}*"
        counters["disclaimer_restored"] = 1

    return None, document, counters


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Enforce the published output invariant on the assembled answer.

    Input state keys:
        result: the rendered answer document, mapped in from the inner graph

    Output state keys (partial dict):
        formatted_output: the cleared document, or the withheld stub
        result:           kept in step with formatted_output
        status:           SUCCESS, or ERROR when the answer is withheld
        error_log:        set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }

        result = state.get("result") or ""

        if not result or not str(result).strip():
            # Nothing was generated — pass the empty result through unchanged.
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        violation, document, counters = _security_gate_output(str(result), state)

        if violation:
            logger.error("PostProcessNode: answer withheld at the output boundary (%s)", violation)
            emit_trace_event("output_withheld", {"violation": violation}, state)
            emit_progress(OUTPUT_BLOCKED)
            return {
                "formatted_output": _WITHHELD_DOCUMENT,
                "result": _WITHHELD_DOCUMENT,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: answer withheld at the output boundary ({violation})"],
            }

        if counters["caller_echo"]:
            emit_trace_event("output_caller_text_redacted", {"redactions": counters["caller_echo"]}, state)
        if counters["identifiers"]:
            emit_trace_event("output_identifiers_redacted", {"redactions": counters["identifiers"]}, state)
        if counters["disclaimer_restored"]:
            emit_trace_event("output_disclaimer_restored", {}, state)

        emit_trace_event("post_process_complete", {"output_chars": len(document)}, state)

        return {
            "formatted_output": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
