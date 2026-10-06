# The output boundary: the published invariant, enforced in both directions.
#
# Every layer is probed twice — once with the form it exists to catch, and once
# with ordinary catalog text that must survive byte-identical. The second half
# matters as much as the first: a boundary that deletes published support
# hotlines or mangles a product code has silently removed the answer.
#
# Nodes are invoked as node(state) so the framework's own gates run first, the
# way they do in production. The trust-gate rejection lives in
# test_trust_gate.py.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.post_process_node import _security_gate_output
from src.nodes.post_process_node import PostProcessNode
from src.output_schema import INFORMATIONAL_DISCLAIMER

_ANSWER_BODY = (
    "[1] Unlimited Plus plan pricing: the plan costs 6,980 yen per month and includes "
    "30GB of tethering. Support hotline 0120-000-111, store code 5G-1000, from 2026."
)

_CLEAN_DOCUMENT = (
    "# Telecom Plan & Service Answer\n\n"
    "The following passages from the plan and service catalog address this question:\n\n"
    f"{_ANSWER_BODY}\n\n"
    "## Sources\n"
    "- [1] Unlimited Plus plan pricing (Consumer Plan Catalog)\n\n"
    "---\n\n"
    f"*{INFORMATIONAL_DISCLAIMER}*"
)

# Credential-shaped fixtures are assembled at runtime so no credential literal
# ever sits in the repository.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestCleanDocument:
    def test_a_clean_answer_passes_through_byte_identical(self):
        result = PostProcessNode()(_make_state(_CLEAN_DOCUMENT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the enum.
        assert not isinstance(result["status"], AgentStatus)
        assert result["formatted_output"] == _CLEAN_DOCUMENT

    def test_an_empty_result_is_not_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestCredentialLayer:
    """Layer 1 — a credential shape withholds the whole document."""

    def _assert_withheld(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("withheld" in str(e) for e in result["error_log"])
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "withheld at the output boundary" in result["formatted_output"]

    def test_an_api_key_is_withheld(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Answer\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_withheld(result, secret)

    def test_a_credential_assignment_is_withheld(self):
        result = PostProcessNode()(_make_state("# Answer\n\ninternal note: password=super_secret_value_123\n"))
        self._assert_withheld(result, "super_secret_value_123")

    def test_a_web_token_is_withheld(self):
        result = PostProcessNode()(_make_state(f"# Answer\n\nsession token {_FAKE_JWT}\n"))
        self._assert_withheld(result, _FAKE_JWT)

    def test_a_bearer_token_is_withheld(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Answer\n\nauthorization: {secret}\n"))
        self._assert_withheld(result, secret)

    def test_credentials_are_scanned_before_anything_rewrites_the_document(self):
        """A pattern scan must see the text as assembled.

        The identifier screen replaces digit runs. If it ran first it could
        break a token into pieces the credential scanner no longer matches, so
        the scan is deliberately the first layer.
        """
        secret = "sk-" + "1" * 20
        violation, _, _ = _security_gate_output(f"# Answer\n\n{secret}\n", {})
        assert violation == "api_key"

    def test_nested_strings_are_scanned_too(self):
        violation = _security_gate_output("x", {})[0]
        assert violation is None
        from src.nodes.post_process_node import _scan_credentials

        assert _scan_credentials({"a": ["clean", {"b": "sk-ABCDEF0123456789abcdef"}]}) == "api_key"


class TestCallerEchoLayer:
    """Layer 2 — the caller's own question must not come back in the answer."""

    def test_a_verbatim_echo_of_the_question_is_replaced(self):
        question = "What is the monthly price and data allowance for the Unlimited Plus plan?"
        document = _CLEAN_DOCUMENT.replace("The following passages", f"{question} The following passages")
        result = PostProcessNode()(_make_state(document, search_query=question))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert question not in result["formatted_output"]

    def test_a_short_fragment_is_not_treated_as_an_echo(self):
        """Short caller strings collide with ordinary catalog wording."""
        result = PostProcessNode()(_make_state(_CLEAN_DOCUMENT, search_query="plan"))
        assert result["formatted_output"] == _CLEAN_DOCUMENT

    def test_every_blocked_field_is_checked(self):
        from src.nodes.post_process_node import _BLOCKED_FIELDS, _redact_caller_echo

        assert set(_BLOCKED_FIELDS) == {"search_query", "validated_input", "user_input"}
        fragment = "a question long enough to count as an echo"
        for field in _BLOCKED_FIELDS:
            cleaned, count = _redact_caller_echo(f"body {fragment} tail", {field: fragment})
            assert count == 1 and fragment not in cleaned


class TestIdentifierLayer:
    """Layer 3 — subscriber identifiers out, published catalog detail intact."""

    @pytest.mark.parametrize(
        "leak",
        [
            "090-1234-5678",
            "08012345678",
            "+81-90-1234-5678",
            "subscriber@example.com",
            "8981100012345678901",
            "123-45-6789",
        ],
    )
    def test_subscriber_identifiers_are_removed(self, leak):
        document = _CLEAN_DOCUMENT.replace("30GB of tethering", f"30GB of tethering, contact {leak}")
        result = PostProcessNode()(_make_state(document))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert leak not in result["formatted_output"]

    @pytest.mark.parametrize(
        "published",
        ["0120-000-111", "6,980", "30GB", "5G-1000", "2026"],
    )
    def test_published_catalog_detail_survives(self, published):
        """The screen is narrower on the way out on purpose.

        A catalog publishes hotlines, prices, data volumes, product codes and
        years; removing them would delete the answer the caller asked for.
        """
        result = PostProcessNode()(_make_state(_CLEAN_DOCUMENT))
        assert published in result["formatted_output"]


class TestGroundingAndDisclaimerLayer:
    """Layer 4 — every marker resolves, and the disclaimer is always present."""

    def test_a_dangling_citation_marker_withholds_the_answer(self):
        document = _CLEAN_DOCUMENT.replace("[1] Unlimited Plus plan pricing:", "[1] Unlimited Plus plan pricing [4]:")
        result = PostProcessNode()(_make_state(document))
        assert result["status"] == AgentStatus.ERROR.value
        assert "ungrounded_citation" in result["error_log"][0]

    def test_a_resolved_citation_marker_is_accepted(self):
        result = PostProcessNode()(_make_state(_CLEAN_DOCUMENT))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_a_missing_disclaimer_is_restored(self):
        document = _CLEAN_DOCUMENT.replace(f"\n\n---\n\n*{INFORMATIONAL_DISCLAIMER}*", "")
        result = PostProcessNode()(_make_state(document))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert INFORMATIONAL_DISCLAIMER in result["formatted_output"]

    def test_an_answer_with_no_sources_section_and_no_markers_is_accepted(self):
        document = (
            "# Telecom Plan & Service Answer\n\nThe catalog does not cover this question.\n\n"
            f"---\n\n*{INFORMATIONAL_DISCLAIMER}*"
        )
        result = PostProcessNode()(_make_state(document))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == document


class TestLayerCounters:
    """Each layer reports separately, so an operator can tell which one fired."""

    def test_counters_are_zero_for_a_clean_document(self):
        violation, document, counters = _security_gate_output(_CLEAN_DOCUMENT, {})
        assert violation is None
        assert document == _CLEAN_DOCUMENT
        assert counters == {"caller_echo": 0, "identifiers": 0, "disclaimer_restored": 0}

    def test_counters_record_each_layer_that_fired(self):
        question = "What is the monthly price and data allowance for the Unlimited Plus plan?"
        document = _CLEAN_DOCUMENT.replace(
            "30GB of tethering", f"30GB of tethering, contact 090-1234-5678. {question}"
        ).replace(f"\n\n---\n\n*{INFORMATIONAL_DISCLAIMER}*", "")
        violation, _, counters = _security_gate_output(document, {"search_query": question})
        assert violation is None
        assert counters == {"caller_echo": 1, "identifiers": 1, "disclaimer_restored": 1}
