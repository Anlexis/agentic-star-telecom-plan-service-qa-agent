"""AgentCore Platform v1.0"""

# The published shape of an answer — declared once, in one place.
#
# OutputFormatNode RENDERS documents in this shape; PostProcessNode ENFORCES
# it at the boundary. Both read the constants below, so the rendered document
# and the enforced invariant cannot drift apart.
#
# The invariant, stated in full:
#
#   Every answer this agent emits
#     (a) carries the standing informational disclaimer, and
#     (b) carries a Sources section in which every [n] marker used in the
#         answer body resolves to a listed source, and
#     (c) contains no direct subscriber identifier and no credential-shaped
#         string.
#
# Note on rounding: some agents in this family publish monetary AGGREGATES and
# round them onto a coarse grid so that individual figures cannot be recovered.
# That does not apply here and is deliberately not done. This agent answers
# questions about published plan pricing, where "¥6,980 per month" is the
# answer — rounding it to ¥7,000 would make the agent wrong rather than
# discreet. Nothing here is an aggregate over private records; every figure is
# quoted from a catalog passage that is cited alongside it.

import re

# Document structure.
ANSWER_HEADING = "# Telecom Plan & Service Answer"
SOURCES_HEADING = "## Sources"
NO_SOURCES_LINE = "- none (no catalog passage cleared the relevance threshold)"

# The standing disclaimer, attached to every answer. It records what this agent
# is and is not: a reader of the catalog, never a view onto a live account or
# billing system.
INFORMATIONAL_DISCLAIMER = (
    "This answer is generated from the plan and service catalog for "
    "informational purposes only. Pricing, promotions, taxes, and coverage "
    "details may vary by account, region, and current offers — confirm final "
    "terms with a customer service representative or your account portal "
    "before making a purchase or plan change."
)

# A citation marker in the answer body: [1], [2], ...
CITATION_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

# A listed source in the Sources section: "- [1] Title (source)".
SOURCE_ENTRY_RE = re.compile(r"^- \[(\d{1,3})\]", re.MULTILINE)

# The Sources heading only counts when it OPENS a line. A heading is a
# line-initial construct, and caller-supplied passages are flattened to a
# single line before they render — so this is also what stops a passage that
# contains the heading text from splitting the document somewhere else and
# hiding the markers that follow it.
_SOURCES_SPLIT_RE = re.compile(r"^" + re.escape(SOURCES_HEADING) + r"[ \t]*$", re.MULTILINE)


def split_document(document: str) -> tuple[str, str]:
    """Split a rendered answer into (body, sources section)."""
    parts = _SOURCES_SPLIT_RE.split(document, maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def cited_refs(document: str) -> set[str]:
    """Every citation marker used anywhere in the document body."""
    body, _ = split_document(document)
    return set(CITATION_MARKER_RE.findall(body))


def listed_refs(document: str) -> set[str]:
    """Every reference number listed in the Sources section."""
    _, sources = split_document(document)
    return set(SOURCE_ENTRY_RE.findall(sources))
