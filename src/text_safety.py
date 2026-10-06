"""AgentCore Platform v1.0"""

# Text screens shared by the input boundary and the output boundary.
#
# Two directions, deliberately different:
#
#   strip_subscriber_identifiers()  runs on text a SUBSCRIBER typed. It is
#       broad: anything shaped like a phone number, an account/SIM number or
#       an e-mail is removed before the text is written to State, because a
#       subscriber pasting "my number is 090-1234-5678" must not have that
#       number persisted or echoed.
#
#   redact_published_identifiers()  runs on text about to leave the agent. It
#       is deliberately NARROWER: a plan catalog legitimately publishes
#       support hotlines (0120-…, 0800-…) and short numeric codes, and
#       redacting those would silently delete the answer the caller asked
#       for. It targets subscriber-identifying forms only — mobile numbers,
#       e-mail addresses, long account/SIM/device digit runs, and national
#       identifier shapes.
#
# Both directions are exercised in both senses by the test suite: hostile
# forms are removed, and ordinary catalog sentences survive byte-identical.

import re
from typing import List, Pattern

REDACTION = "[REDACTED]"

# Control characters (tab and newline excepted) never survive into State.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")

# ── Input-side screen (broad) ────────────────────────────────────────────────
_SUBSCRIBER_INPUT_PATTERNS: List[Pattern[str]] = [
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Grouped account / phone-shaped runs: three digit groups, optionally
    # separated by a hyphen or a space (090-1234-5678, 0120 456 7890).
    re.compile(r"\b\d{2,4}[- ]?\d{3,4}[- ]?\d{2,8}\b"),
    # Ungrouped runs long enough to be an account, SIM or device identifier.
    # The grouped pattern above cannot match these: it caps at 16 digits and
    # then requires a word boundary, which a longer run never provides.
    re.compile(r"\b\d{7,}\b"),
    # A national-identifier shape a subscriber may paste into a support
    # question (three-two-four with a separator). The grouped pattern above
    # requires a three-or-four-digit middle group and misses it.
    re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
]

# ── Output-side screen (subscriber forms only) ───────────────────────────────
_PUBLISHED_IDENTIFIER_PATTERNS: List[Pattern[str]] = [
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Domestic mobile numbers (070/080/090 prefixes) — hyphenated, spaced or
    # bare. Hotline prefixes (0120/0800) and landline area codes are NOT
    # matched: a catalog publishes those on purpose.
    re.compile(r"\b0[789]0[- ]?\d{4}[- ]?\d{4}\b"),
    # International form of the same.
    re.compile(r"\+81[- ]?[789]0[- ]?\d{4}[- ]?\d{4}\b"),
    # Account / SIM / device identifiers: an unbroken run of 13+ digits (an
    # ICCID is 19-20, an IMEI 15). Prices, years, data volumes and hotline
    # numbers are all far shorter.
    re.compile(r"\b\d{13,}\b"),
    # A national-identifier shape (three-two-four with a separator). No
    # catalog token has this form, so removing it costs nothing on the way out.
    re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
]

# Bracketed digit runs in caller-supplied catalog text are neutralised at
# ingest: the rendered answer uses [n] markers for its own citations, and a
# passage carrying a stray "[3]" would otherwise read as a citation that no
# source backs.
_BRACKETED_DIGITS_RE = re.compile(r"\[(\d{1,3})\]")


def strip_control_chars(text: str) -> str:
    """Remove control characters that have no meaning in a question or a passage."""
    return _CONTROL_CHARS_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run (including newlines) to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_subscriber_identifiers(text: str) -> str:
    """Remove subscriber-identifying tokens from text a subscriber supplied."""
    for pattern in _SUBSCRIBER_INPUT_PATTERNS:
        text = pattern.sub(REDACTION, text)
    return text


def redact_published_identifiers(text: str) -> tuple[str, int]:
    """Remove subscriber-identifying tokens from text about to be published.

    Returns the cleaned text and the number of redactions made, so the caller
    can record an audit event when the screen actually fired.
    """
    total = 0
    for pattern in _PUBLISHED_IDENTIFIER_PATTERNS:
        text, count = pattern.subn(REDACTION, text)
        total += count
    return text, total


def render_safe_text(text: str, limit: int) -> str:
    """Make caller-supplied prose safe to render inside the answer document.

    Control characters go, every whitespace run (newlines included) collapses
    to a single space so caller text cannot forge document structure, stray
    citation markers are turned into plain parentheses, and the result is
    capped at ``limit`` characters.
    """
    cleaned = collapse_whitespace(strip_control_chars(text))
    cleaned = _BRACKETED_DIGITS_RE.sub(r"(\1)", cleaned)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip()
    return cleaned
