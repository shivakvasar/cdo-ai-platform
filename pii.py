"""Lightweight PII scrubbing for anything that leaves this machine or gets
written to logs — the Claude prompts built from workspace data, and the
debug prints in mapper_agent's tool-calling loop.

This is intentionally basic (regex, not a real PII-detection model). It's
the pulled-forward, minimal version of the full governance work planned for
later — good enough to stop an email or phone number from a customer's
spreadsheet showing up in a live app's server logs or in what we send to
the Claude API, not a compliance-grade redaction tool.
"""

import re

# `[\w.+-]+` = the local part before the @ (letters/digits/underscore, plus
# dots, pluses, hyphens — covers things like "first.last+tag@").
# `[\w-]+(?:\.[\w-]+)+` = the domain: one or more dot-separated labels, so it
# matches "example.com" and "mail.example.co.uk" alike.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Matches common US/international phone formats: optional leading "+1",
# optional parens around the area code, and "-", ".", or " " as separators —
# e.g. "555-123-4567", "(555) 123-4567", "+1 555.123.4567", "5551234567".
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)


def scrub_pii(text: str) -> str:
    """Mask emails and phone numbers in `text`, leaving everything else as-is.

    Order matters: emails are masked first so a numeric-looking chunk inside
    an email's domain (rare, but possible) never gets caught by the phone
    regex on the same pass.
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text
