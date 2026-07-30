"""Tests for pii.scrub_pii — the regex-based email/phone masking helper."""

from pii import scrub_pii


def test_masks_a_plain_email():
    assert scrub_pii("Contact: jane.doe@example.com") == "Contact: [EMAIL]"


def test_masks_an_email_with_plus_tag_and_subdomain():
    text = "reach out to first.last+work@mail.example.co.uk please"
    assert scrub_pii(text) == "reach out to [EMAIL] please"


def test_masks_a_hyphenated_phone_number():
    assert scrub_pii("Call 555-123-4567 for support") == "Call [PHONE] for support"


def test_masks_a_parenthesized_phone_number():
    assert scrub_pii("(555) 123-4567") == "[PHONE]"


def test_masks_an_international_style_phone_number():
    assert scrub_pii("+1 555.123.4567") == "[PHONE]"


def test_masks_multiple_matches_in_one_string():
    text = "Email jane@example.com or call 555-123-4567."
    assert scrub_pii(text) == "Email [EMAIL] or call [PHONE]."


def test_leaves_text_without_pii_unchanged():
    text = "Job #4521 is 3 days overdue, invoice total $1,200."
    assert scrub_pii(text) == text


def test_handles_empty_string():
    assert scrub_pii("") == ""
