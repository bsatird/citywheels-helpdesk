"""``search_policies`` -- retrieval quality on the real policy documents."""

from __future__ import annotations

import pytest

from citywheels.tools.policies import (
    load_index,
    search_policies,
    split_markdown,
    tokenize,
)

FAQ_OPEN = "data/policies/faq.md#when-are-you-open"
PRICES_RENTAL = "data/policies/prices.md#bike-rental"
PRICES_DEPOSIT = "data/policies/prices.md#deposit"
PRICES_LATE = "data/policies/prices.md#late-return"
CANCEL_BOOKING = "data/policies/cancellations.md#cancelling-a-booking"
CANCEL_WEATHER = "data/policies/cancellations.md#bad-weather"


def sources(result: dict) -> list[str]:
    return [hit["source"] for hit in result["results"]]


# --- indexing ---------------------------------------------------------------
def test_every_document_is_indexed():
    docs = {section.doc for section in load_index().sections}
    assert docs == {"prices.md", "cancellations.md", "faq.md"}


def test_expected_sections_are_present():
    anchors = {section.source for section in load_index().sections}
    for expected in (FAQ_OPEN, PRICES_RENTAL, PRICES_DEPOSIT, PRICES_LATE, CANCEL_WEATHER):
        assert expected in anchors


def test_split_markdown_keeps_headings_as_sections():
    text = "# Title\n\nIntro line.\n\n## First\n\nBody one.\n\n## Second\n\nBody two.\n"
    sections = split_markdown(text, "doc.md")
    headings = [section.heading for section in sections]
    assert "First" in headings and "Second" in headings
    assert sections[0].doc_title == "Title"  # the intro keeps the document title
    assert sections[-1].text == "Body two."


def test_tokenize_drops_stopwords_and_keeps_hyphens():
    tokens = tokenize("What is the price of an e-bike?")
    assert "what" not in tokens and "the" not in tokens
    assert "e-bike" in tokens and "price" in tokens


# --- retrieval quality ------------------------------------------------------
# The model receives the top 3 passages, so the expected section only has to be
# inside that window. Unambiguous queries must win outright.
@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("What time do you open on Saturday?", FAQ_OPEN),
        ("opening hours", FAQ_OPEN),
        ("How much is an e-bike for a whole day?", PRICES_RENTAL),
        ("late return fee", PRICES_LATE),
        ("what happens if I return the bike late", PRICES_LATE),
        ("storm warning", CANCEL_WEATHER),
        ("Can my 14-year-old rent an e-bike?", "data/policies/faq.md#are-there-age-limits"),
        ("do you sell bikes", "data/policies/faq.md#do-you-sell-bikes"),
        ("can I pay the deposit in cash", "data/policies/faq.md#how-can-i-pay"),
        (
            "how many bikes can I take on a train",
            "data/policies/faq.md#can-i-take-the-bike-on-a-train-or-outside-the-city",
        ),
        ("is there a child seat", "data/policies/prices.md#add-ons"),
        ("What is the phone number?", "data/policies/cancellations.md#contact-the-staff"),
    ],
)
def test_the_right_section_is_ranked_first(query, expected_top):
    result = search_policies(query, top_k=3)
    assert result["found"] is True
    assert sources(result)[0] == expected_top, sources(result)


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        # both sections are relevant, the passage only has to be in the top 3
        ("what does it cost to rent a bike for a week", PRICES_RENTAL),
        ("how much deposit for an electric bike", PRICES_DEPOSIT),
        ("can I cancel my booking", CANCEL_BOOKING),
        ("do I need a helmet", "data/policies/faq.md#do-i-have-to-wear-a-helmet"),
    ],
)
def test_relevant_section_is_in_the_top_three(query, expected_section):
    result = search_policies(query, top_k=3)
    assert result["found"] is True
    assert expected_section in sources(result), sources(result)



# --- behaviour the model depends on -----------------------------------------
def test_passages_carry_their_source():
    result = search_policies("late return fee")
    passage = result["results"][0]
    assert passage["source"].startswith("data/policies/")
    assert passage["section"]
    assert passage["text"]
    assert passage["score"] > 0


def test_off_topic_question_finds_nothing():
    result = search_policies("What is the capital of Australia?")
    assert result["found"] is False
    assert result["results"] == []
    assert "could not find" in result["message"].lower()


def test_empty_query_asks_for_a_question():
    result = search_policies("   ")
    assert result["found"] is False
    assert result["results"] == []


def test_top_k_is_clamped():
    assert len(search_policies("bike", top_k=99)["results"]) <= 5
    assert len(search_policies("bike", top_k=0)["results"]) == 1
    assert len(search_policies("bike", top_k="three")["results"]) == 3


def test_faq_facts_are_in_the_index():
    # the answer the model must be able to quote for "when do you open"
    text = search_policies("opening hours saturday")["results"][0]["text"]
    assert "Monday to Saturday" in text
    assert "08:00" in text
