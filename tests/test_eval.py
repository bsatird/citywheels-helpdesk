"""Tests for the evaluation harness itself (``eval/run_eval.py``).

The evaluation set is data, so it can rot: a question can point at a booking that
was edited, an expected amount can stop matching the refund rules, or a case can
lose its id. These tests keep the harness honest and make sure its scoring really
fails a bad answer instead of quietly passing everything.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from citywheels.agent import AgentResult
from citywheels.tools.refunds import calculate_refund

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.json"

REQUIRED_KEYS = {"id", "question", "expect_tools", "expect_amounts", "note"}
CHECK_KEYS = {"tools", "amounts", "phrases", "forbidden", "refusal", "grounded"}
BOOKING_ID_RE = re.compile(r"CW-\d{4}")
MONEY_FIELDS = (
    "total_refund_eur",
    "rental_refund_eur",
    "deposit_refund_eur",
    "booking_fee_refund_eur",
    "rental_price_eur",
    "deposit_eur",
)
CASE_IDS = [case["id"] for case in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))]



@pytest.fixture(scope="module")
def harness():
    """``eval/run_eval.py`` as a module (``eval`` is not a package)."""
    spec = importlib.util.spec_from_file_location("citywheels_eval", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["citywheels_eval"] = module  # dataclass field defaults need this
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cases(harness):
    return harness.load_cases(QUESTIONS_PATH)


def result_of(harness, *, answer: str, tools=(), unsupported=()) -> AgentResult:
    return AgentResult(
        question="q",
        answer=answer,
        tool_calls=list(tools),
        unsupported_amounts=list(unsupported),
    )


# --------------------------------------------------------------------------- #
# the question set
# --------------------------------------------------------------------------- #


def test_question_file_is_well_formed(cases):
    assert len(cases) >= 12
    seen_ids = set()
    for case in cases:
        assert REQUIRED_KEYS <= set(case), case["id"]
        assert case["id"] not in seen_ids, f"duplicate id {case['id']}"
        seen_ids.add(case["id"])
        assert case["question"].strip().endswith("?")
        assert case["note"].strip()
        assert isinstance(case["expect_tools"], list)
        for group in case.get("expect_phrases", []):
            assert isinstance(group, list) and group, case["id"]
            assert all(isinstance(option, str) for option in group)
        for amount in case["expect_amounts"]:
            assert isinstance(amount, (int, float)) and amount > 0, case["id"]


def test_question_set_covers_every_tool_and_both_outcomes(cases):
    tools = {tool for case in cases for tool in case["expect_tools"]}
    assert {"search_policies", "get_booking", "calculate_refund"} <= tools
    assert sum(1 for case in cases if case.get("expect_refusal")) >= 2
    assert sum(1 for case in cases if "calculate_refund" in case["expect_tools"]) >= 4


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_refund_expectations_match_the_real_tool(cases, case_id):
    """Every amount promised by a refund question must come out of the tool."""
    case = case_by_id(cases, case_id)
    if "calculate_refund" not in case["expect_tools"] or not case["expect_amounts"]:
        pytest.skip("not a refund question")

    found = BOOKING_ID_RE.search(case["question"])
    assert found, f"{case['id']} must name a booking number"
    refund = calculate_refund(found.group())
    assert "error" not in refund

    money_values = [float(refund[field]) for field in MONEY_FIELDS]
    missing = [
        amount
        for amount in case["expect_amounts"]
        if not any(abs(amount - value) <= 0.011 for value in money_values)
    ]
    assert not missing, (
        f"{case['id']}: expected {missing} but calculate_refund returned "
        f"{dict(zip(MONEY_FIELDS, money_values))}"
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def case_by_id(cases, case_id):
    return next(case for case in cases if case["id"] == case_id)


def test_numbers_in_reads_the_usual_money_formats(harness):
    assert harness.numbers_in("You get 340 EUR back") == [340.0]
    assert harness.numbers_in("That is 1 234,50 EUR in total") == [1234.5]
    assert harness.numbers_in("40,5 EUR") == [40.5]
    assert harness.numbers_in("We open at 08:00") == [8.0, 0.0]
    assert harness.numbers_in("no digits here") == []
    assert harness.numbers_in("") == []


def test_has_amount_uses_a_one_cent_tolerance(harness):
    assert harness.has_amount([339.995], 340)
    assert not harness.has_amount([339.9], 340)


def test_is_refusal_recognises_a_decline(harness):
    assert harness.is_refusal("Sorry, I can only answer questions about CityWheels.")
    assert harness.is_refusal("We don't rent tandems.")
    assert harness.is_refusal("That is outside what I can check here.")
    assert not harness.is_refusal("An e-bike costs 40 EUR per day.")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_forbidden_phrases_are_not_contradicted_by_the_refund_tool(cases, case_id):
    """A case may not forbid a word that the tool output the model copies contains.

    Q04 forbids "cash" because a voucher refund never becomes cash. That only works
    while ``calculate_refund`` does not itself say "cash" - this is how the wording
    bug was found (the tool note said "never as cash", so every faithful answer
    failed the check).
    """
    case = case_by_id(cases, case_id)
    forbidden = case.get("forbid_phrases") or []
    if not forbidden or "calculate_refund" not in case["expect_tools"]:
        pytest.skip("no refund tool / nothing forbidden")

    found = BOOKING_ID_RE.search(case["question"])
    assert found, f"{case['id']} must name a booking number"
    dump = json.dumps(calculate_refund(found.group()), ensure_ascii=False).lower()
    leaked = [phrase for phrase in forbidden if phrase.lower() in dump]
    assert not leaked, f"{case['id']}: the tool output itself contains {leaked}"


def test_score_case_accepts_a_good_answer(harness, cases):
    case = case_by_id(cases, "Q01")
    result = result_of(
        harness,
        answer="You get 340 EUR back: 40 EUR of the rental price and the 300 EUR deposit to your card.",
        tools=["calculate_refund"],
    )
    checks = harness.score_case(case, result)
    assert set(checks) <= CHECK_KEYS
    assert all(checks.values()), checks


def test_score_case_fails_when_the_money_is_wrong(harness, cases):
    case = case_by_id(cases, "Q01")
    result = result_of(
        harness,
        answer="You get 320 EUR back to your card.",
        tools=["calculate_refund"],
    )
    checks = harness.score_case(case, result)
    assert checks["amounts"] is False
    assert checks["tools"] is True
    assert checks["phrases"] is True


def test_score_case_fails_when_the_tool_was_not_used(harness, cases):
    case = case_by_id(cases, "Q01")
    result = result_of(
        harness, answer="You get 340 EUR and the 300 EUR deposit to your card."
    )
    assert harness.score_case(case, result)["tools"] is False


def test_score_case_flags_a_forbidden_phrase(harness, cases):
    case = case_by_id(cases, "Q04")  # a voucher booking must never promise cash
    result = result_of(
        harness,
        answer="You get 73 EUR in cash instead of the voucher.",
        tools=["calculate_refund"],
    )
    checks = harness.score_case(case, result)
    assert checks["forbidden"] is False
    assert checks["phrases"] is True


def test_a_negated_forbidden_word_is_not_a_claim(harness):
    """\"not as cash\" mentions the word but promises the opposite, so it is allowed."""
    assert harness.mentions_forbidden("You get the money in cash.", "cash")
    assert not harness.mentions_forbidden("It comes back as a voucher, not as cash.", "cash")
    assert not harness.mentions_forbidden("The refund is never paid out in cash.", "cash")
    assert not harness.mentions_forbidden("A voucher rather than cash.", "cash")
    assert not harness.mentions_forbidden("Instead of cash you get a voucher.", "cash")


def test_a_negation_far_away_does_not_excuse_the_claim(harness):
    text = (
        "The booking exists and was paid by voucher. There is no doubt about that. "
        "You will get the 73 EUR in cash at the counter."
    )
    assert harness.mentions_forbidden(text, "cash")


def test_score_case_accepts_a_negated_forbidden_word(harness, cases):
    case = case_by_id(cases, "Q04")
    result = result_of(
        harness,
        answer=(
            "You get 73 EUR as a new voucher, not as cash: 23 EUR of the rental "
            "price and the 50 EUR deposit."
        ),
        tools=["calculate_refund"],
    )
    checks = harness.score_case(case, result)
    assert checks["forbidden"] is True
    assert all(checks.values()), checks


def test_score_case_flags_an_ungrounded_amount(harness, cases):
    case = case_by_id(cases, "Q13")  # out of scope, no amount may appear
    result = result_of(harness, answer="Canberra.", unsupported=[42.0])
    checks = harness.score_case(case, result)
    assert checks["grounded"] is False
    assert checks["refusal"] is False


def test_refusal_check_only_applies_to_out_of_scope_cases(harness, cases):
    result = result_of(
        harness,
        answer="You get 340 EUR back to your card.",
        tools=["calculate_refund"],
    )
    assert "refusal" not in harness.score_case(case_by_id(cases, "Q01"), result)


def test_case_result_passes_only_when_every_check_passes(harness):
    evaluated = harness.CaseResult(
        case_id="Q01", question="q", checks={"tools": True, "amounts": False}
    )
    assert evaluated.passed is False
    evaluated.checks["amounts"] = True
    assert evaluated.passed is True
    evaluated.error = "model exploded"
    assert evaluated.passed is False


def test_fake_model_never_scores_full_marks(harness, cases):
    """The grader has to discriminate, otherwise the score is meaningless."""
    factory, (provider, _model) = harness.build_agent_factory(
        fake=True, max_iterations=1
    )
    results = harness.run_cases(cases, factory, quiet=True)
    stats = harness.summarize(results, cases)
    assert provider == "fake"
    assert stats["passed"] < stats["cases"]
    assert stats["errors"] == 0


# --------------------------------------------------------------------------- #
# reporting and the command line
# --------------------------------------------------------------------------- #


def passing_result(harness, case_id, answer="340 EUR to your card"):
    return harness.CaseResult(
        case_id=case_id,
        question="q",
        answer=answer,
        tools=["calculate_refund"],
        checks={
            "tools": True,
            "amounts": True,
            "phrases": True,
            "forbidden": True,
            "grounded": True,
        },
    )


def test_render_markdown_reports_a_clean_run(harness, cases):
    results = [passing_result(harness, "Q01")]
    report = harness.render_markdown(
        results, [case_by_id(cases, "Q01")], provider="p", model="m", fake=True
    )
    assert "# Evaluation results" in report
    assert "| Q01 |" in report
    assert "1/1 correct (100%)" in report
    assert "Plumbing check only" in report
    assert "## Failures in detail" not in report
    assert "How the score is computed" in report


def test_render_markdown_explains_a_failure(harness, cases):
    failing = passing_result(harness, "Q01", answer="You get 320 EUR to your card")
    failing.checks["amounts"] = False
    failing.unsupported = [320.0]
    report = harness.render_markdown(
        [failing], [case_by_id(cases, "Q01")], provider="p", model="m", fake=False
    )
    assert "## Failures in detail" in report
    assert "**Failed checks:** amounts" in report
    assert "unsupported amounts: 320" in report
    assert "0/1 correct (0%)" in report
    assert "Plumbing check only" not in report


def test_main_with_the_fake_model_writes_the_report(harness, tmp_path, capsys):
    destination = tmp_path / "results.md"
    code = harness.main(
        ["--fake", "--only", "q13,Q14", "--quiet", "--out", str(destination)]
    )
    assert code == 0
    report = destination.read_text(encoding="utf-8")
    assert "| Q13 |" in report
    assert "| Q14 |" in report
    assert "| Q01 |" not in report
    assert "fake" in report
    assert "correct (" in capsys.readouterr().out


def test_main_prints_json_without_writing(harness, tmp_path, capsys):
    destination = tmp_path / "never.md"
    code = harness.main(["--fake", "--only", "Q13", "--json", "--out", str(destination)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "fake-model"
    assert payload["stats"]["cases"] == 1
    assert payload["results"][0]["id"] == "Q13"
    assert not destination.exists()


def test_main_rejects_unknown_case_ids(harness):
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--fake", "--only", "Q99", "--dry-run"])
    assert excinfo.value.code == 2


def test_load_cases_rejects_an_empty_file(harness, tmp_path):
    empty = tmp_path / "questions.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        harness.load_cases(empty)


# --------------------------------------------------------------------------- #
# resuming an interrupted run
# --------------------------------------------------------------------------- #


def graded_result(harness, case, *, answer=None, error=None):
    return harness.CaseResult(
        case_id=case["id"],
        question=case["question"],
        answer=answer if answer is not None else "Sorry, that is outside CityWheels.",
        checks={
            "tools": True,
            "amounts": True,
            "phrases": True,
            "forbidden": True,
            "grounded": True,
            "refusal": True,
        },
        error=error,
        note=case.get("note", ""),
    )


def test_case_fingerprint_follows_the_content_of_a_case(harness, cases):
    case = case_by_id(cases, "Q13")
    assert harness.case_fingerprint(case) == harness.case_fingerprint(dict(case))
    assert harness.case_fingerprint(case) != harness.case_fingerprint(
        {**case, "note": "edited later"}
    )


def test_a_graded_case_is_reused_instead_of_asked_again(harness, tmp_path, cases):
    cached, other = case_by_id(cases, "Q13"), case_by_id(cases, "Q14")
    cache = tmp_path / "cache.jsonl"
    harness.store_result(cache, cached, graded_result(harness, cached))
    harness.store_result(cache, other, graded_result(harness, other))
    destination = tmp_path / "results.md"

    code = harness.main(
        [
            "--fake",
            "--only",
            "Q13,Q14",
            "--quiet",
            "--resume",
            "--cache",
            str(cache),
            "--out",
            str(destination),
        ]
    )

    assert code == 0
    report = destination.read_text(encoding="utf-8")
    assert "Sorry, that is outside CityWheels." in report  # the cached answer
    assert "(fake model)" not in report  # the model was never asked
    assert "2/2 correct (100%)" in report
    assert report.index("| Q13 |") < report.index("| Q14 |")  # question order kept
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_the_report_says_how_many_answers_were_reused(harness, cases):
    report = harness.render_markdown(
        [passing_result(harness, "Q01")],
        [case_by_id(cases, "Q01")],
        provider="p",
        model="m",
        fake=False,
        reused=1,
    )
    assert "1 of these answers were reused" in report


def test_a_case_that_changed_is_asked_again(harness, tmp_path, cases):
    case = case_by_id(cases, "Q13")
    cache = tmp_path / "cache.jsonl"
    harness.store_result(
        cache, {**case, "note": "old wording"}, graded_result(harness, case)
    )
    destination = tmp_path / "results.md"

    code = harness.main(
        ["--fake", "--only", "Q13", "--quiet", "--resume", "--cache", str(cache),
         "--out", str(destination)]
    )

    assert code == 0
    assert "(fake model)" in destination.read_text(encoding="utf-8")


def test_a_rate_limited_case_is_asked_again(harness, tmp_path, cases):
    """An error row is not a grade: the model never answered, so it is retried."""
    case = case_by_id(cases, "Q13")
    cache = tmp_path / "cache.jsonl"
    harness.store_result(
        cache, case, graded_result(harness, case, error="429 rate limited")
    )
    destination = tmp_path / "results.md"

    code = harness.main(
        ["--fake", "--only", "Q13", "--quiet", "--resume", "--cache", str(cache),
         "--out", str(destination)]
    )

    assert code == 0
    report = destination.read_text(encoding="utf-8")
    assert "(fake model)" in report
    assert "429 rate limited" not in report
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2  # retry appended


def test_a_half_written_cache_line_is_skipped(harness, tmp_path, cases):
    case = case_by_id(cases, "Q13")
    cache = tmp_path / "cache.jsonl"
    harness.store_result(cache, case, graded_result(harness, case))
    with cache.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "Q13", "fingerpr')  # a run killed mid-write

    rows = harness.read_cache(cache)
    assert set(rows) == {"Q13"}
    assert harness.reusable(case, rows) is not None


def test_the_cache_is_ignored_without_resume(harness, tmp_path, cases):
    case = case_by_id(cases, "Q13")
    cache = tmp_path / "cache.jsonl"
    harness.store_result(cache, case, graded_result(harness, case))
    destination = tmp_path / "results.md"

    code = harness.main(
        ["--fake", "--only", "Q13", "--quiet", "--cache", str(cache),
         "--out", str(destination)]
    )

    assert code == 0
    assert "(fake model)" in destination.read_text(encoding="utf-8")
