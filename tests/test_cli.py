"""The command-line front end, driven by a fake model."""

from __future__ import annotations

import io
import sys

import pytest

import citywheels.cli as cli
from citywheels.agent import MAX_ITERATIONS_FALLBACK
from citywheels.llm import LLMError
from citywheels.testing import FakeLLM, text_turn, tool_turn


@pytest.fixture
def wired(monkeypatch):
    """Point the CLI at a fake model; the returned helper scripts its turns."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")

    def _wire(*turns, repeat_last: bool = False) -> FakeLLM:
        llm = FakeLLM(list(turns), repeat_last=repeat_last)
        monkeypatch.setattr(cli, "build_client", lambda: llm)
        return llm

    return _wire


def test_one_question_prints_the_answer(wired, capsys):
    wired(
        tool_turn(("search_policies", {"query": "opening hours"})),
        text_turn("We are open Monday to Saturday, 08:00-20:00."),
    )
    assert cli.main(["When do you open on Saturday?"]) == 0

    captured = capsys.readouterr()
    assert "Monday to Saturday" in captured.out


def test_verbose_mode_shows_the_tool_trace(wired, capsys):
    wired(text_turn("Nothing to do."))
    assert cli.main(["--verbose", "hello"]) == 0

    captured = capsys.readouterr()
    assert "[trace] provider=openai model=fake-model" in captured.err
    assert "[trace] 1 model turn(s), stopped: final_answer" in captured.out


def test_verbose_mode_lists_tool_calls(wired, capsys):
    wired(
        tool_turn(("get_booking", {"booking_id": "CW-1001"})),
        text_turn("CW-1001 is a city bike."),
    )
    assert cli.main(["-v", "tell me about CW-1001"]) == 0

    out = capsys.readouterr().out
    assert "[trace] -> get_booking(booking_id='CW-1001')" in out


def test_piped_stdin_is_used_as_the_question(wired, monkeypatch, capsys):
    wired(text_turn("Yes, you can pay by card."))
    monkeypatch.setattr(sys, "stdin", io.StringIO("Can I pay by card?\n"))

    assert cli.main([]) == 0
    assert "pay by card" in capsys.readouterr().out


def test_no_question_prints_the_help(wired, monkeypatch, capsys):
    wired()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    assert cli.main([]) == 2
    assert "usage: citywheels" in capsys.readouterr().out


def test_unknown_provider_is_a_configuration_error(wired, monkeypatch, capsys):
    wired()
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")

    assert cli.main(["hello"]) == 1
    assert "Configuration error" in capsys.readouterr().err


def test_model_errors_are_reported_with_the_staff_contact(wired, monkeypatch, capsys):
    class BrokenModel:
        def chat(self, messages, tools=None):
            raise LLMError("quota exhausted")

    monkeypatch.setattr(cli, "build_client", lambda: BrokenModel())

    assert cli.main(["hello"]) == 1
    captured = capsys.readouterr()
    assert "quota exhausted" in captured.err
    assert "contact the staff" in captured.out


def test_interactive_session_remembers_the_previous_question(wired, monkeypatch, capsys):
    llm = wired(text_turn("The first answer."), text_turn("The second answer."))
    answers = iter(["first question", "second question", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main(["--interactive"]) == 0

    out = capsys.readouterr().out
    assert "CityWheels help desk agent" in out
    assert "The first answer." in out and "The second answer." in out
    second_call = llm.requests[1]["messages"]
    assert {"role": "assistant", "content": "The first answer."} in second_call


def test_interactive_quit_leaves_at_once(wired, monkeypatch):
    wired()
    answers = iter([":quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    assert cli.main(["-i"]) == 0


def test_max_iterations_argument_is_honoured(wired, capsys):
    wired(tool_turn(("get_current_time", {})), repeat_last=True)

    assert cli.main(["--max-iterations", "2", "what time is it"]) == 0
    assert MAX_ITERATIONS_FALLBACK in capsys.readouterr().out
