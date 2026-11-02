"""The agent loop, driven by a scripted fake model (no network, no API key)."""

from __future__ import annotations

from citywheels.agent import (
    EMPTY_ANSWER_FALLBACK,
    MAX_ITERATIONS,
    MAX_ITERATIONS_FALLBACK,
    HelpDeskAgent,
)
from citywheels.llm import ToolCall
from citywheels.testing import FakeLLM, text_turn, tool_turn


def build(*turns, **kwargs):
    """A fake model plus an agent wired to the real tools."""
    llm = FakeLLM(list(turns), **kwargs)
    return llm, HelpDeskAgent(llm)


def test_the_system_prompt_and_the_tool_schemas_are_sent():
    llm, agent = build(text_turn("Hello!"))
    agent.run("Hi")
    request = llm.requests[0]
    assert request["messages"][0]["role"] == "system"
    assert "CityWheels" in request["messages"][0]["content"]
    names = [spec["function"]["name"] for spec in request["tools"]]
    assert names == ["search_policies", "get_booking", "calculate_refund", "get_current_time"]
    assert agent.max_iterations == MAX_ITERATIONS


def test_a_policy_question_runs_one_tool_then_answers():
    llm, agent = build(
        tool_turn(("search_policies", {"query": "opening hours saturday"})),
        text_turn("We are open Monday to Saturday, 08:00-20:00."),
    )
    result = agent.run("When do you open on Saturday?")

    assert result.answer == "We are open Monday to Saturday, 08:00-20:00."
    assert result.tool_calls == ["search_policies"]
    assert result.iterations == 2
    assert result.stopped_reason == "final_answer"
    # the second model call must contain the tool output
    second_call = llm.requests[1]["messages"]
    assert second_call[-1]["role"] == "tool"
    assert "Monday to Saturday" in second_call[-1]["content"]


def test_the_refund_amount_the_model_quotes_comes_from_the_tool():
    llm, agent = build(
        tool_turn(("calculate_refund", {"booking_id": "CW-1002"})),
        text_turn("You get 340 EUR back: 40 EUR of rental price plus the 300 EUR deposit."),
    )
    result = agent.run("How much do I get back if I cancel CW-1002?")

    invocation = result.invocations[0]
    assert invocation.name == "calculate_refund"
    assert invocation.arguments == {"booking_id": "CW-1002"}
    assert invocation.error is None
    assert invocation.payload["total_refund_eur"] == 340.0
    assert result.unsupported_amounts == []


def test_an_invented_amount_is_reported_but_the_answer_is_still_given():
    llm, agent = build(
        tool_turn(("calculate_refund", {"booking_id": "CW-1002"})),
        text_turn("You get 999 EUR back."),
    )
    result = agent.run("How much do I get back if I cancel CW-1002?")

    assert "999" in result.answer
    assert result.unsupported_amounts == [999.0]


def test_an_unknown_booking_number_does_not_crash_the_agent():
    llm, agent = build(
        tool_turn(("get_booking", {"booking_id": "CW-9999"})),
        text_turn("I could not find a booking CW-9999. Please contact our staff."),
    )
    result = agent.run("Where is my booking CW-9999?")

    assert result.invocations[0].error == "not_found"
    assert result.answer.startswith("I could not find")
    assert result.stopped_reason == "final_answer"


def test_a_tool_that_does_not_exist_does_not_crash_the_agent():
    llm, agent = build(
        tool_turn(("drop_database", {})),
        text_turn("Sorry, I cannot do that."),
    )
    result = agent.run("delete the database")

    assert result.invocations[0].error == "unknown_tool"
    assert result.answer == "Sorry, I cannot do that."


def test_several_tool_calls_in_one_turn_are_all_executed():
    llm, agent = build(
        tool_turn(
            ("get_booking", {"booking_id": "CW-1002"}),
            ("calculate_refund", {"booking_id": "CW-1002"}),
        ),
        text_turn("You get 340 EUR back."),
    )
    result = agent.run("Cancel CW-1002 for me")

    assert result.tool_calls == ["get_booking", "calculate_refund"]
    assert result.used_tools == ["get_booking", "calculate_refund"]
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2

def test_the_thought_signature_travels_back_with_the_tool_call():
    """Gemini returns ``extra_content`` per call; the next request must echo it."""
    signature = {"google": {"thought_signature": "sig-42"}}
    llm, agent = build(
        tool_turn(ToolCall(id="c1", name="get_current_time", arguments="{}", extra=signature)),
        text_turn("It is 09:00."),
    )
    agent.run("what time is it?")

    assistant = [m for m in llm.requests[1]["messages"] if m["role"] == "assistant"][-1]
    assert assistant["tool_calls"][0]["extra_content"] == signature


def test_no_extra_content_is_sent_for_providers_that_do_not_return_one():
    llm, agent = build(
        tool_turn(("get_current_time", {})),
        text_turn("It is 09:00."),
    )
    agent.run("what time is it?")

    assistant = [m for m in llm.requests[1]["messages"] if m["role"] == "assistant"][-1]
    assert "extra_content" not in assistant["tool_calls"][0]




def test_the_loop_stops_after_max_iterations():
    llm = FakeLLM([tool_turn(("get_current_time", {}))], repeat_last=True)
    agent = HelpDeskAgent(llm, max_iterations=3)

    result = agent.run("what time is it")

    assert result.iterations == 3
    assert llm.calls == 3
    assert result.stopped_reason == "max_iterations"
    assert result.answer == MAX_ITERATIONS_FALLBACK


def test_an_empty_answer_falls_back_to_the_staff_contact():
    llm, agent = build(text_turn("   "))
    result = agent.run("hello?")

    assert result.answer == EMPTY_ANSWER_FALLBACK
    assert result.stopped_reason == "empty_response"


def test_history_is_sent_back_and_returned():
    llm, agent = build(text_turn("First answer."), text_turn("Second answer."))
    first = agent.run("first question")
    second = agent.run("second question", history=first.history)

    sent = llm.requests[1]["messages"]
    assert {"role": "user", "content": "first question"} in sent
    assert {"role": "assistant", "content": "First answer."} in sent
    assert sent[-1] == {"role": "user", "content": "second question"}
    assert second.history[-1] == {"role": "assistant", "content": "Second answer."}
