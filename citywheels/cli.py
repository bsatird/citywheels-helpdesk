"""``python -m citywheels "your question"`` -- the command-line front end.

Two ways to use it:

    python -m citywheels "How much will I get back if I cancel CW-1002?"
    python -m citywheels --interactive          # a small chat with memory

``--verbose`` prints every tool call the model made, which is the quickest way to
see what the agent is doing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .agent import AgentResult, HelpDeskAgent
from .config import SUPPORT_EMAIL, SUPPORT_PHONE, resolve_now
from .llm import LLMClient, LLMError, LLMSettings, build_client

BANNER = """CityWheels help desk agent
Type your question, or ':help' for commands. ':quit' to leave."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citywheels",
        description="Ask the CityWheels bike rental help desk a question.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The customer question. Omit it to use --interactive.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start a chat session that remembers earlier questions.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the tool calls the model made and the guardrail warnings.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum number of model turns per question (default: 6).",
    )
    return parser


def _print_trace(result: AgentResult, *, verbose: bool) -> None:
    if not verbose:
        return
    print(f"[trace] {result.iterations} model turn(s), stopped: {result.stopped_reason}")
    for invocation in result.invocations:
        arguments = ", ".join(f"{k}={v!r}" for k, v in invocation.arguments.items())
        marker = "!" if invocation.error else "->"
        print(f"[trace] {marker} {invocation.name}({arguments})")
        print(f"[trace]    {invocation.short()}")
    if result.unsupported_amounts:
        amounts = ", ".join(f"{amount:g}" for amount in result.unsupported_amounts)
        print(f"[trace] WARNING: amounts not backed by any tool result: {amounts}")


def ask(agent: HelpDeskAgent, question: str, *, verbose: bool) -> str:
    """Run one question, print the answer and return it."""
    result = agent.run(question)
    _print_trace(result, verbose=verbose)
    print(result.answer)
    return result.answer


def interactive(agent: HelpDeskAgent, *, verbose: bool) -> int:
    """A tiny REPL that keeps the conversation, so follow-up questions work."""
    print(BANNER)
    history: list[dict] = []
    show_trace = verbose
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        command = question.lower()
        if command in {":quit", ":q", "exit", "quit"}:
            return 0
        if command == ":help":
            print(":reset  forget the conversation   :now   show the shop's date")
            print(":trace  toggle tool tracing      :quit  leave")
            continue
        if command == ":reset":
            history = []
            print("Conversation cleared.")
            continue
        if command == ":now":
            print(resolve_now().isoformat(timespec="seconds"))
            continue
        if command == ":trace":
            show_trace = not show_trace
            print(f"Tool tracing {'on' if show_trace else 'off'}.")
            continue

        try:
            result = agent.run(question, history=history)
        except LLMError as exc:
            print(f"\nAgent: sorry, the model is not available right now ({exc}).")
            print(f"Please contact the staff: {SUPPORT_EMAIL} / {SUPPORT_PHONE}.")
            continue
        _print_trace(result, verbose=show_trace)
        print(f"\nAgent: {result.answer}")
        history = result.history


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()

    if not question and not args.interactive:
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        if not question:
            parser.print_help()
            return 2

    try:
        settings = LLMSettings.from_env()
    except LLMError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(
            f"[trace] provider={settings.provider} model={settings.model} "
            f"now={resolve_now().isoformat(timespec='seconds')}",
            file=sys.stderr,
        )

    try:
        client = build_client()
    except LLMError as exc:  # pragma: no cover - defensive
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    agent = HelpDeskAgent(client, max_iterations=max(1, args.max_iterations))

    try:
        if args.interactive or not question:
            return interactive(agent, verbose=args.verbose)
        ask(agent, question, verbose=args.verbose)
        return 0
    except LLMError as exc:
        print(f"\nModel error: {exc}", file=sys.stderr)
        print(f"Please contact the staff: {SUPPORT_EMAIL} / {SUPPORT_PHONE}.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
