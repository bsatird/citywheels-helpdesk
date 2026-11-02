"""Run the agent over ``eval/questions.json`` and write ``eval/results.md``.

    python eval/run_eval.py                 # grade with the configured provider
    python eval/run_eval.py --fake          # plumbing check, no network, no key
    python eval/run_eval.py --only Q01,Q04  # just a few cases
    python eval/run_eval.py --resume        # keep going after an interrupted run

Each question lists what a correct answer must contain: the tools that have to be
used, the amounts that have to appear (computed by the tools, never by the model),
phrases that must or must not show up, and whether the agent has to refuse. The
score is the number of fully correct answers, plus how often the model quoted a
euro amount that no tool produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # running the file directly, without an install
    sys.path.insert(0, str(ROOT))

from citywheels.agent import AgentResult, HelpDeskAgent  # noqa: E402
from citywheels.config import resolve_now  # noqa: E402
from citywheels.llm import LLMError, LLMSettings, build_client  # noqa: E402
from citywheels.testing import FakeLLM, text_turn  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.md"
CACHE_DIR = ROOT / "eval" / ".cache"

CHECK_LABELS = {
    "tools": "tools",
    "amounts": "amounts",
    "phrases": "phrases",
    "forbidden": "no-invention",
    "refusal": "refusal",
    "grounded": "grounded",
}

#: An out-of-scope question counts as declined when the answer says any of these.
REFUSAL_MARKERS = (
    "cannot",
    "can not",
    "can't",
    "unable",
    "not able",
    "sorry",
    "contact",
    "outside",
    "not related",
    "not covered",
    "not something",
    "not in the citywheels",
    "only answer",
    "only help",
    "don't have",
    "do not have",
    "no information",
    "don't know",
    "do not know",
    "don't rent",
    "do not rent",
    "don't offer",
    "do not offer",
)

_NUMBER_RE = re.compile(r"\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")

#: Words that turn a forbidden phrase into a statement of its opposite ("not cash").
NEGATION_CUES = (
    "no",
    "not",
    "never",
    "cannot",
    "can not",
    "can't",
    "don't",
    "doesn't",
    "isn't",
    "won't",
    "without",
    "rather than",
    "instead of",
    "other than",
)

#: How far back to look for a cue ("it comes back as a voucher, not as cash").
NEGATION_WINDOW = 40


def _is_negated(window: str) -> bool:
    words = re.sub(r"[^a-z' ]+", " ", window)
    padded = f" {words} "
    return any(f" {cue} " in padded for cue in NEGATION_CUES)


def mentions_forbidden(answer: str, phrase: str) -> bool:
    """True when ``phrase`` appears in ``answer`` as a claim rather than a negation.

    Substring matching alone cannot tell "it comes back as a voucher, not as cash"
    from "you get the money in cash": both contain "cash". Only an occurrence with
    no negation cue in front of it counts, and that is what the ``no-invention``
    check uses.
    """
    lowered = (answer or "").lower()
    needle = (phrase or "").lower()
    if not needle:
        return False
    start = lowered.find(needle)
    while start != -1:
        window = lowered[max(0, start - NEGATION_WINDOW) : start]
        if not _is_negated(window):
            return True
        start = lowered.find(needle, start + 1)
    return False


def numbers_in(text: str) -> list[float]:
    """Every number written in ``text`` (``340 EUR``, ``1 234,50``, ``08:00`` -> 8, 0)."""
    values: list[float] = []
    for raw in _NUMBER_RE.findall(text or ""):
        cleaned = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            values.append(float(cleaned))
        except ValueError:  # pragma: no cover - the regex only matches numbers
            continue
    return values


def has_amount(numbers: list[float], expected: float, tolerance: float = 0.011) -> bool:
    return any(abs(value - expected) <= tolerance for value in numbers)


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str = ""
    tools: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    seconds: float = 0.0
    unsupported: list[float] = field(default_factory=list)
    error: str | None = None
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.checks.values())


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score_case(case: dict, result: AgentResult) -> dict[str, bool]:
    """Which requirements of one question were met."""
    answer = result.answer or ""
    lowered = answer.lower()
    numbers = numbers_in(answer)

    checks = {
        "tools": all(tool in result.tool_calls for tool in case.get("expect_tools", [])),
        "amounts": all(
            has_amount(numbers, amount) for amount in case.get("expect_amounts", [])
        ),
        "phrases": all(
            any(alternative.lower() in lowered for alternative in group)
            for group in case.get("expect_phrases", [])
        ),
        "forbidden": not any(
            mentions_forbidden(answer, phrase) for phrase in case.get("forbid_phrases", [])
        ),
        "grounded": not result.unsupported_amounts,
    }
    if case.get("expect_refusal"):
        checks["refusal"] = is_refusal(answer)
    return checks

def run_cases(
    cases: list[dict],
    agent_factory,
    *,
    quiet: bool = False,
    on_result=None,
) -> list[CaseResult]:
    """Run every case. ``on_result(case, result)`` is called once per case, as soon as
    it is graded, so a caller can persist progress before the next case starts."""
    results: list[CaseResult] = []
    for case in cases:
        agent = agent_factory(case)
        started = time.perf_counter()
        try:
            agent_result = agent.run(case["question"])
        except LLMError as exc:
            failed = CaseResult(
                case_id=case["id"],
                question=case["question"],
                error=str(exc),
                note=case.get("note", ""),
            )
            results.append(failed)
            if on_result is not None:
                on_result(case, failed)
            if not quiet:
                print(f"{case['id']}: ERROR {exc}", file=sys.stderr, flush=True)
            continue
        elapsed = time.perf_counter() - started
        result = CaseResult(
            case_id=case["id"],
            question=case["question"],
            answer=agent_result.answer,
            tools=agent_result.used_tools,
            checks=score_case(case, agent_result),
            seconds=elapsed,
            unsupported=agent_result.unsupported_amounts,
            note=case.get("note", ""),
        )
        results.append(result)
        if on_result is not None:
            on_result(case, result)
        if not quiet:
            met = sum(1 for value in result.checks.values() if value)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"{case['id']}: {status} {met}/{len(result.checks)} "
                f"({elapsed:.1f}s, tools: {', '.join(result.tools) or 'none'})",
                file=sys.stderr,
                flush=True,
            )
    return results


# --------------------------------------------------------------------------- #
# progress cache: one line per graded case, so a killed run loses nothing
# --------------------------------------------------------------------------- #


def case_fingerprint(case: dict) -> str:
    """Short hash of a question, used to invalidate a cached grade when it changes."""
    payload = json.dumps(case, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def default_cache_path(provider: str, model: str) -> Path:
    """One cache per provider+model, next to the report (``eval/.cache/``, ignored by git)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{provider}-{model}").strip("-")
    return CACHE_DIR / f"{slug or 'model'}.jsonl"


def read_cache(path: Path) -> dict[str, dict]:
    """``{case id: row}`` from a cache file; the last row for an id wins.

    A killed run can leave a half-written last line, so unreadable lines are skipped
    instead of crashing - the point of the file is to survive exactly that.
    """
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id"):
            rows[str(row["id"])] = row
    return rows


def row_from(case: dict, result: CaseResult) -> dict:
    return {
        "id": result.case_id,
        "fingerprint": case_fingerprint(case),
        "question": result.question,
        "answer": result.answer,
        "tools": result.tools,
        "checks": result.checks,
        "seconds": round(result.seconds, 2),
        "unsupported": result.unsupported,
        "error": result.error,
        "note": result.note,
    }


def result_from(row: dict) -> CaseResult:
    return CaseResult(
        case_id=str(row["id"]),
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        tools=list(row.get("tools") or []),
        checks=dict(row.get("checks") or {}),
        seconds=float(row.get("seconds") or 0.0),
        unsupported=list(row.get("unsupported") or []),
        error=row.get("error"),
        note=row.get("note", ""),
    )


def store_result(path: Path, case: dict, result: CaseResult) -> None:
    """Append one graded case and flush it to disk before the next request starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row_from(case, result), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reusable(case: dict, rows: dict[str, dict]) -> CaseResult | None:
    """The cached grade of ``case``, if it can be reused.

    A row is reused only when the question still hashes the same (otherwise the
    expectation changed and the old answer proves nothing) and when it is not an
    error: a rate-limited case was never graded, so it has to be asked again.
    """
    row = rows.get(case["id"])
    if not row or row.get("error"):
        return None
    if row.get("fingerprint") != case_fingerprint(case):
        return None
    return result_from(row)


def _mark(result: CaseResult, key: str) -> str:
    if result.error:
        return "-"
    value = result.checks.get(key)
    if value is None:
        return "."
    return "yes" if value else "**NO**"


def _one_line(text: str, limit: int = 220) -> str:
    flat = " ".join((text or "").split()).replace("|", "/")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def summarize(results: list[CaseResult], cases: list[dict]) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    scored = [result for result in results if result.error is None]
    tool_cases = [result for result in scored if by_id[result.case_id].get("expect_tools")]
    return {
        "cases": len(results),
        "passed": sum(1 for result in results if result.passed),
        "errors": sum(1 for result in results if result.error),
        "tool_cases": len(tool_cases),
        "tool_cases_ok": sum(1 for result in tool_cases if result.checks.get("tools")),
        "unsupported_amounts": sum(len(result.unsupported) for result in scored),
        "seconds": round(sum(result.seconds for result in scored), 1),
    }

def render_markdown(
    results: list[CaseResult],
    cases: list[dict],
    *,
    provider: str,
    model: str,
    fake: bool,
    reused: int = 0,
) -> str:
    stats = summarize(results, cases)
    rate = 100.0 * stats["passed"] / stats["cases"] if stats["cases"] else 0.0
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = ["# Evaluation results", ""]
    if fake:
        lines.append(
            "> **Plumbing check only.** These numbers come from a fake model "
            "(`--fake`), they say nothing about the real model."
        )
        lines.append("")
    lines.append(f"- Model: `{provider}` / `{model}`")
    lines.append(f"- Shop's fixed \"now\": `{resolve_now().isoformat(timespec='seconds')}`")
    lines.append(f"- Run at {stamp}, generated by `python eval/run_eval.py`")
    if reused:
        lines.append(
            f"- {reused} of these answers were reused from an earlier run of the same "
            "question (`--resume`); a cached answer is kept only while its question is "
            "unchanged"
        )
    lines.append(
        f"- **Score: {stats['passed']}/{stats['cases']} correct ({rate:.0f}%)**, "
        f"{stats['tool_cases_ok']}/{stats['tool_cases']} used the required tools, "
        f"{stats['unsupported_amounts']} unsupported euro amounts, "
        f"{stats['seconds']}s total"
    )
    lines.append("")
    lines.append(
        "| ID | tools | amounts | phrases | no-invention | refusal | grounded "
        "| result | answer |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        cells = [_mark(result, key) for key in CHECK_LABELS]
        outcome = "ERROR" if result.error else ("PASS" if result.passed else "FAIL")
        lines.append(
            f"| {result.case_id} | "
            + " | ".join(cells)
            + f" | {outcome} | {_one_line(result.answer)} |"
        )
    lines.append("")

    failures = [result for result in results if not result.passed]
    if failures:
        lines.append("## Failures in detail")
        lines.append("")
        for result in failures:
            lines.append(f"### {result.case_id}")
            lines.append("")
            lines.append(f"**Question:** {result.question}")
            if result.error:
                lines.append(f"**Error:** {result.error}")
                lines.append("")
                continue
            lines.append(f"**Tools used:** {', '.join(result.tools) or 'none'}")
            failed = [CHECK_LABELS[key] for key, ok in result.checks.items() if not ok]
            lines.append(f"**Failed checks:** {', '.join(failed)}")
            lines.append(f"**A correct answer needs:** {result.note}")
            if result.unsupported:
                amounts = ", ".join(f"{amount:g}" for amount in result.unsupported)
                lines.append(f"**Guardrail:** unsupported amounts: {amounts}")
            lines.append("")
            lines.append(f"> {_one_line(result.answer, 600)}")
            lines.append("")

    lines.append("## How the score is computed")
    lines.append("")
    lines.append(
        "- `tools`: every tool in `expect_tools` was actually called.\n"
        "- `amounts`: every expected euro amount appears in the answer (1 cent tolerance).\n"
        "- `phrases`: every group of alternatives has at least one match (case-insensitive).\n"
        "- `no-invention`: none of the `forbid_phrases` appears as a claim (a negated\n"
        "  mention such as \"as a voucher, not as cash\" does not count).\n"
        "- `refusal`: out-of-scope questions are declined instead of answered.\n"
        "- `grounded`: every euro amount in the answer appears in some tool result.\n"
        "- A case passes only when all of its checks pass."
    )
    lines.append("")
    return "\n".join(lines)

def load_cases(path: Path | str = QUESTIONS_PATH) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{path} must contain a non-empty JSON list of questions")
    return cases


def build_agent_factory(*, fake: bool, max_iterations: int):
    """One agent per case (the fake model needs a fresh script each time)."""
    if fake:

        def factory(_case):
            return HelpDeskAgent(
                FakeLLM([text_turn("(fake model) I cannot answer that.")]),
                max_iterations=max_iterations,
            )

        return factory, ("fake", "fake-model")

    settings = LLMSettings.from_env()
    client = build_client()
    agent = HelpDeskAgent(client, max_iterations=max_iterations)
    return (lambda _case: agent), (settings.provider, settings.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Grade the CityWheels agent on eval/questions.json.",
    )
    parser.add_argument("--only", help="comma-separated case ids, e.g. Q01,Q04")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="report path")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use a fake model to check the harness itself (no network, no key)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON, write nothing")
    parser.add_argument("--dry-run", action="store_true", help="do not write the report")
    parser.add_argument("--quiet", action="store_true", help="no per-case progress lines")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse graded cases from the cache and save each new one as it finishes, "
        "so an interrupted run only repeats what it never graded",
    )
    parser.add_argument(
        "--cache",
        help="cache file for --resume (default: eval/.cache/<provider>-<model>.jsonl)",
    )
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument(
        "--min-interval",
        type=float,
        metavar="SECONDS",
        help="minimum gap between two model calls (free tiers allow ~5 per minute)",
    )
    args = parser.parse_args(argv)

    if args.min_interval is not None:
        os.environ["LLM_MIN_INTERVAL"] = str(args.min_interval)

    cases = load_cases()
    if args.only:
        wanted = {part.strip().upper() for part in args.only.split(",") if part.strip()}
        cases = [case for case in cases if case["id"].upper() in wanted]
        if not cases:
            parser.error("no case matches --only")

    try:
        factory, (provider, model) = build_agent_factory(
            fake=args.fake, max_iterations=max(1, args.max_iterations)
        )
    except LLMError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    cache_path = Path(args.cache) if args.cache else default_cache_path(provider, model)
    reused: dict[str, CaseResult] = {}
    if args.resume:
        rows = read_cache(cache_path)
        for case in cases:
            previous = reusable(case, rows)
            if previous is not None:
                reused[case["id"]] = previous
        if reused and not args.quiet:
            print(
                f"{len(reused)} case(s) reused from {cache_path}",
                file=sys.stderr,
                flush=True,
            )

    pending = [case for case in cases if case["id"] not in reused]
    on_result = (
        (lambda case, result: store_result(cache_path, case, result))
        if args.resume
        else None
    )
    fresh = run_cases(pending, factory, quiet=args.quiet, on_result=on_result)
    graded = {**reused, **{result.case_id: result for result in fresh}}
    results = [graded[case["id"]] for case in cases]
    stats = summarize(results, cases)

    if args.json:
        payload = {
            "provider": provider,
            "model": model,
            "fake": args.fake,
            "stats": stats,
            "results": [
                {
                    "id": result.case_id,
                    "passed": result.passed,
                    "checks": result.checks,
                    "tools": result.tools,
                    "unsupported_amounts": result.unsupported,
                    "answer": result.answer,
                    "error": result.error,
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1 if stats["errors"] else 0

    report = render_markdown(
        results, cases, provider=provider, model=model, fake=args.fake, reused=len(reused)
    )
    rate = 100.0 * stats["passed"] / stats["cases"] if stats["cases"] else 0.0
    if not args.dry_run:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report + "\n", encoding="utf-8")
        print(f"Report written to {destination}")
    print(
        f"{stats['passed']}/{stats['cases']} correct ({rate:.0f}%), "
        f"{stats['unsupported_amounts']} unsupported euro amounts, "
        f"{stats['errors']} error(s)"
    )
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
