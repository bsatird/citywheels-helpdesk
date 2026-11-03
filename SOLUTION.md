# CityWheels help-desk agent — how it was built

This is a CLI agent that answers customer questions for a small bike rental shop.
The model never states a fact it did not read from a tool result, and it never does
arithmetic: prices, deposits and refunds are computed in Python with `Decimal`.

* **Provider:** Google Gemini free tier (`gemini-3.5-flash` by default, the lite
  model for the first evaluation run; both are compared in §5) through the
  OpenAI-compatible endpoint. No framework, ~120 lines of `requests` in
  `citywheels/llm.py`.
* **Tests:** `python -m pytest` -> 247 tests, 226 passing and 21 skipped, no network
  and no key needed for any of them.
* **Evaluation:** `python eval/run_eval.py` -> `eval/results.md`, 15 questions,
  15/15 correct with 0 unsupported euro amounts; a second run on the default model
  is committed as `eval/results-gemini-3.5-flash.md`.

## 1. How to install and run it

```bash
git clone https://github.com/bsatird/citywheels-helpdesk && cd citywheels-helpdesk
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                              # then put your key into GEMINI_API_KEY
python -m citywheels "How much do I get back if I cancel CW-1002 today?"
```

`.env` is the only place a key lives; it is in `.gitignore` and the repository
contains just `.env.example` with empty values. `GEMINI_API_KEY` is the variable the
Gemini preset reads; `LLM_API_KEY` overrides it (and is the one to use for the other
providers).

```bash
python -m citywheels -i                                     # interactive, keeps memory
python -m citywheels -v "Can my 14-year-old rent an e-bike?" # prints every tool call
python -m citywheels --max-iterations 10 "..."               # more model turns

python -m pytest                       # the test suite
python eval/run_eval.py                # the evaluation, writes eval/results.md
python eval/run_eval.py --fake         # harness only, no key needed
```

`CITYWHEELS_NOW` pins the shop's clock (default `2026-06-10T09:00:00+02:00`), which
is what makes "cancel today" testable: every refund rule is relative to that instant.


## 2. Model and provider

Gemini, free tier: `gemini-3.5-flash` by default and `gemini-flash-lite-latest`
for the evaluation committed in `eval/results.md`. Reasons:

* It is genuinely free, so the task can be run by anyone reviewing it.
* It supports the `tools` / `tool_calls` protocol, so the same request shape works
  for Gemini, OpenAI, GitHub Models and a local Ollama - `LLM_PROVIDER` swaps the
  preset, `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` override anything.
* Tool calling is reliable enough that the model picks the right tool without
  examples in the prompt, which keeps the system prompt short.

Why two models: the free tier of this project allows 5 requests per minute and 20
per day on `gemini-2.5-flash` (both numbers come from the `429` body, not from
guessing), and one answer costs 2-3 requests, so 15 questions do not fit in a day.
The lite model has its own, much larger bucket; the behaviour is identical, so the
report names the model it used.

I wrote the HTTP client instead of using LangChain/LangGraph because the loop is
about 40 lines: send messages + tool schemas, run the requested tools, append the
results, repeat. A framework would hide the one part of the system the task is
actually about, and every provider I need speaks `/chat/completions` anyway.

Three provider details worth knowing:

* Free Gemini keys are rate limited per model and per minute, so a burst of
  questions hits `429`. The client retries with exponential backoff and honours the
  delay the provider asks for. Gemini does not send a `Retry-After` header - the
  delay is inside the JSON error body (`"retryDelay": "23s"`), so the client looks
  there too. `LLM_MIN_INTERVAL` (or `eval/run_eval.py --min-interval`) spaces the
  calls out instead of retrying.
* Gemini 3.x signs every function call with an opaque `thought_signature` inside
  `extra_content` and rejects the next request with `HTTP 400` if the assistant
  message does not carry it back. `ToolCall.extra` keeps it, `agent.py` echoes it,
  and the key is only sent when the provider actually returned one - OpenAI and
  Ollama would reject the unknown field. This was found by running the real API,
  not by reading the docs: the first eval run after switching to a 3.x model failed
  on 15/15 cases with "Function call is missing a thought_signature".
* Responses can carry a `null` content next to the tool calls; `ChatResult`
  normalises that to an empty string so the loop does not crash.


## 3. How the agent works

One question goes through a loop of at most six model turns
(`citywheels/agent.py`):

1. The system prompt (`citywheels/prompts.py`) plus the four tool schemas are sent to
   the model.
2. If the reply has `tool_calls`, each one is executed by
   `citywheels/tools/registry.py`, the JSON result is appended as a `tool` message,
   and the loop asks the model again.
3. If the reply has no tool calls, its text is the answer. An empty reply and a
   burnt-out budget both fall back to "please contact the staff", so the customer
   never sees a stack trace or a blank screen.

The tools are deliberately few and each one owns one kind of fact:

| Tool | Answers | Where the truth lives |
| --- | --- | --- |
| `search_policies` | prices, deposits, opening hours, age limits, payment, late return, bad weather | BM25 over `data/policies/*.md`, one hit per `##` section, with the section reference in the result |
| `get_booking` | one booking: dates, bike, status, price, deposit | `data/bookings.json`, `found: false` for unknown numbers |
| `calculate_refund` | what a cancellation returns, right now or in N hours | the rules in `data/policies/cancellations.md`, computed with `Decimal` |
| `get_current_time` | "is my booking tomorrow?" | `CITYWHEELS_NOW` |

Three decisions do most of the work against hallucinations:

* **Money is never computed by the model.** `calculate_refund` returns
  `rental_refund_eur`, `deposit_refund_eur`, `total_refund_eur` and a ready
  `explanation`, and the prompt tells the model to copy those numbers verbatim. It
  also routes every money question through the calculator - including "can I still
  get something back?" and the cases where the answer must be zero - so the tool,
  not the model, decides that nothing is refunded. The 24 h / 48 h boundaries, the
  never-refunded booking fee, the always-refunded deposit and the
  voucher-instead-of-cash rule are unit-tested in Python (`tests/test_refunds.py`),
  not trusted to a language model. When a fee would have to be worked out for a
  detail the customer has not given yet (which bike was late), the prompt asks the
  model to state the rule and ask instead of doing the multiplication itself.
* **Every failure is a tool result, not an exception.** Unknown tool, unparsable
  arguments, unexpected error, `found: false` - all come back as JSON with an
  `error`/`message` key that the model is told to report. A bad booking number
  cannot crash the CLI.
* **The answer is checked after the fact.** `citywheels/guardrails.py` extracts
  every euro amount from the answer and compares it with every number that came
  back from a tool - JSON numbers and amounts written in policy prose alike. Any
  amount that no tool can justify lands in `AgentResult.unsupported_amounts`,
  prints as a warning with `-v`, and counts as a failed `grounded` check in the
  evaluation. It warns instead of blocking on purpose: rewriting a customer's
  answer behind their back felt worse than showing the operator that the model
  drifted.

The CLI (`citywheels/cli.py`) has three modes: one question, `-i` for a REPL that
keeps the transcript (the answer is appended to the history, so "and if I cancel
tomorrow instead?" has context), and `-v` to print every tool call with its result
and the guardrail warning. `--json` gives the same result as a JSON object.

Tests run without a key or a network: `citywheels/testing.py` contains a scripted
`FakeLLM` that returns prepared turns, so the loop, the registry, the refund maths,
the retrieval, the guardrail and the evaluation harness are all covered by ordinary
pytest cases, and the real provider is only used by the evaluation run.


## 4. How I used AI coding assistants

Almost all of the code in this repository was written with an AI assistant
(Cline, running Claude/GPT-class models) in VS Code, in small steps: I described a
piece of behaviour, reviewed the diff, ran the tests, and kept or reverted the
result. What I did myself is the part that decides whether the thing is correct:
the tool contracts (what each tool returns), the refund rules read from
`data/policies/cancellations.md`, the eval questions with their expected answers,
and every judgement about whether an answer was acceptable.

The assistant was genuinely good at boilerplate I did not want to type (the BM25
scoring, the CLI argument parsing, the Markdown report) and bad at two things:
assuming its code worked without running it, and inventing facts in prose. Five
concrete cases where the output was wrong:

**1. The final answer was never stored in the transcript.** The first version of
`HelpDeskAgent.run` appended the tool results to `messages` but returned the final
answer without appending it, so in `-i` mode the follow-up "and what if I cancel
tomorrow instead?" reached the model without the previous answer. The code looked
right and every single-question example worked. I noticed only when I wrote a test
that asserted the returned `history` ends with the assistant turn - it failed. The
fix is three lines in `agent.py`, but the lesson is the test, not the fix: a manual
check would never have caught it.

**2. `FakeLLM.calls` did not count repeated calls.** The scripted fake model
recorded tool calls in a list that was deduplicated, so a test that expected the
model to ask for the same tool twice (a retry after `found: false`) passed while
the real agent would have been wrong. This is the failure mode of a fake model in
general: it can make the tests agree with a bug. Fixed by recording every call and
adding a test that asserts the count, not just the set.

**3. A confidently invented number in the documentation.** The README said the
suite was "190 tests". The real number was 217. Nothing in the assistant's output
looked suspicious - it was prose, and prose is where hallucinations live. I now run
the documented command and compare its output with the doc before committing.

**4. Assuming the standard `Retry-After` header.** The retry logic read
`response.headers["Retry-After"]` and the unit tests passed, because they used a
fake response carrying that header. The first real evaluation run instead produced
`429` after `429`: Gemini sends no `Retry-After` at all and puts the delay in the
JSON body (`"retryDelay": "23s"`, "Please retry in 23.6s"). The fake test had
encoded my own wrong assumption, which is exactly what a fake tends to do. Fixed by
parsing the body too, and by rewriting the test with the payload captured from the
provider.

**5. A duplicated test block after a bad edit.** A multi-part edit of
`tests/test_eval.py` was applied twice, leaving a copy of the same functions in the
file. The assistant reported the edit as successful. Running the suite is what
caught it, not reading the diff.

What I changed about my own workflow because of the above: every claim in the docs
is a command I ran, every test that mattered was written after seeing the behaviour
fail once, and any number in the evaluation results comes from a real
`eval/results.md` rather than from an estimate.


## 5. Evaluation results

`python eval/run_eval.py` writes `eval/results.md`; the committed report is the run
below (15 questions, shop clock pinned to `2026-06-10T09:00:00+02:00`,
`gemini-flash-lite-latest`, 168.6 s, run with `--min-interval 6` to stay inside the
free-tier request limit - any `429` is retried with the delay the API asks for):

> **Score: 15/15 correct (100%)**, 13/13 used the required tools, **0 unsupported
> euro amounts**.

That report is `gemini-flash-lite-latest`, chosen for quota reasons, while the code
ships with `gemini-3.5-flash`. The same 15 questions were later run on the default
model and committed next to it as `eval/results-gemini-3.5-flash.md`; *Two models,
one harness* at the end of this section has the comparison.

What that says in practice:

* Every refund answer quoted the amounts `calculate_refund` returned, to the cent,
  across the cases the rules distinguish: the 50 % bracket (Q01), a cancelled booking
  with a deposit (Q02), "less than 24 h, so no rental price back" (Q03), a voucher
  paid booking that comes back as a voucher (Q04), an already cancelled booking
  (Q15) and one that is running now (Q05). The "nothing is refunded" answers call the
  calculator too and quote the zero instead of working it out themselves.
* No answer contained a euro amount that no tool produced - the guardrail stayed
  empty for all 15 questions, which is the property I care about most.
* The out-of-scope question (Q13), the unknown booking (Q06) and "do you rent
  tandems or scooters?" (Q14) were all declined and handed to the staff contact
  instead of being answered from the model's own knowledge.
* Retrieval picked the right section for prices, opening hours, age limits, the
  deposit, late return and bad weather.

### The road to 15/15: three runs, only one of them about code

| Run | Score | What was really wrong |
| --- | --- | --- |
| 1 | 11/15, 7 unsupported amounts | the guardrail read only JSON, so amounts copied from policy prose looked invented; the refund tool's own text said "never as cash", which Q04 forbids |
| 2 | 13/15 | two prompt gaps (Q05, Q14); Q11 was graded against a total the model refuses to compute |
| 3 | 15/15, 0 unsupported amounts | the rules and the negation-aware check below |

The second run's failures are worth more than the score, because each one was a
decision about *where* to fix something:

**Q05 - "CW-1008 is running right now. Can I cancel it and get money back?"** The
answer was correct ("status `in_progress` ... cannot be cancelled, nothing is
refunded") but the model reached it from `get_booking` + `search_policies` and never
called `calculate_refund`. The prompt said "any refund amount: call
`calculate_refund`", and this question asks about money without naming an amount, so
the rule did not fire. That is a prompt gap, not a tool gap: the rule now covers any
question about money coming back, "even when you expect the answer to be zero".

**Q14 - "Do you also rent tandems or scooters?"** The model answered "tandems and
scooters are not listed in our fleet" from the price list but never escalated, so the
`refusal` check failed. The information was right, the hand-off was missing. The
refusal section now says that when we do not offer something, say so *and* give the
staff contact.

**Q11 - "I returned my bike three hours late. What does that cost?"** is the one case
where the expectation was wrong, not the answer. The prompt says money is never
computed by the model, so the honest answer states the "1.5x hourly price" rule and
asks which bike was late - it cannot know the hourly price without knowing the bike.
The case now accepts that ("depends", "which bike"), and a second prompt rule tells
the model to state the rule and ask rather than do the multiplication.

### The first run: two harness bugs, not model failures

The first full run of this evaluation scored **11/15 with 7 unsupported amounts**, and
the difference was in the measuring equipment:

* Q04 failed on every attempt because `calculate_refund` itself said the voucher
  refund is "never as cash", and "cash" is the phrase the case forbids. The tool
  text now says "not as money on a card", and a new test
  (`test_forbidden_phrases_are_not_contradicted_by_the_refund_tool`) checks that no
  case forbids a word its own tool output contains. The evaluator also stopped
  treating a negated mention as a claim (`mentions_forbidden`), because "comes back
  as a voucher, not as cash" is the *correct* answer to Q04 and substring matching
  failed it.
* Q08 ("40 EUR for an e-bike per day") and Q10 ("150 EUR deposit") were flagged as
  ungrounded although both amounts were in the policy text: the guardrail only read
  numbers out of JSON, and `search_policies` returns prose. It now reads euro
  amounts from prose too, with tests for both directions.
* Q11 (late return) had the same cause: the answer quoted the "1.5x hourly price"
  rule, and both the multiplier and the hourly prices are prose, not JSON fields.
  Seven amounts were flagged in that run; the ones I checked by hand (the 40 EUR
  daily price and the 150 EUR deposit) were figures the model had copied correctly
  out of a tool's prose, which is why the fix was in the guardrail, not the prompt.

Earlier still, an evaluation run on a Gemini 3.x model failed on **15/15**
questions with `HTTP 400 ... Function call is missing a thought_signature`; that is
what the `ToolCall.extra` / `extra_content` round-trip in `agent.py` exists for.

The honest reading of 15/15: the check that would catch a real bug is "0 unsupported
euro amounts", and it held on all 15 answers in every run once the guardrail could
read prose. The score itself moved for reasons I can name - two harness bugs, two
prompt rules, one expectation I had written too strictly - which is the useful part:
a score that only goes up because the model got better would tell me nothing about
the system around it. One run of one model is not a guarantee, and the failures that
remain possible are exactly the ones this harness cannot see (tone, a subtly wrong
paraphrase that keeps the right numbers).

### Limits of this measurement

One run, fifteen questions, one model, one temperature setting. A single run
cannot separate model flakiness from a regression, and 15 questions cover the
tool-selection and fact-copying behaviour, not tone. Everything the score is based
on is deterministic string and number matching (`eval/run_eval.py`), so the checks
are inspectable and cannot be talked into passing - but they also cannot judge
whether an answer *reads* well.

The harness is also built for an unreliable terminal and a small free quota: every
graded case is appended to `eval/.cache/` as soon as it is scored (`--resume`), so an
interrupted run continues where it stopped instead of paying for the same questions
twice. The committed report is a complete run - no answer in it is reused - but the
next run of the same unchanged questions would cost zero requests, which is why the
report says how many answers came from the cache when that happens.


### Two models, one harness

The same harness, the same 15 questions and the same fixed clock, run against the
second model:

| | `gemini-flash-lite-latest` | `gemini-3.5-flash` (the default) |
| --- | --- | --- |
| Score | 15/15 | 15/15 |
| Unsupported euro amounts | 0 | 0 |
| Sum of the 15 per-question timings | 168.6 s | 480.1 s |
| Committed report | `eval/results.md` | `eval/results-gemini-3.5-flash.md` |

The timing column is not the two models racing: `seconds` in a report is the sum of
the per-question timings, and on the bigger model that sum is dominated by the waits
for `429` retries. The free tier allows about 20 requests per day **per Google
project** - two API keys inside one project share that budget, which is why the run
went through several sessions with `--resume` and `--min-interval 13`. A full run
costs 30-45 requests, so requests, not seconds, are the unit the quota measures. What
the table does establish is the property I wanted from a second model: two different
models agreeing on 15/15 and on zero unsupported amounts makes the score look like a
property of the harness, the prompt and the tools, not of one model's luck.

The second run also found a bug in the *test*. Q11 ("I returned my bike three hours
late") failed on `gemini-3.5-flash` with an answer that gave the 1.5x hourly rule,
said the fee comes out of the deposit and asked *"which type of bike?"*. Asking which
type of bike **is** asking which bike, but the case's `expect_phrases` listed only
"depends", "which bike", "what bike" and "bike type", so a correct answer scored as a
miss. The expectation was widened (`eval/questions.json`, with a dated note); the
model was not touched. That is the second time in this project where the measuring
equipment was the broken part - the first was the guardrail that could not read prose
- and it is why the 15/15 above is worth more than the number: the score moves when I
can name why.

What the comparison does not say: that the bigger model is better. The questions are
written to be answerable from the tools, both models cleared them, and the failure
modes this harness cannot see - tone, a fluent paraphrase that keeps the numbers -
stay invisible on both.

## 6. Limitations

* **Retrieval is lexical.** `search_policies` is BM25 over a handful of Markdown
  files with a hand-written synonym list. It is fast and explainable and it is
  enough for four documents; it would fail on a real shop's document pile (PDFs,
  tables, 300 pages) where embeddings and a reranker are needed.
* **The shop clock is fake.** `CITYWHEELS_NOW` is a single fixed instant, so "today"
  is the same for every user and there is no timezone or daylight-saving handling
  beyond the offset in the data.
* **Nothing is mutated.** `calculate_refund` computes what a cancellation would
  return; it does not cancel. A real cancellation needs a write path, an audit log
  and idempotency, and none of that is here.
* **The guardrail warns, it does not block.** If the model quotes an amount no tool
  produced, the answer is still returned (with `unsupported_amounts` filled in and a
  warning printed with `-v`). I preferred visibility over silently rewriting the
  model's answer; a stricter mode would re-ask with the tool output attached.
* **The evaluation is small.** 15 questions, one run, one model. It measures tool
  selection, key facts and refusals, not tone, and a single run cannot separate
  model flakiness from a real regression.
* **Free-tier throughput.** ~5 requests/minute and ~20 requests/day on the free
  Gemini keys means a full eval run takes minutes and a real shop would need a paid
  key (or a response cache) to serve a queue of customers.
* **The answers are English and the documents are English.** The prompt asks the
  model to answer in the customer's language, but nothing guarantees that a Polish
  or German answer is as accurate as the English one.
* **No streaming, no conversation persistence.** History lives in the process, so
  `-i` memory dies with the terminal.

## 7. What I would do next

1. **Make cancellation real**: `cancel_booking` writing to `data/bookings.json`
   (or a small SQLite file) with a status transition, an audit trail and an
   idempotency key per request, so the agent can do the thing it currently only
   prices.
2. **Blocking guardrail**: when `unsupported_amounts` is non-empty, send the answer
   back to the model with the actual tool payloads and ask for a corrected version;
   keep the warning path for a second failure.
3. **Retrieval that scales**: chunk by heading and paragraph, hybrid BM25 +
   embeddings, a reranker, and metadata filters (document, section, effective date)
   so that "your rules from July" can be answered.
4. **A bigger, honest evaluation**: 50-80 questions, three runs per question to
   measure stability, an LLM-as-judge for phrasing next to the deterministic
   checks, and recorded provider responses (VCR cassettes) so CI can run the real
   answers without a key or a rate limit.
5. **More rules in Python**: opening hours, late-return fees and deposit rules are
   currently read from text; a small calculator tool per rule would remove the last
   place where the model could do arithmetic in its head.
6. **Operational bits**: response cache for repeated questions, `--json` output for
   other tools, cost/latency tracing, and an HTTP wrapper reusing the same tools.
7. **Record the real answers.** The `--resume` cache already keeps graded answers
   per model; turning it into proper cassettes (question -> recorded response) would
   let CI replay a real run with no key and no rate limit, and would make the
   15-question score reproducible on any machine. It would also let the suite check
   the *answers* of the last real run, not just the scoring of hand-written strings.
