# CityWheels help desk agent

A command-line AI assistant for the CityWheels bike rental shop. It answers
customer questions with **tool calling**: the model picks a tool, this repo runs
it and feeds the result back, and the loop repeats until the model writes the
final answer.

Everything the agent says about prices, rules, bookings and refunds comes from
`data/`. All the money is computed in Python, never by the model.

```
$ python -m citywheels "How much do I get back if I cancel CW-1002 today?"
You get back 340 EUR in total: 40 EUR of the 80 EUR rental price (50%, because
the rental starts in 33 hours) plus the 300 EUR deposit, paid back to the same
card within 5 working days. The 5 EUR booking fee is not refunded.
```

## Quick start

```bash
git clone https://github.com/bsatird/citywheels-helpdesk && cd citywheels-helpdesk
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your key into GEMINI_API_KEY
python -m citywheels "What time do you open on Saturday?"
```

The free Google Gemini tier is the default (`gemini-3.5-flash`): create a key at
<https://aistudio.google.com/apikey> and set it in `.env`.

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
```

Any OpenAI-compatible endpoint works; `LLM_PROVIDER` also accepts `openai`,
`github` and `ollama` (local). See `.env.example` for the variables and defaults.
The `github` preset is kept as an example only: GitHub Models itself was retired
on 2026-07-30, so that host no longer answers - point `LLM_BASE_URL` at another
compatible service to use it.

Free Gemini keys are tight: the API answers `429` both per minute (~5 requests)
and per day (about 20 requests *per project* - the number is in the error body,
and two keys inside one project share that budget). One agent answer is 2-3
requests. Nothing breaks - the client honours the delay the provider asks for
and retries - but a full 15-question evaluation does not fit in a 20-request day,
so it has to be spread over sessions with `--resume`, or run on the lite model:

```bash
LLM_MODEL=gemini-flash-lite-latest python eval/run_eval.py --min-interval 6
```

The default model's report (`eval/results-gemini-3.5-flash.md`) was collected the
other way: several sessions with `--min-interval 13`, each one asking only the
questions that had no cached answer yet.

`--resume` is there for the flaky end of that: every graded answer is appended to
`eval/.cache/<provider>-<model>.jsonl` as soon as it is scored, so an interrupted
run (a closed terminal, a `429`, a Ctrl-C) is continued by running the same command
again instead of paying for the same questions twice. A cached answer is reused only
while its question is unchanged, and never when the row is an error. Delete
`eval/.cache/` to force fresh answers.

Set `LLM_MIN_INTERVAL=13` (seconds between calls) to avoid the per-minute `429`
in normal use.

## Ways to run it

```bash
python -m citywheels "Can my 14-year-old rent an e-bike?"   # one question
python -m citywheels -i                                    # chat with memory
python -m citywheels -v "Where is CW-4242?"                # show every tool call
echo "How much is a deposit for an e-bike?" | python -m citywheels   # stdin
python -m citywheels --max-iterations 10 "..."             # more model turns
```

In the interactive mode `:help`, `:reset`, `:now`, `:trace` and `:quit` work.
Exit codes: `0` answered, `1` configuration or model error, `2` no question.

## Tests

```bash
python -m pytest          # 247 tests, 226 passed, 21 skipped, no network, no API key
```

The suite covers every refund rule including the 48 h / 24 h boundaries, booking
lookup normalisation, policy search ranking, the tool dispatcher, the guardrail,
the agent loop (driven by a scripted fake model), the CLI, the consistency of
`data/bookings.json` with `data/policies/prices.md`, and the evaluation harness
itself.

## Evaluation

```bash
python eval/run_eval.py                 # real model, writes eval/results.md
python eval/run_eval.py --only Q01,Q04  # a subset
python eval/run_eval.py --json          # machine-readable, writes nothing
python eval/run_eval.py --fake          # plumbing check, no key needed
```

`eval/questions.json` holds 15 questions with their expected outcome: which tool
must be called, which amounts and phrases must appear, which must not, and which
questions the agent has to decline. Two runs are committed: `eval/results.md`
(15/15 correct, 0 unsupported euro amounts on `gemini-flash-lite-latest`) and
`eval/results-gemini-3.5-flash.md` (the same 15/15 on the default model, which is how
the optional "compare two approaches" part is covered). `LLM_MODEL` runs the harness on
another model and `--out` keeps the earlier report intact; `SOLUTION.md` explains what
failed and why.

## How it works

```
question -> HelpDeskAgent.run()
              |  system prompt + 4 tool schemas
              v
            model  --(tool calls)-->  execute_tool()  -> JSON payload
              ^                            |
              +----------- result --------+
              |
              +--(no tool calls)--> guardrail -> answer
```

* **`citywheels/agent.py`** — the loop. At most `max_iterations` (default 6)
  model turns, then a fallback answer. `stopped_reason` is `final_answer`,
  `empty_response` or `max_iterations`.
* **`citywheels/tools/`** — the four tools plus the dispatcher. `execute_tool`
  **never raises**: unknown tools, broken JSON, wrong argument names and bugs
  inside a tool all come back as `{"error": ...}` payloads the model can read.
* **`citywheels/guardrails.py`** — collects every euro amount a tool returned and
  flags amounts in the answer that no tool can justify
  (`AgentResult.unsupported_amounts`, printed as a warning with `-v`).
* **`citywheels/prompts.py`** — the instructions: use a tool before answering,
  copy `calculate_refund` numbers verbatim, never invent a booking or an amount,
  decline and point to the staff when the documents do not cover something.
* **`citywheels/config.py`** — paths, contact details and the fixed "now",
  overridable with `CITYWHEELS_NOW` so answers and tests are repeatable.
* **`citywheels/testing.py`** — `FakeLLM`, a scripted stand-in for the model.

### Tools

| Tool | Input | What it does |
| --- | --- | --- |
| `search_policies` | `query`, `top_k` (1–5, default 3) | BM25 search over the Markdown in `data/policies/`, split into `##` sections. Synonyms are expanded on the query only, so the documents stay clean. Returns passages with their file and section anchor. |
| `get_booking` | `booking_id` | Looks up one booking in `data/bookings.json`. Accepts `CW-1002`, `cw-1002`, `CW1002`, `1002`. Unknown numbers return `found: false`, never an exception. |
| `calculate_refund` | `booking_id`, optional `cancel_in_hours` | Applies the cancellation rules with `Decimal`, rounded half-up to cents: ≥48 h full rental price, ≥24 h half, <24 h none; the booking fee is never refunded, the deposit always is; `in_progress` / `completed` / `cancelled` cannot be cancelled. `cancel_in_hours` answers "what if I cancel tomorrow instead?". |
| `get_current_time` | – | The shop's fixed "now". |

## Project layout

```
citywheels/          agent, CLI, tools, prompts, guardrails, fake model
data/                bookings.json and the policy Markdown (the only source of truth)
eval/                questions.json, run_eval.py, results*.md (one report per model)
tests/               pytest suite
SOLUTION.md          design notes, evaluation results, limitations, next steps
```

## The data

* `data/policies/prices.md` — bike types, rental prices, add-ons, deposits, late return fees.
* `data/policies/cancellations.md` — cancelling, changing, refunds, bad weather.
* `data/policies/faq.md` — opening hours, age limits, payment, helmets, accidents, contact details.
* `data/bookings.json` — example bookings with the statuses `confirmed`, `in_progress`, `completed` and `cancelled`.

The bookings are set in June 2026, so the shop's "current time" is fixed at
**2026-06-10 09:00 Europe/Warsaw (UTC+02:00)**; change it with `CITYWHEELS_NOW`.