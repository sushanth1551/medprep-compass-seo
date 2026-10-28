# MedPrep Compass — SEO Blog Draft Workflow

A local tool that drafts blog articles for a medical-exam preparation site, checks each draft with
transparent automatic rules, helps you review its sources, and measures token use. It has a web
interface and a command line.

**It only creates drafts. It never publishes anything.** Automatic checks cannot guarantee search
rankings or factual accuracy, and nothing here medically validates an article. Publication always
needs a qualified human.

## 1. Set up (Windows PowerShell)

Run these from inside the `medprep-compass-seo` folder.

```powershell
python -m venv .venv                         # first time only
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
Copy-Item .env.example .env                  # then edit .env yourself
```

`.env` is git-ignored. Put keys **only** in `.env`: never in chat, source code, tests, or this README.

### Choose a mode in `.env`

| Mode | Settings |
|---|---|
| Mock (default, free, offline) | `AI_PROVIDER=mock` |
| Groq | `AI_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_MODEL` |
| Claude | `AI_PROVIDER=claude`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |

Model ids are used exactly as written; the app never swaps models. Pick a current id from the
provider's model list. Groq and Claude requests may cost money. Mock output is fixed placeholder text.

Optional settings: `MAX_OUTPUT_TOKENS` (default 8000), `GROQ_REASONING_EFFORT` (`low`/`medium`/`high`),
`PROMPT_STYLE` (`structured` default, `optimized`, `baseline`), `REGENERATE_FAILED_SECTIONS=true`,
`GROQ_TIMEOUT_SECONDS`, `ANTHROPIC_TIMEOUT_SECONDS`.

## 2. Start the web app

```powershell
python -m app.web
```

Open **http://127.0.0.1:8000** (health check: http://127.0.0.1:8000/health). Stop it with Ctrl+C.

1. Enter a **topic**, an optional **keyword** (empty = reuse the topic) and a **length**:
   short (about 500-600 words), medium (about 900), long (about 1,500). These are targets, not guarantees.
2. Press **Generate draft** (one request; the button disables itself while it runs).
3. Read the article, **Status**, and the **Quality report**.
4. Optionally press **Check source links** (free; fetches the cited pages, no AI).
5. Press **Download Markdown** or **Download PDF**. Both contain exactly the revision on screen.

Command line (same workflow, no browser): `python -m app.main`.

### What the page shows

- **Status** — only states that are really reached: *Draft generated*, *Automated checks completed*,
  *Source verification incomplete*, *Needs human review*, *Approved for export*,
  *Approved for publication by a human*. The last two come only from buttons **you** press for that
  exact revision; they record your decision and the app still publishes nothing.
- **Quality report** — separate sections (not one score): completeness, SEO structure, keyword use,
  readability, sources and facts, human-review flags, export readiness. The SEO list is an internal
  checklist, not a Google ranking score.
- **Run details** — provider, exact model, token usage, model calls, elapsed time, cost.
- **Revisions** — *Rewrite this section* regenerates **one** section as a new revision (a visible,
  confirmed, paid request). The original stays untouched; checks are recalculated; sources and
  approvals start empty again.
- **Re-run automatic checks** — recomputes the free checks on the saved text without changing the article.

Export is blocked if the article looks incomplete (missing sections or FAQ, far below the length
target, empty parts). Other warnings are shown next to the download buttons so you can decide.

## 3. Where files are stored

| What | Where |
|---|---|
| Article (Markdown) | `output/<name>.md` (never overwritten; `-2`, `-3`, or `-rev2` is added) |
| SEO report for that file | `output/<name>.seo-report.json` (includes run id, revision, SHA-256 of the article) |
| Web record (checks, sources, approvals) | `output/<name>.web.json` |
| Per-run metrics (also failed runs) | `reports/runs/run-*.json` |
| Benchmarks | `reports/benchmarks/` |

The article, report, metrics and exports are tied together by a **run ID** and a **hash**. If the article
file is edited later, the page shows a "stale report" problem and blocks approval. Drafts from earlier
versions of the app are marked *legacy*: viewable and exportable, but not approvable.

## 4. Output limit, words and tokens

A **word** and a **token** differ: English averages roughly 1.3-1.5 tokens per word (a rule of thumb),
and the JSON wrapper adds more. `MAX_OUTPUT_TOKENS` limits **tokens**; the length choice only *asks* the
model for about that many words. Reasoning models such as `openai/gpt-oss-20b` also spend output tokens
on thinking, so the default is 8000 (Groq documents a 65,536-token maximum for that model). The form warns
if your limit looks too small for the chosen length.

A reply is treated as cut off only when the provider says so (`finish_reason = "length"` for Groq,
`max_tokens` for Claude). Then **no draft is saved, nothing is retried**, and the message says how many
tokens were used. A missing or unknown finish reason is not treated as truncation. Separately, the
completeness check looks at sections, FAQ, title/meta, length against the target, and whether the
text ends cleanly.

**Missing sections.** Sometimes a model finishes normally but leaves sections out (or repeats a JSON key,
which plain JSON parsing would silently collapse to the last copy). The app merges repeated keys, then, if
planned sections are still missing, makes **one** extra request that writes only the missing sections (never
repeating existing ones). That follow-up is bounded (`MAX_CONTINUATION_CALLS`, default 1, `0` turns it off,
maximum 2), counted in the model-call total and token usage, and shown on the page. If fewer than half the
sections exist afterwards the draft is **not saved**; milder gaps are saved but blocked from export.
Unusual replies write a diagnostics file to `reports/diagnostics/` (finish reason, token counts and the
model's raw reply text; never prompts or keys) so the cause can be proven instead of guessed.

Troubleshooting: *cut off* -> raise `MAX_OUTPUT_TOKENS`, restart, try again (optionally set
`GROQ_REASONING_EFFORT=low`). *Model not accepted* -> check `GROQ_MODEL` in your Groq console.
*Missing credentials* -> the page names the missing variable. *Rate limit / timeout / network* -> wait and
retry by hand; the app never retries a paid request automatically.

## 5. Sources and facts

The model may propose references, but it can invent them. **Check source links** only establishes:
whether each URL is well-formed, whether the page can be fetched (public websites only), and a weak
word-overlap hint between the claim and the page. It **cannot** establish that a source supports a claim, is
current, or is medically correct, so its best result is "links reachable", never "verified". Some sites
(for example usmle.org) may refuse automated requests (HTTP 403); that is shown honestly as not reached.
Deterministic checks also flag factual-sounding sentences, and the model's own `review_flags`, for a human to check.
Source checking is kept separate from the SEO checklist.

## 6. Tokens and cost

- **actual** — counts returned by the provider; **estimated** — text length / 4 (mock only);
  **unavailable** — the API gave no numbers, so none are invented. They are never compared as equal.
- Cost comes from `pricing.json` (USD per 1,000,000 tokens, key `provider/model-id`). It ships **empty**, so cost
  shows **unknown** (never 0) until you add rates from the provider's pricing page and the date you checked.
- Groq prompt caching is automatic for a few models and reported in `cached_tokens`; no saving is claimed
  without measured usage. Mock benchmark numbers are estimates and prove nothing about real savings.

## 7. Tests, lint and benchmarks

```powershell
python -m pytest                 # no test uses the network or a real key
python -m ruff check .
python -m app.benchmark          # mock benchmark: free, offline, labelled as estimates
python -m app.benchmark --live --i-understand-this-spends-credits   # real, paid: 6 requests
```

The live benchmark is never run automatically.

## 8. Security

Keys live only in `.env` on the server; no page, response, saved file or log contains them. The server binds
to `127.0.0.1` only and has **no login**. Source fetching refuses private/local addresses. Do not expose this
app to a network without adding authentication and HTTPS.

## 9. Architecture

```
topic -> app/web.py or app/main.py -> app/config.py (provider, limits)
  -> app/workflow.py: validate -> outline -> provider writes article (providers/*.py, prompts.py)
  -> app/seo_checks.py + app/metrics.py (completeness) -> save article + report + metrics
  -> app/store.py (record, hashes, consistency) -> app/quality.py (separate sections + status)
  -> app/sources.py (link checks) -> app/revision.py (one-section revisions) -> app/pdf_export.py
```

## 10. Optional deployment (not done, needs your approval)

Nothing is deployed. If you choose to later: keep `.env` out of source control and set `AI_PROVIDER`,
the key and model as the host's secret variables; start with
`python -m uvicorn app.web:app --host 0.0.0.0 --port $PORT` (behind HTTPS); use `GET /health` as the health check;
add authentication first; use persistent storage for `output/` and `reports/`.

## Known limitations

- Medical content is never validated automatically; a qualified person must review it.
- Source checks are weak hints; claim detection is pattern matching.
- The PDF uses built-in fonts (non-Western characters become `?`); no tables or images.
- Single-user, local; one generation at a time; token savings are only claimed from real measured runs.
