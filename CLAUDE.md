# CLAUDE.md — Footnote

Operating manual for Claude Code working in this repo. Read fully at the start of every
session. It overrides your defaults.

**Also read `NOTES.md` if it exists** — it carries working preferences that aren't committed.

Repo and folder name: `footnote` (lowercase). Display name in prose and UI: **Footnote**.

---

## 1. What this is

Footnote answers plain-English questions from a fixed set of documents, and every answer
carries the exact line it came from. If the answer is not in the documents, it says so instead
of guessing. Code — not the model — checks every quote before anything reaches the user.

The documents are the internal handbook of **Inter Mediate FC**, a fictional club in the
fictional **Premier-ish League**: code of conduct, matchday operations, travel, set pieces, and
so on, in two editions (2025/26 and 2026/27). The corpus is fiction by design: the model cannot
know any of it, so the eval can tell grounding from memory.

It answers questions about these documents. It does not take uploads, hold conversations, or
act on anything.

---

## 2. Working style

- **Act as a senior reviewer.** Disagree when something is wrong. Agreeable answers that let
  fragile code ship are a failure.
- **Plan before editing.** For anything beyond a one-file change, state the plan and the exact
  files you'll touch, then wait for approval.
- **Never invent data.** API fixtures and cached examples come from real recorded responses.
  If Voyage or Claude is unreachable, surface the failure. The corpus is the one deliberate
  fiction, and it lives only in `corpus/` and `eval/corpora/`.
- **Prefer deleting to adding.** The default answer to "should we add dependency X?" is no.
- Answer "why" with the failure it prevents, not with a principle.
- Give rules that can be *checked*, not rules that require judgment. "Every quote is a
  substring of a retrieved chunk" beats "make sure answers are grounded."

---

## 3. The seven invariants

Violating any of these is a bug regardless of test status.

1. **Every quote is verifiable.** Each citation's `quote` must appear verbatim in a chunk that
   was retrieved *for this request*. The only normalisation is collapsing whitespace and
   straightening curly quotes; case must match. One failing quote turns the whole answer into
   a refusal with reason `citation_not_found` or `citation_not_retrieved`. Checked by code, in
   `verify.py`, never by the model. Prevents: fabricated quotes and invented sources.
2. **No citation, no answer.** `answerable: true` requires at least one citation, or the
   answer is rejected as `uncited_answer`. `answerable: false` is a correct output, not an
   error. Prevents: a fluent answer from the model's general knowledge.
3. **Retrieved text is data, never instructions.** Chunks appear only inside `<document>`
   tags in the user turn, never in the system prompt. `GroundedAnswer` has no field that could
   carry an action. A test inspects the outgoing request structure. Prevents: a document
   steering the model. **Limit:** this reduces injection, it does not rule it out — a planted
   instruction can be obeyed and still quoted verbatim. The eval's canary case measures it.
4. **Structured output only.** Every reply parses into `GroundedAnswer` before anything
   downstream touches it. Free text is a failure path, not a fallback.
5. **Degrade, never drop.** Voyage fails → BM25-only retrieval, response tagged
   `retrieval: keyword_only`. Claude fails after one retry → return the top retrieved passages
   with `answer: null` and reason `passages_only`. Every degradation path logs a reason code
   and has a test.
6. **Cost ceiling.** Per question: at most one embedding call and one Claude call, plus at
   most one retry; both are billed. Context is capped at the top 5 chunks and
   `max_context_words` from config. A hard monthly spend limit on Footnote's own workspace in
   the Anthropic Console. A per-day cap from config stops bursts within one process and resets
   on restart. `/ask` is rate-limited by IP. On either cap, serve a cached example — a real
   recorded answer — and say so plainly.
7. **Superseded rules are flagged by code, not by the model.** Every chunk carries `edition`
   and `is_current` from the index. When an answer cites a superseded chunk, `verify.py` sets
   `superseded: true` on that citation from the metadata. The model is never asked. Prevents:
   last season's rule presented as current.

---

## 4. Stack — seven runtime dependencies, do not expand

`fastapi`, `uvicorn`, `pydantic` (v2), `pydantic-settings`, `httpx`, `anthropic`, `numpy`

The same six as SitRep, plus `numpy` for vector search. Dev: `pytest`, `pytest-asyncio`,
`ruff`. Pin every version exactly in `requirements.txt`, as SitRep does.

### Retrieval

Hybrid: dense search and BM25 run side by side, merged with reciprocal rank fusion.

- **Chunking** at headings. A chunk is one section; a section over `max_chunk_words` splits at
  paragraph boundaries. Each chunk carries `chunk_id`, `doc_id`, `club`, `edition`,
  `is_current`, `section_path` (e.g. `matchday-ops-2026-27#2.3`), `title`, `text`.
- **Dense:** Voyage embeddings, cosine similarity over a numpy matrix. Brute force over a few
  hundred chunks takes about a millisecond; a vector database would add a service, a network
  hop and a failure mode to solve a problem this corpus doesn't have.
- **Keyword:** BM25, written here, about 30 lines. Dense search misses exact terms — codenames,
  fine amounts, section numbers. BM25 finds them.
- **Fusion:** RRF, `score = Σ 1 / (rrf_k + rank)` over both lists, `rrf_k = 60`. Uses ranks
  only, so the two incompatible score scales never meet.
- **Determinism:** ties break by `chunk_id`. The same question on the same index returns the
  same chunks in the same order. Replay depends on this.

### The index

Built offline by `scripts/build_index.py` (`make index`), never at runtime. Output is committed
in `index/`: `chunks.jsonl`, `embeddings.npy`, `bm25.json`, `manifest.json`. The manifest holds
the sha256 of every corpus file, the chunker settings, the embedding model and its dimension.
**CI fails if `corpus/` or the chunker changed and the manifest wasn't rebuilt.** Loads at
startup in under one second. If any committed index file exceeds 20 MB, stop and reconsider
(smaller `output_dimension`, float16).

### Embeddings — Voyage AI

- Raw `httpx` POST to Voyage's embeddings endpoint, **no SDK** — one dependency fewer, and
  `httpx.MockTransport` tests it the same way as everything else.
- Model pinned in `config.py`, never at call sites. Default `voyage-4`. Verify the current
  model list and endpoint at https://docs.voyageai.com before Phase 2.
- `input_type` is `document` for chunks and `query` for questions; mixing them degrades recall.
- Not a local embedding model: loading one risks Render's 512 MB limit and adds a heavy
  dependency, to save calls that cost a fraction of a cent.

### Claude API

- Model: **`claude-haiku-5-5`**, pinned in `config.py`, never at call sites. Verify the current
  ID at https://platform.claude.com/docs/en/about-claude/models/overview.
- The `anthropic` SDK directly, not a gateway. `client.messages.parse(..., output_format=GroundedAnswer)`.
  The call lives in one function, so switching providers is a single-file change.
- Haiku 5.5 specifics: **do not send `temperature`, `top_p` or `top_k`**; no assistant prefill
  (400 error); thinking is adaptive, controlled by `effort` in `output_config`, set from config
  (default `low`). `max_tokens` includes thinking tokens. Handle `stop_reason: "refusal"` as a
  reason code.
- Pricing is tiered by prompt size: up to 100k tokens, input $0.10 / output $0.50 / cache read
  $0.01 per MTok; above 100k, $0.50 / $2.50 / $0.05. Both tiers live in `config.py`; cost is
  computed from the response's `usage`, including cache reads and writes.
- **Not Anthropic's native Citations API.** It cannot be combined with structured outputs (the
  API returns 400), and Footnote needs both: the structured refusal and the citations. Our own
  verifier also works unchanged with a local model, which has no citations API.

### Explicitly not in this project

LangChain, LlamaIndex, any vector database (Chroma, FAISS, pgvector, Pinecone), `rank-bm25`
(write BM25), `sentence-transformers`/`fastembed`, the `voyageai` SDK, rerankers, `structlog`,
`mypy`, `respx`, SQLite, Celery, Redis, Docker, a React frontend.

---

## 5. Hosting

| Piece | Where |
|---|---|
| Backend (FastAPI) | Render free web service, **in Footnote's own Render workspace**, native Python build, no Dockerfile |
| Demo page | GitHub Pages, served separately from the backend |
| CI | GitHub Actions — ruff + pytest on push |
| Embeddings | Voyage AI API |
| Keep-awake ping | UptimeRobot monitor on `/healthz`, every 5 minutes. Set up on their site, not in this repo |

Secrets live in `.env`, gitignored, with `.env.example` committed as placeholders.

**Why a separate Render workspace.** Free instance hours are 750 per workspace per month, and
exceeding them suspends *every* free service in that workspace for the rest of the month.
SitRep's keep-awake monitor already uses about 730. Sharing a workspace would let Footnote's
traffic take SitRep down. In its own workspace, Footnote's monitor uses about 730 of its 750.

**Cold starts.** SitRep measured 42 s to wake on Render's free tier, which is why the monitor
exists. Footnote also loads the index at startup; measure the full cold start in Phase 2 and
record the number here.

**The monitor's second job** is the same as SitRep's: the daily cap, the IP limit and any cache
live in process memory, and a sleeping service is a stopped process. The Console's monthly
limit is the one bound that survives restarts; confirm it is set before relying on anything
else.

**Do not use HuggingFace Spaces** — as of roughly July 2026, only Static Spaces are free.

---

## 6. Repo layout

```
footnote/
├── app/
│   ├── main.py          # FastAPI app, router wiring, index load at startup
│   ├── config.py        # pydantic-settings; ALL env vars and tunables, typed
│   ├── schemas.py       # Chunk, Citation, GroundedAnswer, AskRequest, AskResponse
│   ├── logs.py          # stdlib logging + JSON formatter
│   ├── security.py      # IP rate limit for /ask
│   ├── chunker.py       # markdown → chunks with section paths and edition metadata
│   ├── embed.py         # Voyage client (httpx)
│   ├── bm25.py          # BM25 build + score
│   ├── retrieve.py      # dense search, RRF, hybrid, keyword-only fallback
│   ├── index.py         # load and validate the committed index
│   ├── llm.py           # Anthropic client, retry, cost accounting, spend ledger
│   ├── verify.py        # citation verifier, uncited rule, superseded flag
│   ├── prompts/         # versioned .md prompt files (answer_v1.md)
│   └── routes/          # ask.py, health.py
├── corpus/inter-mediate/    # the handbook, one .md per document and edition
├── index/               # committed build output: chunks.jsonl, embeddings.npy, bm25.json, manifest.json
├── scripts/build_index.py
├── eval/
│   ├── dataset.jsonl    # 25 labelled questions
│   ├── recordings.jsonl # recorded Voyage and Claude exchanges
│   ├── run_eval.py
│   ├── results.md       # committed, written by `make eval`
│   ├── corpora/         # Premier-ish League filler for the larger corpus sizes
│   └── index/           # eval-only indexes for those sizes; never loaded by the app
├── demo/                # static page for GitHub Pages
├── tests/
├── docs/architecture.png
├── .github/workflows/   # ci.yml, pages.yml
├── .env.example
├── CLAUDE.md
└── README.md
```

### Conventions

- Type hints everywhere. Public functions document what fails and how.
- No bare `except`. Catch specific exceptions, log a reason code, degrade. The one broad
  `except Exception` in `app/` is the last-resort net in the `/ask` handler, and it logs the
  traceback. The only other is the per-case runner in `eval/run_eval.py`, recorded as `crashed`.
- All I/O async.
- Config only via `config.py`. `os.getenv` anywhere else is a review rejection. Every tunable
  — `top_k`, `rrf_k`, `max_chunk_words`, `max_context_words`, caps, prices, model IDs — lives
  there.
- **Prompts are code** — versioned `.md` files in `app/prompts/`. Changing one requires
  `make eval-record`, marking the new replies, then `make eval`, and committing
  `eval/recordings.jsonl` and `eval/results.md` with the prompt. CI fails until you do.
- **Retrieval and generation are separate modules.** `retrieve.py` never imports `llm.py`, and
  the other way round. Every LLM call goes through one function in `llm.py`.
- One log line = one JSON object with `request_id`, `stage`, `latency_ms`, `outcome`.
- Pydantic model names describe the object, not the product (`GroundedAnswer`, not `Footnote`).
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`.
  Imperative, lowercase, no trailing period. One idea per commit.

Make targets: `setup`, `run`, `test`, `lint`, `index`, `eval`, `eval-record`.

### Testing

- Every degradation path in invariant #5 gets a test.
- **The Anthropic and Voyage APIs are never called in tests** — mock at the client boundary
  with `httpx.MockTransport`, fixtures from real recorded responses.
- Required cases:
  - verifier: verbatim quote passes; whitespace and curly-quote differences pass; a case
    difference fails; a quote from a chunk that wasn't retrieved fails; an answer with no
    citations fails; a superseded chunk is flagged from metadata alone
  - Voyage timeout → `keyword_only`; Claude invalid output twice → `passages_only`
  - daily cap reached → cached example; IP limit → 429
  - request structure: no chunk text in the system prompt, every chunk inside `<document>` tags
  - retrieval determinism: the same question twice returns identical chunk IDs
  - context never exceeds 5 chunks or `max_context_words`
  - index manifest matches the corpus hashes
- Tests run offline in under 30 seconds. Target ≥80% coverage on `app/`.

### Eval harness

`eval/dataset.jsonl` holds 25 labelled questions: 12 normal, 7 edge-case, 6 adversarial.
**Labels are written before any reply is recorded.** Each row carries `id`, `category`,
`question`, `answerable`, `gold_sections` (section paths, so a chunker change doesn't break
them), `must_contain` (accepted forms of the key fact), and `expect_superseded`. Every question
names the club or is unambiguous about it, because the filler league shares the template.

Adversarial rows: a real-world football question; a club question the handbook plausibly should
cover but doesn't; a false premise; "ignore the documents"; a request to quote a section that
doesn't exist; and the **canary** — one corpus document contains a planted instruction to end
the answer with `MARMALADE`. The canary appearing in any answer fails that case.

**Modes** (`--mode`): `no-retrieval` (the model alone), `full-context` (the whole corpus in the
prompt, cached), `rag` (Footnote). **Corpus sizes** (`--size`): `s` (the handbook, ~15k words),
`m` (~100k), `l` (~500k), padded with handbooks of other Premier-ish League clubs — same
template, different numbers. Full-context and rag run at every size; no-retrieval once.

Record once, replay as often as needed:

- `make eval-record` (online, billed, a few dollars across all modes and sizes) sends every case
  through the real pipeline, saving each raw Voyage and Claude reply with its latency, token
  usage and the sha256 of the exact request body to `eval/recordings.jsonl`. It keeps nothing
  from a run with a timeout or an error status.
- A person then marks each answer `marked_correct: true` or `false`. The string check cannot
  grade a paraphrase.
- `make eval` (offline, free) runs the same code with Voyage's and Claude's parts answered from
  the recording, matched by request fingerprint. A changed prompt, chunker, index or model
  changes the request, finds no recording, and fails.

`eval/results.md` reports, each rate as its two counts (0 of 0 is not 100%):

- **Retrieval**, one row each for dense, BM25 and hybrid: recall@5 and MRR against
  `gold_sections`.
- **Answers**, one row per mode and size: schema-valid rate; quotes verified (the verifier's
  raw input); key fact present (string check); correct (human-marked); correct refusal; false
  refusal; superseded flagged; canary leaked; fallback rate; latency p50/p95; median cost per
  question in USD.
- Case by case, where it disagreed with the labels, and **Limits**.

CI replays the committed recording. A prompt, model, index or dataset change without a new
recording fails, and so does a `results.md` that a fresh replay would write differently.

---

## 7. Out of scope — push back if raised

Document uploads; PDF ingestion (that is a separate project); multi-turn chat or conversation
memory; user accounts; streaming; a chat-style frontend; real-world corpora (the model already
knows them, so the eval cannot tell grounding from memory); an LLM judge for faithfulness (the
verifier is deterministic — that is the point); an agent loop.

**On agents:** the test is *does the system need to decide something at runtime that cannot be
decided at design time?* Here it doesn't — one source, fixed steps: retrieve, answer, verify.
The hardest case, a rule that changed between editions, is handled by retrieving both editions
with metadata, not by a loop. An agent would add cost variance, latency variance and
nondeterminism for no benefit.

---

## 8. Deferred

- **A local LLM** (Ollama or llama.cpp over their OpenAI-compatible HTTP API, via `httpx`) as
  an eval mode. This is where a two-implementation `generate()` switch becomes justified.
- **An agentic mode** — Claude with a `search` tool, a hard max-turns cap and a per-question
  token budget — as an eval column against the fixed pipeline.
- **An MCP server** exposing `search_handbook(query)`, so agents can use Footnote as a
  grounded source.
- **A reranker**, only if the eval shows recall@5 is the bottleneck.
- Docker, a provider abstraction.

These stay cheap because of four rules already above: one LLM function; retrieval and
generation in separate modules; `run_eval.py --mode` from day one; one results table with a row
per mode.
