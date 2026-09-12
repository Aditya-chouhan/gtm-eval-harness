# gtm-eval-harness

Measurement tools for AI outputs in a go-to-market stack — precision with
intervals, stratified breakdowns, and chance-corrected judge-vs-human
agreement.

**Status:** the metrics core, stratified reporting, and a deterministic
numeric-grounding checker are built and tested (87 tests), and have produced
two real results on real repositories. The judge-model slot for non-numeric
claims is defined but not implemented — see [Roadmap](#roadmap). Everything
claimed below is in the repo now.

---

## The problem

AI-generated GTM output gets trusted or distrusted on vibes. A brief reads
fluent, so it ships. Then someone finds a fabricated fact in an email that went
to a real prospect, and the whole category loses credibility — which is roughly
the [11x story](https://techcrunch.com/), and why "AI SDR" is now a phrase
buyers flinch at.

The gap is not model quality. It is that almost nobody attaches a number to
"how often is this wrong, and wrong in what way." Doing that is unglamorous and
it is the actual skill: turning an unpredictable system into a governable one.

## What this does

**Reports rates with intervals, never bare.** `60.5%` from 43 samples and
`60.5%` from 4,300 samples are different claims. Every proportion ships with a
Wilson score interval, so sample size is visible inside the claim rather than
buried in a methods note.

**Refuses to return flattering defaults.** Precision over zero predictions is
`None`, not `0.0`. Accuracy without observed true negatives is `None`, not a
number computed over a population the system never ran against. A system that
flagged nothing has undefined precision, and saying so is the honest answer.

**Breaks results down by group and flags concentration.** If most of the sample
came from one source, the pooled number is largely a statement about that
source. The report says so, in the output, unprompted.

**Chance-corrects agreement.** When an LLM judge is validated against human
labels, raw agreement is inflated by the base rate — two raters who both mostly
say "no" agree ~90% of the time while sharing no signal. Cohen's kappa is
reported alongside.

## The first result: re-analysing a published number, including my own

[`groundtruth`](https://github.com/Aditya-chouhan/groundtruth) scanned 30 public
repositories for a specific defect pattern, hand-checked all 43 candidates
against the live GitHub API, and published **26 true positives / 17 false
positives = 60.5% precision**, false positives included rather than hidden.

That is already more honest than most published numbers. This harness asks the
next two questions of it:

```
PRECISION
  point estimate : 60.5%  (26/43)
  95% interval   : [45.6%, 73.6%] (95% CI)
  interval width : 28.1 percentage points

PRECISION BY REPOSITORY
  mem0ai/mem0             78.6%  (22/28)  [60.5%, 89.8%] (95% CI)
  pallets/flask            0.0%  (0/5)   [0.0%, 43.4%] (95% CI)
  run-llama/llama_index   60.0%  (3/5)   [23.1%, 88.2%] (95% CI)
  expressjs/express        0.0%  (0/2)   [0.0%, 65.8%] (95% CI)
  streamlit/streamlit      0.0%  (0/2)   [0.0%, 65.8%] (95% CI)
  fastify/fastify        100.0%  (1/1)   [20.7%, 100.0%] (95% CI)

SAMPLE SHAPE
  65% of the sample is a single group (mem0ai/mem0).
  WARNING: the pooled figure above is substantially a statement
  about mem0ai/mem0 and should not be presented as a general rate.
```

Two findings, both of which change how that 60.5% should be read:

1. **The interval is 28 points wide.** At n=43 the result cannot distinguish
   "about half" from "about three quarters." The point estimate was never
   wrong, but on its own it implies a precision the sample does not support.

2. **The detector mostly works on one codebase.** It scores 78.6% on
   `mem0ai/mem0` — which supplies 65% of the sample — and 0-for-5 on Flask,
   0-for-2 on Express, 0-for-2 on Streamlit. The pooled 60.5% is close to a
   weighted average of "good on mem0" and "no signal anywhere else." That is a
   materially different conclusion from "60% precision on open-source repos."

Neither finding required new data. Both were sitting in the committed JSON.

Reproduce it:

```bash
python scripts/analyze_groundtruth.py [path/to/groundtruth]
```

No network access, no API key. Every number recomputes from
`output/verified_findings.json` in a groundtruth checkout.

## The second result: can a reader verify the numbers in a README?

A write-up citing figures a reader cannot find in the repository is asking to
be taken on trust. `scripts/check_readme_grounding.py` extracts every numeric
claim from a README and checks it against that repo's **committed** data files
(found via `git ls-files`, because the question is what a reader of the
*published* repo can verify).

Historical snapshot run across six repositories:

> **Snapshot warning:** this table and
> `output/readme-grounding-sweep.txt` predate later repository corrections and
> evidence receipts, including Salesforce execution receipts and the Clay cost
> rewrite. They are retained as historical research output and must not be read
> as a current ranking. Re-run the checker against pinned commits before citing
> a current rate.

| Repository | Claims verifiable from committed data |
|---|---|
| `n8n-gtm-orchestration` | **33/34 (97%)** [85.1%, 99.5%] |
| `gtm-data-quality-monitor` | 20/24 (83%) [64.1%, 93.3%] |
| `clay-enrichment-waterfall` | 93/116 (80%) [72.0%, 86.4%] |
| `groundtruth` | 20/31 (65%) [46.9%, 78.9%] |
| `salesforce-gtm-org` | **10/40 (25%)** [14.2%, 40.2%] |
| `revenue-engine-system` | not checkable — no git repo present locally |

At the time of that snapshot, `salesforce-gtm-org` had no committed deploy log
or raw test receipt. That is no longer the repository's current state: dated
machine-readable and human-readable Apex test output has since been committed.
The 25% figure describes the old snapshot and should not be used as a current
assessment.

**Rates are not comparable across repos.** A repository that commits rich
output will score high; one that commits configuration only will score low
regardless of whether its claims are true. The number is a prompt to look, not
a verdict — and version pins, retry counts and install sizes will always flag.
`n8n-gtm-orchestration`'s single miss is "n8n is a 2.5 GB install", which is
correct and simply not something a run log would contain.

```bash
python scripts/check_readme_grounding.py ../some-repo [../another-repo ...]
```

## Run it

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest tests/ -q     # 87 tests
```

Tested on Python 3.9.6 — the system Python that ships with macOS. The metrics
core imports nothing outside the standard library, so it runs anywhere Python
does.

## What this doesn't do

- **It does not call a model.** Numeric claims are labelled deterministically,
  which is exactly why this runs at zero cost with no API key. Sending prompts
  to a model and scoring the responses is not built.
- **It only checks numeric claims.** Whether a sentence fairly characterises a
  source needs a reader or a judge model. `LLMClaimChecker` defines that slot
  and raises `NotImplementedError` rather than pretending to fill it.
- **"Supported" means the value appears in a source, not that it is used
  correctly.** A brief that swaps two real figures between quarters passes this
  check. The limitation has a test pinning it in place
  (`test_documented_limitation_swapped_numbers_pass`) so it stays known rather
  than becoming a surprise.
- **It has no opinion on whether your labels are right.** Garbage ground truth
  produces confident garbage metrics. Human labelling quality is the input this
  cannot check for you.
- **The groundtruth re-analysis is not independent validation of groundtruth.**
  Same author, same underlying run. It is a sharper reading of a published
  result, not a second opinion on it.
- **Wilson intervals assume independent samples.** The 28 candidates from
  `mem0ai/mem0` come from one codebase with one set of conventions and are
  plausibly correlated, which means the true interval is likely *wider* than
  the one reported, not narrower.

## Roadmap

Specified, not yet built:

- **Prompt runner** — versioned prompts, a golden dataset, precision/recall per
  prompt version, cost and latency per call.
- **LLM-as-judge validation** — judge labels scored against human labels using
  the `cohens_kappa` already implemented here. No judge output gets reported as
  a rate before this exists.
- **Same-kind swap detection** — the documented blind spot above. Catching it
  means checking the association between a figure and its subject, not just the
  figure's presence, which likely needs the judge slot.

## Design notes

The three decisions worth arguing about are documented at the top of
[`src/gtm_eval/metrics.py`](src/gtm_eval/metrics.py): undefined is not zero,
point estimates never ship alone, and agreement is measured against chance.

One test in this suite caught a real bug during development: the concentration
warning used `>=` against its 0.5 threshold, which fired on a perfectly
balanced two-group sample — the one case that should never warn. Fixed to `>`;
the test that caught it is `test_no_warning_on_balanced_sample`.

MIT licensed.
