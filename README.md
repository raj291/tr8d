# TR8D paper-trading research lab

TR8D is the first executable slice of the paper-trading agent design. It is a
research simulator—not a broker, financial adviser, or real-money trading
system.

The current MVP deliberately focuses on the foundations that must be correct
before news, RAG, or an LLM is allowed to influence a trade:

- point-in-time daily price snapshots;
- no use of the current day's open or close when making a decision;
- an expanding-window logistic-regression baseline;
- deterministic position and cash limits;
- fractional-share paper execution at the daily open with slippage;
- SQLite audit records for runs, decisions, trades, and portfolio snapshots;
- reproducible demo data and automated invariant tests.
- versioned data manifests and baseline-relative performance reports.

## Run it

Python 3.11+ and NumPy are required.

```bash
python3 -m tr8d demo --database var/tr8d.db --days 800
python3 -m tr8d inspect --database var/tr8d.db
python3 -m unittest discover -s tests -v
```

The demo creates deterministic synthetic daily data for `XLK`, `XLE`, and
`XLF`, trains only on observations preceding each replay date, and gives each
strategy an independent $10 wallet. Its JSON report includes return, maximum
drawdown, annualized volatility, turnover, trade count, and cash/buy-and-hold/
equal-weight comparisons.

## Ingest downloaded Stooq data

Download one daily CSV per symbol from Stooq, then normalize and fingerprint
the files:

```bash
python3 -m tr8d ingest-stooq \
  XLK=data/raw/xlk.csv XLE=data/raw/xle.csv XLF=data/raw/xlf.csv \
  --output data/processed/stooq_prices.csv

python3 -m tr8d replay data/processed/stooq_prices.csv --database var/stooq.db
```

Every ingestion/replay writes a JSON manifest containing the source, symbols,
date range, row count, and a SHA-256 digest of canonicalized rows.

## Compare numerical models

Install the optional ML dependency and run the expanding-window evaluation:

```bash
python3 -m pip install -e '.[ml]'
python3 -m tr8d evaluate-models data/processed/stooq_prices.csv \
  --output var/model-evaluation.json
```

Each fold trains on the past, calibrates probabilities on a later dedicated
period, and measures only the still-later test period. The report compares
logistic regression and XGBoost using accuracy, balanced accuracy, Brier score,
log loss, and expected calibration error. Its `promotion_candidate` is an
evaluation result—not automatic deployment authority.

## Ingest evidence and explain large moves

SEC access requires an identifying user agent with a contact email. GDELT DOC
search is limited to its recent rolling window.

```bash
python3 -m tr8d fetch-sec --cik 0000320193 --symbol AAPL \
  --user-agent 'tr8d research contact@example.com' \
  --output data/documents/aapl-sec.jsonl

python3 -m tr8d fetch-gdelt --query 'Apple Inc' --symbol AAPL \
  --start 2026-08-12T00:00:00+00:00 --end 2026-08-13T21:00:00+00:00 \
  --user-agent 'tr8d research contact@example.com' \
  --output data/documents/aapl-news.jsonl

python3 -m tr8d explain-move prices.csv data/documents/aapl-news.jsonl \
  --symbol AAPL --sector XLK --market SPY --date 2026-08-13 \
  --output var/aapl-2026-08-13-explanation.json
```

The explainer runs only after close, rejects evidence that was unavailable by
the close, decomposes the move against sector and market returns, and reports
likely driver categories, competing explanations, confidence, and an
unexplained fraction. Article metadata is evidence—not proof of causality.

## Temporal retrieval and agent memory

The offline retrieval baseline uses deterministic hashed token vectors and a
small transparent finance lexicon. It is useful for testing temporal filters,
ranking, persistence, and tool contracts without downloading a model; it is
not presented as a replacement for FinBERT or BGE embeddings.

```bash
python3 -m tr8d index-evidence data/documents/aapl-news.jsonl --database var/tr8d.db
python3 -m tr8d search-evidence --query 'raised guidance earnings' --symbol AAPL \
  --decision-time 2026-08-13T13:20:00+00:00 --database var/tr8d.db

python3 -m tr8d write-memory --agent pattern --kind episodic \
  --text 'Positive guidance was followed by a weak close' \
  --created-at 2026-08-13T21:00:00+00:00 --available-at 2026-08-13T21:00:00+00:00
python3 -m tr8d search-memory --agent pattern --query 'guidance weak close' \
  --decision-time 2026-08-14T13:20:00+00:00
```

Evidence retrieval enforces `available_at <= decision_time`, limits repeated
sources, and requires a matching symbol. Memory retrieval additionally requires
an exact agent namespace, preventing another strategy's experience or a future
outcome from entering the current decision context. Semantic lessons require
at least three distinct supporting decision IDs, preventing a single lucky
trade from being promoted into durable strategy knowledge.

To replay a real CSV:

```bash
python3 -m tr8d replay prices.csv --database var/tr8d.db
```

Expected columns:

```text
symbol,date,open,close,available_at
```

`available_at` is optional. If omitted, a row becomes available only after its
market close. Dates must use ISO format. The replay refuses duplicated rows,
non-positive prices, inconsistent timestamps, or insufficient history.

## Safety boundary

The package has no brokerage adapter, credentials, order-routing endpoint, or
arbitrary network/SQL tool. The predictor proposes a paper action; the risk
governor can reject it; only the simulator can mutate wallet state.

## Next milestones

1. Replace synthetic/CSV-only ingestion with versioned Stooq imports.
2. Add PostgreSQL and immutable data manifests.
3. Add XGBoost and calibration once its walk-forward report beats the baseline.
4. Add SEC/GDELT ingestion and the post-close large-move explainer.
5. Add pgvector memory/RAG, then a structured-output LLM behind the same risk
   boundary.
