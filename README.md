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
