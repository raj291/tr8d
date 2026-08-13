from __future__ import annotations

import argparse
import json
import sqlite3

from .data import load_price_csv, load_stooq_csv, synthetic_prices, write_normalized_csv
from .manifest import create_manifest
from .evaluation import evaluate_models, write_model_report
from .replay import replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-in-time paper-trading research simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run deterministic synthetic replay")
    demo.add_argument("--database", default="var/tr8d.db")
    demo.add_argument("--days", type=int, default=800)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--manifest-directory", default="data/manifests")
    real = sub.add_parser("replay", help="replay an OHLC CSV")
    real.add_argument("csv")
    real.add_argument("--database", default="var/tr8d.db")
    real.add_argument("--manifest-directory", default="data/manifests")
    ingest = sub.add_parser("ingest-stooq", help="normalize downloaded Stooq daily CSV files")
    ingest.add_argument("inputs", nargs="+", metavar="SYMBOL=PATH")
    ingest.add_argument("--output", default="data/processed/stooq_prices.csv")
    ingest.add_argument("--manifest-directory", default="data/manifests")
    evaluate = sub.add_parser("evaluate-models", help="compare calibrated models with temporal folds")
    evaluate.add_argument("csv", nargs="?", help="normalized price CSV; omit to use synthetic data")
    evaluate.add_argument("--demo-days", type=int, default=320)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--output", default="var/model-evaluation.json")
    inspect = sub.add_parser("inspect", help="show latest run results")
    inspect.add_argument("--database", default="var/tr8d.db")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "demo":
        result = replay(synthetic_prices(args.days, args.seed), args.database, "synthetic", args.seed, manifest_directory=args.manifest_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "replay":
        result = replay(load_price_csv(args.csv), args.database, args.csv, manifest_directory=args.manifest_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "ingest-stooq":
        bars = []
        for value in args.inputs:
            if "=" not in value:
                raise SystemExit(f"invalid input {value!r}; expected SYMBOL=PATH")
            symbol, path = value.split("=", 1)
            bars.extend(load_stooq_csv(path, symbol))
        write_normalized_csv(bars, args.output)
        manifest = create_manifest(bars, "stooq").write(args.manifest_directory)
        print(json.dumps({"output": args.output, "manifest": str(manifest), "rows": len(bars)}, indent=2))
    elif args.command == "evaluate-models":
        bars = load_price_csv(args.csv) if args.csv else synthetic_prices(args.demo_days, args.seed)
        source = args.csv or "synthetic"
        report = evaluate_models(bars, seed=args.seed)
        report_path = write_model_report(report, bars, source, args.output)
        print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    else:
        connection = sqlite3.connect(args.database)
        rows = connection.execute(
            """SELECT p.agent_id, p.equity, p.cash, p.trading_date
            FROM portfolio_snapshots p JOIN (
              SELECT agent_id, MAX(trading_date) day FROM portfolio_snapshots GROUP BY agent_id
            ) x ON x.agent_id=p.agent_id AND x.day=p.trading_date ORDER BY p.agent_id"""
        ).fetchall()
        for agent, equity, cash, day in rows:
            print(f"{agent:14} equity=${equity:.4f} cash=${cash:.4f} as_of={day}")


if __name__ == "__main__":
    main()
