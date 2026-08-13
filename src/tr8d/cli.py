from __future__ import annotations

import argparse
import json
import sqlite3

from .data import load_price_csv, synthetic_prices
from .replay import replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-in-time paper-trading research simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run deterministic synthetic replay")
    demo.add_argument("--database", default="var/tr8d.db")
    demo.add_argument("--days", type=int, default=800)
    demo.add_argument("--seed", type=int, default=7)
    real = sub.add_parser("replay", help="replay an OHLC CSV")
    real.add_argument("csv")
    real.add_argument("--database", default="var/tr8d.db")
    inspect = sub.add_parser("inspect", help="show latest run results")
    inspect.add_argument("--database", default="var/tr8d.db")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "demo":
        result = replay(synthetic_prices(args.days, args.seed), args.database, "synthetic", args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "replay":
        result = replay(load_price_csv(args.csv), args.database, args.csv)
        print(json.dumps(result, indent=2, sort_keys=True))
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
