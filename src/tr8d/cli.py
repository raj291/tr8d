from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime

from .data import load_price_csv, load_stooq_csv, synthetic_prices, write_normalized_csv
from .decision import orchestrate_decision
from .documents import (
    EvidenceClient,
    EvidenceFetchError,
    read_documents,
    write_documents,
)
from .domain import Prediction, Wallet
from .evaluation import evaluate_models, write_model_report
from .explain import explain_large_move, write_explanation
from .manifest import create_manifest
from .memory import create_memory, rank_memories
from .replay import replay
from .retrieval import chunk_document, rank_chunks
from .store import Store


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
    sec = sub.add_parser("fetch-sec", help="fetch SEC submissions with point-in-time timestamps")
    sec.add_argument("--cik", required=True)
    sec.add_argument("--symbol", required=True)
    sec.add_argument("--user-agent", required=True, help="application name and contact email")
    sec.add_argument("--output", required=True)
    sec.add_argument("--database", default="var/tr8d.db")
    gdelt = sub.add_parser("fetch-gdelt", help="fetch recent GDELT article metadata")
    gdelt.add_argument("--query", required=True)
    gdelt.add_argument("--symbol", required=True)
    gdelt.add_argument("--start", required=True, help="ISO timestamp")
    gdelt.add_argument("--end", required=True, help="ISO timestamp")
    gdelt.add_argument("--user-agent", required=True)
    gdelt.add_argument("--output", required=True)
    gdelt.add_argument("--database", default="var/tr8d.db")
    explain = sub.add_parser("explain-move", help="produce evidence-backed post-close move decomposition")
    explain.add_argument("prices")
    explain.add_argument("documents")
    explain.add_argument("--symbol", required=True)
    explain.add_argument("--sector", required=True)
    explain.add_argument("--market", required=True)
    explain.add_argument("--date", required=True)
    explain.add_argument("--output", required=True)
    explain.add_argument("--database", default="var/tr8d.db")
    index = sub.add_parser("index-evidence", help="chunk, score, and index evidence")
    index.add_argument("documents")
    index.add_argument("--database", default="var/tr8d.db")
    search = sub.add_parser("search-evidence", help="retrieve point-in-time evidence")
    search.add_argument("--query", required=True)
    search.add_argument("--symbol", required=True)
    search.add_argument("--decision-time", required=True)
    search.add_argument("--database", default="var/tr8d.db")
    remember = sub.add_parser("write-memory", help="write an agent-namespaced memory")
    remember.add_argument("--agent", required=True)
    remember.add_argument("--kind", choices=("episodic", "semantic"), required=True)
    remember.add_argument("--text", required=True)
    remember.add_argument("--created-at", required=True)
    remember.add_argument("--available-at", required=True)
    remember.add_argument("--importance", type=float, default=0.5)
    remember.add_argument("--supporting-decision", action="append", default=[])
    remember.add_argument("--database", default="var/tr8d.db")
    recall = sub.add_parser("search-memory", help="retrieve temporal agent memory")
    recall.add_argument("--agent", required=True)
    recall.add_argument("--query", required=True)
    recall.add_argument("--decision-time", required=True)
    recall.add_argument("--database", default="var/tr8d.db")
    synthesize = sub.add_parser("synthesize-decision", help="build and gate a structured paper-trade proposal")
    synthesize.add_argument("--agent", required=True)
    synthesize.add_argument("--symbol", required=True)
    synthesize.add_argument("--decision-time", required=True)
    synthesize.add_argument("--bull-probability", type=float, required=True)
    synthesize.add_argument("--expected-return", type=float, required=True)
    synthesize.add_argument("--cash", type=float, default=10.0)
    synthesize.add_argument("--positions", default="{}", help="JSON symbol-to-quantity mapping")
    synthesize.add_argument("--marks", required=True, help="JSON symbol-to-price mapping from completed data")
    synthesize.add_argument("--data-quality", type=float, default=1.0)
    synthesize.add_argument("--database", default="var/tr8d.db")
    inspect = sub.add_parser("inspect", help="show latest run results")
    inspect.add_argument("--database", default="var/tr8d.db")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
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
        elif args.command == "fetch-sec":
            documents = EvidenceClient(args.user_agent).sec_submissions(args.cik, args.symbol)
            path = write_documents(documents, args.output)
            store = Store(args.database)
            store.documents(documents)
            store.commit()
            print(json.dumps({"output": str(path), "documents": len(documents)}, indent=2))
        elif args.command == "fetch-gdelt":
            client = EvidenceClient(args.user_agent)
            documents = client.gdelt_articles(
                args.query, args.symbol, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
            )
            path = write_documents(documents, args.output)
            store = Store(args.database)
            store.documents(documents)
            store.commit()
            print(json.dumps({"output": str(path), "documents": len(documents)}, indent=2))
        elif args.command == "explain-move":
            bars = load_price_csv(args.prices)
            target_date = datetime.fromisoformat(args.date).date()
            by_key = {(bar.symbol, bar.trading_date): bar for bar in bars}
            history = [bar for bar in bars if bar.symbol == args.symbol and bar.trading_date < target_date]
            explanation = explain_large_move(
                by_key[(args.symbol, target_date)], by_key[(args.sector, target_date)],
                by_key[(args.market, target_date)], history, read_documents(args.documents), args.sector,
            )
            path = write_explanation(explanation, args.output)
            store = Store(args.database)
            store.explanation(explanation)
            store.commit()
            print(json.dumps({"output": str(path), **explanation.as_dict()}, indent=2, sort_keys=True))
        elif args.command == "index-evidence":
            documents = read_documents(args.documents)
            chunks = [chunk for document in documents for chunk in chunk_document(document)]
            store = Store(args.database)
            store.documents(documents)
            store.chunks(chunks)
            store.commit()
            print(json.dumps({"documents": len(documents), "chunks": len(chunks)}, indent=2))
        elif args.command == "search-evidence":
            store = Store(args.database)
            results = rank_chunks(
                store.load_chunks(), args.query, args.symbol,
                datetime.fromisoformat(args.decision_time),
            )
            print(json.dumps([{
                "chunk_id": result.chunk.id, "document_id": result.chunk.document_id,
                "score": round(result.score, 8), "text": result.chunk.text,
                "available_at": result.chunk.available_at.isoformat(),
                "sentiment": result.chunk.sentiment_label,
            } for result in results], indent=2))
        elif args.command == "write-memory":
            memory = create_memory(
                args.agent, args.kind, args.text, datetime.fromisoformat(args.created_at),
                datetime.fromisoformat(args.available_at), args.importance,
                tuple(args.supporting_decision),
            )
            store = Store(args.database)
            store.memory(memory)
            store.commit()
            print(json.dumps({"memory_id": memory.id, "agent_id": memory.agent_id}, indent=2))
        elif args.command == "search-memory":
            store = Store(args.database)
            memories = rank_memories(
                store.load_memories(), args.agent, args.query,
                datetime.fromisoformat(args.decision_time),
            )
            print(json.dumps([{
                "memory_id": memory.id, "kind": memory.kind, "text": memory.text,
                "available_at": memory.available_at.isoformat(), "importance": memory.importance,
            } for memory in memories], indent=2))
        elif args.command == "synthesize-decision":
            store = Store(args.database)
            outcome = orchestrate_decision(
                agent_id=args.agent, symbol=args.symbol.upper(),
                decision_time=datetime.fromisoformat(args.decision_time),
                prediction=Prediction(args.bull_probability, args.expected_return),
                wallet=Wallet(args.cash, {key.upper(): float(value) for key, value in json.loads(args.positions).items()}),
                marks={key.upper(): float(value) for key, value in json.loads(args.marks).items()},
                chunks=store.load_chunks(), memories=store.load_memories(),
                data_quality=args.data_quality,
            )
            store.decision_outcome(outcome)
            store.commit()
            print(json.dumps({
                "decision_id": outcome.context.decision_id,
                "snapshot_hash": outcome.context.snapshot_hash,
                "proposal": asdict(outcome.proposal),
                "risk": asdict(outcome.risk),
                "approved": outcome.approved,
                "gate_reason": outcome.gate_reason,
                "provider": outcome.context.provider_name,
                "fallback_used": outcome.fallback_used,
                "tools": [trace.name for trace in outcome.tool_traces],
            }, indent=2, sort_keys=True))
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
    except EvidenceFetchError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
