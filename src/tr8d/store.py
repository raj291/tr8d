from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .decision import DecisionOutcome
from .documents import Document
from .domain import PriceBar
from .explain import MoveExplanation
from .memory import AgentMemory
from .retrieval import DocumentChunk

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, started_at TEXT NOT NULL, seed INTEGER NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
  run_id TEXT NOT NULL, symbol TEXT NOT NULL, trading_date TEXT NOT NULL,
  open REAL NOT NULL, close REAL NOT NULL, available_at TEXT NOT NULL,
  PRIMARY KEY (run_id, symbol, trading_date), FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS dataset_manifests (
  run_id TEXT PRIMARY KEY, version TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  manifest_path TEXT NOT NULL, manifest_json TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, agent_id TEXT NOT NULL,
  symbol TEXT NOT NULL, decision_date TEXT NOT NULL, action TEXT NOT NULL,
  confidence REAL NOT NULL, expected_return REAL NOT NULL, notional REAL NOT NULL,
  approved INTEGER NOT NULL, reason TEXT NOT NULL, feature_json TEXT NOT NULL,
  UNIQUE(run_id, agent_id, symbol, decision_date)
);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id INTEGER NOT NULL,
  fill_price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL,
  FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  run_id TEXT NOT NULL, agent_id TEXT NOT NULL, trading_date TEXT NOT NULL,
  cash REAL NOT NULL, equity REAL NOT NULL, positions_json TEXT NOT NULL,
  PRIMARY KEY (run_id, agent_id, trading_date)
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, source_type TEXT NOT NULL, external_id TEXT NOT NULL,
  title TEXT NOT NULL, url TEXT NOT NULL, published_at TEXT NOT NULL,
  available_at TEXT NOT NULL, ingested_at TEXT NOT NULL,
  symbols_json TEXT NOT NULL, event_tags_json TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS move_explanations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, trading_date TEXT NOT NULL,
  explanation_json TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(symbol, trading_date)
);
CREATE TABLE IF NOT EXISTS document_chunks (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
  chunk_text TEXT NOT NULL, available_at TEXT NOT NULL, symbols_json TEXT NOT NULL,
  source_type TEXT NOT NULL, source_domain TEXT NOT NULL, event_tags_json TEXT NOT NULL,
  sentiment_label TEXT NOT NULL, sentiment_score REAL NOT NULL, embedding_json TEXT NOT NULL,
  FOREIGN KEY (document_id) REFERENCES documents(id)
);
CREATE TABLE IF NOT EXISTS agent_memories (
  id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, kind TEXT NOT NULL, memory_text TEXT NOT NULL,
  created_at TEXT NOT NULL, available_at TEXT NOT NULL, importance REAL NOT NULL,
  supporting_decision_ids_json TEXT NOT NULL, embedding_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_runs (
  decision_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, symbol TEXT NOT NULL,
  decision_time TEXT NOT NULL, snapshot_hash TEXT NOT NULL, provider_name TEXT NOT NULL,
  proposal_action TEXT NOT NULL, proposal_json TEXT NOT NULL, risk_json TEXT NOT NULL,
  approved INTEGER NOT NULL, fallback_used INTEGER NOT NULL,
  gate_reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  tool_name TEXT NOT NULL, called_at TEXT NOT NULL, arguments_hash TEXT NOT NULL,
  result_count INTEGER NOT NULL, success INTEGER NOT NULL,
  UNIQUE(decision_id, sequence), FOREIGN KEY (decision_id) REFERENCES decision_runs(decision_id)
);
"""


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def start_run(self, run_id: str, started_at: str, seed: int, source: str) -> None:
        self.connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (run_id, started_at, seed, source))

    def prices(self, run_id: str, bars: list[PriceBar]) -> None:
        self.connection.executemany(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, b.symbol, b.trading_date.isoformat(), b.open, b.close, b.available_at.isoformat()) for b in bars],
        )

    def manifest(self, run_id: str, version: str, digest: str, path: str, manifest_json: str) -> None:
        self.connection.execute(
            "INSERT INTO dataset_manifests VALUES (?, ?, ?, ?, ?)",
            (run_id, version, digest, path, manifest_json),
        )

    def decision(self, values: tuple) -> int:
        cursor = self.connection.execute(
            """INSERT INTO decisions
            (run_id, agent_id, symbol, decision_date, action, confidence, expected_return,
             notional, approved, reason, feature_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        return int(cursor.lastrowid)

    def trade(self, decision_id: int, fill: float, quantity: float, notional: float) -> None:
        self.connection.execute("INSERT INTO trades(decision_id, fill_price, quantity, notional) VALUES (?, ?, ?, ?)", (decision_id, fill, quantity, notional))

    def snapshot(self, run_id: str, agent_id: str, trading_date: str, cash: float, equity: float, positions: dict[str, float]) -> None:
        self.connection.execute("INSERT INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?)", (run_id, agent_id, trading_date, cash, equity, json.dumps(positions, sort_keys=True)))

    def documents(self, documents: list[Document]) -> None:
        self.connection.executemany(
            """INSERT OR REPLACE INTO documents
            (id, source_type, external_id, title, url, published_at, available_at, ingested_at,
             symbols_json, event_tags_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                item.id, item.source_type, item.external_id, item.title, item.url,
                item.published_at.isoformat(), item.available_at.isoformat(), item.ingested_at.isoformat(),
                json.dumps(item.symbols), json.dumps(item.event_tags), json.dumps(item.metadata, sort_keys=True),
            ) for item in documents],
        )

    def explanation(self, explanation: MoveExplanation) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO move_explanations
            (symbol, trading_date, explanation_json, created_at) VALUES (?, ?, ?, ?)""",
            (
                explanation.symbol, explanation.trading_date,
                json.dumps(explanation.as_dict(), sort_keys=True), datetime.now(UTC).isoformat(),
            ),
        )

    def chunks(self, chunks: list[DocumentChunk]) -> None:
        self.connection.executemany(
            """INSERT OR REPLACE INTO document_chunks
            (id, document_id, chunk_index, chunk_text, available_at, symbols_json,
             source_type, source_domain, event_tags_json, sentiment_label, sentiment_score,
             embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                chunk.id, chunk.document_id, chunk.chunk_index, chunk.text,
                chunk.available_at.isoformat(), json.dumps(chunk.symbols), chunk.source_type,
                chunk.source_domain, json.dumps(chunk.event_tags), chunk.sentiment_label,
                chunk.sentiment_score, json.dumps(chunk.embedding),
            ) for chunk in chunks],
        )

    def load_chunks(self) -> list[DocumentChunk]:
        rows = self.connection.execute(
            """SELECT id, document_id, chunk_index, chunk_text, available_at, symbols_json,
            source_type, source_domain, event_tags_json, sentiment_label, sentiment_score,
            embedding_json FROM document_chunks"""
        ).fetchall()
        return [DocumentChunk(
            id=row[0], document_id=row[1], chunk_index=row[2], text=row[3],
            available_at=datetime.fromisoformat(row[4]), symbols=tuple(json.loads(row[5])),
            source_type=row[6], source_domain=row[7], event_tags=tuple(json.loads(row[8])),
            sentiment_label=row[9], sentiment_score=row[10], embedding=tuple(json.loads(row[11])),
        ) for row in rows]

    def memory(self, memory: AgentMemory) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO agent_memories
            (id, agent_id, kind, memory_text, created_at, available_at, importance,
             supporting_decision_ids_json, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id, memory.agent_id, memory.kind, memory.text,
                memory.created_at.isoformat(), memory.available_at.isoformat(), memory.importance,
                json.dumps(memory.supporting_decision_ids), json.dumps(memory.embedding),
            ),
        )

    def load_memories(self) -> list[AgentMemory]:
        rows = self.connection.execute(
            """SELECT id, agent_id, kind, memory_text, created_at, available_at, importance,
            supporting_decision_ids_json, embedding_json FROM agent_memories"""
        ).fetchall()
        return [AgentMemory(
            id=row[0], agent_id=row[1], kind=row[2], text=row[3],
            created_at=datetime.fromisoformat(row[4]), available_at=datetime.fromisoformat(row[5]),
            importance=row[6], supporting_decision_ids=tuple(json.loads(row[7])),
            embedding=tuple(json.loads(row[8])),
        ) for row in rows]

    def decision_outcome(self, outcome: DecisionOutcome) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO decision_runs
            (decision_id, agent_id, symbol, decision_time, snapshot_hash, provider_name,
             proposal_action, proposal_json, risk_json, approved, fallback_used, gate_reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                outcome.context.decision_id, outcome.context.agent_id, outcome.context.symbol,
                outcome.context.decision_time.isoformat(), outcome.context.snapshot_hash,
                outcome.context.provider_name, outcome.proposal.action,
                json.dumps(asdict(outcome.proposal), sort_keys=True),
                json.dumps(asdict(outcome.risk), sort_keys=True), int(outcome.approved),
                int(outcome.fallback_used), outcome.gate_reason, datetime.now(UTC).isoformat(),
            ),
        )
        self.connection.execute("DELETE FROM tool_calls WHERE decision_id = ?", (outcome.context.decision_id,))
        self.connection.executemany(
            """INSERT INTO tool_calls
            (decision_id, sequence, tool_name, called_at, arguments_hash, result_count, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(
                outcome.context.decision_id, sequence, trace.name, trace.called_at.isoformat(),
                trace.arguments_hash, trace.result_count, int(trace.success),
            ) for sequence, trace in enumerate(outcome.tool_traces)],
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
