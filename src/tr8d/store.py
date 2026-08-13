from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .domain import PriceBar


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

    def commit(self) -> None:
        self.connection.commit()
