from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime

from .domain import Action, Proposal, Wallet
from .memory import AgentMemory, create_memory
from .risk import RiskGovernor
from .simulator import PaperSimulator
from .store import Store


class ExecutionRejected(RuntimeError):
    pass


class WalletAlreadyExists(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionReceipt:
    decision_id: str
    agent_id: str
    symbol: str
    trading_date: str
    action: str
    approved_notional: float
    fill_price: float
    quantity: float
    wallet_version: int
    idempotent_replay: bool = False


@dataclass(frozen=True)
class PostCloseReceipt:
    agent_id: str
    trading_date: str
    cash: float
    equity: float
    memories_created: int
    idempotent_replay: bool = False


class PaperExecutionEngine:
    def __init__(self, store: Store, simulator: PaperSimulator | None = None, governor: RiskGovernor | None = None):
        self.store = store
        self.simulator = simulator or PaperSimulator()
        self.governor = governor or RiskGovernor()

    def initialize_wallet(self, agent_id: str, initial_cash: float = 10.0) -> Wallet:
        if not agent_id.strip() or initial_cash <= 0:
            raise ValueError("agent and positive initial cash are required")
        connection = self.store.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO live_wallets VALUES (?, ?, ?, 0, ?, ?)",
                (agent_id, initial_cash, initial_cash, now, now),
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise WalletAlreadyExists(f"wallet already exists for agent {agent_id}") from error
        return Wallet(initial_cash, {})

    def load_wallet(self, agent_id: str) -> Wallet:
        row = self.store.connection.execute(
            "SELECT cash FROM live_wallets WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise ExecutionRejected(f"wallet does not exist for agent {agent_id}")
        positions = dict(self.store.connection.execute(
            "SELECT symbol, quantity FROM live_positions WHERE agent_id = ? AND quantity > 0",
            (agent_id,),
        ).fetchall())
        return Wallet(float(row[0]), {symbol: float(quantity) for symbol, quantity in positions.items()})

    def _existing_receipt(self, decision_id: str) -> ExecutionReceipt | None:
        row = self.store.connection.execute(
            """SELECT decision_id, agent_id, symbol, trading_date, action, approved_notional,
            fill_price, quantity, wallet_version FROM paper_executions WHERE decision_id = ?""",
            (decision_id,),
        ).fetchone()
        return ExecutionReceipt(*row, idempotent_replay=True) if row else None

    def execute_decision(self, decision_id: str, open_price: float, executed_at: datetime | None = None) -> ExecutionReceipt:
        if open_price <= 0:
            raise ValueError("open price must be positive")
        executed_at = executed_at or datetime.now(UTC)
        if executed_at.tzinfo is None:
            raise ValueError("executed_at must be timezone-aware")
        connection = self.store.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._existing_receipt(decision_id)
            if existing:
                connection.commit()
                return existing
            row = connection.execute(
                """SELECT agent_id, symbol, decision_time, proposal_json, risk_json, approved,
                context_json FROM decision_runs WHERE decision_id = ?""",
                (decision_id,),
            ).fetchone()
            if row is None:
                raise ExecutionRejected("decision does not exist")
            agent_id, symbol, decision_time, proposal_json, risk_json, approved, context_json = row
            parsed_decision_time = datetime.fromisoformat(decision_time)
            if executed_at < parsed_decision_time:
                raise ExecutionRejected("execution cannot precede the decision")
            if not approved:
                raise ExecutionRejected("decision was not approved")
            proposal_data, original_risk, context = json.loads(proposal_json), json.loads(risk_json), json.loads(context_json)
            wallet = self.load_wallet(agent_id)
            marks = {key: float(value) for key, value in context.get("marks", {}).items()}
            if symbol not in marks:
                raise ExecutionRejected("decision snapshot is missing its completed-data mark")
            proposal = Proposal(
                symbol=symbol, action=Action(proposal_data["action"]),
                notional=float(proposal_data["notional"]), confidence=float(proposal_data["confidence"]),
                reason="; ".join(proposal_data["thesis"]),
            )
            current_risk = self.governor.assess(proposal, wallet, marks)
            if not current_risk.allowed:
                raise ExecutionRejected(f"execution-time risk rejection: {current_risk.reason}")
            notional = min(float(original_risk["approved_notional"]), current_risk.approved_notional)
            updated, fill, quantity = self.simulator.execute(wallet, proposal, notional, open_price)
            version_row = connection.execute(
                "SELECT version FROM live_wallets WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            new_version = int(version_row[0]) + 1
            connection.execute(
                "UPDATE live_wallets SET cash = ?, version = ?, updated_at = ? WHERE agent_id = ?",
                (updated.cash, new_version, executed_at.isoformat(), agent_id),
            )
            for position_symbol, position_quantity in updated.quantities.items():
                if position_quantity <= 1e-12:
                    connection.execute(
                        "DELETE FROM live_positions WHERE agent_id = ? AND symbol = ?",
                        (agent_id, position_symbol),
                    )
                else:
                    connection.execute(
                        """INSERT INTO live_positions(agent_id, symbol, quantity) VALUES (?, ?, ?)
                        ON CONFLICT(agent_id, symbol) DO UPDATE SET quantity = excluded.quantity""",
                        (agent_id, position_symbol, position_quantity),
                    )
            trading_date = parsed_decision_time.date().isoformat()
            connection.execute(
                "INSERT INTO paper_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id, agent_id, symbol, trading_date, proposal.action.value,
                    notional, fill, quantity, executed_at.isoformat(), new_version,
                ),
            )
            connection.commit()
            return ExecutionReceipt(
                decision_id, agent_id, symbol, trading_date, proposal.action.value,
                notional, fill, quantity, new_version,
            )
        except Exception:
            connection.rollback()
            raise

    def post_close(
        self, agent_id: str, trading_date: date, marks: dict[str, float], available_at: datetime,
    ) -> PostCloseReceipt:
        if available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if available_at.date() < trading_date:
            raise ValueError("post-close availability cannot precede the trading date")
        connection = self.store.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                """SELECT cash, equity, marks_json FROM live_portfolio_snapshots
                WHERE agent_id = ? AND trading_date = ?""",
                (agent_id, trading_date.isoformat()),
            ).fetchone()
            if existing:
                if json.loads(existing[2]) != marks:
                    raise ExecutionRejected("post-close snapshot already exists with different marks")
                connection.commit()
                return PostCloseReceipt(agent_id, trading_date.isoformat(), existing[0], existing[1], 0, True)
            wallet = self.load_wallet(agent_id)
            executions = connection.execute(
                """SELECT decision_id, symbol, action, fill_price FROM paper_executions
                WHERE agent_id = ? AND trading_date = ?""",
                (agent_id, trading_date.isoformat()),
            ).fetchall()
            required_symbols = set(wallet.quantities) | {row[1] for row in executions}
            missing = required_symbols - set(marks)
            if missing or any(value <= 0 for value in marks.values()):
                raise ExecutionRejected(f"valid close marks required for all positions: {sorted(missing)}")
            equity = wallet.equity(marks)
            connection.execute(
                "INSERT INTO live_portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id, trading_date.isoformat(), available_at.isoformat(), wallet.cash,
                    equity, json.dumps(wallet.quantities, sort_keys=True), json.dumps(marks, sort_keys=True),
                ),
            )
            memories: list[AgentMemory] = []
            for decision_id, symbol, action, fill_price in executions:
                close_price = marks[symbol]
                signed_return = close_price / fill_price - 1 if action == "BUY" else fill_price / close_price - 1
                text = (
                    f"{action} {symbol} at {fill_price:.6f}; close {close_price:.6f}; "
                    f"same-day signed return {signed_return:+.4%}."
                )
                memory = create_memory(
                    agent_id, "episodic", text, available_at, available_at,
                    min(1.0, 0.4 + abs(signed_return) * 10), (decision_id,),
                )
                self.store.memory(memory)
                memories.append(memory)
            connection.commit()
            return PostCloseReceipt(agent_id, trading_date.isoformat(), wallet.cash, equity, len(memories))
        except Exception:
            connection.rollback()
            raise
