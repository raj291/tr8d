from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .decision import (
    DecisionProvider,
    default_decision_provider,
    orchestrate_decision,
)
from .documents import Document
from .domain import Prediction, PriceBar
from .execution import PaperExecutionEngine
from .manifest import create_manifest
from .memory import rank_memories
from .retrieval import chunk_document
from .store import Store


class WorkflowAlreadyExists(RuntimeError):
    pass


DEMO_TRADING_DATE = date(2026, 8, 14)
DEMO_DECISION_TIME = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
DEMO_EXECUTION_TIME = datetime(2026, 8, 14, 13, 30, tzinfo=UTC)
DEMO_CLOSE_TIME = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _demo_bars() -> list[PriceBar]:
    return [
        PriceBar(
            "AAPL", date(2026, 8, 13), 99.0, 100.0,
            datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
        ),
        PriceBar("AAPL", DEMO_TRADING_DATE, 101.0, 103.0, DEMO_CLOSE_TIME),
    ]


def _demo_document() -> Document:
    available_at = DEMO_DECISION_TIME - timedelta(hours=1)
    return Document(
        id="demo-aapl-guidance-2026-08-14",
        source_type="synthetic-demo",
        external_id="demo-guidance-1",
        title="Synthetic AAPL guidance evidence",
        url="demo://aapl/guidance",
        published_at=available_at,
        available_at=available_at,
        ingested_at=available_at,
        symbols=("AAPL",),
        event_tags=("guidance", "earnings"),
        metadata={"synthetic": True, "domain": "demo.local"},
        content="AAPL raised guidance after strong earnings growth beat expectations.",
    )


def _workflow_id(agent_id: str, provider_name: str) -> str:
    payload = f"agent-demo-v1|{agent_id}|{provider_name}|{DEMO_TRADING_DATE.isoformat()}"
    return f"workflow-{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _audit_counts(store: Store) -> dict[str, int]:
    tables = (
        "runs", "prices", "dataset_manifests", "documents", "document_chunks",
        "decision_runs", "tool_calls", "live_wallets", "live_positions",
        "paper_executions", "live_portfolio_snapshots", "agent_memories", "workflow_runs",
    )
    return {
        table: int(store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }


def run_agent_demo(
    database: str | Path = ":memory:", agent_id: str = "pattern-demo",
    provider: DecisionProvider | None = None,
) -> dict:
    """Run one complete paper-agent lifecycle using Laya by default."""
    provider = provider or default_decision_provider()
    store = Store(database)
    workflow_id = _workflow_id(agent_id, provider.name)
    existing = store.connection.execute(
        "SELECT status, report_json FROM workflow_runs WHERE id = ?", (workflow_id,)
    ).fetchone()
    if existing:
        if existing[0] == "COMPLETED" and existing[1]:
            report = json.loads(existing[1])
            store.close()
            return report
        store.close()
        raise WorkflowAlreadyExists(f"workflow {workflow_id} already exists with status {existing[0]}")

    started_at = datetime.now(UTC).isoformat()
    store.connection.execute(
        "INSERT INTO workflow_runs VALUES (?, ?, ?, ?, 'RUNNING', ?, NULL, NULL, NULL, NULL, NULL)",
        (workflow_id, agent_id, DEMO_TRADING_DATE.isoformat(), provider.name, started_at),
    )
    store.commit()
    try:
        bars = _demo_bars()
        prior, current = bars
        manifest = create_manifest(bars, "synthetic-agent-demo")
        dataset_run_id = f"dataset-{manifest.content_sha256[:20]}"
        store.start_run(dataset_run_id, started_at, 0, "synthetic-agent-demo")
        store.manifest(
            dataset_run_id, manifest.version, manifest.content_sha256,
            "embedded://agent-demo", json.dumps(asdict(manifest), sort_keys=True),
        )
        store.prices(dataset_run_id, bars)

        document = _demo_document()
        chunks = chunk_document(document)
        store.documents([document])
        store.chunks(chunks)
        store.commit()

        engine = PaperExecutionEngine(store)
        engine.initialize_wallet(agent_id, 10.0)
        wallet_before = engine.load_wallet(agent_id)
        outcome = orchestrate_decision(
            agent_id=agent_id,
            symbol="AAPL",
            decision_time=DEMO_DECISION_TIME,
            prediction=Prediction(0.70, 0.005),
            wallet=wallet_before,
            marks={"AAPL": prior.close},
            chunks=store.load_chunks(),
            memories=store.load_memories(),
            provider=provider,
            data_reference={
                "dataset_run_id": dataset_run_id,
                "manifest_sha256": manifest.content_sha256,
                "decision_mark_date": prior.trading_date.isoformat(),
                "execution_bar_date": current.trading_date.isoformat(),
            },
        )
        store.decision_outcome(outcome)
        store.commit()

        execution = None
        if outcome.approved:
            execution = engine.execute_decision(
                outcome.context.decision_id, current.open, DEMO_EXECUTION_TIME,
            )
        close = engine.post_close(
            agent_id, DEMO_TRADING_DATE, {"AAPL": current.close}, current.available_at,
        )
        wallet_after = engine.load_wallet(agent_id)
        all_memories = store.load_memories()
        recalled_before_close = rank_memories(
            all_memories, agent_id, "AAPL guidance return", DEMO_CLOSE_TIME - timedelta(microseconds=1),
        )
        recalled_at_close = rank_memories(
            all_memories, agent_id, "AAPL guidance return", DEMO_CLOSE_TIME,
        )
        report = {
            "status": "COMPLETED",
            "mode": "paper-only",
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "provider": {
                "name": provider.name,
                "fallback_used": outcome.fallback_used,
                "metadata": outcome.provider_metadata,
                "laya_installed_version": _installed_version("laya"),
            },
            "market_data": {
                "source": "synthetic-agent-demo",
                "real_time": False,
                "manifest_sha256": manifest.content_sha256,
                "decision_mark": prior.close,
                "open": current.open,
                "close": current.close,
            },
            "evidence": {
                "synthetic": True,
                "document_id": document.id,
                "available_at": document.available_at.isoformat(),
                "retrieved_chunk_ids": [result.chunk.id for result in outcome.context.evidence],
            },
            "decision": {
                "id": outcome.context.decision_id,
                "action": outcome.proposal.action,
                "approved": outcome.approved,
                "approved_notional": outcome.risk.approved_notional,
                "gate_reason": outcome.gate_reason,
                "tool_trace": [trace.name for trace in outcome.tool_traces],
                "data_reference": outcome.context.data_reference,
            },
            "execution": asdict(execution) if execution else None,
            "post_close": asdict(close),
            "wallet": {
                "cash": wallet_after.cash,
                "positions": wallet_after.quantities,
                "equity": wallet_after.equity({"AAPL": current.close}),
            },
            "memory": {
                "created": close.memories_created,
                "visible_before_close": [memory.id for memory in recalled_before_close],
                "visible_at_close": [memory.id for memory in recalled_at_close],
                "texts": [memory.text for memory in all_memories],
            },
            "safety": {
                "broker_adapter": False,
                "real_money": False,
                "risk_rechecked_at_execution": execution is not None,
                "temporal_evidence_gate": True,
            },
            "audit_counts": _audit_counts(store),
        }
        serialized = json.dumps(report, sort_keys=True)
        store.connection.execute(
            """UPDATE workflow_runs SET status = 'COMPLETED', completed_at = ?, decision_id = ?,
            report_json = ? WHERE id = ?""",
            (datetime.now(UTC).isoformat(), outcome.context.decision_id, serialized, workflow_id),
        )
        store.commit()
        return report
    except Exception as error:
        store.connection.rollback()
        store.connection.execute(
            """UPDATE workflow_runs SET status = 'FAILED', completed_at = ?, error_type = ?,
            error_message = ? WHERE id = ?""",
            (datetime.now(UTC).isoformat(), type(error).__name__, str(error), workflow_id),
        )
        store.commit()
        raise
    finally:
        store.close()
