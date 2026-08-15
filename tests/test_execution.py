import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from tr8d.decision import orchestrate_decision
from tr8d.documents import Document
from tr8d.domain import Prediction, Wallet
from tr8d.execution import ExecutionRejected, PaperExecutionEngine, WalletAlreadyExists
from tr8d.retrieval import chunk_document
from tr8d.store import Store


def approved_outcome(agent_id: str = "pattern"):
    decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
    document = Document(
        id="execution-doc", source_type="sec", external_id="filing",
        title="Apple raised guidance after strong earnings", url="https://sec.example/filing",
        published_at=decision_time - timedelta(hours=1),
        available_at=decision_time - timedelta(hours=1), ingested_at=decision_time,
        symbols=("AAPL",), event_tags=("guidance", "earnings"), metadata={},
        content="Apple raised guidance after strong earnings growth beat expectations.",
    )
    chunk = chunk_document(document)[0]
    outcome = orchestrate_decision(
        agent_id, "AAPL", decision_time, Prediction(0.7, 0.005),
        Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [],
    )
    return document, chunk, outcome


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name) / "audit.db")
        self.engine = PaperExecutionEngine(self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def save_decision(self, agent_id: str = "pattern"):
        document, chunk, outcome = approved_outcome(agent_id)
        self.store.documents([document])
        self.store.chunks([chunk])
        self.store.decision_outcome(outcome)
        self.store.commit()
        return outcome

    def test_wallet_cannot_be_silently_reset(self):
        self.engine.initialize_wallet("pattern", 10.0)
        with self.assertRaisesRegex(WalletAlreadyExists, "already exists"):
            self.engine.initialize_wallet("pattern", 20.0)
        self.assertEqual(self.engine.load_wallet("pattern").cash, 10.0)

    def test_approved_decision_executes_exactly_once(self):
        outcome = self.save_decision()
        self.engine.initialize_wallet("pattern", 10.0)
        executed_at = datetime(2026, 8, 14, 13, 30, tzinfo=UTC)
        first = self.engine.execute_decision(outcome.context.decision_id, 101.0, executed_at)
        wallet_after_first = self.engine.load_wallet("pattern")
        second = self.engine.execute_decision(outcome.context.decision_id, 101.0, executed_at)
        self.assertFalse(first.idempotent_replay)
        self.assertTrue(second.idempotent_replay)
        self.assertEqual(wallet_after_first, self.engine.load_wallet("pattern"))
        self.assertAlmostEqual(wallet_after_first.cash, 8.0)
        self.assertGreater(wallet_after_first.quantities["AAPL"], 0)

    def test_execution_rechecks_risk_against_current_wallet(self):
        outcome = self.save_decision()
        self.engine.initialize_wallet("pattern", 1.0)
        receipt = self.engine.execute_decision(
            outcome.context.decision_id, 101.0, datetime(2026, 8, 14, 13, 30, tzinfo=UTC)
        )
        self.assertAlmostEqual(receipt.approved_notional, 0.35)
        self.assertAlmostEqual(self.engine.load_wallet("pattern").cash, 0.65)

    def test_invalid_execution_rolls_back_wallet(self):
        outcome = self.save_decision()
        self.engine.initialize_wallet("pattern", 10.0)
        before = self.engine.load_wallet("pattern")
        with self.assertRaisesRegex(ValueError, "open price"):
            self.engine.execute_decision(outcome.context.decision_id, 0)
        self.assertEqual(self.engine.load_wallet("pattern"), before)

    def test_execution_cannot_precede_decision(self):
        outcome = self.save_decision()
        self.engine.initialize_wallet("pattern", 10.0)
        with self.assertRaisesRegex(ExecutionRejected, "precede"):
            self.engine.execute_decision(
                outcome.context.decision_id, 101.0,
                outcome.context.decision_time - timedelta(minutes=1),
            )
        self.assertEqual(self.engine.load_wallet("pattern"), Wallet(10.0, {}))

    def test_unapproved_decision_cannot_execute(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.5, 0.0),
            Wallet(10.0, {}), {"AAPL": 100.0}, [], [],
        )
        self.store.decision_outcome(outcome)
        self.store.commit()
        self.engine.initialize_wallet("pattern", 10.0)
        with self.assertRaisesRegex(ExecutionRejected, "not approved"):
            self.engine.execute_decision(outcome.context.decision_id, 101.0)

    def test_decision_record_is_idempotent(self):
        outcome = self.save_decision()
        self.store.decision_outcome(outcome)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM decision_runs").fetchone()[0], 1)

    def test_post_close_creates_snapshot_and_temporal_memory(self):
        outcome = self.save_decision()
        self.engine.initialize_wallet("pattern", 10.0)
        self.engine.execute_decision(
            outcome.context.decision_id, 101.0, datetime(2026, 8, 14, 13, 30, tzinfo=UTC)
        )
        available_at = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)
        receipt = self.engine.post_close("pattern", date(2026, 8, 14), {"AAPL": 103.0}, available_at)
        memories = self.store.load_memories()
        self.assertEqual(receipt.memories_created, 1)
        self.assertGreater(receipt.equity, 10.0)
        self.assertEqual(memories[0].available_at, available_at)
        replay = self.engine.post_close("pattern", date(2026, 8, 14), {"AAPL": 103.0}, available_at)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(len(self.store.load_memories()), 1)


if __name__ == "__main__":
    unittest.main()
