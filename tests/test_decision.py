import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tr8d.decision import orchestrate_decision, validate_proposal
from tr8d.documents import Document
from tr8d.domain import Prediction, Wallet
from tr8d.retrieval import chunk_document
from tr8d.store import Store


def evidence_chunk(available_at: datetime):
    document = Document(
        id="doc-1", source_type="sec", external_id="filing-1",
        title="Apple raised guidance after strong earnings",
        url="https://sec.example/filing", published_at=available_at,
        available_at=available_at, ingested_at=available_at, symbols=("AAPL",),
        event_tags=("guidance", "earnings"), metadata={},
        content="Apple raised guidance after strong earnings growth beat expectations.",
    )
    return document, chunk_document(document)[0]


class MalformedProvider:
    name = "malformed"

    def structured_response(self, context):
        return {"action": "BUY"}


class DecisionTests(unittest.TestCase):
    def test_valid_evidence_backed_buy_reaches_risk_governor(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.7, 0.005),
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [],
        )
        self.assertEqual(outcome.proposal.action, "BUY")
        self.assertTrue(outcome.approved)
        self.assertEqual(outcome.risk.approved_notional, 2.0)
        self.assertEqual(
            [trace.name for trace in outcome.tool_traces],
            ["get_wallet", "get_price_model_prediction", "search_evidence", "search_agent_memory", "get_risk_assessment"],
        )

    def test_future_evidence_produces_hold(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, future = evidence_chunk(decision_time + timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.75, 0.01),
            Wallet(10.0, {}), {"AAPL": 100.0}, [future], [],
        )
        self.assertEqual(outcome.proposal.action, "HOLD")
        self.assertFalse(outcome.approved)
        self.assertFalse(outcome.fallback_used)

    def test_malformed_provider_fails_closed(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.75, 0.01),
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [], MalformedProvider(),
        )
        self.assertEqual(outcome.proposal.action, "HOLD")
        self.assertFalse(outcome.approved)
        self.assertTrue(outcome.fallback_used)

    def test_low_data_quality_blocks_otherwise_valid_trade(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.75, 0.01),
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [], data_quality=0.4,
        )
        self.assertFalse(outcome.approved)
        self.assertEqual(outcome.gate_reason, "data quality below gate")

    def test_proposal_schema_rejects_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_proposal({"symbol": "AAPL", "extra": True}, "AAPL")

    def test_missing_mark_fails_before_provider(self):
        with self.assertRaisesRegex(ValueError, "mark"):
            orchestrate_decision(
                "pattern", "AAPL", datetime(2026, 8, 14, 13, 20, tzinfo=UTC),
                Prediction(0.7, 0.005), Wallet(10.0, {}), {}, [], [],
            )

    def test_decision_and_tool_traces_are_persisted(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        document, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.7, 0.005),
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [],
        )
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "audit.db")
            store.documents([document])
            store.chunks([chunk])
            store.decision_outcome(outcome)
            store.commit()
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM decision_runs").fetchone()[0], 1)
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0],
                len(outcome.tool_traces),
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
