import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tr8d.decision import (
    DeterministicDecisionProvider,
    LayaDecisionProvider,
    orchestrate_decision,
    validate_proposal,
)
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
            provider=DeterministicDecisionProvider(),
        )
        self.assertEqual(outcome.proposal.action, "BUY")
        self.assertTrue(outcome.approved)
        self.assertEqual(outcome.risk.approved_notional, 2.0)
        self.assertEqual(
            [trace.name for trace in outcome.tool_traces],
            [
                "get_wallet", "get_price_model_prediction", "search_evidence",
                "search_agent_memory", "structured_decision_provider", "get_risk_assessment",
            ],
        )

    def test_laya_is_default_and_offline_testable_with_injected_runner(self):
        class FakeRunner:
            def decide(self, state, schema, return_details):
                self.state = state
                return SimpleNamespace(
                    values={"action": "A"}, confidence={"action": 0.82},
                    probabilities={"action": {"A": 0.82, "B": 0.05, "C": 0.13}},
                    routing={"model": "fake"},
                )

        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        provider = LayaDecisionProvider(FakeRunner(), model_id="fake")
        with patch("tr8d.decision.default_decision_provider", return_value=provider):
            outcome = orchestrate_decision(
                "pattern", "AAPL", decision_time, Prediction(0.7, 0.005),
                Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [],
            )
        self.assertTrue(outcome.approved)
        self.assertEqual(outcome.proposal.action, "BUY")
        self.assertEqual(outcome.provider_metadata["effective_action"], "BUY")
        self.assertEqual(outcome.provider_metadata["routing"], {"model": "fake"})

    def test_laya_low_confidence_fails_closed(self):
        class FakeRunner:
            def decide(self, state, schema, return_details):
                return SimpleNamespace(
                    values={"action": "A"}, confidence={"action": 0.55},
                    probabilities={"action": {}}, routing=None,
                )

        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, chunk = evidence_chunk(decision_time - timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.7, 0.005),
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [], LayaDecisionProvider(FakeRunner()),
        )
        self.assertEqual(outcome.proposal.action, "HOLD")
        self.assertFalse(outcome.approved)
        self.assertEqual(outcome.provider_metadata["selected_action"], "BUY")
        self.assertEqual(outcome.provider_metadata["effective_action"], "HOLD")
        self.assertEqual(outcome.provider_metadata["fail_closed_reason"], "confidence below threshold")
        self.assertTrue(outcome.provider_metadata["llm_escalation_required"])
        self.assertEqual(
            outcome.provider_metadata["llm_escalation_reason"],
            "laya confidence below threshold",
        )

    def test_future_evidence_produces_hold(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        _, future = evidence_chunk(decision_time + timedelta(hours=1))
        outcome = orchestrate_decision(
            "pattern", "AAPL", decision_time, Prediction(0.75, 0.01),
            Wallet(10.0, {}), {"AAPL": 100.0}, [future], [],
            provider=DeterministicDecisionProvider(),
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
            Wallet(10.0, {}), {"AAPL": 100.0}, [chunk], [],
            provider=DeterministicDecisionProvider(), data_quality=0.4,
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
            provider=DeterministicDecisionProvider(),
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
