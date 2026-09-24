import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from tr8d.decision import (
    DeterministicDecisionProvider,
    LayaDecisionProvider,
    orchestrate_decision,
)
from tr8d.domain import Prediction, Wallet
from tr8d.store import Store
from tr8d.teacher import TeacherReview, export_pending_reviews, process_pending_reviews


class FakeTeacher:
    name = "fake-llm-teacher"

    def __init__(self):
        self.calls = 0

    def review(self, request):
        self.calls += 1
        return TeacherReview(
            action_probabilities={"A": 0.2, "B": 0.1, "C": 0.7},
            preferred_action="C",
            reasoning=f"Background review of {request['decision_id']}",
        )


class BackgroundTeacherTests(unittest.TestCase):
    @staticmethod
    def create_outcome(provider):
        return orchestrate_decision(
            "pattern", "AAPL", datetime(2026, 8, 14, 13, 20, tzinfo=UTC),
            Prediction(0.7, 0.005), Wallet(10.0, {}), {"AAPL": 100.0}, [], [],
            provider=provider,
        )

    def test_uncertain_laya_queues_review_until_worker_runs(self):
        class UncertainRunner:
            def decide(self, state, schema, return_details):
                return SimpleNamespace(
                    values={"action": "A"}, confidence={"action": 0.42},
                    probabilities={"action": {"A": 0.42, "B": 0.27, "C": 0.31}},
                    routing={"model": "test"},
                )

        outcome = self.create_outcome(LayaDecisionProvider(UncertainRunner(), model_id="test"))
        self.assertEqual(outcome.proposal.action, "HOLD")
        self.assertTrue(outcome.provider_metadata["llm_escalation_required"])
        provider = FakeTeacher()
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "teacher.db")
            store.decision_outcome(outcome)
            store.commit()

            self.assertEqual(provider.calls, 0)
            self.assertEqual(store.llm_review_counts()["PENDING"], 1)
            exported = export_pending_reviews(store, Path(directory) / "jobs.jsonl")
            row = json.loads(exported.read_text(encoding="utf-8"))
            self.assertEqual(row["decision_id"], outcome.context.decision_id)
            self.assertEqual(row["request"]["escalation_reason"], "laya confidence below threshold")
            self.assertEqual(row["request"]["laya_confidence"], 0.42)

            result = process_pending_reviews(store, provider)
            self.assertEqual(result, {"completed": 1, "failed": 0})
            self.assertEqual(provider.calls, 1)
            self.assertEqual(store.llm_review_counts()["COMPLETED"], 1)
            response = store.connection.execute(
                "SELECT response_json FROM llm_review_jobs WHERE decision_id = ?",
                (outcome.context.decision_id,),
            ).fetchone()[0]
            self.assertEqual(json.loads(response)["preferred_action"], "C")
            store.close()

    def test_confident_non_laya_decision_does_not_queue_review(self):
        outcome = self.create_outcome(DeterministicDecisionProvider())
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "teacher.db")
            store.decision_outcome(outcome)
            store.commit()
            self.assertEqual(store.llm_review_counts()["PENDING"], 0)
            store.close()

    def test_confident_laya_does_not_queue_review(self):
        class ConfidentRunner:
            def decide(self, state, schema, return_details):
                return SimpleNamespace(
                    values={"action": "C"}, confidence={"action": 0.81},
                    probabilities={"action": {"A": 0.1, "B": 0.09, "C": 0.81}},
                    routing={"model": "test"},
                )

        outcome = self.create_outcome(LayaDecisionProvider(ConfidentRunner(), model_id="test"))
        self.assertFalse(outcome.provider_metadata["llm_escalation_required"])
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "teacher.db")
            store.decision_outcome(outcome)
            store.commit()
            self.assertEqual(store.llm_review_counts()["PENDING"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
