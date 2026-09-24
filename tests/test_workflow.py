import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tr8d.decision import DeterministicDecisionProvider
from tr8d.workflow import run_agent_demo


class AgentWorkflowTests(unittest.TestCase):
    def test_complete_demo_is_audited_and_temporally_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "agent-demo.db"
            with patch("urllib.request.urlopen", side_effect=AssertionError("network is forbidden")):
                report = run_agent_demo(database, provider=DeterministicDecisionProvider())

            self.assertEqual(report["status"], "COMPLETED")
            self.assertEqual(report["mode"], "paper-only")
            self.assertFalse(report["market_data"]["real_time"])
            self.assertTrue(report["evidence"]["synthetic"])
            self.assertEqual(report["decision"]["action"], "BUY")
            self.assertTrue(report["decision"]["approved"])
            self.assertEqual(report["execution"]["approved_notional"], 2.0)
            self.assertAlmostEqual(report["wallet"]["cash"], 8.0)
            self.assertAlmostEqual(report["wallet"]["equity"], 10.037566394002038)
            self.assertEqual(report["memory"]["visible_before_close"], [])
            self.assertEqual(len(report["memory"]["visible_at_close"]), 1)
            self.assertEqual(report["audit_counts"]["decision_runs"], 1)
            self.assertEqual(report["audit_counts"]["paper_executions"], 1)
            self.assertEqual(report["audit_counts"]["live_portfolio_snapshots"], 1)
            self.assertEqual(report["audit_counts"]["tool_calls"], 6)
            self.assertEqual(report["audit_counts"]["llm_review_jobs"], 1)

    def test_complete_demo_replay_does_not_place_a_second_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "agent-demo.db"
            provider = DeterministicDecisionProvider()
            first = run_agent_demo(database, provider=provider)
            second = run_agent_demo(database, provider=provider)
            self.assertEqual(first, second)
            self.assertEqual(second["audit_counts"]["paper_executions"], 1)
            self.assertEqual(second["audit_counts"]["agent_memories"], 1)


if __name__ == "__main__":
    unittest.main()
