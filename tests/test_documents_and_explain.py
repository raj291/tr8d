import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from tr8d.documents import (
    EvidenceClient,
    EvidenceFetchError,
    parse_gdelt_articles,
    parse_sec_submissions,
    read_documents,
    write_documents,
)
from tr8d.domain import PriceBar
from tr8d.explain import explain_large_move
from tr8d.store import Store


class DocumentAndExplainTests(unittest.TestCase):
    def test_sec_columnar_payload_preserves_acceptance_time(self):
        payload = {
            "cik": "320193", "name": "Apple Inc.",
            "filings": {"recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-02"], "acceptanceDateTime": ["2026-01-02T12:34:56.000Z"],
                "form": ["8-K"], "primaryDocument": ["aapl-8k.htm"],
            }},
        }
        documents = parse_sec_submissions(payload, "AAPL", datetime(2026, 1, 3, tzinfo=UTC))
        self.assertEqual(documents[0].available_at, datetime(2026, 1, 2, 12, 34, 56, tzinfo=UTC))
        self.assertEqual(documents[0].metadata["form"], "8-K")

    def test_gdelt_round_trip_and_event_tags(self):
        payload = {"articles": [{
            "url": "https://example.com/story", "title": "Apple raises guidance after earnings",
            "seendate": "20260813T120000Z", "domain": "example.com", "language": "English",
        }]}
        documents = parse_gdelt_articles(payload, "AAPL", datetime(2026, 8, 13, 12, 1, tzinfo=UTC))
        self.assertEqual(documents[0].event_tags, ("earnings", "guidance"))
        with tempfile.TemporaryDirectory() as directory:
            path = write_documents(documents, Path(directory) / "docs.jsonl")
            self.assertEqual(read_documents(path), documents)

    def test_move_explanation_excludes_after_close_evidence(self):
        target = date(2026, 8, 13)
        prior = [
            PriceBar("AAPL", target - timedelta(days=30-index), 100, 100.5, datetime.combine(target - timedelta(days=30-index), datetime.min.time(), tzinfo=UTC))
            for index in range(20)
        ]
        stock = PriceBar("AAPL", target, 100, 108, datetime(2026, 8, 13, 21, tzinfo=UTC))
        sector = PriceBar("XLK", target, 100, 102, datetime(2026, 8, 13, 21, tzinfo=UTC))
        market = PriceBar("SPY", target, 100, 101, datetime(2026, 8, 13, 21, tzinfo=UTC))
        before, after = parse_gdelt_articles({"articles": [
            {"url": "https://example.com/before", "title": "Apple raises guidance", "seendate": "20260813T180000Z"},
            {"url": "https://example.com/after", "title": "Apple earnings analysis", "seendate": "20260813T220000Z"},
        ]}, "AAPL", datetime(2026, 8, 14, tzinfo=UTC))
        result = explain_large_move(stock, sector, market, prior, [before, after], "technology")
        self.assertTrue(result.is_large_move)
        self.assertEqual(result.evidence_ids, (before.id,))
        self.assertAlmostEqual(result.idiosyncratic_return, 0.06)

    def test_evidence_fetch_retries_then_fails_cleanly(self):
        client = EvidenceClient("tr8d test@example.com", timeout=0.01, retries=1)
        with patch("tr8d.documents.urlopen", side_effect=TimeoutError), patch("tr8d.documents.clock.sleep") as sleep:
            with self.assertRaisesRegex(EvidenceFetchError, "after 2 attempts"):
                client._json("https://data.sec.gov/test")
            sleep.assert_called_once()

    def test_gdelt_query_requires_temporal_bounds(self):
        client = EvidenceClient("tr8d test@example.com")
        naive_start = datetime(2026, 8, 12, tzinfo=UTC).replace(tzinfo=None)
        naive_end = datetime(2026, 8, 13, tzinfo=UTC).replace(tzinfo=None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            client.gdelt_articles("Apple", "AAPL", naive_start, naive_end)

    def test_documents_and_explanation_are_audited(self):
        payload = {"articles": [{
            "url": "https://example.com/story", "title": "Apple raises guidance",
            "seendate": "20260813T180000Z",
        }]}
        documents = parse_gdelt_articles(payload, "AAPL", datetime(2026, 8, 13, 18, 1, tzinfo=UTC))
        target = date(2026, 8, 13)
        prior = [
            PriceBar("AAPL", target - timedelta(days=30-index), 100, 100.5, datetime.combine(target - timedelta(days=30-index), datetime.min.time(), tzinfo=UTC))
            for index in range(20)
        ]
        bars = [PriceBar(symbol, target, 100, close, datetime(2026, 8, 13, 20, tzinfo=UTC)) for symbol, close in (("AAPL", 108), ("XLK", 102), ("SPY", 101))]
        explanation = explain_large_move(*bars, prior, documents)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "audit.db")
            store.documents(documents)
            store.explanation(explanation)
            store.commit()
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM move_explanations").fetchone()[0], 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
