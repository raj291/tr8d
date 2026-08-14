import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tr8d.documents import Document
from tr8d.memory import create_memory, rank_memories
from tr8d.retrieval import chunk_document, lexical_sentiment, rank_chunks
from tr8d.store import Store


def document(external_id: str, title: str, available_at: datetime, source: str = "gdelt") -> Document:
    return Document(
        id=external_id, source_type=source, external_id=external_id, title=title,
        url=f"https://example.com/{external_id}", published_at=available_at,
        available_at=available_at, ingested_at=available_at, symbols=("AAPL",),
        event_tags=("guidance",), metadata={"domain": f"{source}.example.com"}, content=title,
    )


class RetrievalAndMemoryTests(unittest.TestCase):
    def test_sentiment_baseline_is_transparent(self):
        result = lexical_sentiment("Strong growth beat but regulatory risk")
        self.assertEqual(result.positive_hits, 3)
        self.assertEqual(result.negative_hits, 1)
        self.assertEqual(result.label, "positive")

    def test_retrieval_excludes_future_documents(self):
        decision_time = datetime(2026, 8, 13, 13, 20, tzinfo=UTC)
        past = document("past", "Apple raised guidance after strong earnings", decision_time - timedelta(hours=1))
        future = document("future", "Apple raised guidance again", decision_time + timedelta(hours=1), "sec")
        results = rank_chunks(
            chunk_document(past) + chunk_document(future), "raised guidance earnings", "AAPL", decision_time,
        )
        self.assertEqual([result.chunk.document_id for result in results], ["past"])

    def test_memory_is_agent_namespaced_and_temporal(self):
        decision_time = datetime(2026, 8, 14, 13, 20, tzinfo=UTC)
        created = decision_time - timedelta(days=1)
        own = create_memory("pattern", "episodic", "guidance preceded weakness", created, created, 0.8)
        other = create_memory("news", "episodic", "guidance preceded strength", created, created, 1.0)
        future = create_memory("pattern", "episodic", "future outcome", created, decision_time + timedelta(hours=1), 1.0)
        results = rank_memories([own, other, future], "pattern", "guidance weakness", decision_time)
        self.assertEqual(results, [own])

    def test_chunks_and_memories_round_trip_through_store(self):
        timestamp = datetime(2026, 8, 13, 12, tzinfo=UTC)
        source = document("doc", "Strong growth and raised guidance", timestamp)
        chunks = chunk_document(source)
        memory = create_memory(
            "pattern", "semantic", "Guidance needs confirmation", timestamp, timestamp,
            0.7, ("dec-1", "dec-2", "dec-3"),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "audit.db")
            store.documents([source])
            store.chunks(chunks)
            store.memory(memory)
            store.commit()
            self.assertEqual(store.load_chunks(), chunks)
            self.assertEqual(store.load_memories(), [memory])
            store.close()

    def test_semantic_memory_needs_multiple_supporting_decisions(self):
        timestamp = datetime(2026, 8, 13, 12, tzinfo=UTC)
        with self.assertRaisesRegex(ValueError, "three supporting"):
            create_memory("pattern", "semantic", "One trade is not a lesson", timestamp, timestamp, 0.8, ("dec-1",))


if __name__ == "__main__":
    unittest.main()
