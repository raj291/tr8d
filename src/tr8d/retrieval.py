from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .documents import Document

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
POSITIVE_WORDS = frozenset({
    "beat", "beats", "growth", "higher", "improved", "profit", "record", "raised",
    "strong", "surge", "upgrade", "upgraded", "wins",
})
NEGATIVE_WORDS = frozenset({
    "cut", "decline", "downgrade", "downgraded", "fraud", "investigation", "loss",
    "miss", "misses", "recall", "risk", "subpoena", "warning", "weak",
})


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class Sentiment:
    label: str
    score: float
    positive_hits: int
    negative_hits: int


def lexical_sentiment(text: str) -> Sentiment:
    tokens = tokenize(text)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (positive - negative) / max(1, positive + negative)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return Sentiment(label, float(score), positive, negative)


class HashEmbedding:
    """Deterministic offline retrieval baseline; not a semantic finance model."""

    def __init__(self, dimensions: int = 128):
        if dimensions < 16:
            raise ValueError("embedding dimensions must be at least 16")
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        vector = np.zeros(self.dimensions, dtype=float)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    available_at: datetime
    symbols: tuple[str, ...]
    source_type: str
    source_domain: str
    event_tags: tuple[str, ...]
    sentiment_label: str
    sentiment_score: float
    embedding: tuple[float, ...]


def chunk_document(
    document: Document, embedder: HashEmbedding | None = None, words_per_chunk: int = 120, overlap: int = 20,
) -> list[DocumentChunk]:
    if overlap < 0 or words_per_chunk <= overlap:
        raise ValueError("chunk size must exceed non-negative overlap")
    embedder = embedder or HashEmbedding()
    words = (document.content or document.title).split()
    chunks = []
    step = words_per_chunk - overlap
    for index, start in enumerate(range(0, len(words), step)):
        text = " ".join(words[start:start + words_per_chunk]).strip()
        if not text:
            continue
        sentiment = lexical_sentiment(text)
        chunk_id = hashlib.sha256(f"{document.id}|{index}|{text}".encode()).hexdigest()
        chunks.append(DocumentChunk(
            id=chunk_id, document_id=document.id, chunk_index=index, text=text,
            available_at=document.available_at, symbols=document.symbols,
            source_type=document.source_type, source_domain=str(document.metadata.get("domain", "")),
            event_tags=document.event_tags, sentiment_label=sentiment.label,
            sentiment_score=sentiment.score, embedding=embedder.embed(text),
        ))
        if start + words_per_chunk >= len(words):
            break
    return chunks


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


def rank_chunks(
    chunks: list[DocumentChunk], query: str, symbol: str, decision_time: datetime,
    limit: int = 6, per_source_limit: int = 2, embedder: HashEmbedding | None = None,
) -> list[RetrievalResult]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    embedder = embedder or HashEmbedding()
    query_vector = np.asarray(embedder.embed(query))
    query_terms = set(tokenize(query))
    scored = []
    for chunk in chunks:
        if chunk.available_at > decision_time or symbol.upper() not in chunk.symbols:
            continue
        vector_score = float(query_vector @ np.asarray(chunk.embedding))
        chunk_terms = set(tokenize(chunk.text))
        lexical_score = len(query_terms & chunk_terms) / max(1, len(query_terms))
        source_bonus = 0.12 if chunk.source_type == "sec" else 0.0
        event_bonus = min(0.08, len(chunk.event_tags) * 0.02)
        scored.append(RetrievalResult(chunk, 0.62 * vector_score + 0.3 * lexical_score + source_bonus + event_bonus))
    scored.sort(key=lambda result: (-result.score, result.chunk.available_at, result.chunk.id))
    selected, source_counts = [], {}
    for result in scored:
        source = result.chunk.source_domain or result.chunk.source_type
        if source_counts.get(source, 0) >= per_source_limit:
            continue
        selected.append(result)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= limit:
            break
    return selected
