from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .retrieval import HashEmbedding


@dataclass(frozen=True)
class AgentMemory:
    id: str
    agent_id: str
    kind: str
    text: str
    created_at: datetime
    available_at: datetime
    importance: float
    supporting_decision_ids: tuple[str, ...]
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.available_at.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        if self.available_at < self.created_at:
            raise ValueError("memory cannot be available before creation")
        if self.kind not in {"episodic", "semantic"}:
            raise ValueError("memory kind must be episodic or semantic")
        if self.kind == "semantic" and len(set(self.supporting_decision_ids)) < 3:
            raise ValueError("semantic memory requires at least three supporting decisions")
        if not 0 <= self.importance <= 1:
            raise ValueError("memory importance must be between 0 and 1")


def create_memory(
    agent_id: str, kind: str, text: str, created_at: datetime, available_at: datetime,
    importance: float, supporting_decision_ids: tuple[str, ...] = (), embedder: HashEmbedding | None = None,
) -> AgentMemory:
    embedder = embedder or HashEmbedding()
    memory_id = hashlib.sha256(
        f"{agent_id}|{kind}|{text}|{created_at.isoformat()}|{available_at.isoformat()}".encode()
    ).hexdigest()
    return AgentMemory(
        id=memory_id, agent_id=agent_id, kind=kind, text=text, created_at=created_at,
        available_at=available_at, importance=importance,
        supporting_decision_ids=supporting_decision_ids, embedding=embedder.embed(text),
    )


def rank_memories(
    memories: list[AgentMemory], agent_id: str, query: str, decision_time: datetime,
    limit: int = 5, embedder: HashEmbedding | None = None,
) -> list[AgentMemory]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    embedder = embedder or HashEmbedding()
    query_vector = np.asarray(embedder.embed(query))
    eligible = [
        memory for memory in memories
        if memory.agent_id == agent_id and memory.available_at <= decision_time
    ]
    eligible.sort(
        key=lambda memory: (
            -(0.8 * float(query_vector @ np.asarray(memory.embedding)) + 0.2 * memory.importance),
            memory.available_at, memory.id,
        )
    )
    return eligible[:limit]
