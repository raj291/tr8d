from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, Literal, Protocol

from .domain import Action, Prediction, Proposal, RiskDecision, Wallet
from .memory import AgentMemory, rank_memories
from .retrieval import DocumentChunk, RetrievalResult, rank_chunks
from .risk import RiskGovernor


@dataclass(frozen=True)
class ToolTrace:
    name: str
    called_at: datetime
    arguments_hash: str
    result_count: int
    success: bool


@dataclass(frozen=True)
class DecisionContext:
    decision_id: str
    agent_id: str
    symbol: str
    decision_time: datetime
    prediction: Prediction
    wallet: Wallet
    marks: dict[str, float]
    evidence: tuple[RetrievalResult, ...]
    memories: tuple[AgentMemory, ...]
    data_quality: float
    estimated_friction_bps: float
    snapshot_hash: str
    provider_name: str
    data_reference: dict[str, str]


@dataclass(frozen=True)
class StructuredProposal:
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    notional: float
    confidence: float
    thesis: tuple[str, ...]
    counter_thesis: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    expected_direction: Literal["BULL", "BEAR", "FLAT"]
    invalidation_conditions: tuple[str, ...]


@dataclass(frozen=True)
class DecisionOutcome:
    context: DecisionContext
    proposal: StructuredProposal
    risk: RiskDecision
    approved: bool
    gate_reason: str
    tool_traces: tuple[ToolTrace, ...]
    fallback_used: bool
    provider_metadata: dict[str, Any] = dataclass_field(default_factory=dict)


class DecisionProvider(Protocol):
    name: str

    def structured_response(self, context: DecisionContext) -> dict: ...


class DecisionProviderUnavailable(RuntimeError):
    pass


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class DecisionTools:
    """Small allowlisted tool surface used to construct one immutable context."""

    def __init__(
        self, chunks: list[DocumentChunk], memories: list[AgentMemory], wallet: Wallet,
        prediction: Prediction, marks: dict[str, float], risk_governor: RiskGovernor | None = None,
    ):
        self.chunks = chunks
        self.memories = memories
        self.wallet = wallet
        self.prediction = prediction
        self.marks = marks
        self.risk_governor = risk_governor or RiskGovernor()
        self.traces: list[ToolTrace] = []

    def _trace(self, name: str, arguments: dict, count: int, success: bool = True) -> None:
        self.traces.append(ToolTrace(name, datetime.now(UTC), _hash_payload(arguments), count, success))

    def get_wallet(self) -> Wallet:
        self._trace("get_wallet", {}, 1)
        return self.wallet

    def get_price_model_prediction(self, symbol: str, decision_time: datetime) -> Prediction:
        self._trace("get_price_model_prediction", {"symbol": symbol, "decision_time": decision_time}, 1)
        return self.prediction

    def search_evidence(self, query: str, symbol: str, decision_time: datetime) -> list[RetrievalResult]:
        results = rank_chunks(self.chunks, query, symbol, decision_time)
        self._trace("search_evidence", {"query": query, "symbol": symbol, "decision_time": decision_time}, len(results))
        return results

    def search_agent_memory(self, agent_id: str, query: str, decision_time: datetime) -> list[AgentMemory]:
        results = rank_memories(self.memories, agent_id, query, decision_time)
        self._trace("search_agent_memory", {"agent_id": agent_id, "query": query, "decision_time": decision_time}, len(results))
        return results

    def get_risk_assessment(self, proposal: StructuredProposal) -> RiskDecision:
        domain_proposal = Proposal(
            symbol=proposal.symbol, action=Action(proposal.action), notional=proposal.notional,
            confidence=proposal.confidence, reason="; ".join(proposal.thesis),
        )
        result = self.risk_governor.assess(domain_proposal, self.wallet, self.marks)
        self._trace("get_risk_assessment", {"proposal": asdict(proposal)}, 1)
        return result


class DeterministicDecisionProvider:
    """Fail-closed provider used until a local/API LLM passes the same gates."""

    name = "deterministic-v1"

    def structured_response(self, context: DecisionContext) -> dict:
        probability = context.prediction.bull_probability
        negative_evidence = sum(result.chunk.sentiment_score < 0 for result in context.evidence)
        positive_evidence = sum(result.chunk.sentiment_score > 0 for result in context.evidence)
        held_value = context.wallet.quantities.get(context.symbol, 0.0) * context.marks.get(context.symbol, 0.0)
        friction = context.estimated_friction_bps / 10_000
        if (
            probability >= 0.60 and context.prediction.expected_return > 2 * friction
            and context.evidence and negative_evidence == 0
        ):
            action, direction, notional = "BUY", "BULL", 2.0
            thesis = ("Price-model confidence clears the entry gate.", "Retrieved evidence is not negatively scored.")
        elif probability <= 0.40 and held_value >= 0.25 and positive_evidence == 0:
            action, direction, notional = "SELL", "BEAR", min(2.0, held_value)
            thesis = ("Price-model bearish probability clears the exit gate.",)
        else:
            action, direction, notional = "HOLD", "FLAT", 0.0
            thesis = ("Available edge or evidence is insufficient.",)
        return {
            "symbol": context.symbol, "action": action, "notional": notional,
            "confidence": max(probability, 1 - probability), "thesis": thesis,
            "counter_thesis": ("The numerical relationship may not persist in the current regime.",),
            "evidence_ids": tuple(result.chunk.id for result in context.evidence[:3]),
            "expected_direction": direction,
            "invalidation_conditions": ("New contradictory filing or news becomes available.",),
        }


class LayaDecisionProvider:
    """Default advisory decision maker; deterministic gates retain final authority."""

    name = "laya-v1"
    _schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["A", "B", "C"],
                "description": "Choose A for BUY, B for SELL, or C for HOLD.",
            },
        },
    }

    def __init__(self, runner: Any, confidence_threshold: float = 0.60, model_id: str = "auto"):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Laya confidence threshold must be between 0 and 1")
        self.runner = runner
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        try:
            package_version = version("laya")
        except PackageNotFoundError:
            package_version = "not-installed"
        self.audit_metadata: dict[str, Any] = {
            "package_version": package_version,
            "model_id": model_id,
            "confidence_threshold": confidence_threshold,
        }

    def structured_response(self, context: DecisionContext) -> dict:
        state = {
            "instruction": "Choose a conservative paper-trading action; C is the safe default.",
            "symbol": context.symbol,
            "bull_probability": context.prediction.bull_probability,
            "expected_return": context.prediction.expected_return,
            "cash": context.wallet.cash,
            "held_quantity": context.wallet.quantities.get(context.symbol, 0.0),
            "evidence": [
                {
                    "text": result.chunk.text,
                    "sentiment": result.chunk.sentiment_label,
                    "available_at": result.chunk.available_at.isoformat(),
                }
                for result in context.evidence[:3]
            ],
        }
        result = self.runner.decide(state, schema=self._schema, return_details=True)
        label = result.values.get("action", "C")
        confidence = float(result.confidence.get("action", 0.0))
        labels = {"A": "BUY", "B": "SELL", "C": "HOLD"}
        if label not in labels:
            raise ValueError("Laya returned an unsupported action label")
        selected_action = labels[label]
        action = selected_action
        fail_closed_reason = None
        if confidence < self.confidence_threshold:
            fail_closed_reason = "confidence below threshold"
            action = "HOLD"
        elif not context.evidence:
            fail_closed_reason = "no eligible evidence"
            action = "HOLD"
        direction = {"BUY": "BULL", "SELL": "BEAR", "HOLD": "FLAT"}[action]
        self.audit_metadata = {
            **self.audit_metadata,
            "confidence": confidence,
            "selected_label": label,
            "selected_action": selected_action,
            "effective_action": action,
            "fail_closed_reason": fail_closed_reason,
            "probabilities": result.probabilities.get("action", {}),
            "routing": result.routing,
        }
        return {
            "symbol": context.symbol,
            "action": action,
            "notional": 0.0 if action == "HOLD" else 2.0,
            "confidence": confidence,
            "thesis": ("Laya selected a bounded action from point-in-time context.",),
            "counter_thesis": ("The advisory classifier is not validated as a price model.",),
            "evidence_ids": tuple(result.chunk.id for result in context.evidence[:3]) if action != "HOLD" else (),
            "expected_direction": direction,
            "invalidation_conditions": ("New contradictory evidence or a risk-limit breach occurs.",),
        }


def load_laya_provider(
    model_id: str = "auto", device: str | None = None, confidence_threshold: float = 0.60,
) -> LayaDecisionProvider:
    try:
        from laya import Router
    except ImportError as error:
        raise DecisionProviderUnavailable(
            "Laya is the default decision provider; install it with: "
            "python -m pip install laya==0.3.20"
        ) from error
    models = None if model_id == "auto" else {"english": model_id}
    return LayaDecisionProvider(
        Router(models=models, device=device),
        confidence_threshold=confidence_threshold,
        model_id=model_id,
    )


@lru_cache(maxsize=1)
def default_decision_provider() -> LayaDecisionProvider:
    return load_laya_provider()


def validate_proposal(payload: dict, expected_symbol: str) -> StructuredProposal:
    required = {
        "symbol", "action", "notional", "confidence", "thesis", "counter_thesis",
        "evidence_ids", "expected_direction", "invalidation_conditions",
    }
    if set(payload) != required:
        raise ValueError(f"proposal schema mismatch: expected exactly {sorted(required)}")
    for field in ("thesis", "counter_thesis", "evidence_ids", "invalidation_conditions"):
        if not isinstance(payload[field], (list, tuple)) or not all(isinstance(item, str) for item in payload[field]):
            raise ValueError(f"proposal {field} must be a string sequence")
    proposal = StructuredProposal(
        symbol=str(payload["symbol"]).upper(), action=str(payload["action"]).upper(),
        notional=float(payload["notional"]), confidence=float(payload["confidence"]),
        thesis=tuple(payload["thesis"]), counter_thesis=tuple(payload["counter_thesis"]),
        evidence_ids=tuple(payload["evidence_ids"]), expected_direction=str(payload["expected_direction"]).upper(),
        invalidation_conditions=tuple(payload["invalidation_conditions"]),
    )
    if proposal.symbol != expected_symbol.upper():
        raise ValueError("proposal symbol does not match context")
    if proposal.action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("proposal action is invalid")
    if proposal.expected_direction not in {"BULL", "BEAR", "FLAT"}:
        raise ValueError("proposal direction is invalid")
    expected_directions = {"BUY": "BULL", "SELL": "BEAR", "HOLD": "FLAT"}
    if expected_directions[proposal.action] != proposal.expected_direction:
        raise ValueError("proposal action and direction disagree")
    if not 0 <= proposal.confidence <= 1 or proposal.notional < 0:
        raise ValueError("proposal confidence or notional is out of range")
    if proposal.action == "HOLD" and proposal.notional != 0:
        raise ValueError("HOLD notional must be zero")
    if proposal.action != "HOLD" and (proposal.notional <= 0 or not proposal.thesis):
        raise ValueError("trade proposal requires positive notional and thesis")
    if not proposal.counter_thesis or not proposal.invalidation_conditions:
        raise ValueError("proposal requires counter-thesis and invalidation conditions")
    return proposal


def decision_gate(context: DecisionContext, proposal: StructuredProposal) -> tuple[bool, str]:
    if context.data_quality < 0.60:
        return False, "data quality below gate"
    if any(result.chunk.available_at > context.decision_time for result in context.evidence):
        return False, "future evidence detected"
    retrieved_ids = {result.chunk.id for result in context.evidence}
    if not set(proposal.evidence_ids).issubset(retrieved_ids):
        return False, "proposal cites evidence outside retrieved context"
    if proposal.action != "HOLD" and not proposal.evidence_ids:
        return False, "trade proposal requires evidence"
    return True, "decision gate passed"


def orchestrate_decision(
    agent_id: str, symbol: str, decision_time: datetime, prediction: Prediction, wallet: Wallet,
    marks: dict[str, float], chunks: list[DocumentChunk], memories: list[AgentMemory],
    provider: DecisionProvider | None = None, data_quality: float = 1.0,
    estimated_friction_bps: float = 10.0, data_reference: dict[str, str] | None = None,
) -> DecisionOutcome:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not 0 <= prediction.bull_probability <= 1:
        raise ValueError("bull probability must be between 0 and 1")
    if not 0 <= data_quality <= 1:
        raise ValueError("data quality must be between 0 and 1")
    if estimated_friction_bps < 0:
        raise ValueError("estimated friction cannot be negative")
    if wallet.cash < 0:
        raise ValueError("wallet cash cannot be negative")
    held_symbols = {held for held, quantity in wallet.quantities.items() if quantity > 0}
    missing_marks = held_symbols - set(marks)
    invalid_marks = {held for held in held_symbols if held in marks and marks[held] <= 0}
    if missing_marks or invalid_marks:
        raise ValueError(f"positive marks required for all holdings: {sorted(missing_marks | invalid_marks)}")
    if symbol not in marks or marks[symbol] <= 0:
        raise ValueError("a positive completed-data mark is required for the symbol")
    provider = provider or default_decision_provider()
    data_reference = dict(data_reference or {})
    tools = DecisionTools(chunks, memories, wallet, prediction, marks)
    wallet_snapshot = tools.get_wallet()
    model_prediction = tools.get_price_model_prediction(symbol, decision_time)
    query = f"{symbol} material event guidance earnings regulatory"
    evidence = tuple(tools.search_evidence(query, symbol, decision_time))
    recalled = tuple(tools.search_agent_memory(agent_id, query, decision_time))
    snapshot_payload = {
        "agent_id": agent_id, "symbol": symbol, "decision_time": decision_time,
        "prediction": asdict(model_prediction), "wallet": asdict(wallet_snapshot), "marks": marks,
        "evidence_ids": [result.chunk.id for result in evidence],
        "memory_ids": [memory.id for memory in recalled], "data_quality": data_quality,
        "estimated_friction_bps": estimated_friction_bps, "provider_name": provider.name,
        "data_reference": data_reference,
    }
    snapshot_hash = _hash_payload(snapshot_payload)
    context = DecisionContext(
        decision_id=f"dec-{snapshot_hash[:20]}", agent_id=agent_id, symbol=symbol,
        decision_time=decision_time, prediction=model_prediction, wallet=wallet_snapshot,
        marks=dict(marks), evidence=evidence, memories=recalled, data_quality=data_quality,
        estimated_friction_bps=estimated_friction_bps, snapshot_hash=snapshot_hash,
        provider_name=provider.name, data_reference=data_reference,
    )
    fallback_used = False
    try:
        proposal = validate_proposal(provider.structured_response(context), symbol)
        tools._trace("structured_decision_provider", {"provider": provider.name}, 1)
    except Exception as error:  # noqa: BLE001 - untrusted provider failures must fail closed
        tools._trace(
            "structured_decision_provider",
            {"provider": provider.name, "error_type": type(error).__name__},
            0,
            success=False,
        )
        fallback_used = True
        proposal = validate_proposal(DeterministicDecisionProvider().structured_response(
            replace(context, evidence=(), memories=())
        ), symbol)
    gate_allowed, gate_reason = decision_gate(context, proposal)
    provider_metadata = dict(getattr(provider, "audit_metadata", {}))
    if not gate_allowed or proposal.action == "HOLD":
        outcome_reason = gate_reason if not gate_allowed else "hold proposal"
        risk = RiskDecision(False, outcome_reason)
        return DecisionOutcome(
            context, proposal, risk, False, outcome_reason, tuple(tools.traces),
            fallback_used, provider_metadata,
        )
    risk = tools.get_risk_assessment(proposal)
    return DecisionOutcome(
        context, proposal, risk, risk.allowed, gate_reason, tuple(tools.traces),
        fallback_used, provider_metadata,
    )
