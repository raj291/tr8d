from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from .domain import Action, PriceBar, Proposal, Wallet
from .features import bars_by_symbol, build_features, training_rows
from .model import LogisticBaseline
from .risk import RiskGovernor
from .simulator import PaperSimulator
from .store import Store


@dataclass(frozen=True)
class Strategy:
    agent_id: str
    buy_threshold: float
    sell_threshold: float
    target_notional: float


STRATEGIES = (
    Strategy("pattern", 0.58, 0.42, 2.50),
    Strategy("conservative", 0.64, 0.36, 2.00),
    Strategy("active", 0.54, 0.46, 3.00),
)


def _proposal(strategy: Strategy, symbol: str, probability: float, expected: float, holding_value: float) -> Proposal:
    if probability >= strategy.buy_threshold and expected > 0:
        return Proposal(symbol, Action.BUY, strategy.target_notional, probability, "baseline bullish gate")
    if probability <= strategy.sell_threshold and holding_value > 0:
        return Proposal(symbol, Action.SELL, min(strategy.target_notional, holding_value), 1 - probability, "baseline bearish exit")
    return Proposal(symbol, Action.HOLD, 0.0, max(probability, 1 - probability), "edge below gate")


def replay(bars: list[PriceBar], database: str, source: str, seed: int = 7, warmup: int = 80) -> dict[str, float]:
    grouped = bars_by_symbol(bars)
    if not grouped:
        raise ValueError("no prices supplied")
    if any(len(history) < warmup + 1 for history in grouped.values()):
        raise ValueError(f"each symbol requires at least {warmup + 1} rows")
    digest = hashlib.sha256(
        "".join(f"{b.symbol}|{b.trading_date}|{b.open}|{b.close}|{b.available_at}" for b in bars).encode()
    ).hexdigest()[:12]
    run_id = f"run-{digest}-{seed}"
    store = Store(database)
    store.start_run(run_id, datetime.now(timezone.utc).isoformat(), seed, source)
    store.prices(run_id, bars)
    wallets = {strategy.agent_id: Wallet(10.0, {}) for strategy in STRATEGIES}
    governor = RiskGovernor()
    simulator = PaperSimulator()
    dates = sorted({bar.trading_date for bar in bars})
    for decision_date in dates[warmup:]:
        today = {b.symbol: b for b in bars if b.trading_date == decision_date}
        close_marks = {symbol: bar.close for symbol, bar in today.items()}
        # Risk and proposal construction use only the previous completed close.
        # The current open is revealed solely to the execution simulator.
        decision_marks = {
            symbol: [bar.close for bar in grouped[symbol] if bar.trading_date < decision_date][-1]
            for symbol in today
        }
        for symbol, current in today.items():
            history = grouped[symbol]
            features = build_features(history, decision_date)
            if not all(source_date < decision_date for source_date in features.source_dates):
                raise AssertionError("future feature leakage")
            x, y, returns = training_rows(history, decision_date)
            model = LogisticBaseline().fit(x, y, returns)
            prediction = model.predict(features.values)
            for strategy in STRATEGIES:
                wallet = wallets[strategy.agent_id]
                holding = wallet.quantities.get(symbol, 0.0) * decision_marks[symbol]
                proposal = _proposal(strategy, symbol, prediction.bull_probability, prediction.expected_return, holding)
                risk = governor.assess(proposal, wallet, decision_marks)
                decision_id = store.decision((
                    run_id, strategy.agent_id, symbol, decision_date.isoformat(), proposal.action.value,
                    proposal.confidence, prediction.expected_return, proposal.notional, int(risk.allowed),
                    risk.reason, json.dumps(features.values),
                ))
                if risk.allowed:
                    wallet, fill, quantity = simulator.execute(wallet, proposal, risk.approved_notional, current.open)
                    wallets[strategy.agent_id] = wallet
                    store.trade(decision_id, fill, quantity, risk.approved_notional)
        for strategy in STRATEGIES:
            wallet = wallets[strategy.agent_id]
            marks = dict(close_marks)
            for held_symbol in wallet.quantities:
                if held_symbol not in marks:
                    prior = [b.close for b in grouped[held_symbol] if b.trading_date < decision_date]
                    marks[held_symbol] = prior[-1]
            store.snapshot(run_id, strategy.agent_id, decision_date.isoformat(), wallet.cash, wallet.equity(marks), wallet.quantities)
        store.commit()
    final_marks = {symbol: history[-1].close for symbol, history in grouped.items()}
    return {agent_id: round(wallet.equity(final_marks), 4) for agent_id, wallet in wallets.items()}
