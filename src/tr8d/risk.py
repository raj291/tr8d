from __future__ import annotations

from dataclasses import dataclass

from .domain import Action, Proposal, RiskDecision, Wallet


@dataclass(frozen=True)
class RiskPolicy:
    max_single_position_pct: float = 0.35
    max_total_invested_pct: float = 0.70
    min_cash_reserve_pct: float = 0.20
    min_order_notional: float = 0.25


class RiskGovernor:
    def __init__(self, policy: RiskPolicy | None = None):
        self.policy = policy or RiskPolicy()

    def assess(self, proposal: Proposal, wallet: Wallet, marks: dict[str, float]) -> RiskDecision:
        if proposal.action is Action.HOLD:
            return RiskDecision(False, "hold proposal")
        equity = wallet.equity(marks)
        if equity <= 0:
            return RiskDecision(False, "non-positive equity")
        if proposal.notional < self.policy.min_order_notional:
            return RiskDecision(False, "below minimum order")
        if proposal.action is Action.SELL:
            price = marks.get(proposal.symbol)
            owned_value = wallet.quantities.get(proposal.symbol, 0.0) * price if price else 0.0
            if price is None or owned_value <= 0:
                return RiskDecision(False, "no long position to sell")
            return RiskDecision(True, "approved", min(proposal.notional, owned_value))

        current_value = wallet.quantities.get(proposal.symbol, 0.0) * marks.get(proposal.symbol, 0.0)
        invested = equity - wallet.cash
        maximum = min(
            proposal.notional,
            equity * self.policy.max_single_position_pct - current_value,
            equity * self.policy.max_total_invested_pct - invested,
            wallet.cash - equity * self.policy.min_cash_reserve_pct,
        )
        maximum = round(maximum, 8)
        if maximum < self.policy.min_order_notional:
            return RiskDecision(False, "position, exposure, or reserve limit")
        return RiskDecision(True, "approved", maximum)
