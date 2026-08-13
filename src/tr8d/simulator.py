from __future__ import annotations

from dataclasses import replace

from .domain import Action, Proposal, Wallet


class PaperSimulator:
    def __init__(self, slippage_bps: float = 10.0):
        self.slippage_bps = slippage_bps

    def execute(self, wallet: Wallet, proposal: Proposal, approved_notional: float, open_price: float) -> tuple[Wallet, float, float]:
        if open_price <= 0 or approved_notional <= 0:
            raise ValueError("execution requires positive price and notional")
        quantities = dict(wallet.quantities)
        if proposal.action is Action.BUY:
            fill = open_price * (1 + self.slippage_bps / 10_000)
            quantity = approved_notional / fill
            cash = wallet.cash - approved_notional
            quantities[proposal.symbol] = quantities.get(proposal.symbol, 0.0) + quantity
        elif proposal.action is Action.SELL:
            fill = open_price * (1 - self.slippage_bps / 10_000)
            quantity = min(quantities.get(proposal.symbol, 0.0), approved_notional / fill)
            cash = wallet.cash + quantity * fill
            quantities[proposal.symbol] = quantities.get(proposal.symbol, 0.0) - quantity
        else:
            raise ValueError("HOLD is not executable")
        if cash < -1e-8 or quantities.get(proposal.symbol, 0.0) < -1e-12:
            raise AssertionError("execution invariant violated")
        return replace(wallet, cash=max(0.0, cash), quantities=quantities), fill, quantity
