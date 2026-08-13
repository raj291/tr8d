import unittest

from tr8d.domain import Action, Proposal, Wallet
from tr8d.risk import RiskGovernor
from tr8d.simulator import PaperSimulator


class RiskAndSimulatorTests(unittest.TestCase):
    def test_buy_cannot_create_negative_cash(self):
        wallet = Wallet(10.0, {})
        proposal = Proposal("XLK", Action.BUY, 12.0, 0.8, "test")
        risk = RiskGovernor().assess(proposal, wallet, {"XLK": 100.0})
        self.assertTrue(risk.allowed)
        self.assertLessEqual(risk.approved_notional, 3.5)
        result, _, _ = PaperSimulator().execute(wallet, proposal, risk.approved_notional, 100.0)
        self.assertGreaterEqual(result.cash, 0)

    def test_zero_cash_with_position_is_not_bankrupt(self):
        wallet = Wallet(0.0, {"XLK": 0.101})
        self.assertAlmostEqual(wallet.equity({"XLK": 100.0}), 10.1)

    def test_cannot_sell_without_position(self):
        proposal = Proposal("XLK", Action.SELL, 1.0, 0.8, "test")
        risk = RiskGovernor().assess(proposal, Wallet(10.0, {}), {"XLK": 100.0})
        self.assertFalse(risk.allowed)


if __name__ == "__main__":
    unittest.main()
