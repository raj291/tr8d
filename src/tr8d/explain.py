from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np

from .documents import EASTERN, Document
from .domain import PriceBar


@dataclass(frozen=True)
class MoveExplanation:
    symbol: str
    trading_date: str
    open_close_return: float
    market_return: float
    sector_return: float
    idiosyncratic_return: float
    is_large_move: bool
    likely_driver_categories: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    alternative_explanation: str
    confidence: str
    unexplained_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)


def _return(bar: PriceBar) -> float:
    return bar.close / bar.open - 1


def explain_large_move(
    stock: PriceBar, sector: PriceBar, market: PriceBar, prior_stock_bars: list[PriceBar],
    documents: list[Document], sector_name: str = "",
) -> MoveExplanation:
    if not (stock.trading_date == sector.trading_date == market.trading_date):
        raise ValueError("stock, sector, and market bars must share a date")
    prior_returns = np.asarray([_return(bar) for bar in prior_stock_bars if bar.trading_date < stock.trading_date][-20:])
    if len(prior_returns) < 20:
        raise ValueError("large-move explanation requires 20 completed prior bars")
    stock_return, sector_return, market_return = _return(stock), _return(sector), _return(market)
    threshold = max(0.02, 2 * float(np.std(prior_returns)))
    residual = stock_return - sector_return
    prior_date = max(bar.trading_date for bar in prior_stock_bars if bar.trading_date < stock.trading_date)
    evidence_start = datetime.combine(prior_date, time(16, 0), tzinfo=EASTERN).astimezone(UTC)
    market_close = datetime.combine(stock.trading_date, time(16, 0), tzinfo=EASTERN).astimezone(UTC)
    candidates = [
        document for document in documents
        if evidence_start < document.available_at <= market_close
        and stock.symbol in document.symbols
    ]
    candidates.sort(key=lambda item: (item.source_type != "sec", -len(item.event_tags), item.available_at))
    candidates = candidates[:6]
    categories = tuple(dict.fromkeys(tag for document in candidates for tag in document.event_tags))
    explained_scale = min(abs(residual), sum(0.15 for document in candidates if document.event_tags) * abs(residual))
    unexplained = 1.0 if abs(residual) < 1e-12 else max(0.0, 1 - explained_scale / abs(residual))
    if not candidates:
        confidence = "LOW"
    elif any(document.source_type == "sec" for document in candidates) or len({document.metadata.get("domain") for document in candidates}) >= 2:
        confidence = "HIGH" if categories else "MEDIUM"
    else:
        confidence = "MEDIUM" if categories else "LOW"
    if abs(sector_return) >= abs(residual):
        alternative = f"The move may be primarily sector-wide ({sector_name or sector.symbol})."
    elif abs(market_return) >= abs(residual):
        alternative = "The move may be primarily market-wide."
    else:
        alternative = "Observed evidence may not fully explain the company-specific residual."
    return MoveExplanation(
        symbol=stock.symbol, trading_date=stock.trading_date.isoformat(),
        open_close_return=round(stock_return, 8), market_return=round(market_return, 8),
        sector_return=round(sector_return, 8), idiosyncratic_return=round(residual, 8),
        is_large_move=abs(stock_return) >= threshold, likely_driver_categories=categories,
        evidence_ids=tuple(document.id for document in candidates), alternative_explanation=alternative,
        confidence=confidence, unexplained_fraction=round(unexplained, 8),
    )


def write_explanation(explanation: MoveExplanation, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(explanation.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
