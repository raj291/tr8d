from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .domain import PriceBar


@dataclass(frozen=True)
class DataManifest:
    version: str
    source: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    row_count: int
    content_sha256: str
    created_at: str

    def write(self, directory: str | Path) -> Path:
        target = Path(directory) / f"{self.version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def bar_digest(bars: list[PriceBar]) -> str:
    canonical = "\n".join(
        f"{bar.symbol}|{bar.trading_date.isoformat()}|{bar.open:.10f}|{bar.close:.10f}|{bar.available_at.isoformat()}"
        for bar in sorted(bars, key=lambda item: (item.trading_date, item.symbol))
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_manifest(bars: list[PriceBar], source: str) -> DataManifest:
    if not bars:
        raise ValueError("cannot manifest an empty dataset")
    digest = bar_digest(bars)
    dates = [bar.trading_date for bar in bars]
    return DataManifest(
        version=f"prices-{min(dates).isoformat()}-{max(dates).isoformat()}-{digest[:12]}",
        source=source,
        symbols=tuple(sorted({bar.symbol for bar in bars})),
        start_date=min(dates).isoformat(),
        end_date=max(dates).isoformat(),
        row_count=len(bars),
        content_sha256=digest,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
