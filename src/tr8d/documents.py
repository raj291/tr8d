from __future__ import annotations

import hashlib
import json
import re
import time as clock
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

EVENT_PATTERNS = {
    "earnings": ("earnings", "quarterly results", "10-q", "10-k"),
    "guidance": ("guidance", "outlook", "forecast", "profit warning"),
    "merger": ("merger", "acquisition", "acquire", "takeover"),
    "regulatory": ("regulator", "regulatory", "sec investigation", "antitrust", "fda"),
    "legal": ("lawsuit", "litigation", "court", "settlement", "subpoena"),
    "product": ("product launch", "launches", "unveils", "recall"),
    "macro": ("inflation", "interest rate", "federal reserve", "jobs report", "tariff"),
}


class EvidenceFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Document:
    id: str
    source_type: str
    external_id: str
    title: str
    url: str
    published_at: datetime
    available_at: datetime
    ingested_at: datetime
    symbols: tuple[str, ...]
    event_tags: tuple[str, ...]
    metadata: dict
    content: str = ""

    def __post_init__(self) -> None:
        for field in (self.published_at, self.available_at, self.ingested_at):
            if field.tzinfo is None:
                raise ValueError("document timestamps must be timezone-aware")
        if self.available_at < self.published_at:
            raise ValueError("available_at cannot precede published_at")

    def to_json(self) -> str:
        payload = asdict(self)
        for key in ("published_at", "available_at", "ingested_at"):
            payload[key] = payload[key].isoformat()
        return json.dumps(payload, sort_keys=True)


def event_tags(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text.lower())
    return tuple(tag for tag, patterns in EVENT_PATTERNS.items() if any(pattern in normalized for pattern in patterns))


def _document_id(source_type: str, external_id: str) -> str:
    return hashlib.sha256(f"{source_type}|{external_id}".encode()).hexdigest()


def _parse_sec_acceptance(value: str, filing_date: str) -> datetime:
    if value:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    filed = date.fromisoformat(filing_date)
    return datetime.combine(filed + timedelta(days=1), time(0, 0), tzinfo=EASTERN).astimezone(UTC)


def parse_sec_submissions(payload: dict, symbol: str, ingested_at: datetime | None = None) -> list[Document]:
    recent = payload.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    required = ("filingDate", "form", "primaryDocument")
    if any(len(recent.get(key, [])) != len(accessions) for key in required):
        raise ValueError("SEC recent filing arrays have inconsistent lengths")
    accepted = recent.get("acceptanceDateTime", [""] * len(accessions))
    if len(accepted) not in (0, len(accessions)):
        raise ValueError("SEC acceptance timestamps have inconsistent lengths")
    accepted = accepted or [""] * len(accessions)
    cik = str(payload.get("cik", "")).zfill(10)
    company = payload.get("name", symbol)
    now = ingested_at or datetime.now(UTC)
    documents = []
    for index, accession in enumerate(accessions):
        accession_compact = accession.replace("-", "")
        primary = recent["primaryDocument"][index]
        form = recent["form"][index]
        available = _parse_sec_acceptance(accepted[index], recent["filingDate"][index])
        title = f"{company} {form} filing"
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_compact}/{quote(primary)}"
        documents.append(Document(
            id=_document_id("sec", accession), source_type="sec", external_id=accession,
            title=title, url=url, published_at=available, available_at=available,
            ingested_at=now, symbols=(symbol.upper(),), event_tags=event_tags(f"{title} {form}"),
            metadata={"cik": cik, "form": form, "filing_date": recent["filingDate"][index]},
        ))
    return documents


def _parse_gdelt_datetime(value: str) -> datetime:
    for pattern in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unsupported GDELT timestamp: {value}")


def parse_gdelt_articles(payload: dict, symbol: str, ingested_at: datetime | None = None) -> list[Document]:
    now = ingested_at or datetime.now(UTC)
    documents = []
    for article in payload.get("articles", []):
        url = article.get("url", "").strip()
        title = article.get("title", "").strip()
        seen = article.get("seendate", "")
        if not url or not title or not seen:
            continue
        available = _parse_gdelt_datetime(seen)
        documents.append(Document(
            id=_document_id("gdelt", url), source_type="gdelt", external_id=url,
            title=title, url=url, published_at=available, available_at=available,
            ingested_at=now, symbols=(symbol.upper(),), event_tags=event_tags(title),
            metadata={
                key: article[key] for key in ("domain", "language", "sourcecountry", "socialimage")
                if article.get(key)
            },
        ))
    return documents


class EvidenceClient:
    def __init__(self, user_agent: str, timeout: float = 20.0, retries: int = 1):
        if "@" not in user_agent:
            raise ValueError("user agent must identify an application and contact email")
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries

    def _json(self, url: str) -> dict:
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt == self.retries:
                    raise EvidenceFetchError(f"evidence source unavailable after {attempt + 1} attempts: {type(error).__name__}") from error
                clock.sleep(0.25 * (attempt + 1))
        raise AssertionError("unreachable")

    def sec_submissions(self, cik: str, symbol: str) -> list[Document]:
        normalized = str(int(cik)).zfill(10)
        return parse_sec_submissions(self._json(f"https://data.sec.gov/submissions/CIK{normalized}.json"), symbol)

    def gdelt_articles(self, query: str, symbol: str, start: datetime, end: datetime, max_records: int = 75) -> list[Document]:
        if not 1 <= max_records <= 250:
            raise ValueError("GDELT max_records must be between 1 and 250")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("GDELT start and end must be timezone-aware")
        if start >= end:
            raise ValueError("GDELT start must precede end")
        parameters = urlencode({
            "query": query, "mode": "artlist", "format": "json", "sort": "datedesc",
            "maxrecords": max_records, "startdatetime": start.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
        })
        return parse_gdelt_articles(self._json(f"https://api.gdeltproject.org/api/v2/doc/doc?{parameters}"), symbol)


def write_documents(documents: list[Document], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    unique = {document.id: document for document in documents}
    target.write_text("".join(document.to_json() + "\n" for document in unique.values()), encoding="utf-8")
    return target


def read_documents(path: str | Path) -> list[Document]:
    documents = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        for key in ("published_at", "available_at", "ingested_at"):
            payload[key] = datetime.fromisoformat(payload[key])
        payload["symbols"] = tuple(payload["symbols"])
        payload["event_tags"] = tuple(payload["event_tags"])
        documents.append(Document(**payload))
    return documents
