"""Market data: daily prices from Yahoo Finance, cached to CSV beside a manifest.

``fetch_prices`` downloads daily closes, dividends and splits for a list of tickers with
yfinance and checks them: every ticker must have prices from within a week of the
window's start and a price on every day any of them traded (so no fund joins, leaves
or drops out part way through the history, and no day is lost to every fund's
returns), with positive, finite closes and non-negative dividends. ``write_cache``
writes them to a CSV file and a manifest beside it (``prices.csv`` and
``prices.manifest.json``): the window, each ticker's rows, first and last dates and
dividend count, and the file's SHA-256.
``load_cache`` reads the file back and refuses it unless it matches its manifest, so
every later step works offline on exactly the data the manifest describes.

The cache holds, per ticker and trading day:

``close``
    Yahoo's close, adjusted for splits but not for dividends. Total returns are
    computed from it and the dividends rather than taken from Yahoo's adjusted close,
    which is recomputed on each request: two fetches seconds apart gave adjusted
    closes differing by up to 2 parts in a million on 26,377 of 43,760 rows, while
    every other field was identical. Without it the same data gives the same file.
``dividend``, ``capital_gain``
    Cash paid per share on that day (the ex-date), split-adjusted like ``close``.
``split``
    The split ratio on that day, 0 on days without one (yfinance's convention).
``volume``
    Shares traded.

Run ``python -m portfolio_risk.data fetch`` to download the default universe into
``data/prices.csv``, and ``python -m portfolio_risk.data verify`` to check a cache.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str


UNIVERSE: tuple[Asset, ...] = (
    Asset("SPY", "SPDR S&P 500 ETF Trust", "US large-cap equity"),
    Asset("IWM", "iShares Russell 2000 ETF", "US small-cap equity"),
    Asset("EFA", "iShares MSCI EAFE ETF", "Developed ex-US equity"),
    Asset("EEM", "iShares MSCI Emerging Markets ETF", "Emerging-market equity"),
    Asset("AGG", "iShares Core US Aggregate Bond ETF", "US investment-grade bonds"),
    Asset("TLT", "iShares 20+ Year Treasury Bond ETF", "Long US Treasuries"),
    Asset("GLD", "SPDR Gold Shares", "Gold"),
    Asset("VNQ", "Vanguard Real Estate ETF", "US real estate (REITs)"),
)
DEFAULT_TICKERS: tuple[str, ...] = tuple(asset.ticker for asset in UNIVERSE)

# Both ends inclusive. GLD, the youngest fund, listed in November 2004.
DEFAULT_START = date(2005, 1, 1)
DEFAULT_END = date(2026, 9, 30)
DEFAULT_CACHE = Path("data") / "prices.csv"

# A ticker whose first price comes later than this after the window's start joined
# part way through; a week covers the holidays at the start of a year.
MAX_START_DELAY = timedelta(days=7)

COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "close",
    "dividend",
    "capital_gain",
    "split",
    "volume",
)
# yfinance's names for the fields kept, by cache column.
_YAHOO_FIELDS = {
    "close": "Close",
    "dividend": "Dividends",
    "capital_gain": "Capital Gains",
    "split": "Stock Splits",
    "volume": "Volume",
}
# Fields yfinance gives only for funds: a request for stocks alone (AAPL) has none.
_OPTIONAL_FIELDS = {"capital_gain"}

_TICKER = re.compile(r"^\^?[A-Z0-9][A-Z0-9.=-]{0,14}$")


class DataError(ValueError):
    """Prices that cannot be used: a failed fetch, bad values, or a damaged cache."""


def validate_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    """The tickers as given, checked: at least one, each a plausible symbol, no repeats."""
    if isinstance(tickers, str):
        raise DataError("tickers must be a list of symbols, not one string")
    tickers = tuple(tickers)
    if not tickers:
        raise DataError("no tickers given")
    for ticker in tickers:
        if not isinstance(ticker, str) or not _TICKER.match(ticker):
            raise DataError(f"not a ticker symbol: {ticker!r} (use upper case, e.g. 'SPY')")
    repeated = sorted({t for t in tickers if tickers.count(t) > 1})
    if repeated:
        raise DataError(f"tickers repeated: {', '.join(repeated)}")
    return tickers


def _validate_window(start: date, end: date) -> None:
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise DataError(f"{name} must be a date, not {value!r}")
    if start > end:
        raise DataError(f"the window starts ({start}) after it ends ({end})")


def fetch_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    *,
    download: Callable[..., pd.DataFrame] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Daily prices for ``tickers`` from ``start`` to ``end`` inclusive, checked.

    Returns one row per ticker and trading day, with the columns in ``COLUMNS``, sorted
    by ticker (in the order given) and date. ``end`` must be before ``today`` (by
    default the date in New York), because during a session Yahoo's bar for the day
    holds the latest trade, not the close, and a later fetch would differ.
    ``download`` stands in for ``yfinance.download`` and is called with the same
    arguments.
    """
    tickers = validate_tickers(tickers)
    _validate_window(start, end)
    today = today or pd.Timestamp.now(tz="America/New_York").date()
    if end >= today:
        raise DataError(
            f"the window ends on {end}, not before today ({today} in New York); the day's "
            "bar may still be trading, so end the window on an earlier day"
        )
    if download is None:
        import yfinance

        download = yfinance.download
    try:
        raw = download(
            list(tickers),
            start=start.isoformat(),
            # yfinance's end is exclusive.
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=True,
            progress=False,
            multi_level_index=True,
        )
    except Exception as exc:  # network, rate limit, or yfinance's own errors
        raise DataError(f"the download failed: {type(exc).__name__}: {exc}") from exc
    return _from_yahoo(raw, tickers, start, end)


def _from_yahoo(raw: Any, tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataError(f"Yahoo returned no prices for {', '.join(tickers)} from {start} to {end}")
    if raw.columns.nlevels != 2:
        raise DataError("expected yfinance's (field, ticker) columns")
    fields = set(raw.columns.get_level_values(0))
    missing_fields = [
        yahoo
        for column, yahoo in _YAHOO_FIELDS.items()
        if yahoo not in fields and column not in _OPTIONAL_FIELDS
    ]
    if missing_fields:
        raise DataError(f"the download has no {', '.join(missing_fields)} column")
    # A timezone-aware index gives each day's date in its own (the exchange's) zone.
    raw = raw.set_axis(pd.DatetimeIndex(raw.index).normalize(), axis=0)
    if raw.index.has_duplicates:
        raise DataError("the download repeats a date")

    frames = []
    missing = []
    for ticker in tickers:
        if ("Close", ticker) not in raw.columns or raw[("Close", ticker)].isna().all():
            missing.append(ticker)
            continue
        columns = {}
        for column, yahoo in _YAHOO_FIELDS.items():
            if (yahoo, ticker) in raw.columns:
                columns[column] = raw[(yahoo, ticker)].astype("float64")
            else:
                columns[column] = pd.Series(0.0, index=raw.index)
        frame = pd.DataFrame(columns)
        frame = frame[frame["close"].notna()]
        frame.insert(0, "ticker", ticker)
        frame.insert(0, "date", frame.index.date)
        frames.append(frame.reset_index(drop=True))
    if missing:
        raise DataError(
            f"Yahoo returned no prices for {', '.join(missing)} "
            f"from {start} to {end} (check the symbol, or whether it traded then)"
        )
    prices = pd.concat(frames, ignore_index=True)
    validate_prices(prices, tickers, start, end)
    return prices


def validate_prices(prices: pd.DataFrame, tickers: Sequence[str], start: date, end: date) -> None:
    """Refuse prices that later steps would misuse; raise ``DataError`` naming the problem."""
    if list(prices.columns) != list(COLUMNS):
        raise DataError(f"expected the columns {', '.join(COLUMNS)}, got {list(prices.columns)}")
    found = list(dict.fromkeys(prices["ticker"]))
    if found != list(tickers):
        raise DataError(f"expected the tickers {', '.join(tickers)} in that order, got {found}")
    for column in COLUMNS[2:]:
        values = prices[column]
        bad = ~values.map(lambda v: isinstance(v, float) and math.isfinite(v))
        if bad.any():
            row = prices[bad].iloc[0]
            raise DataError(
                f"{row['ticker']} on {row['date']}: {column} is {row[column]}, not a number"
            )
    bad = prices["close"] <= 0
    if bad.any():
        row = prices[bad].iloc[0]
        raise DataError(f"{row['ticker']} on {row['date']}: close {row['close']} is not positive")
    for column in ("dividend", "capital_gain", "split", "volume"):
        bad = prices[column] < 0
        if bad.any():
            row = prices[bad].iloc[0]
            raise DataError(f"{row['ticker']} on {row['date']}: {column} {row[column]} is negative")
    for ticker, group in prices.groupby("ticker", sort=False):
        days = list(group["date"])
        if any(not isinstance(d, date) or isinstance(d, datetime) for d in days):
            raise DataError(f"{ticker}: a date is not a calendar date")
        if any(b <= a for a, b in zip(days, days[1:], strict=False)):
            raise DataError(f"{ticker}: dates are repeated or out of order")
        if days[0] < start or days[-1] > end:
            raise DataError(
                f"{ticker}: prices from {days[0]} to {days[-1]}, outside {start} to {end}"
            )
        if days[0] - start > MAX_START_DELAY:
            raise DataError(
                f"{ticker}: the first price is on {days[0]}, more than "
                f"{MAX_START_DELAY.days} days after the window starts ({start}), so it would "
                "join part way through; start the window later or leave it out"
            )
    # Every fund on every trading day: a day one fund lacks is dropped from every fund's
    # returns, and a fund that stops or pauses would shorten the whole history.
    calendar = set(prices["date"])
    for ticker, group in prices.groupby("ticker", sort=False):
        missing = sorted(calendar - set(group["date"]))
        if missing:
            more = f" and on {len(missing) - 1} more" if len(missing) > 1 else ""
            raise DataError(
                f"{ticker}: no price on {missing[0]}{more} of the days the others traded; "
                "every fund needs a price on every day, so shorten the window or leave it out"
            )


def wide(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    """One column of the cache as a date-by-ticker table, tickers in their order."""
    if column not in COLUMNS[2:]:
        raise DataError(f"not a price column: {column!r}")
    order = list(dict.fromkeys(prices["ticker"]))
    table = prices.pivot(index="date", columns="ticker", values=column)
    table = table[order]
    table.index = pd.DatetimeIndex(table.index, name="date")
    table.columns.name = None
    return table


def manifest_path(cache: Path) -> Path:
    return cache.with_name(cache.stem + ".manifest.json")


def check_output(path: Path | None) -> None:
    """Refuse a ``--json`` path that cannot be written as a file, before a command runs:
    a folder, or a path under a file."""
    if path is None:
        return
    path = Path(path)
    if path.is_dir():
        raise ValueError(f"{path} is a folder; --json takes a file to write")
    for parent in path.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"{path} cannot be written: {parent} is a file")
            break


def _csv_bytes(prices: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    out = prices.copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _summary(prices: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary = {}
    for ticker, group in prices.groupby("ticker", sort=False):
        summary[ticker] = {
            "rows": len(group),
            "first": group["date"].iloc[0].isoformat(),
            "last": group["date"].iloc[-1].isoformat(),
            "dividends": int((group["dividend"] > 0).sum()),
            "capital_gains": int((group["capital_gain"] > 0).sum()),
            "splits": int((group["split"] > 0).sum()),
        }
    return summary


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_cache(
    prices: pd.DataFrame,
    cache: Path,
    *,
    start: date,
    end: date,
    source: str,
    fetched_at: datetime | None = None,
) -> dict[str, Any]:
    """Write ``prices`` to ``cache`` and its manifest beside it; return the manifest."""
    # Whole numbers (a volume, say) are written as the floats they are read back as.
    prices = prices.astype(
        {c: "float64" for c in COLUMNS[2:] if pd.api.types.is_integer_dtype(prices.get(c))}
    )
    tickers = tuple(dict.fromkeys(prices["ticker"]))
    validate_prices(prices, tickers, start, end)
    data = _csv_bytes(prices)
    fetched_at = fetched_at or datetime.now(UTC)
    manifest = {
        "source": source,
        "fetched_at": fetched_at.astimezone(UTC).isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "file": cache.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(prices),
        "tickers": _summary(prices),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(cache, data)
    text = json.dumps(manifest, indent=2) + "\n"
    _write_atomic(manifest_path(cache), text.encode("utf-8"))
    return manifest


def read_manifest(cache: Path) -> dict[str, Any]:
    cache = Path(cache)
    if cache.is_dir():
        raise DataError(f"{cache} is a folder; the cache is a file, such as {DEFAULT_CACHE}")
    path = manifest_path(cache)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DataError(f"{path} is missing; a cache is only read beside its manifest") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataError(f"{path} cannot be read: {exc}") from None
    required = {"start", "end", "file", "sha256", "rows", "tickers"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise DataError(f"{path} is not a prices manifest (needs {', '.join(sorted(required))})")
    if not isinstance(manifest["tickers"], dict) or not manifest["tickers"]:
        raise DataError(f"{path} lists no tickers")
    return manifest


def load_cache(cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """The cached prices, refused unless they match their manifest exactly."""
    cache = Path(cache)
    manifest = read_manifest(cache)
    try:
        data = cache.read_bytes()
    except FileNotFoundError:
        raise DataError(
            f"{cache} is missing; fetch it with: python -m portfolio_risk.data fetch"
        ) from None
    except OSError as exc:
        raise DataError(f"{cache} cannot be read: {exc}") from None
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"]:
        raise DataError(
            f"{cache} does not match its manifest (SHA-256 {digest[:12]}..., the manifest "
            f"says {str(manifest['sha256'])[:12]}...); fetch it again, or restore the file"
        )
    try:
        start = date.fromisoformat(manifest["start"])
        end = date.fromisoformat(manifest["end"])
        prices = pd.read_csv(
            io.BytesIO(data),
            # Read as text, or a ticker of digits alone would come back as a number.
            dtype={"date": str, "ticker": str} | {column: "float64" for column in COLUMNS[2:]},
            keep_default_na=False,
            # The default parser reads some numbers a last bit off what was written
            # (8,269 of the 43,760 closes of the 2005-2026 cache); this one is exact.
            float_precision="round_trip",
        )
    except (ValueError, TypeError) as exc:
        raise DataError(f"{cache} cannot be parsed: {exc}") from None
    if list(prices.columns) != list(COLUMNS):
        raise DataError(f"{cache}: expected the columns {', '.join(COLUMNS)}")
    try:
        prices["date"] = [date.fromisoformat(d) for d in prices["date"].astype(str)]
    except ValueError as exc:
        raise DataError(f"{cache}: {exc}") from None
    validate_prices(prices, tuple(manifest["tickers"]), start, end)
    if _summary(prices) != manifest["tickers"] or len(prices) != manifest["rows"]:
        raise DataError(f"{cache}: rows or dates differ from its manifest")
    return prices


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {text!r}") from None


def _print_summary(manifest: dict[str, Any]) -> None:
    print(f"{manifest['rows']} rows, {manifest['start']} to {manifest['end']}")
    for ticker, info in manifest["tickers"].items():
        print(
            f"  {ticker:<6} {info['rows']:>5} days  {info['first']} to {info['last']}  "
            f"{info['dividends']} dividends"
        )
    print(f"SHA-256 {manifest['sha256']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.data", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download prices from Yahoo into the cache")
    fetch.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="comma-separated symbols (default: the eight-fund universe)",
    )
    fetch.add_argument("--start", type=_parse_date, default=DEFAULT_START)
    fetch.add_argument("--end", type=_parse_date, default=DEFAULT_END, help="inclusive")
    fetch.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    verify = commands.add_parser("verify", help="check a cache against its manifest")
    verify.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)

    try:
        if args.command == "fetch":
            import yfinance

            previous = None
            if manifest_path(args.out).exists():
                try:
                    previous = read_manifest(args.out)
                except DataError:
                    previous = None
            tickers = [t.strip() for t in args.tickers.split(",")]
            prices = fetch_prices(tickers, args.start, args.end)
            data = _csv_bytes(prices)
            same = previous is not None and (
                previous["sha256"],
                previous["start"],
                previous["end"],
            ) == (hashlib.sha256(data).hexdigest(), args.start.isoformat(), args.end.isoformat())
            if same:
                # The manifest is left as it was, so a fetch of the same data (in a fresh
                # clone, say, where only the manifest is committed) changes no tracked file.
                _write_atomic(args.out, data)
                manifest = previous
            else:
                manifest = write_cache(
                    prices,
                    args.out,
                    start=args.start,
                    end=args.end,
                    source=f"Yahoo Finance, via yfinance {yfinance.__version__}",
                )
            _print_summary(manifest)
            if previous is not None:
                print(
                    "Same data as the previous manifest, which is left as it was."
                    if same
                    else "Differs from the previous manifest."
                )
        else:
            load_cache(args.cache)
            _print_summary(read_manifest(args.cache))
            print(f"{args.cache} matches its manifest.")
    except (DataError, OSError) as exc:  # OSError: the cache or manifest cannot be written
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
