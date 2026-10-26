"""Bid-ask spreads estimated from daily highs, lows and closes, and how far they miss.

Yahoo's daily bars carry each day's high and low beside the close, and two published
estimators turn those into a bid-ask spread: a day's range and the close's place in
it widen with the spread. ``fetch_ranges`` downloads the highs, lows and closes for
the funds, ``write_cache`` writes them to ``data/ranges.csv`` beside a manifest, as
the price cache is, and ``load_ranges`` reads them back, refused unless the file
matches the manifest.

The cache holds, per ticker and trading day, Yahoo's ``high``, ``low`` and ``close``,
adjusted for splits as the price cache's close is (and that close, to the bit), so
that a day's three prices are on one basis. Every day's low is at most its close and
its close at most its high.

Run ``python -m portfolio_risk.spreads fetch`` to download the ranges for the price
cache's window into ``data/ranges.csv``, and ``python -m portfolio_risk.spreads
verify`` to check a cache.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_risk.data import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    DataError,
    _parse_date,
    _validate_window,
    _write_atomic,
    manifest_path,
    validate_tickers,
)

DEFAULT_CACHE = Path("data") / "ranges.csv"
COLUMNS: tuple[str, ...] = ("date", "ticker", "high", "low", "close")
PRICE_COLUMNS = COLUMNS[2:]
# yfinance's names for them.
_YAHOO_FIELDS = {"high": "High", "low": "Low", "close": "Close"}


def fetch_ranges(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    *,
    download: Callable[..., pd.DataFrame] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Daily highs, lows and closes for ``tickers`` from ``start`` to ``end`` inclusive,
    checked; one row per ticker and trading day, as ``data.fetch_prices`` gives prices.
    ``download`` stands in for ``yfinance.download``."""
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
    if not isinstance(raw, pd.DataFrame) or raw.empty or raw.columns.nlevels != 2:
        raise DataError(f"Yahoo returned no (field, ticker) bars for {', '.join(tickers)}")
    raw = raw.set_axis(pd.DatetimeIndex(raw.index).normalize(), axis=0)
    frames = []
    for ticker in tickers:
        missing = [y for y in _YAHOO_FIELDS.values() if (y, ticker) not in raw.columns]
        if missing:
            raise DataError(f"the download has no {', '.join(missing)} for {ticker}")
        frame = pd.DataFrame(
            {
                column: raw[(yahoo, ticker)].astype("float64")
                for column, yahoo in _YAHOO_FIELDS.items()
            }
        )
        frame = frame[frame["close"].notna()]
        frame.insert(0, "ticker", ticker)
        frame.insert(0, "date", frame.index.date)
        frames.append(frame.reset_index(drop=True))
    ranges = pd.concat(frames, ignore_index=True)
    validate_ranges(ranges, tickers, start, end)
    return ranges


def validate_ranges(ranges: pd.DataFrame, tickers: Sequence[str], start: date, end: date) -> None:
    """Refuse ranges the estimators would misread; raise ``DataError`` naming the problem."""
    if list(ranges.columns) != list(COLUMNS):
        raise DataError(f"expected the columns {', '.join(COLUMNS)}, got {list(ranges.columns)}")
    found = list(dict.fromkeys(ranges["ticker"]))
    if found != list(tickers):
        raise DataError(f"expected the tickers {', '.join(tickers)} in that order, got {found}")
    for column in PRICE_COLUMNS:
        bad = ~ranges[column].map(lambda v: isinstance(v, float) and math.isfinite(v) and v > 0)
        if bad.any():
            row = ranges[bad].iloc[0]
            raise DataError(
                f"{row['ticker']} on {row['date']}: {column} is {row[column]}, "
                "not a positive number"
            )
    bad = (ranges["low"] > ranges["close"]) | (ranges["close"] > ranges["high"])
    if bad.any():
        row = ranges[bad].iloc[0]
        raise DataError(
            f"{row['ticker']} on {row['date']}: the close {row['close']} is outside the day's "
            f"range, {row['low']} to {row['high']}"
        )
    for ticker, group in ranges.groupby("ticker", sort=False):
        days = list(group["date"])
        if any(not isinstance(d, date) or isinstance(d, datetime) for d in days):
            raise DataError(f"{ticker}: a date is not a calendar date")
        if any(b <= a for a, b in zip(days, days[1:], strict=False)):
            raise DataError(f"{ticker}: dates are repeated or out of order")
        if days[0] < start or days[-1] > end:
            raise DataError(
                f"{ticker}: bars from {days[0]} to {days[-1]}, outside {start} to {end}"
            )


def _csv_bytes(ranges: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    out = ranges.copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def write_cache(
    ranges: pd.DataFrame,
    cache: Path,
    *,
    start: date,
    end: date,
    source: str,
    fetched_at: datetime | None = None,
) -> dict[str, Any]:
    """Write ``ranges`` to ``cache`` and its manifest beside it; return the manifest."""
    tickers = tuple(dict.fromkeys(ranges["ticker"]))
    validate_ranges(ranges, tickers, start, end)
    data = _csv_bytes(ranges)
    fetched_at = fetched_at or datetime.now(UTC)
    manifest = {
        "source": source,
        "fetched_at": fetched_at.astimezone(UTC).isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "file": cache.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(ranges),
        "tickers": {
            str(t): {
                "rows": len(group),
                "first": group["date"].iloc[0].isoformat(),
                "last": group["date"].iloc[-1].isoformat(),
            }
            for t, group in ranges.groupby("ticker", sort=False)
        },
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
        raise DataError(f"{path} is not a ranges manifest (needs {', '.join(sorted(required))})")
    if not isinstance(manifest["tickers"], dict) or not manifest["tickers"]:
        raise DataError(f"{path} lists no tickers")
    return manifest


def load_ranges(cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """The cached ranges, refused unless they match their manifest exactly."""
    cache = Path(cache)
    manifest = read_manifest(cache)
    try:
        data = cache.read_bytes()
    except FileNotFoundError:
        raise DataError(
            f"{cache} is missing; fetch it with: python -m portfolio_risk.spreads fetch"
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
        ranges = pd.read_csv(
            io.BytesIO(data),
            dtype={"date": str, "ticker": str} | {column: "float64" for column in PRICE_COLUMNS},
            keep_default_na=False,
            float_precision="round_trip",
        )
        if list(ranges.columns) != list(COLUMNS):
            raise DataError(f"{cache}: expected the columns {', '.join(COLUMNS)}")
        ranges["date"] = [date.fromisoformat(d) for d in ranges["date"]]
    except (ValueError, TypeError) as exc:
        raise DataError(f"{cache} cannot be parsed: {exc}") from None
    tickers = tuple(manifest["tickers"])
    validate_ranges(ranges, tickers, start, end)
    if len(ranges) != manifest["rows"]:
        raise DataError(f"{cache}: {len(ranges)} rows, the manifest says {manifest['rows']}")
    return ranges


def _print_summary(manifest: dict[str, Any]) -> None:
    print(f"{manifest['rows']} rows, {manifest['start']} to {manifest['end']}")
    for ticker, info in manifest["tickers"].items():
        print(f"  {ticker:<6} {info['rows']:>5} days  {info['first']} to {info['last']}")
    print(f"SHA-256 {manifest['sha256']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.spreads", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download highs, lows and closes into the cache")
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
            ranges = fetch_ranges(tickers, args.start, args.end)
            data = _csv_bytes(ranges)
            same = previous is not None and (
                previous["sha256"],
                previous["start"],
                previous["end"],
            ) == (hashlib.sha256(data).hexdigest(), args.start.isoformat(), args.end.isoformat())
            if same:
                # As for prices: the same data leaves the committed manifest as it was.
                _write_atomic(args.out, data)
                manifest = previous
            else:
                manifest = write_cache(
                    ranges,
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
            load_ranges(args.cache)
            _print_summary(read_manifest(args.cache))
            print(f"{args.cache} matches its manifest.")
    except (DataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
