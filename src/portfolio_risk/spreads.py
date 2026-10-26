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

``abdi_ranaldo``
    Abdi and Ranaldo (2017): with ``c`` the log close and ``eta`` the mean of the log
    high and log low, ``S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]``; a month's estimate
    is ``sqrt(max(4 m, 0))``, ``m`` the mean of the products over the two-day windows
    starting in it.
``corwin_schultz``
    Corwin and Schultz (2012): from ``beta``, the sum of the two days' squared log
    ranges, and ``gamma``, the squared log range over both days, ``alpha = (sqrt(2
    beta) - sqrt(beta)) / k - sqrt(gamma / k)`` with ``k = 3 - 2 sqrt(2)``, and
    ``S = 2 (e^alpha - 1) / (1 + e^alpha)``; the second day's high and low are moved by
    any overnight gap past the first day's close, negative two-day estimates count as
    0, and a month's estimate is their mean.

Both assume that the spread, not the efficient price's movement within the day, sets
the close's place in the day's range. ``simulate`` builds bars from a known spread
(390 trades a day at the bid or the ask around a random-walk mid-price) to see how far
that holds: they recover a spread that is wide beside the day's volatility, and read
one that is narrow beside it as many times wider, with many months at zero.

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

import numpy as np
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


def _bars(high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    for series in (high, low, close):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("the highs, lows and closes are Series on a date index")
    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("the highs, lows and closes must be on the same dates")
    if len(close) < 2:
        raise ValueError("needs at least two days of bars")
    values = np.array([high, low, close], dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("the highs, lows and closes must be positive numbers")
    if (low > close).any() or (close > high).any():
        raise ValueError("a close is outside its day's range")


def _windows(values: pd.Series, skip: pd.DatetimeIndex | None) -> pd.Series:
    """Two-day figures indexed by their first day: the last day, which has no second,
    dropped, and so are the windows whose second day is in ``skip`` (a distribution or
    split day, when the close moves for other reasons)."""
    second = values.index[1:]
    values = values.iloc[:-1]
    if skip is not None:
        values = values[~second.isin(pd.DatetimeIndex(skip))]
    return values


def _monthly(values: pd.Series) -> pd.Series:
    return values.groupby(values.index.to_period("M")).mean()


def abdi_ranaldo(
    high: pd.Series, low: pd.Series, close: pd.Series, skip: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Monthly spread estimates, as a share of the price, by Abdi and Ranaldo (see
    above); two-day windows ending on a day in ``skip`` are left out."""
    _bars(high, low, close)
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    products = _windows((c - eta) * (c - eta.shift(-1)), skip)
    return np.sqrt(np.maximum(4.0 * _monthly(products), 0.0)).rename("spread")


def corwin_schultz(
    high: pd.Series, low: pd.Series, close: pd.Series, skip: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Monthly spread estimates, as a share of the price, by Corwin and Schultz (see
    above); two-day windows ending on a day in ``skip`` are left out."""
    _bars(high, low, close)
    high_next, low_next = high.shift(-1), low.shift(-1)
    # An overnight gap past the first day's close moves the second day's range back.
    down = (low_next - close).clip(lower=0.0)
    up = (close - high_next).clip(lower=0.0)
    high_next, low_next = high_next - down + up, low_next - down + up
    beta = np.log(high / low) ** 2 + np.log(high_next / low_next) ** 2
    gamma = np.log(np.maximum(high, high_next) / np.minimum(low, low_next)) ** 2
    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    two_day = (2.0 * np.expm1(alpha) / (1.0 + np.exp(alpha))).clip(lower=0.0)
    return _monthly(_windows(two_day, skip)).rename("spread")


def simulate(
    spread: float,
    daily_volatility: float,
    days: int,
    *,
    trades_per_day: int = 390,
    seed: int = 0,
    start: str = "2006-02-01",
) -> pd.DataFrame:
    """Daily bars from a known spread: the mid-price a random walk with
    ``daily_volatility`` over each day, every trade at the ask, ``mid (1 + spread / 2)``,
    or the bid, ``mid (1 - spread / 2)``, at random; the high and low are the day's
    highest and lowest trade, the close its last."""
    if not 0.0 <= spread < 2.0 or not daily_volatility >= 0.0:
        raise ValueError("needs a spread from 0 to 2 and a volatility of at least 0")
    if days < 2 or trades_per_day < 1:
        raise ValueError("needs at least two days and one trade a day")
    rng = np.random.default_rng(seed)
    steps = days * trades_per_day
    log_mid = np.cumsum(rng.normal(0.0, daily_volatility / math.sqrt(trades_per_day), steps))
    side = rng.choice([-1.0, 1.0], steps)
    prices = (np.exp(log_mid) * (1.0 + side * spread / 2.0)).reshape(days, trades_per_day)
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {"high": prices.max(axis=1), "low": prices.min(axis=1), "close": prices[:, -1]},
        index=index,
    )


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
