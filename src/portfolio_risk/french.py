"""Fama-French factors from Kenneth French's data library, cached to CSV beside a manifest.

``fetch_factors`` downloads three files from the library and checks them against each
other: the daily three-factor file (``Mkt-RF``, ``SMB``, ``HML``, ``RF``), the daily
five-factor file (2x3 sorts: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``RF``)
and the monthly three-factor file, for its one-month Treasury bill rate. ``write_cache``
writes them to ``data/factors.csv`` and ``load_factors`` reads them back, refused unless
the file matches the manifest beside it, as the price cache is.

The cache holds, per trading day, the library's figures as it publishes them, in percent:

``mkt_rf``, ``smb``, ``hml``, ``rf``
    From the daily three-factor file. ``Mkt-RF``, ``HML`` and ``RF`` are the same in
    the five-factor file on every day of the 2005-2026 window, and a fetch checks that.
``smb5``, ``rmw``, ``cma``
    From the daily five-factor file, whose ``SMB`` is built from three sorts rather
    than one and differs from the three-factor file's.
``rf_month``
    The month's one-month Treasury bill rate from the monthly file, on every day of it.

The daily ``rf`` is published to two decimals of a percent (0.00, 0.01 or 0.02 on every
day of 2005-2026), so it rounds a rate of 1% a year to nothing and one of 4.25% to 5%.
``daily_risk_free`` gives instead the simple daily rate that compounds to the month's
rate over the month's trading days, which is how the library builds its daily rate
before rounding it. Months are therefore kept whole: a window starts on the first of a
month and ends on the last day of one.

Run ``python -m portfolio_risk.french fetch`` to download the factors for the price
cache's window into ``data/factors.csv``, and ``python -m portfolio_risk.french verify``
to check a cache.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import io
import json
import math
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk.data import (
    DEFAULT_END,
    DEFAULT_START,
    EXACT_FLOATS,
    DataError,
    _parse_date,
    _write_atomic,
    manifest_path,
)

LIBRARY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "daily3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "daily5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "monthly3": "F-F_Research_Data_Factors_CSV.zip",
}
DEFAULT_CACHE = Path("data") / "factors.csv"

COLUMNS: tuple[str, ...] = ("date", "mkt_rf", "smb", "hml", "rf", "smb5", "rmw", "cma", "rf_month")
FACTOR_COLUMNS = COLUMNS[1:]
# The columns each file must have, as the library names them.
_HEADERS = {
    "daily3": ("Mkt-RF", "SMB", "HML", "RF"),
    "daily5": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    "monthly3": ("Mkt-RF", "SMB", "HML", "RF"),
}
# The figures both daily files publish, which must agree.
_SHARED = ("Mkt-RF", "HML", "RF")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _unzip(name: str, data: bytes) -> str:
    """The text of the one CSV file in a library download."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise DataError(f"{name} holds {len(members)} CSV files, not one")
        return archive.read(members[0]).decode("ascii")
    except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise DataError(f"{name} cannot be read: {exc}") from None


def parse_table(name: str, text: str, columns: Sequence[str], date_width: int) -> pd.DataFrame:
    """The first table in a library file: its rows of figures, indexed by their date label.

    A file opens with a few lines of notes, then a header line (``,Mkt-RF,SMB,...``),
    then one row per period labelled ``YYYYMMDD`` (daily) or ``YYYYMM`` (monthly). The
    table ends at the first line that is not such a row; monthly files go on to a table
    of annual figures, which is not read.
    """
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith(",")), None)
    if header is None:
        raise DataError(f"{name} has no table header")
    found = tuple(c.strip() for c in lines[header].split(",")[1:])
    if found != tuple(columns):
        raise DataError(f"{name}: expected the columns {', '.join(columns)}, got {list(found)}")
    labels, values = [], []
    for line in lines[header + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts[0]) != date_width or not parts[0].isdigit():
            break
        if len(parts) != len(columns) + 1:
            raise DataError(f"{name}: the row for {parts[0]} has {len(parts) - 1} figures")
        try:
            row = [float(p) for p in parts[1:]]
        except ValueError:
            raise DataError(
                f"{name}: the row for {parts[0]} holds a figure that is not a number"
            ) from None
        if not all(math.isfinite(v) for v in row):
            raise DataError(f"{name}: the row for {parts[0]} holds a figure that is not finite")
        labels.append(parts[0])
        values.append(row)
    if not labels:
        raise DataError(f"{name} has no rows under its header")
    if any(b <= a for a, b in zip(labels, labels[1:], strict=False)):
        raise DataError(f"{name}: dates are repeated or out of order")
    return pd.DataFrame(values, index=labels, columns=list(columns))


def _vintage(text: str) -> str:
    """The file's first line, which names the CRSP database it was built from."""
    return text.splitlines()[0].strip() if text else ""


def _validate_window(start: date, end: date) -> None:
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise DataError(f"{name} must be a date, not {value!r}")
    if start > end:
        raise DataError(f"the window starts ({start}) after it ends ({end})")
    if start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]:
        raise DataError(
            f"the window {start} to {end} is not whole months; the risk-free rate is "
            "monthly, so start on the first of a month and end on the last day of one"
        )


def fetch_factors(
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    *,
    download: Callable[[str], bytes] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """The library's daily factors from ``start`` to ``end`` inclusive, checked.

    Returns one row per trading day the library has in the window, with the columns in
    ``COLUMNS`` (percent, as published), and each file's first line, which names the
    CRSP database it was built from. The library publishes whole months, a month or two
    behind, so the data may end before the window does. ``download`` stands in for an
    HTTP GET of a URL and returns the body.
    """
    _validate_window(start, end)
    download = download or _download
    tables, vintages = {}, {}
    for key, name in FILES.items():
        try:
            body = download(LIBRARY + name)
        except Exception as exc:  # network errors, HTTP errors, timeouts
            raise DataError(f"the download of {name} failed: {type(exc).__name__}: {exc}") from exc
        text = _unzip(name, body)
        vintages[key] = _vintage(text)
        width = 6 if key.startswith("monthly") else 8
        tables[key] = parse_table(name, text, _HEADERS[key], width)
    first, last = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    daily3, daily5 = (
        table[(table.index >= first) & (table.index <= last)]
        for table in (tables["daily3"], tables["daily5"])
    )
    if daily3.empty:
        raise DataError(f"the library has no daily factors from {start} to {end}")
    if not daily3.index.equals(daily5.index):
        days = sorted(set(daily3.index) ^ set(daily5.index))
        raise DataError(
            f"the three- and five-factor files cover different days in the window (first "
            f"{days[0]}, {len(days)} in all)"
        )
    for column in _SHARED:
        differ = daily3.index[daily3[column] != daily5[column]]
        if len(differ):
            raise DataError(
                f"{column} differs between the three- and five-factor files on {differ[0]} "
                f"and {len(differ) - 1} more days"
            )
    monthly = tables["monthly3"]["RF"]
    months = daily3.index.str[:6]
    missing = sorted(set(months) - set(monthly.index))
    if missing:
        raise DataError(f"the monthly file has no risk-free rate for {missing[0]}")
    try:
        days = [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in daily3.index]
    except ValueError as exc:
        raise DataError(f"the daily files hold a date that is not one: {exc}") from None
    factors = pd.DataFrame(
        {
            "date": days,
            "mkt_rf": daily3["Mkt-RF"].to_numpy(),
            "smb": daily3["SMB"].to_numpy(),
            "hml": daily3["HML"].to_numpy(),
            "rf": daily3["RF"].to_numpy(),
            "smb5": daily5["SMB"].to_numpy(),
            "rmw": daily5["RMW"].to_numpy(),
            "cma": daily5["CMA"].to_numpy(),
            "rf_month": monthly.loc[months].to_numpy(),
        }
    )
    validate_factors(factors, start, end)
    return factors, vintages


def validate_factors(factors: pd.DataFrame, start: date, end: date) -> None:
    """Refuse factors that later steps would misuse; raise ``DataError`` naming the problem."""
    if list(factors.columns) != list(COLUMNS):
        raise DataError(f"expected the columns {', '.join(COLUMNS)}, got {list(factors.columns)}")
    if factors.empty:
        raise DataError("no factors")
    days = list(factors["date"])
    if any(not isinstance(d, date) or isinstance(d, datetime) for d in days):
        raise DataError("a date is not a calendar date")
    if any(b <= a for a, b in zip(days, days[1:], strict=False)):
        raise DataError("dates are repeated or out of order")
    if days[0] < start or days[-1] > end:
        raise DataError(f"factors from {days[0]} to {days[-1]}, outside {start} to {end}")
    for column in FACTOR_COLUMNS:
        values = factors[column]
        bad = ~values.map(lambda v: isinstance(v, float) and math.isfinite(v))
        if bad.any():
            row = factors[bad].iloc[0]
            raise DataError(f"{row['date']}: {column} is {row[column]}, not a number")
        # A daily return of -100% or worse, in percent.
        if (values <= -100.0).any():
            row = factors[values <= -100.0].iloc[0]
            raise DataError(f"{row['date']}: {column} is {row[column]}%, not above -100%")
    months = pd.Series([(d.year, d.month) for d in days])
    varies = factors.groupby(months.to_numpy())["rf_month"].nunique() > 1
    if varies.any():
        year, month = varies[varies].index[0]
        raise DataError(f"{year}-{month:02d}: the month's risk-free rate differs between days")


def daily_risk_free(factors: pd.DataFrame) -> pd.Series:
    """The simple daily rate that compounds to each month's rate over its trading days.

    ``factors`` is ``load_factors``' frame (fractions, indexed by date); the rate is a
    fraction, on the same dates.
    """
    month = factors.index.to_period("M")
    days = pd.Series(1, index=factors.index).groupby(month).transform("size")
    rate = np.expm1(np.log1p(factors["rf_month"].to_numpy()) / days.to_numpy())
    return pd.Series(rate, index=factors.index, name="rf")


def _csv_bytes(factors: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    out = factors.copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def write_cache(
    factors: pd.DataFrame,
    cache: Path,
    *,
    start: date,
    end: date,
    vintages: dict[str, str],
    fetched_at: datetime | None = None,
) -> dict[str, Any]:
    """Write ``factors`` to ``cache`` and its manifest beside it; return the manifest."""
    validate_factors(factors, start, end)
    data = _csv_bytes(factors)
    fetched_at = fetched_at or datetime.now(UTC)
    manifest = {
        "source": "Kenneth R. French Data Library",
        "files": {key: LIBRARY + name for key, name in FILES.items()},
        "vintages": dict(vintages),
        "fetched_at": fetched_at.astimezone(UTC).isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "file": cache.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(factors),
        "first": factors["date"].iloc[0].isoformat(),
        "last": factors["date"].iloc[-1].isoformat(),
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
    required = {"start", "end", "file", "sha256", "rows", "first", "last"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise DataError(f"{path} is not a factors manifest (needs {', '.join(sorted(required))})")
    return manifest


def load_factors(cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """The cached factors as fractions, indexed by date; refused unless they match their
    manifest exactly."""
    cache = Path(cache)
    manifest = read_manifest(cache)
    try:
        data = cache.read_bytes()
    except FileNotFoundError:
        raise DataError(
            f"{cache} is missing; fetch it with: python -m portfolio_risk.french fetch"
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
        factors = pd.read_csv(
            io.BytesIO(data),
            dtype={"date": str} | {column: "float64" for column in FACTOR_COLUMNS},
            keep_default_na=False,
            **EXACT_FLOATS,
        )
        if list(factors.columns) != list(COLUMNS):
            raise DataError(f"{cache}: expected the columns {', '.join(COLUMNS)}")
        factors["date"] = [date.fromisoformat(d) for d in factors["date"]]
    except (ValueError, TypeError) as exc:
        raise DataError(f"{cache} cannot be parsed: {exc}") from None
    validate_factors(factors, start, end)
    if (
        len(factors) != manifest["rows"]
        or factors["date"].iloc[0].isoformat() != manifest["first"]
        or factors["date"].iloc[-1].isoformat() != manifest["last"]
    ):
        raise DataError(f"{cache}: rows or dates differ from its manifest")
    out = factors.set_index(pd.DatetimeIndex(factors["date"], name="date"))
    return out[list(FACTOR_COLUMNS)] / 100.0


def _print_summary(manifest: dict[str, Any]) -> None:
    print(
        f"{manifest['rows']} days, {manifest['first']} to {manifest['last']} "
        f"(window {manifest['start']} to {manifest['end']})"
    )
    for key, vintage in manifest.get("vintages", {}).items():
        print(f"  {key:<9} {vintage}")
    print(f"SHA-256 {manifest['sha256']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.french", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download the factors into the cache")
    fetch.add_argument(
        "--start", type=_parse_date, default=DEFAULT_START, help="the first day of a month"
    )
    fetch.add_argument("--end", type=_parse_date, default=DEFAULT_END, help="a month's last day")
    fetch.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    verify = commands.add_parser("verify", help="check a cache against its manifest")
    verify.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)

    try:
        if args.command == "fetch":
            factors, vintages = fetch_factors(args.start, args.end)
            previous = None
            if manifest_path(args.out).exists():
                try:
                    previous = read_manifest(args.out)
                except DataError:
                    previous = None
            data = _csv_bytes(factors)
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
                    factors, args.out, start=args.start, end=args.end, vintages=vintages
                )
            _print_summary(manifest)
            if previous is not None:
                print(
                    "Same data as the previous manifest, which is left as it was."
                    if same
                    else "Differs from the previous manifest."
                )
        else:
            load_factors(args.cache)
            _print_summary(read_manifest(args.cache))
            print(f"{args.cache} matches its manifest.")
    except (DataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
