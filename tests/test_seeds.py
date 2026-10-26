"""A seed fixes a run: in any process, alongside any other work, and apart from every
other seed."""

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import spreads
from portfolio_risk.data import COLUMNS, write_cache
from portfolio_risk.montecarlo import BLOCK, NormalModel, StudentTModel, simulate


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01 * (1 + i), len(days)))
        for day, value in zip(days, close, strict=True):
            rows.append((day.date(), ticker, float(value), 0.0, 0.0, 0.0, 1e6))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def run(module: str, argv: list[str], hash_seed: str) -> subprocess.CompletedProcess:
    """The command in a fresh interpreter, with its own string hashing."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    return subprocess.run(
        [sys.executable, "-m", f"portfolio_risk.{module}", *argv],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "module, extra",
    [
        # Three models, and a last block of 200 paths.
        ("montecarlo", ["--paths", "1200", "--horizon", "21", "--seed", "5"]),
        ("scenarios", ["--mix", "equal", "--shuffles", "20", "--seed", "5"]),
    ],
)
def test_a_seed_gives_the_same_file_in_any_process(tmp_path: Path, module, extra) -> None:
    cache = tmp_path / "prices.csv"
    write_cache(prices(), cache, start=date(2023, 1, 1), end=date(2024, 12, 31), source="t")
    written = []
    for hash_seed in ("0", "1", "12345"):
        out = tmp_path / f"{module}-{hash_seed}.json"
        done = run(module, ["--cache", str(cache), *extra, "--json", str(out)], hash_seed)
        assert done.returncode == 0, done.stderr
        written.append(out.read_bytes())
    assert written[0] == written[1] == written[2]
    # Another seed gives another file.
    other = tmp_path / "other.json"
    argv = ["--cache", str(cache), *extra[:-1], "6", "--json", str(other)]
    assert run(module, argv, "0").returncode == 0
    assert other.read_bytes() != written[0]


def test_no_two_seeds_share_a_stream() -> None:
    # Each block of paths has its own stream spawned from the seed: across 20 seeds of
    # two blocks each, no path repeats another's first day.
    model = NormalModel(("A",), np.array([0.0]), np.array([[1e-4]]))
    first = np.concatenate(
        [simulate(model, {"A": 1.0}, 2, 2 * BLOCK, seed).returns[:, 0] for seed in range(20)]
    )
    assert len(np.unique(first)) == len(first)


def test_the_models_of_one_run_draw_the_same_normals() -> None:
    # The t scales the normal model's draw of each day by its own mixing draw, so on one
    # seed the two move in the same direction on every day: their difference is the
    # tails, not the luck of the draw.
    mean, covariance = np.array([0.0002]), np.array([[1e-4]])
    normal = simulate(NormalModel(("A",), mean, covariance), {"A": 1.0}, 50, 1_500, seed=3)
    t = simulate(StudentTModel(("A",), mean, covariance, 4.0), {"A": 1.0}, 50, 1_500, seed=3)
    shocks = np.log1p(normal.returns) - mean, np.log1p(t.returns) - mean
    assert (np.sign(shocks[0]) == np.sign(shocks[1])).all()
    assert not np.allclose(*shocks)


def test_simulated_bars_are_fixed_by_their_seed() -> None:
    a = spreads.simulate(0.001, 0.01, 30, trades_per_day=50, seed=8)
    b = spreads.simulate(0.001, 0.01, 30, trades_per_day=50, seed=8)
    c = spreads.simulate(0.001, 0.01, 30, trades_per_day=50, seed=9)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert not (a.to_numpy() == c.to_numpy()).any()
