import re
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import portfolio_risk

ROOT = Path(__file__).resolve().parents[1]


def test_the_installed_version_is_the_package_s() -> None:
    assert version("portfolio-risk-simulation") == portfolio_risk.__version__


def test_pyproject_takes_its_version_from_the_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "portfolio_risk.__version__"


def test_it_imports_from_a_source_tree_that_was_never_installed(tmp_path: Path) -> None:
    # A copy of the package alone, so no install metadata sits beside it, and
    # -S keeps site-packages, where the installed copy's metadata is, off the path.
    shutil.copytree(ROOT / "src" / "portfolio_risk", tmp_path / "portfolio_risk")
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        "import portfolio_risk; print(portfolio_risk.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == portfolio_risk.__version__


def test_constraints_pin_every_declared_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["dependencies"] + [
        d for extra in pyproject["project"]["optional-dependencies"].values() for d in extra
    ]
    pins = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            assert version, f"not an exact pin: {line}"
            pins[name.lower()] = version
    for requirement in declared:
        name = re.split(r"[\[<>=!~ ;]", requirement, maxsplit=1)[0].lower()
        assert name in pins, f"constraints.txt does not pin {name}"


def test_ci_s_lowest_job_installs_pyproject_s_lower_bounds() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    bounds = {}
    for requirement in pyproject["project"]["dependencies"]:
        name, _, version = requirement.partition(">=")
        bounds[name] = version
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lowest = [line for line in workflow.splitlines() if "pip install" in line and "==" in line]
    assert len(lowest) == 1
    assert dict(re.findall(r'"([a-z]+)==([0-9.]+)"', lowest[0])) == bounds
