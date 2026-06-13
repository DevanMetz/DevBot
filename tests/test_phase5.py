"""Phase 5: verify README env-var docs are complete and match the code."""
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_exists_and_contains_safety_model():
    """The README file should exist and document the safety model."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.is_file(), "README.md not found"
    content = readme.read_text(encoding="utf-8")
    assert "Safety model" in content, "README.md missing 'Safety model' section"


# ---------------------------------------------------------------------------
# Helpers: extract DEVBOT_* env-var names from the README config table
# ---------------------------------------------------------------------------

def _readme_table_vars(readme_path: Path) -> set[str]:
    """Parse the README Configuration table and return all DEVBOT_* var names."""
    text = readme_path.read_text(encoding="utf-8")
    # Find the Configuration section
    m = re.search(r'## Configuration\n\n\| .*?\n\n', text, re.DOTALL)
    assert m, "Could not find '## Configuration' section with a table in README"
    table_text = m.group(0)
    # Each row starts with | `VAR_NAME` | ...
    vars_found = set()
    for line in table_text.splitlines():
        match = re.match(r'^\| `([^`]+)`', line.strip())
        if match:
            name = match.group(1)
            if name.startswith("DEVBOT_") or name == "DEEPSEEK_API_KEY":
                vars_found.add(name)
    return vars_found


# ---------------------------------------------------------------------------
# Helpers: extract DEVBOT_* env-var names from Python source files
# ---------------------------------------------------------------------------

def _source_vars(file_paths: list[Path]) -> set[str]:
    """Scan Python files for os.environ[...] / os.environ.get(...) of DEVBOT_* vars."""
    names: set[str] = set()
    # Matches: os.environ.get("DEVBOT_X", ...)  or  os.environ["DEVBOT_X"]
    # Also int(os.environ.get(...))
    pattern = re.compile(
        r'os\.environ(?:\.get)?\(\s*["\'](DEVBOT_[A-Z_]+)["\']'
    )
    for fp in file_paths:
        if not fp.is_file():
            continue
        content = fp.read_text(encoding="utf-8")
        for m in pattern.finditer(content):
            names.add(m.group(1))
    return names


# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------

def test_every_readme_env_var_exists_in_code():
    """No invented env vars in README — every listed DEVBOT_* must be in the code."""
    readme = PROJECT_ROOT / "README.md"
    readme_vars = _readme_table_vars(readme)

    src_files = [
        PROJECT_ROOT / "devbot" / "agent.py",
        PROJECT_ROOT / "devbot" / "swarm.py",
        PROJECT_ROOT / "devbot" / "devlog.py",
    ]
    code_vars = _source_vars(src_files)

    # DEEPSEEK_API_KEY is not a DEVBOT_* var — exclude it for the exact match.
    readme_devbot = {v for v in readme_vars if v.startswith("DEVBOT_")}

    # Every DEVBOT_* in the README must be found in the code.
    missing_in_code = readme_devbot - code_vars
    assert not missing_in_code, (
        f"README lists env vars not found in code: {sorted(missing_in_code)}"
    )


def test_every_code_env_var_exists_in_readme():
    """No undocumented env vars — every DEVBOT_* in code must be in the README table."""
    readme = PROJECT_ROOT / "README.md"
    readme_vars = _readme_table_vars(readme)

    src_files = [
        PROJECT_ROOT / "devbot" / "agent.py",
        PROJECT_ROOT / "devbot" / "swarm.py",
        PROJECT_ROOT / "devbot" / "devlog.py",
    ]
    code_vars = _source_vars(src_files)

    # Every DEVBOT_* in the code must be in the README.
    missing_in_readme = code_vars - readme_vars
    assert not missing_in_readme, (
        f"Code uses env vars not documented in README: {sorted(missing_in_readme)}"
    )
