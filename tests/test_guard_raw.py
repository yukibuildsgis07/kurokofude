"""関所フックの判定を、実際に標準入力へ JSON を渡して確かめる。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "guard_raw.py"
BLOCK = 2
ALLOW = 0


def run_hook(tool_input: dict, stdin: str | None = None) -> subprocess.CompletedProcess:
    payload = stdin if stdin is not None else json.dumps(
        {"tool_name": "Any", "tool_input": tool_input, "cwd": str(ROOT)}
    )
    return subprocess.run(
        [sys.executable, str(HOOK)], input=payload, capture_output=True, text=True, encoding="utf-8"
    )


@pytest.mark.parametrize(
    "tool_input",
    [
        {"file_path": "data/raw/a.tif"},
        {"file_path": str(ROOT / "data" / "raw" / "a.tif")},
        {"file_path": "data/foo/../raw/a.tif"},
        {"file_path": "data\\raw\\a.tif"},
        {"file_path": "data/RAW/a.tif"},
        {"path": "data/raw"},
        {"pattern": "data/raw/**"},
        {"pattern": "**/raw/*.tif"},
        {"pattern": "*.tif", "path": "data/raw"},
        {"pattern": "name", "glob": "raw/**"},
        {"command": "cat data/raw/a.tif"},
        {"command": "cd data/raw && ls"},
        {"command": "cd data && cat raw/a.tif"},
        {"command": "cat data/ra''w/a.tif"},
        {"command": 'cat "data/ra"w/a.tif'},
        {"command": "cat data/r*/a.tif"},
        {"command": "cat data/ra?/a.tif"},
        {"command": "python x.py --in=data/raw/a.tif"},
        {"command": "type data\\raw\\a.tif"},
    ],
)
def test_blocks_protected_access(tool_input):
    result = run_hook(tool_input)
    assert result.returncode == BLOCK
    assert "mask_drawing" in result.stderr


@pytest.mark.parametrize(
    "tool_input",
    [
        {"file_path": "docs/design.md"},
        {"file_path": "data/masked/a.tif"},
        {"file_path": "src/kurokofude/rawness.py"},
        {"pattern": "**/*.py"},
        {"command": "pytest"},
        {"command": "ls data/masked"},
        {"command": "cat src/*.py"},
    ],
)
def test_allows_other_access(tool_input):
    assert run_hook(tool_input).returncode == ALLOW


def test_symlink_to_protected_dir_is_blocked(tmp_path):
    protected = tmp_path / "data" / "raw"
    protected.mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(protected, target_is_directory=True)
    assert run_hook({"file_path": str(link / "a.tif")}).returncode == BLOCK


def test_unreadable_input_fails_closed():
    assert run_hook({}, stdin="not json").returncode == BLOCK
