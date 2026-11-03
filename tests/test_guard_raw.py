"""関所フックの判定を、実際に標準入力へ JSON を渡して確かめる。"""

import json
import os
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


@pytest.mark.parametrize(
    "command",
    [
        "kurokofude mask data/raw/a.tif",
        "kurokofude mask data/raw/a.tif --out data/masked",
        "kurokofude batch data/raw",
        "kurokofude eval --truth data/truth --result data/masked --image data/raw",
        "python tools/make_dummy/make.py --out data/raw --count 5",
        "python tools\\make_dummy\\make.py --out data\\raw --count 5",
    ],
)
def test_allowed_commands_pass_with_protected_path(command):
    assert run_hook({"command": command}).returncode == ALLOW


@pytest.mark.parametrize(
    "command",
    [
        "kurokofude mask data/raw/a.tif; cat data/raw/a.tif",
        "kurokofude mask data/raw/a.tif && cat data/raw/a.tif",
        "kurokofude mask data/raw/a.tif | cat data/raw/a.tif",
        "kurokofude mask data/raw/a.tif\ncat data/raw/a.tif",
        "kurokofude mask $(cat data/raw/a.txt)",
        "kurokofude mask `cat data/raw/a.txt`",
        "kurokofude mask data/raw/a.tif > data/raw/b.txt",
        "kurokofude maskx data/raw/a.tif",
        "kurokofude review data/raw/a.tif",
        "kurokofude evalx --image data/raw/a.tif",
        "cat data/raw/a.tif kurokofude mask",
        "FOO=1 kurokofude mask data/raw/a.tif",
        "python tools/make_dummy/other.py --out data/raw",
        "python -c open('data/raw/a.tif')",
    ],
)
def test_other_commands_stay_blocked(command):
    assert run_hook({"command": command}).returncode == BLOCK


@pytest.mark.parametrize(
    "command",
    [
        "grep -r x data",
        "ls data",
        "ls data/",
        "find data -name a.tif",
        "ls data/source",
        "ls data/masked/..",
        "cat data/masked/../source/a.txt",
        "ls d*/*",
        "python -c \"open('data/' + 'raw' + '/a.tif')\"",
        "python -c \"open('data/'+'ra'+'w/a.tif')\"",
        "ls DATA",
    ],
)
def test_parent_of_protected_dir_is_blocked_unless_open_child_follows(command):
    assert run_hook({"command": command}).returncode == BLOCK


@pytest.mark.parametrize(
    "command",
    [
        "ls data/masked",
        "cat data/truth/a.truth.json",
        "rm data/cache/a.json",
        "python x.py data/ref/a.geojson",
        "cat metadata/a.txt",
        "python tools/make_dummy/make.py --source data/source --out tests/fixtures",
    ],
)
def test_open_children_and_other_words_pass(command):
    assert run_hook({"command": command}).returncode == ALLOW


def test_project_placed_under_a_folder_named_like_the_parent(tmp_path):
    """プロジェクトの置き場所の親に data があっても、それだけでは止めない。"""
    project = tmp_path / "data" / "kurokofude"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "default.yaml").write_text(
        "hook_open_dirs:\n  - data/masked\n", encoding="utf-8"
    )
    payload = json.dumps({"tool_input": {"command": "pytest"}, "cwd": str(project)})
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project)}
    result = subprocess.run(
        [sys.executable, str(HOOK)], input=payload, capture_output=True, text=True, encoding="utf-8", env=env
    )
    assert result.returncode == ALLOW


def test_settings_deny_read_glob_grep_on_protected_dir():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    deny = settings["permissions"]["deny"]
    for tool in ("Read", "Glob", "Grep"):
        assert f"{tool}(./data/raw/**)" in deny


def test_allow_list_does_not_apply_to_read():
    assert run_hook({"file_path": "data/raw/kurokofude mask"}).returncode == BLOCK


def test_symlink_to_protected_dir_is_blocked(tmp_path):
    protected = tmp_path / "data" / "raw"
    protected.mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(protected, target_is_directory=True)
    assert run_hook({"file_path": str(link / "a.tif")}).returncode == BLOCK


def test_unreadable_input_fails_closed():
    assert run_hook({}, stdin="not json").returncode == BLOCK
