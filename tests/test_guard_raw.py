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
        "ls data/masked/..",
        "cat data/masked/../other/a.txt",
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
        "ls data/source",
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


@pytest.mark.parametrize(
    "tool_input",
    [
        {"command": "curl http://127.0.0.1:8765/api/drawings"},
        {"command": "curl localhost:8765"},
        {"command": "wget http://LOCALHOST:8765/"},
        {"command": "python -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:8765')\""},
        {"url": "http://localhost:8765/api/drawings/a/image"},
        {"url": "http://[::1]:8765/"},
    ],
)
def test_review_server_is_for_people_only(tool_input):
    result = run_hook(tool_input)
    assert result.returncode == BLOCK
    assert "open_review" in result.stderr


@pytest.mark.parametrize(
    "tool_input",
    [
        {"command": "curl http://127.0.0.1:18765/"},
        {"command": "echo 8765"},
        {"command": "curl https://example.com/localhost"},
        {"url": "https://docs.example.com/port-8765"},
    ],
)
def test_other_ports_and_hosts_pass(tool_input):
    assert run_hook(tool_input).returncode == ALLOW


def test_review_preview_cache_is_protected():
    """表示用の PNG は塗る前の画像そのもの。置き場所の名前でフックが効く。"""
    assert run_hook({"file_path": "data/cache/raw/abc.png"}).returncode == BLOCK
    assert run_hook({"command": "ls data/cache/raw"}).returncode == BLOCK
    assert run_hook({"command": "ls data/cache"}).returncode == ALLOW


def test_settings_cover_webfetch_and_preview_cache():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert "WebFetch" in settings["hooks"]["PreToolUse"][0]["matcher"].split("|")
    assert "Read(./data/cache/raw/**)" in settings["permissions"]["deny"]


@pytest.mark.parametrize(
    "tool_input",
    [
        {"command": "grep -o '\"confirmed\": *[a-z]*' data/masked/dummy_001.chiban.json"},
        {"command": "grep -E '[a-z]+' docs/design.md"},
        {"pattern": "[a-z]*"},
        {"command": "ls r*"},
    ],
)
def test_regex_without_path_separator_is_not_treated_as_glob(tool_input):
    """`[a-z]*` のような正規表現だけでは止めない（Q15）。"""
    assert run_hook(tool_input).returncode == ALLOW


@pytest.mark.parametrize(
    "tool_input,rule",
    [
        ({"command": "ls data/raw/*.tif"}, "保護フォルダへのパス"),
        ({"pattern": "data/raw/*.tif"}, "保護フォルダへのパス"),
        ({"file_path": "data/raw/a.tif"}, "保護フォルダへのパス"),
        ({"command": "cat data/r*/a.tif"}, "保護フォルダを指すグロブ"),
        ({"command": "ls data/[r]aw"}, "保護フォルダを指すグロブ"),
        ({"command": "ls raw*"}, "保護フォルダを指すグロブ"),
        ({"command": "grep -r x data"}, "保護フォルダの親"),
        ({"command": "curl http://127.0.0.1:8765/"}, "確認画面への接続"),
    ],
)
def test_block_message_names_the_rule(tool_input, rule):
    """止めたときは、どの規則で止めたかを1行添える（Q15）。"""
    result = run_hook(tool_input)
    assert result.returncode == BLOCK
    assert f"止めた規則：{rule}" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "kurokofude mask data/raw/a.tif --force",
        "kurokofude mask --force data/raw/a.tif",
        "kurokofude batch data/raw --force",
        "kurokofude mask tests/fixtures/a.tif --force",
        "kurokofude mask data/raw/a.tif --fo",
        "kurokofude mask data/raw/a.tif --f",
        "kurokofude mask data/raw/a.tif '--force'",
        ".venv/bin/kurokofude mask tests/fixtures/a.tif --force",
        "python -m kurokofude.cli batch tests/fixtures --force",
        "cd tests; kurokofude mask fixtures/a.tif --force",
    ],
)
def test_forced_overwrite_is_for_people_only(command):
    """確定の上書きは、許可リストのコマンドでも止める（§19）。"""
    result = run_hook({"command": command})
    assert result.returncode == BLOCK
    assert "止めた規則：確定の上書きは人だけ" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "kurokofude mask data/raw/a.tif",
        "kurokofude batch data/raw --out data/masked",
        "kurokofude review data/masked/a.tif",
        "git push --force",
        "kurokofude eval --truth data/truth/a.json --result data/masked/a.json --image data/raw/a.tif",
        "kurokofude mask data/raw/a.tif --config configs/force.yaml",
    ],
)
def test_commands_without_forced_overwrite_still_pass(command):
    assert run_hook({"command": command}).returncode == ALLOW


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
