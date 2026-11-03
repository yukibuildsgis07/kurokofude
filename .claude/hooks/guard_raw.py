#!/usr/bin/env python3
"""黒子筆の関所（設計書 §4.1）。塗る前の図面へのアクセスを PreToolUse で止める。

標準ライブラリだけで動かす（フックは venv の外の Python で走ることがある）。
判定できないときは止める側に倒す。終了コード 1 は Claude Code では「止めない」扱いなので、
例外もすべて終了コード 2 にする。
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

BLOCK_EXIT_CODE = 2
ALLOW_EXIT_CODE = 0
MESSAGE = "このファイルは黒子筆を通す前の図面です。`mask_drawing` ツールを使ってください"
DEFAULT_PROTECTED_DIRS = ("data/raw",)
CONFIG_RELATIVE_PATH = Path("configs") / "default.yaml"
PATH_FIELDS = ("file_path", "path", "notebook_path")
PATTERN_FIELDS = ("pattern", "glob")
GLOB_CHARS = "*?["
ALLOWED_COMMANDS_KEY = "hook_allowed_commands"
# 許可リストのコマンドに含めてはいけない文字（後ろに別のコマンドをつなぐ抜け道になる）
COMMAND_CHAIN_CHARS = ";&|<>`$()\n\r"
# シェル上でパスの区切りになりうる文字
TOKEN_SPLIT = re.compile(r"[\s=:;,|&()<>{}`$]+")


def read_config_list(project_dir: Path, key: str) -> list[str]:
    """configs/default.yaml の最上位のリストを読む。PyYAML に依存しないよう行単位で読む。"""
    config_path = project_dir / CONFIG_RELATIVE_PATH
    if not config_path.is_file():
        return []
    values = []
    in_block = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            in_block = True
        elif in_block and line.strip().startswith("- "):
            values.append(line.strip()[2:].split("#")[0].strip().strip("'\""))
        elif in_block and line.strip():
            in_block = False
    return values


def load_protected_names(project_dir: Path) -> set[str]:
    """保護ディレクトリの末尾名を集める。既定値は常に含める。"""
    dirs = list(DEFAULT_PROTECTED_DIRS) + read_config_list(project_dir, "protected_dirs")
    return {split_components(d)[-1] for d in dirs if d}


def command_is_allowed(command: str, allowed: list[str]) -> bool:
    """許可リストの語で始まり、別のコマンドをつなげない形のときだけ通す（設計書 §4.1）。"""
    if any(ch in command for ch in COMMAND_CHAIN_CHARS):
        return False
    words = command.replace("\\", "/").split()
    return any(words[: len(prefix.split())] == prefix.split() for prefix in allowed if prefix)


def split_components(text: str) -> list[str]:
    """Windows の区切りも含めて分解し、大文字小文字を畳む（Mac/Windows は区別しないため）。"""
    return [c for c in text.replace("\\", "/").casefold().split("/") if c not in ("", ".")]


def component_is_protected(component: str, names: set[str]) -> bool:
    if component in names:
        return True
    # `r*` や `ra?` のような狙い撃ちのグロブは止める。`*` `**` 単独は通常の検索なので通す
    if any(ch in component for ch in GLOB_CHARS) and component.strip("*"):
        return any(fnmatch.fnmatchcase(name, component) for name in names)
    return False


def text_hits_protected(text: str, names: set[str]) -> bool:
    return any(component_is_protected(c, names) for c in split_components(text))


def path_hits_protected(raw_path: str, cwd: Path, names: set[str]) -> bool:
    """`../` と絶対パス（normpath）、シンボリックリンク（realpath）の迂回を防ぐ。"""
    if text_hits_protected(raw_path, names):
        return True
    joined = cwd / os.path.expanduser(raw_path.replace("\\", "/"))
    return text_hits_protected(os.path.normpath(joined), names) or text_hits_protected(
        os.path.realpath(joined), names
    )


def command_hits_protected(command: str, cwd: Path, names: set[str]) -> bool:
    """シェルを解釈せず、語に分けて判定する。`ra""w` のような引用符の分断も潰す。"""
    unquoted = command.replace('"', "").replace("'", "")
    for text in {command, unquoted}:
        for token in TOKEN_SPLIT.split(text):
            if token and path_hits_protected(token, cwd, names):
                return True
    return False


def should_block(payload: dict) -> bool:
    tool_input = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    names = load_protected_names(project_dir)

    for field in PATH_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value and path_hits_protected(value, cwd, names):
            return True
    for field in PATTERN_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value and text_hits_protected(value, names):
            return True
    command = tool_input.get("command")
    if not isinstance(command, str) or not command_hits_protected(command, cwd, names):
        return False
    return not command_is_allowed(command, read_config_list(project_dir, ALLOWED_COMMANDS_KEY))


def main() -> int:
    try:
        blocked = should_block(json.load(sys.stdin))
    except Exception as exc:  # 判定不能は止める
        print(f"{MESSAGE}（関所の判定に失敗: {type(exc).__name__}）", file=sys.stderr)
        return BLOCK_EXIT_CODE
    if blocked:
        print(MESSAGE, file=sys.stderr)
        return BLOCK_EXIT_CODE
    return ALLOW_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
