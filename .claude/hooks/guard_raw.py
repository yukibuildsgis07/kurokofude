#!/usr/bin/env python3
"""黒子筆の関所（設計書 §4.1、§3.1）。塗る前の図面と確認画面へのアクセスを PreToolUse で止める。

標準ライブラリだけで動かす（フックは venv の外の Python で走ることがある）。
判定できないときは止める側に倒す。終了コード 1 は Claude Code では「止めない」扱いなので、
例外もすべて終了コード 2 にする。止めたときは、どの規則で止めたかを1行添える（Q15）。
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
REVIEW_MESSAGE = "確認画面は人が見るための画面です。Claude は接続できません。`open_review` ツールでブラウザを開いてください"
RULE_PATH = "保護フォルダへのパス"
RULE_GLOB = "保護フォルダを指すグロブ"
RULE_PARENT = "保護フォルダの親（data）を、開いてよいフォルダ以外で指すパス"
RULE_REVIEW = "確認画面への接続"
RULE_ERROR = "関所の判定に失敗"
DEFAULT_PROTECTED_DIRS = ("data/raw",)
CONFIG_RELATIVE_PATH = Path("configs") / "default.yaml"
PATH_FIELDS = ("file_path", "path", "notebook_path")
PATTERN_FIELDS = ("pattern", "glob")
REVIEW_FIELDS = ("command", "url")
GLOB_CHARS = "*?["
PATH_SEPARATORS = ("/", "\\")
ALLOWED_COMMANDS_KEY = "hook_allowed_commands"
OPEN_DIRS_KEY = "hook_open_dirs"
REVIEW_SECTION = "review"
REVIEW_PORT_KEY = "port"
DEFAULT_REVIEW_PORT = "8765"
# 確認画面に届きうるローカルのホスト名
LOCAL_HOSTS = ("localhost", "127.0.0.1", "127.1", "0.0.0.0", "[::1]", "::1")
INDENT = (" ", "\t")
# 許可リストのコマンドに含めてはいけない文字（後ろに別のコマンドをつなぐ抜け道になる）
COMMAND_CHAIN_CHARS = ";&|<>`$()\n\r"
# シェル上でパスの区切りになりうる文字
TOKEN_SPLIT = re.compile(r"[\s=:;,|&()<>{}`$]+")


def read_config_lines(project_dir: Path) -> list[str]:
    config_path = project_dir / CONFIG_RELATIVE_PATH
    return config_path.read_text(encoding="utf-8").splitlines() if config_path.is_file() else []


def read_config_list(project_dir: Path, key: str) -> list[str]:
    """configs/default.yaml の最上位のリストを読む。PyYAML に依存しないよう行単位で読む。"""
    values = []
    in_block = False
    for line in read_config_lines(project_dir):
        if line.startswith(f"{key}:"):
            in_block = True
        elif in_block and line.strip().startswith("- "):
            values.append(line.strip()[2:].split("#")[0].strip().strip("'\""))
        elif in_block and line.strip():
            in_block = False
    return values


def read_review_port(project_dir: Path) -> str:
    """configs/default.yaml の review: の下の port: を読む。"""
    in_block = False
    for line in read_config_lines(project_dir):
        if line.startswith(f"{REVIEW_SECTION}:"):
            in_block = True
        elif in_block and line[:1] in INDENT and line.strip().startswith(f"{REVIEW_PORT_KEY}:"):
            return line.split(":", 1)[1].split("#")[0].strip() or DEFAULT_REVIEW_PORT
        elif in_block and line.strip() and line[:1] not in INDENT:
            break
    return DEFAULT_REVIEW_PORT


def hits_review_server(text: str, port: str) -> bool:
    """確認画面（人の道具）への接続。ローカルのホスト名とポートの両方を含めば止める（§3.1）。"""
    folded = text.casefold()
    if not any(host in folded for host in LOCAL_HOSTS):
        return False
    return re.search(rf"(?<!\d){re.escape(port)}(?!\d)", text) is not None


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


def glob_allowed(word: str, names: set[str]) -> bool:
    """グロブとして判定してよい語か（Q15）。パスの区切りを含む語か、保護フォルダ名を含む語だけ。

    `[a-z]*` のような正規表現だけの語は、グロブとしては見ない。
    """
    folded = word.casefold()
    return any(sep in word for sep in PATH_SEPARATORS) or any(name in folded for name in names)


def component_rule(component: str, names: set[str], allow_glob: bool) -> str | None:
    if component in names:
        return RULE_PATH
    # `r*` や `ra?` のような狙い撃ちのグロブは止める。`*` `**` 単独は通常の検索なので通す
    if allow_glob and any(ch in component for ch in GLOB_CHARS) and component.strip("*"):
        if any(fnmatch.fnmatchcase(name, component) for name in names):
            return RULE_GLOB
    return None


def text_rule(text: str, names: set[str], allow_glob: bool) -> str | None:
    for component in split_components(text):
        rule = component_rule(component, names, allow_glob)
        if rule:
            return rule
    return None


def path_rule(raw_path: str, cwd: Path, names: set[str]) -> str | None:
    """`../` と絶対パス（normpath）、シンボリックリンク（realpath）の迂回を防ぐ。

    グロブとして見るかどうかは、元の語で決める（つないだ後のパスには必ず区切りがあるため）。
    """
    allow_glob = glob_allowed(raw_path, names)
    rule = text_rule(raw_path, names, allow_glob)
    if rule:
        return rule
    joined = cwd / os.path.expanduser(raw_path.replace("\\", "/"))
    return text_rule(os.path.normpath(joined), names, allow_glob) or text_rule(os.path.realpath(joined), names, allow_glob)


def load_parent_rule(project_dir: Path) -> tuple[set[str], set[str]]:
    """保護ディレクトリの親の名前（data）と、その下で開いてよい子の名前を返す。"""
    protected = list(DEFAULT_PROTECTED_DIRS) + read_config_list(project_dir, "protected_dirs")
    parents = {split_components(d)[-2] for d in protected if len(split_components(d)) >= 2}
    open_children = {split_components(d)[-1] for d in read_config_list(project_dir, OPEN_DIRS_KEY) if d}
    return parents, open_children


def parent_rule_hits(components: list[str], parents: set[str], open_children: set[str], allow_glob: bool) -> bool:
    """親の名前は、開いてよい子が続く形でなければ止める（`grep -r x data` などの穴を塞ぐ）。"""
    for index, component in enumerate(components):
        if component_rule(component, parents, allow_glob):
            following = components[index + 1] if index + 1 < len(components) else None
            if following not in open_children:
                return True
    return False


def path_inside_project(token: str, cwd: Path, project_dir: Path) -> str | None:
    """正規化してプロジェクト内なら相対パスを返す。置き場所の親にある data などを誤判定しない。"""
    joined = os.path.normpath(cwd / os.path.expanduser(token.replace("\\", "/")))
    try:
        relative = os.path.relpath(joined, os.path.normpath(project_dir))
    except ValueError:  # Windows でドライブが違う
        return None
    return None if relative.startswith("..") else relative


def command_rule(command: str, cwd: Path, project_dir: Path, names: set[str]) -> str | None:
    """シェルを解釈せず、語に分けて判定する。`ra""w` のような引用符の分断も潰す。"""
    parents, open_children = load_parent_rule(project_dir)
    unquoted = command.replace('"', "").replace("'", "")
    for text in {command, unquoted}:
        for token in TOKEN_SPLIT.split(text):
            if not token:
                continue
            rule = path_rule(token, cwd, names)
            if rule:
                return rule
            allow_glob = glob_allowed(token, names | parents)
            if parent_rule_hits(split_components(token), parents, open_children, allow_glob):
                return RULE_PARENT
            relative = path_inside_project(token, cwd, project_dir)
            if relative and parent_rule_hits(split_components(relative), parents, open_children, allow_glob):
                return RULE_PARENT
    return None


def protected_rule(payload: dict) -> str | None:
    """塗る前の図面に関わる規則で止めるなら、その規則の名前を返す。"""
    tool_input = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    names = load_protected_names(project_dir)

    for field in PATH_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value:
            rule = path_rule(value, cwd, names)
            if rule:
                return rule
    for field in PATTERN_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value:
            rule = text_rule(value, names, glob_allowed(value, names))
            if rule:
                return rule
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    rule = command_rule(command, cwd, project_dir, names)
    if rule is None or command_is_allowed(command, read_config_list(project_dir, ALLOWED_COMMANDS_KEY)):
        return None
    return rule


def should_block(payload: dict) -> bool:
    return protected_rule(payload) is not None


def block_reason(payload: dict) -> str | None:
    """止める理由（メッセージと、止めた規則の1行）を返す。止めないときは None。"""
    tool_input = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    port = read_review_port(project_dir)
    for field in REVIEW_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and hits_review_server(value, port):
            return f"{REVIEW_MESSAGE}\n止めた規則：{RULE_REVIEW}"
    rule = protected_rule(payload)
    return f"{MESSAGE}\n止めた規則：{rule}" if rule else None


def main() -> int:
    try:
        reason = block_reason(json.load(sys.stdin))
    except Exception as exc:  # 判定不能は止める
        print(f"{MESSAGE}\n止めた規則：{RULE_ERROR}（{type(exc).__name__}）", file=sys.stderr)
        return BLOCK_EXIT_CODE
    if reason:
        print(reason, file=sys.stderr)
        return BLOCK_EXIT_CODE
    return ALLOW_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
