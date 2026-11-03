"""② 確認画面（設計書 §3、§3.1）。人の道具で、Claude の道具ではない。

127.0.0.1 にだけ bind する。塗る前の画像を扱うのはこの画面だけで、ブラウザはローカル。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import socket
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..config import load_config
from ..mask import write_masked_tiff
from ..ocr import read_image_for_ocr

HOST = "127.0.0.1"
MASKED_DIR = Path("data") / "masked"
PREVIEW_DIR = Path("data") / "cache" / "raw"  # 塗る前の画像そのもの。名前に raw を含めてフックを効かせる
RESULT_SUFFIX = ".chiban.json"
WORLD_FILE_SUFFIX = ".tfw"
TIFF_SUFFIXES = {".tif", ".tiff"}
NAME_PATTERN = re.compile(r"^[\w.\-]+$")
STATUS_REMOVED = "removed"
STATUS_ADDED = "added"
STATUS_CONFIRMED = "confirmed"
ORIGIN_STATUSES = {"auto", "unpaired", "in_title_block", "mixed_masked", "split", "uncertain_chiban", STATUS_ADDED}
MIN_VERTICES = 3
HEALTH_PATH = "/api/health"
HEALTH_MARKER = "kurokofude-review"
STATIC_DIR = Path(__file__).resolve().parent / "static"
HASH_CHUNK_BYTES = 1024 * 1024


class MaskIn(BaseModel):
    quad: list[list[float]]
    status: str
    origin: str | None = None
    chiban: str | None = None
    uncertain_index: int | None = None


class MasksIn(BaseModel):
    masks: list[MaskIn]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class ReviewStore:
    """data/masked/ の塗った図面と、それに対応する塗る前の図面を扱う。"""

    def __init__(self, root: Path, config: dict):
        self.root = root.resolve()
        self.config = config
        self.masked_dir = self.root / MASKED_DIR
        self.preview_dir = self.root / PREVIEW_DIR
        self.protected = [(self.root / d).resolve() for d in config["protected_dirs"]]

    def names(self) -> list[str]:
        return sorted(
            p.stem for p in self.masked_dir.glob("*") if p.suffix.lower() in TIFF_SUFFIXES and self._result_path(p.stem).is_file()
        )

    def _result_path(self, name: str) -> Path:
        return self.masked_dir / f"{name}{RESULT_SUFFIX}"

    def masked_tif(self, name: str) -> Path:
        if not NAME_PATTERN.fullmatch(name):
            raise HTTPException(404, "図面が見つかりません")
        for suffix in TIFF_SUFFIXES:
            path = self.masked_dir / f"{name}{suffix}"
            if path.is_file() and self._result_path(name).is_file():
                return path
        raise HTTPException(404, "図面が見つかりません")

    def load(self, name: str) -> dict:
        self.masked_tif(name)
        return json.loads(self._result_path(name).read_text(encoding="utf-8"))

    def save(self, name: str, result: dict) -> None:
        self._result_path(name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    def raw_drawing(self, name: str) -> Path:
        """chiban.json の source（ファイル名）を、保護ディレクトリの中から探す。"""
        source = self.load(name)["source"]
        if not NAME_PATTERN.fullmatch(source):
            raise HTTPException(404, "塗る前の図面が見つかりません")
        for directory in self.protected:
            direct = directory / source
            if direct.is_file():
                return direct
            found = next((p for p in directory.rglob(source) if p.is_file()), None)
            if found is not None:
                return found
        raise HTTPException(404, "塗る前の図面が見つかりません")

    def preview(self, name: str) -> Path:
        raw = self.raw_drawing(name)
        png = self.preview_dir / f"{_sha256(raw)}.png"
        if not png.is_file():
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", read_image_for_ocr(raw))
            if not ok:
                raise HTTPException(500, "表示用の画像を作れませんでした")
            png.write_bytes(encoded.tobytes())
        return png

    def update_masks(self, name: str, masks: list[MaskIn]) -> dict:
        result = self.load(name)
        uncertain_count = len(result.get("uncertain", []))
        stored = []
        for mask in masks:
            origin = mask.origin or mask.status
            if origin not in ORIGIN_STATUSES:
                raise HTTPException(422, "塗りの種類が不正です")
            if mask.status not in {origin, STATUS_REMOVED, STATUS_ADDED, STATUS_CONFIRMED}:
                raise HTTPException(422, "塗りの状態が不正です")
            if len(mask.quad) < MIN_VERTICES or any(len(point) != 2 for point in mask.quad):
                raise HTTPException(422, "塗りの形が不正です")
            if mask.uncertain_index is not None and not 0 <= mask.uncertain_index < uncertain_count:
                raise HTTPException(422, "数字主体の枠の番号が不正です")
            item = {"quad": mask.quad, "chiban": mask.chiban, "status": mask.status, "origin": origin}
            if mask.uncertain_index is not None:
                item["uncertain_index"] = mask.uncertain_index
            stored.append(item)
        chosen = {item["uncertain_index"] for item in stored if "uncertain_index" in item and item["status"] != STATUS_REMOVED}
        for index, item in enumerate(result.get("uncertain", [])):
            item["decision"] = "mask" if index in chosen else "keep"
        result["masks"] = stored
        result["confirmed"] = False
        result.pop("confirmed_at", None)
        self.save(name, result)
        return result

    def confirm(self, name: str) -> dict:
        """人が決めた塗りで、塗る前の図面から塗り直して上書きする。地番の保護はかけない。"""
        result = self.load(name)
        for mask in result["masks"]:
            mask.setdefault("origin", mask["status"])
            if mask["status"] not in {STATUS_REMOVED, STATUS_ADDED}:
                mask["status"] = STATUS_CONFIRMED
        raw = self.raw_drawing(name)
        out_tif = self.masked_tif(name)
        polygons = [m["quad"] for m in result["masks"] if m["status"] != STATUS_REMOVED]
        write_masked_tiff(raw, out_tif, polygons, self.config["mask"]["color"])
        world_file = raw.with_suffix(WORLD_FILE_SUFFIX)
        if world_file.is_file():
            shutil.copyfile(world_file, out_tif.with_suffix(WORLD_FILE_SUFFIX))
        result["confirmed"] = True
        result["confirmed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.save(name, result)
        return result


def create_app(root: Path, config: dict | None = None) -> FastAPI:
    store = ReviewStore(root, config or load_config())
    app = FastAPI(title="黒子筆 確認画面", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

    @app.get(HEALTH_PATH)
    def health() -> dict:
        return {"app": HEALTH_MARKER}

    @app.get("/api/drawings")
    def list_drawings() -> dict:
        drawings = []
        for name in store.names():
            result = store.load(name)
            drawings.append({"name": name, "confirmed": bool(result.get("confirmed")), "masks": len(result["masks"])})
        return {"drawings": drawings}

    @app.get("/api/drawings/{name}")
    def get_drawing(name: str) -> dict:
        return store.load(name)

    @app.get("/api/drawings/{name}/image")
    def get_image(name: str) -> FileResponse:
        return FileResponse(store.preview(name), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.put("/api/drawings/{name}")
    def put_drawing(name: str, body: MasksIn) -> dict:
        return store.update_masks(name, body.masks)

    @app.post("/api/drawings/{name}/confirm")
    def confirm_drawing(name: str) -> dict:
        return store.confirm(name)

    return app


def review_url(port: int, name: str) -> str:
    return f"http://{HOST}:{port}/?drawing={name}"


def is_running(port: int) -> bool:
    """このポートで黒子筆の確認画面が動いているか。"""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}{HEALTH_PATH}", timeout=1) as response:  # noqa: S310
            return json.loads(response.read()).get("app") == HEALTH_MARKER
    except (OSError, ValueError):
        return False


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def serve_forever(root: Path, port: int, config: dict | None = None) -> None:
    uvicorn.run(create_app(root, config), host=HOST, port=port, log_level="warning")


def ensure_running(root: Path, port: int, config: dict | None = None, wait_seconds: float = 10.0) -> None:
    """動いていなければ、裏のスレッドで起動して応答するまで待つ（MCP サーバーから使う）。"""
    if is_running(port):
        return
    if not port_is_free(port):
        raise RuntimeError(f"ポート {port} が別のプログラムに使われています")
    server = uvicorn.Server(uvicorn.Config(create_app(root, config), host=HOST, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="kurokofude-review").start()
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_running(port):
            return
        time.sleep(0.1)
    raise RuntimeError("確認画面が起動しませんでした")


def open_in_browser(port: int, name: str) -> bool:
    return bool(webbrowser.open(review_url(port, name)))
