# 黒子筆（Kurokofude）

[![test](https://github.com/yukibuildsgis07/kurokofude/actions/workflows/test.yml/badge.svg)](https://github.com/yukibuildsgis07/kurokofude/actions/workflows/test.yml)
[![windows-smoke](https://github.com/yukibuildsgis07/kurokofude/actions/workflows/windows-smoke.yml/badge.svg)](https://github.com/yukibuildsgis07/kurokofude/actions/workflows/windows-smoke.yml)

スキャンした測量図面（TIFF＋ワールドファイル）から、地番の真下にある地権者の氏名だけを手元のPCで塗りつぶし、安心してAIに渡せるようにする「AIに図面を渡す前の関所」です。

## できること

- **氏名だけを塗る**：OCR で文字を見つけ、地番の真下にある文字のかたまりを、傾いたまま塗ります。地番は残します。氏名の文字は読まず、保存もしません。画素の位置は動かさないので、ワールドファイルはそのまま使えます
- **人が確かめる**：確認画面で、塗りを外す・足す・確定ができます
- **Claude は生の図面を見ない**：フックと deny 設定で塗る前の図面への直接アクセスを止め、MCP ツールは塗った図面と地番の一覧だけを返します
- **地番だけで位置合わせ（実験）**：塗った図面の地番の一覧と公図データを照合し、ワールドファイルを作ります。ダミー図面5枚で、わざと約43mずらした位置から始めて、照合のあとの位置のずれは平均 RMS 1.20m（最大 1.95m）でした。地番と公図の筆の対応は 467組中 463組を使いました（1/500 で 1.2m は紙の上で約 2.4mm）

精度の数字はすべて、`tools/make_dummy` で作ったダミー図面で測ったものです（`docs/accuracy_log.md`）。

## インストール

[uv](https://docs.astral.sh/uv/) を使います。Python 3.11 が必要です。

Windows での動作確認は GitHub Actions で自動化しています（`windows-smoke`：インストール、OCR、塗り、採点まで）。

### Mac

```
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

### Windows（PowerShell）

```
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

OCRのモデルは初回実行時に huggingface.co からダウンロードされます。

## 使い方

```
kurokofude mask <図面.tif>
kurokofude batch <図面のフォルダ>
kurokofude eval --truth data/truth/<図面>.truth.json --result data/masked/<図面>.chiban.json --image <図面.tif>
```

図枠の下辺ぜんぶが表題欄の図面（河川境界確定図など）では、`--config configs/bottom_strip.yaml` を付けてください。

### 確認画面

```
kurokofude review data/masked/<図面>.tif
```

ブラウザで確認画面が開きます（`127.0.0.1:8765`。この PC の中だけで動き、インターネットにはつながりません）。塗る前の図面に、塗る場所が色分けして重なります。

- クリック：塗りを外す／戻す。緑の枠（地番と判定できなかった数字）はクリックで「塗る」に切り替え
- Shift＋ドラッグ：塗りを追加（近くの地番の角度に合わせます）
- ドラッグ：移動、ホイール：拡大・縮小
- 「確定して塗り直す」：塗る前の図面から塗り直して `data/masked/` を上書きします。止めるときはターミナルで Ctrl+C

## 既知の制約

- 1桁の地番は検出されにくく、地番と分からないまま塗られることがあります。確認画面での確認が必要です
- OCR が文字として検出しなかった部分は塗られません。確認画面で人が足してください
- 塗り漏れがゼロになるとは限りません。AI に渡す前に確認画面で確かめてください

## Claude Code から使う（MCP）

リポジトリのルートに `.mcp.json` があります。このフォルダで Claude Code を起動すると、MCP サーバー `kurokofude` の承認を求められるので承認してください（`uv` が必要です）。

使えるツールは3つです。どれも、塗る前の図面の中身・パス、氏名の文字列を Claude に返しません。

- `mask_drawing`：`data/raw/` の図面を塗り、塗った図面のパスと地番の一覧を返す
- `open_review`：確認画面をブラウザで開く。URL は Claude に返さず、確定済みかどうかだけを返す
- `list_masked`：`data/masked/` の塗った図面の一覧

「この図面を黒子筆で処理して」と頼むと、スキル `kurokofude` が `mask_drawing` を呼んで結果をまとめます。

## 関所について

塗る前の図面を Claude に渡さないための守りは3層です。MCP の設計（生データを返さない）、フック（`.claude/hooks/guard_raw.py`）、Claude Code 本体の deny 設定（`.claude/settings.json`）です。

フックは誤操作を止める関所で、悪意のある回避は防ぎません。

## トラブルシューティング

### ウイルス対策ソフトが OpenCV のライブラリを隔離する

マカフィーが、uv のキャッシュと `.venv` 内の OpenCV ライブラリ（libaom、libtiff）を誤検出して隔離することがあります。`import cv2` が失敗するようになったら、隔離されていないか確認してください。

1. マカフィーの設定で、次の2つのフォルダを除外に追加します
   - `~/.cache/uv`（`uv cache dir` で場所を確認できます）
   - このリポジトリの `.venv`
2. 隔離されたファイルは壊れたままなので、`uv pip install --reinstall -e ".[dev]"` で入れ直します
3. 次のコマンドがエラーなく終われば復旧しています

```
python -c "import cv2; import tifffile"
```

## 出典

出典：法務省 登記所備付地図データ（G空間情報センター）を加工して作成

## 氏名について

図面に描かれた氏名は、すべて架空のものです。

## ライセンス

MIT
