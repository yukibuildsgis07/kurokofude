# 黒子筆（Kurokofude）

スキャンした測量図面（TIFF＋ワールドファイル）から、地番の真下にある地権者の氏名だけを手元のPCで塗りつぶし、安心してAIに渡せるようにする「AIに図面を渡す前の関所」です。

## インストール

[uv](https://docs.astral.sh/uv/) を使います。Python 3.11 が必要です。

### Mac

```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Windows（PowerShell）

```
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

OCRのモデルは初回実行時に huggingface.co からダウンロードされます。

## 使い方

```
kurokofude mask <図面.tif>
kurokofude batch <図面のフォルダ>
kurokofude eval --truth data/truth/<図面>.truth.json --result data/masked/<図面>.chiban.json --image <図面.tif>
```

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
