# 黒子筆（Kurokofude）設計書 v1

作成：2026-11-02（Fable）／承認：Yuki
この文書はリポジトリの `docs/design.md` にそのまま置く。実装はこの設計に従う。設計を変える必要が出たら、変更点をこの文書に追記してからコードを変える。

---

## 0. 一言で

スキャンした測量図面（TIFF＋ワールドファイル）から、**地番の真下にある地権者の氏名だけ**を手元のPCで塗りつぶし、安心してAIに渡せるようにする「AIに図面を渡す前の関所」。

守る原則：
1. **名前は消す。地番は残す。** 地番が位置合わせの鍵だから
2. **生データはAIに渡さない。** OCRも塗りつぶしもローカルで完結する
3. **画素の位置を動かさない。** ワールドファイル（.tfw）はそのまま使える
4. **氏名は読まない、見つけるだけ。** 氏名の文字認識結果は保存しない

---

## 1. リポジトリ構成

```
kurokofude/
  README.md                 # 日本語。インストール手順（Mac/Windows）、出典、氏名は架空の明記
  LICENSE                   # MIT
  CLAUDE.md                 # Claude Code向けの作業規約
  pyproject.toml            # パッケージ名 kurokofude、CLIエントリ kurokofude
  src/kurokofude/
    config.py       # YAML設定の読み込み・既定値
    ocr.py          # PaddleOCR呼び出し＋結果キャッシュ
    detect.py       # 地番ボックスの判定
    pair.py         # 地番→氏名ボックスのペアリング（幾何）
    mask.py         # 塗りつぶし、.tfwのコピー
    evaluate.py     # 正解JSONとの採点
    cli.py          # kurokofude mask / review / eval / batch
    review/
      app.py        # ②確認画面（FastAPI）
      static/index.html
    mcp_server.py   # ③ MCPサーバー
  tools/make_dummy/         # ダミー図面生成（登記所備付地図 → TIFF＋tfw＋正解JSON）
  tests/
    fixtures/sample_small.tif, .tfw, .truth.json   # CI用の小さな1枚（約1000×700px）
    test_*.py
  .github/workflows/
    test.yml              # ubuntu：単体テスト（OCRなし、高速）
    windows-smoke.yml     # windows-latest：pip install → OCR1枚 → mask → eval
  .claude/
    settings.json         # ③フック
    hooks/guard_raw.py    # フック本体
    skills/georef/SKILL.md          # ④スキル（できれば）
    agents/matcher.md               # ④照合サブエージェント（できれば）
  configs/
    default.yaml          # 1/500公図向けの既定値
  data/
    raw/        # 塗る前（関所の対象。Claudeは読めない）
    masked/     # 塗った後（Claudeが見てよいのはここだけ）
    truth/      # 正解データ（氏名の文字列なし。§6）
    cache/      # OCR結果キャッシュ
  docs/
    design.md             # この文書
    accuracy_log.md       # 精度の推移（日付・設定・3指標）
```

`data/raw/` と `data/masked/` を**ディレクトリで分ける**ことが関所の土台。フックは「パスに `raw/` を含むアクセスを止める」という単純なルールで済む。

---

## 2. 処理パイプライン（① マスキングエンジン）

```
TIFF(+tfw) → [ocr] → 全文字ボックス(4頂点) → [detect] → 地番ボックス群
                                            ↘ [pair] → (地番, 氏名ボックス) のペア
          → [mask] → 塗った TIFF + tfwコピー + 地番一覧JSON
```

### 2.1 ocr.py
- PaddleOCR（検出＋認識）。`enable_mkldnn=False` を必ず指定（CPU推論で落ちる既知不具合の回避）
- 返す形：`[{ "text": str, "score": float, "quad": [[x,y]×4] }]`。quadはOCRが返した**傾いた四角形をそのまま**。水平化しない
- キャッシュ：入力画像のSHA-256をキーに `data/cache/<hash>.json` へ保存。2回目以降はOCRを走らせない（1枚20秒超のため）
- 大きな図面（例：A1相当 7000×5000px）はそのまま投げると落ちる可能性 → 設定 `ocr.max_side` を超える場合は**重なりありのタイルに分割**し、座標を元に戻して結合。タイル境界で二重検出されたボックスはIoU>0.5で統合
- モデルは初回に huggingface.co からダウンロードされる。READMEに明記

### 2.2 detect.py（地番の判定）
地番と判定する条件（すべて設定可能）：
- 正規表現 `^\d{1,5}(-\d{1,4})?$`（枝番あり）。OCRの読み違いで「O→0」「l→1」程度の正規化は入れる
- 桁数：`chiban.min_digits` 〜 `max_digits`（既定 1〜5）
- 除外領域 `exclude_regions`：画像の相対座標（0〜1）の矩形リスト。既定は右下の表題欄 `[0.6, 0.8, 1.0, 1.0]`
- 縮尺・日付パターン（`1:500`, `S=`, `R6`, `2024` など近傍に記号がある）は除外

### 2.3 pair.py（氏名の選択）— 一番の難所
各地番ボックス B について、OCRの全ボックス C から次を満たすものを選ぶ：

1. **角度差** `|angle(C) − angle(B)| ≤ 10°`（angleは四角形の長辺方向）
2. **真下にある**：Bの下辺の中点から、Bの長辺に垂直な方向（図面の傾きに沿った「下」）に C の中心がある。距離は `0 < d ≤ pair.max_gap × Bの高さ`（既定 2.5）
3. **左右の重なり**：Bの長辺方向に投影したとき、BとCの区間が重なる（または中心のずれが `pair.max_lateral × Bの幅` 以内、既定 1.0）
4. 複数候補があれば d が最小のもの。他の地番ボックスは候補から除く（地番は塗らない）

**塗りすぎ対策**：選んだ C の上辺を、**Bの下辺を延長した直線**（＋余白 `pair.cut_margin_px`、既定 2px）で切る。Bと氏名は同じ角度で並ぶので、斜めでもぴったり切れる。切った後の四角形を塗る対象とする。

氏名の認識文字列は**保存しない**（textを捨て、quadだけ使う）。

### 2.4 mask.py
- `cv2.fillPoly` で傾いた四角形を塗る。色は設定（既定：黒）。画像サイズ・解像度・位置は不変
- 出力：`data/masked/<name>.tif`、`.tfw` は元をそのままコピー。GeoTIFFタグがあればそれも維持（tifffile/rasterioで確認）
- 併せて `data/masked/<name>.chiban.json` を出力：
  ```json
  { "source": "<name>.tif", "image_size": [w, h],
    "chiban": [ { "text": "123-4", "center": [x, y], "quad": [[x,y]×4], "score": 0.98, "masked_below": true } ],
    "masks": [ { "quad": [[x,y]×4], "chiban": "123-4", "status": "auto" } ] }
  ```
  ④の照合はこの `chiban[].center` を使う。`masks[].status` は `auto | confirmed | removed | added`（②で更新）

### 2.5 evaluate.py（3指標）
正解JSON（氏名・地番の傾いた四角形の4頂点）と比較：
- **氏名カバー率**：正解の氏名ポリゴン面積のうち塗られた割合（目標 ≥ 99%）
- **塗りすぎ率**：塗った面積のうち正解の外側（事前実験 37.5% → 目標 ≤ 15%）
- **地番の生存率**：正解の地番ポリゴンのうち1画素も塗られていない割合（目標 100%）
- 補助：地番の検出率（正解地番のうちdetectが拾った割合）、誤検出数

`kurokofude eval --truth <json> --result <chiban.json>` で出力し、結果を `docs/accuracy_log.md` に日付・設定・数値で追記する。**事前実験の数値は使わない。**

### 2.6 CLI
```
kurokofude mask  <in.tif> [--config configs/default.yaml] [--out data/masked/]
kurokofude batch <dir>    # フォルダ一括
kurokofude review <masked.tif>   # ②を起動してブラウザを開く
kurokofude eval  --truth <truth.json> --result <chiban.json>
```

---

## 3. ② 確認画面
- FastAPI＋HTML1枚（canvas）。`localhost:8765`
- 表示：塗る前の画像に、塗り予定の四角形を半透明で重ねる（**この画面だけが生画像を扱う。ブラウザはローカルで、Claudeは関与しない**）
- 操作：四角形クリックで外す／ドラッグで四角形を追加（傾きは近傍の地番に合わせる、なければ水平）／「確定」で `mask.py` を再実行し `status` を更新
- 確定後にだけ `data/masked/` に書く設計にすると流れがきれいだが、①単体でも動くよう `mask` は自動で出力し、②は上書きする

---

## 4. ③ 関所（フック）＋MCPサーバー

### 4.1 フック（.claude/settings.json → hooks/guard_raw.py）
- PreToolUse で `Read | Glob | Grep | Bash | Edit | Write` を対象
- ツール入力（file_path / pattern / command / path）に `raw/`（または設定した保護ディレクトリ）が含まれたら **終了コード2** で止め、stderrに「このファイルは黒子筆を通す前の図面です。`mask_drawing` ツールを使ってください」と返す
- 判定は正規化したパスで行う（`../` や絶対パスでの迂回を防ぐ）
- フックはプロジェクトの `.claude/settings.json` に置き、リポジトリに含める（利用者がcloneすれば効く）
- 2026-11-03 追記（Fable 判断）：Bash には許可リストを設ける。`configs/default.yaml` の `hook_allowed_commands`（`kurokofude mask`、`kurokofude batch`、`python tools/make_dummy/make.py`）で始まるコマンドは、保護ディレクトリを含んでも通す。それ以外は止める。語の単位で先頭を比べ（`kurokofude maskx` は不可）、`;` `&` `|` `<` `>` `` ` `` `$` `(` `)` 改行を含むものは許可しない（後ろに別のコマンドをつなぐ抜け道を防ぐ）。Read など Bash 以外のツールに許可リストはない

### 4.2 MCPサーバー（mcp_server.py、Python SDK FastMCP、stdio）
ツールは3つだけ。**どのツールも生画像のバイト列・パスを返さない。**

| ツール | 入力 | 返すもの |
|---|---|---|
| `mask_drawing(path, config?)` | raw側のパス（文字列として受け取るだけ） | 塗った画像のパス、地番一覧（text, center, score）、マスク数、確認画面のURL |
| `open_review(masked_path)` | 塗り済みのパス | 確認画面のURL。確定済みかどうか |
| `list_masked()` | なし | `data/masked/` の一覧と各ファイルの状態 |

`mask_drawing` は入力パスが保護ディレクトリ配下であることを**サーバー側でも検査**し、それ以外は拒否する（フックとMCPで二重の守り）。

### 4.3 デモでの見せ方（10秒）
「`data/raw/図面.tif` を読んで」→ フックで止まる → Claudeが `mask_drawing` を使う → 塗った図面と地番一覧だけが返る。

---

## 5. ④ 一連の流れ（できれば）
- スキル `georef`：「この図面を幾何補正して」→ mask_drawing → 人の確認待ち → サブエージェント `matcher` に地番一覧と公図GeoJSONを渡す → ワールドファイル作成
- `matcher`：地番文字列で公図の筆を引き当て、画素座標（center）と公共座標（筆ポリゴンの重心）のペアを作る。3組以上で**相似変換**（回転・拡大縮小・平行移動）を最小二乗で解く。残差の大きいペアを外して再計算（RANSAC風）。結果を `.tfw` として書き出し、残差一覧を報告
- デモ：わざとずらした図面が、地番の照合で正しい位置に戻る

---

## 6. ダミー図面（tools/make_dummy）
- 入力：法務省 登記所備付地図データ（G空間情報センター）。**公共座標系の地域**を選ぶ
- 地図XML → GeoJSON（デジタル庁公開の変換ツール、または同等の実装）
- 1/500の図郭で切り出し、筆界線と地番を描画、地番の下に**架空の氏名**（姓名リストからランダム生成）。一部は筆の向きに合わせて斜めに
- スキャン風加工：全体の微小回転（±2°）、ガウスノイズ、かすれ、紙の色
- 出力：`.tif`、`.tfw`（元の公共座標から計算）、`.truth.json`（地番・氏名それぞれの傾いた四角形の4頂点、文字列）
- 表題欄（縮尺 `1:500`、日付、図郭番号）を右下に入れ、誤検出のテストにする
- 2026-11-03 追記（Fable 判断）：正解データは `data/truth/`（`raw/` の外）に置く。`.truth.json` に氏名の文字列は入れず、氏名は四角形の4頂点だけ、地番は四角形の4頂点と文字列を保存する。上の「文字列」はこの変更で地番だけになる
- README に「出典：法務省 登記所備付地図データ（G空間情報センター）を加工して作成」「氏名はすべて架空」を明記
- **変換・画像化のコードは11月2日以降に新規に書く**

---

## 7. Windows対応と CI

- Python 3.11 固定（paddlepaddle の Mac arm64 / Windows 両対応の範囲）
- 依存：`paddlepaddle`, `paddleocr`, `opencv-python-headless`, `numpy`, `tifffile`, `pyyaml`, `fastapi`, `uvicorn`, `mcp`, `shapely`（評価の面積計算）
- `windows-smoke.yml`：`windows-latest`、`pip install -e .`、`actions/cache` でPaddleOCRモデル（`~/.paddlex` 相当）をキャッシュ、`tests/fixtures/sample_small.tif` に対して `kurokofude mask` → `kurokofude eval` を実行し、氏名カバー率がしきい値（まず 90%）を下回ったら失敗
- トリガー：`main` への push と手動（workflow_dispatch）。所要10分程度を想定
- `test.yml`（ubuntu）：OCRを使わない単体テスト（detect / pair / mask / evaluate をOCR結果のJSONフィクスチャで検証）。数十秒
- パスは `pathlib` のみ、`/` を直書きしない。フックの保護ディレクトリ判定も Windows のパス区切りを考慮
- 最終確認：家族のWindows PCで1回、READMEの手順どおりにインストール→mask→review

---

## 8. 設定ファイル（configs/default.yaml）

```yaml
ocr:
  lang: japan
  enable_mkldnn: false
  max_side: 4000        # これを超えたらタイル分割
  tile_overlap: 200
chiban:
  pattern: '^\d{1,5}(-\d{1,4})?$'
  min_digits: 1
  max_digits: 5
exclude_regions:        # 画像に対する相対座標 [x0, y0, x1, y1]
  - [0.6, 0.8, 1.0, 1.0]   # 右下の表題欄
pair:
  max_angle_deg: 10
  max_gap: 2.5          # 地番の高さの倍数
  max_lateral: 1.0      # 地番の幅の倍数
  cut_margin_px: 2
mask:
  color: [0, 0, 0]
protected_dirs:
  - data/raw
```

---

## 9. 優先度と到達目標

| 日 | 到達目標 |
|---|---|
| 11/2 夜 | リポジトリ、CLAUDE.md、ダミー図面生成（まず数枚）、ocr/detect の骨格 |
| 11/3 午後 | ①が通しで動く。eval で3指標が出る。accuracy_log に1行目 |
| 11/4 | 軽め。Windows CI を通す |
| 11/5 夜 | ②確認画面 |
| 11/6 夜 | ③フック＋MCP。全体をつなぐ。④着手できれば |
| 11/8 | 動作確認、デモ動画、README、20:00提出 |

---

## 10. 実装時の追記（設計からの変更点）

- 2026-11-03 §2.1 キャッシュと氏名：OCRの文字列は、`ocr.keep_text_pattern`（数字と、地番の読み違い・縮尺日付の記号だけからなる文字列）に一致するものだけ残し、それ以外は OCR の直後に空文字にする。キャッシュ `data/cache/` とフィクスチャ `*.ocr.json` に氏名の文字列を残さないため（CLAUDE.md「氏名は読まない」、§2.3）
- 2026-11-03 §7 依存：`opencv-python-headless` を外す。`paddleocr>=3` が `opencv-contrib-python` を入れ、同じ `cv2` を奪い合って壊れるため。あわせて paddle の import に必要な `setuptools` を足す。インストールは uv を使う（Yuki 指示）
- 2026-11-03 §6 ダミー図面：地図XMLの筆界点は直書きと参照が混在。座標は `(東, 北) = (zmn:Y, zmn:X)` で持ち、`.tfw` もこの順。全体の傾きは `.tfw` の回転項に入れる。座標系が `公共座標8系` の地図だけを使い（Yuki 指示）、地名は出力に残さない
- 2026-11-03 §1 フィクスチャ：`sample_small` は1枚を縮小せず、図郭の一部を同じ解像度で切り出して作る。長辺1000pxへの縮小では文字が約5pxになりOCRが成立しないため

