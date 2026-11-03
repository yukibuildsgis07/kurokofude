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

---

## 11. 精度と関所の方針変更（2026-11-03 Fable 判断。§2.2〜2.5、§4、§6 より優先）

### 11.1 塗る方針：迷ったら塗る、人が外す
OCRの全ボックスを文字の種類だけで3つに分ける。氏名の文字列は使わず、保存しない。
- (a) 地番：正規表現に合う → 残す
- (b) 数字の割合 ≥ 0.5（`ocr.digit_ratio_min`）だが正規表現に合わない → 残す。`chiban.json` の `uncertain` に `status: "uncertain_chiban"` で記録し、②で人が見る
- (c) それ以外で除外領域の外 → 塗る。地番とペアなら `status: "auto"`、ペアにならなければ `status: "unpaired"`（`chiban` は null）。`mask.unpaired_text: false` で unpaired を塗らない設定にできる

実装の補足：判定には数字の割合が要るので、§10 のキャッシュの規則を次のとおり変える。`ocr.keep_char_class` の文字だけの文字列はそのまま残す。それ以外は、数字の割合が 0.5 以上なら許可外の文字を `□` に置き換え、0.5 未満なら空文字にする。キャッシュにもフィクスチャにも氏名の文字は残らない。除外領域の中のボックスは (a)(b)(c) のどれにもしない。

### 11.2 ペアリング
§2.3 の4を「条件を満たす全部」に変える。氏名が複数の枠に割れていても全部その地番の氏名とする。1つの枠を複数の地番が選んだら近い方に渡す。

### 11.3 切る線と余白（`refine.py`）
- 切る線：地番の下辺と氏名の上辺の間の帯（重なっていれば重なりの帯）で、文字に垂直な方向にインクの投影を取り、インク最小の行で切る。最小が複数行あれば、その中央。帯が狭い、または起伏がなければ、下辺と上辺の中点で切る。`pair.cut_margin_px` は廃止
- 余白を詰める：枠の内側で投影を取り、上下左右を詰める。広げない。余白は `mask.tighten_padding_px`（3px）。筆界線でインクが全行・全列にある向きは結果として詰まらない。unpaired も同じく詰める
- インクは `mask.ink_threshold` より暗い画素

### 11.4 地番の正規化
`/` `\` `|` `ｰ` `－` `‐` `ー` `ノ` を `-` にし、先頭・末尾の記号を落としてから正規表現にかける。正規化で文字列が変わったものは score に `chiban.normalized_score_factor` を掛ける。

### 11.5 評価
- 地番生存率：正解の地番枠の中のインク画素が1つも塗られていなければ生存。塗る前の画像が要るので `kurokofude eval --truth <json> --result <json> --image <塗る前.tif>` とし、`kurokofude eval` を許可リストに足す
- 補助指標：`unpaired_masks`（unpaired の塗り数）、`uncertain_chiban`（(b) の数）

### 11.6 角度（§2.3 の「長辺」を変更）
縦横比が `angle.square_aspect_max`（1.5）以下の枠は、自分の長辺で角度を決めない。近傍の縦横比 `angle.reference_aspect_min`（2）以上の枠を近い順に `angle.neighbors`（3）個取り、その角度の中央値を使う。無ければ図面全体の、同じ条件の枠の角度の中央値。それも無ければ 0°。

### 11.7 ダミー図面
- 1桁・2桁の地番を図面ごとに必ず含める（`source.min_one_digit`、`source.min_two_digit`）。足りなければ一部の筆の地番を置き換え、GeoJSON にも同じ値を入れる
- `.geojson` は `data/ref/`（公図の参照データ。個人情報なし。④で Claude が読む）

### 11.8 関所は3層
フックは「誤操作を止める関所」で「悪意のある回避は防がない」。README に明記する。守りは MCP の設計（生データを返さない）＋フック（誤操作）＋deny 設定（Claude Code 本体）の3層。
- フックの Bash 判定：パスの部品 `data` を含むコマンドは、続く部品が `hook_open_dirs`（`data/masked`、`data/truth`、`data/cache`、`data/ref`）のときだけ通す。許可リストのコマンドは従来どおり通す。`data/source` もこの一覧にないので Bash では止まる（生成は許可リストの `make.py` で読む）
- `.claude/settings.json` の `permissions.deny` に Read/Glob/Grep の保護ディレクトリを入れる

### 11.9 batch
`kurokofude batch <dir>` はフォルダ直下の TIFF を順に塗る。1枚失敗しても続け、失敗があれば終了コード 1。

### 11.10 ダミーの実物化（2026-11-03 Fable 指示。Yuki の現場の実感：地番は筆界線と重ならないように置き、入らなければ引き出し線で外に出す）
注記（地番＋氏名）の置き方を次の順に試す。全筆で 1 を先に試し、余った筆で 2、最後に 3。
1. 筆の内側で、文字枠が筆界線と交差しない位置を、代表点に近い順に探す
2. 筆の外の空いた場所に置き、筆の内側の点から文字枠の端まで細い引き出し線を引く（`render.leader_*`）
3. それでも置けないときだけ線と重ねてよい。`.truth.json` の注記に `overlaps_line: true` を記録する（1、2 は false）

`kurokofude eval` は `excluding_line_overlaps` に、線と重なる注記を除いた氏名カバー率・地番生存率・地番検出率を併記する。変更の前後を `docs/accuracy_log.md` に1行ずつ残す。

---

## 12. 除外領域と塗りの見直し（2026-11-03 Fable 判断。§2.2、§2.5、§8、§11 より優先）

### 12.1 除外領域は地番の誤検出を防ぐだけ
- 既定 `exclude_mode: chiban_only`：領域の中では地番の判定 (a) だけを止める。数字が主体なら (b)、それ以外は領域の中でも塗り、`status: "in_title_block"` にする（表題欄の作成者名・会社名も個人情報。図名など消したくないものは②で人が外す）。領域の中の文字は地番とペアにしない
- `exclude_mode: all`：領域の中のボックスを (a)(b)(c) のどれにもしない（従来の動作）
- 既定値は実物に合わせて `[[0.85, 0.83, 1.0, 1.0]]`（A1 相当の図枠 800×600mm に、右下の表題欄 120×100mm）
- 図枠の下辺ぜんぶが表題欄の図面（河川境界確定図など）向けに、プリセット `configs/bottom_strip.yaml`（`[[0.0, 0.90, 1.0, 1.0]]`）を置く。`--config` で指定する
- `make_dummy` の表題欄も同じ大きさにする。文字は枠に入るように小さくする（`sample_small` は枠が小さいため）

### 12.2 地番を塗らない
すべての塗りから、(a) と (b) の枠を `mask.protect_margin_px`（1px）膨らませたものを shapely の difference で引く。結果は非凸になり、複数に割れることがあるので、`masks` にはポリゴン1つを1項目として入れ、`quad` は頂点数4とは限らない。穴のある形は、穴の位置で2つに割って穴をなくす。すべて消えた塗りは項目にしない。

### 12.3 塗りすぎ率の定義
`overmask_pct`：正解の氏名枠を `evaluate.overmask_margin_px`（3px）膨らませたものの外側に塗った割合。線や他の文字を塗った量を測る。目標 15% は維持。従来の値は `overmask_raw_pct` として併記する。

### 12.4 その他
- §11.4 の正規化の一覧に `.` と `,` を足す（`273.19` → `273-19`）。`,` も OCR 直後に残してよい文字に足す
- §11.8 の `hook_open_dirs` に `data/source` を足す（公開データで個人情報なし。④で読む）

---

## 13. 地番の保護と mixed 枠（2026-11-03 Fable 判断。§11、§12 より優先）

### 13.1 引くのはインクで詰めた枠（`refine.tight_box`）
§12.2 で塗りから引く枠は、OCR の枠ではなくインクで詰めた枠にする。枠の内側で投影を取り、上下左右とも詰める。詰めるだけで広げない。余白は `mask.protect_padding_px`（1px）。その枠を §12.2 のとおり `mask.protect_margin_px` だけ膨らませて引く。

実装の補足：OCR の地番枠の下側には氏名の上端のインクが入るので、単純に最初と最後のインクの行を取ると詰まらない。行は枠の中央を含むインクのかたまりだけを取り、かたまりが枠の端まで続くとき（筆界線がかかる等）は、その半分の中でインク最小の行まで詰める（最小が最大の `mask.core_valley_max_ratio` 以下のときだけ）。列は残した行の範囲で、最初と最後のインクを取る。

### 13.2 mixed 枠（`□` と数字の両方を含む枠）
OCR が氏名と地番を1つにまとめた枠。全5枚の3件は横並びで、上下がくっついたものはなかった（docs/questions.md の記録 11/3）。
1. 文字列が「`□` の並び＋数字側」または「数字側＋`□` の並び」のときだけ分割を試みる。入り組んでいれば分割しない
2. 枠の長辺方向に、文字数の比率で仮の分割位置を置き、その前後 `mixed.search_ratio`（長辺に対する割合）でインクの投影の谷を探す。低いとは、列のインクが最大の `mixed.valley_max_ink_ratio` 以下。低い列が `mixed.min_gap_ratio`×枠の高さ以上続くものだけを谷とし、仮の位置に最も近い谷を選ぶ。数え方は `□` を `mixed.name_char_width`（2）文字分とする（漢字は数字の約2倍の幅）。最初に文字数をそのまま数え、最も長い谷を選んだところ、dummy_001 の 420-10 で姓と名の隙間に分割線が入り、名の字が地番側に残った
3. 数字側の文字列が地番の正規表現に合えば、数字側を地番として残し（`chiban[].status: "split"`、score に `chiban.normalized_score_factor` を掛ける）、`□` 側を塗る（`masks[].status: "split"`）
4. 谷が見つからない、または数字側が地番にならないとき、枠全体を塗る（`masks[].status: "mixed_masked"`）。地番は失われるが、個人情報を残すよりよい
5. 除外領域の中の mixed は地番の判定をしないので、枠全体を塗る
6. どちらも②の確認画面で色を分ける

数字の割合が 0.5 未満の枠は、文字列を保存しないので mixed と判定できず、(c) として全体を塗る。

### 13.3 `masks[].quad` は多角形
頂点数は可変（§12.2）。穴のある塗りは2つに割って、穴のない多角形にする。②はこの形に対応する。

