# Opus への指示：11/2（月）夜

前提：`docs/design.md` と `CLAUDE.md` を読んでから始める。今夜は「通しで動く最小の形」まで。精度の追い込みは明日。

## タスク（この順で）

1. **リポジトリの骨格**
   - `pyproject.toml`（パッケージ `kurokofude`、CLIエントリ `kurokofude = kurokofude.cli:main`、Python 3.11、依存は design.md §7）
   - `LICENSE`（MIT）、`README.md`（見出しと出典・架空氏名の注記だけ先に）、`.gitignore`（`data/cache/`, `data/masked/` は除外、`data/raw/` は生成物なので除外、`tests/fixtures/` は含める）
   - `configs/default.yaml` を design.md §8 のとおり作成
   - `.claude/settings.json` と `.claude/hooks/guard_raw.py`（design.md §4.1）。**最初に作る**。これ以降の作業で raw を誤って読まないため
   - 空の `docs/progress.md`, `docs/accuracy_log.md`, `docs/questions.md`

2. **ダミー図面生成 `tools/make_dummy/`**
   - Yuki がダウンロード済みの登記所備付地図XMLの置き場所を確認する（なければ `docs/questions.md` に書く）
   - XML → GeoJSON の変換（デジタル庁の変換ツールを使うか、同等の処理を書く。コードは新規）
   - 1/500 図郭で切り出し → 筆界線・地番・架空氏名の描画 → スキャン風加工 → `.tif` + `.tfw` + `.truth.json`
   - まず **5枚** 生成して `data/raw/` へ。うち1枚を縮小した版（長辺1000px程度）を `tests/fixtures/sample_small.*` として作る
   - 氏名は姓名リストからランダム合成。斜めの地番・氏名を全体の3割程度入れる

3. **① エンジンの骨格**
   - `ocr.py`：PaddleOCR 呼び出し＋キャッシュ。`sample_small.tif` で1回動かし、返ってきたJSONを `tests/fixtures/sample_small.ocr.json` として保存（単体テスト用フィクスチャ）
   - `detect.py` → `pair.py` → `mask.py` を design.md §2 のとおり。まずは「切る」処理なしでもよい（明日入れる）
   - `evaluate.py` と `cli.py`（mask / eval）
   - `kurokofude mask` → `kurokofude eval` が通しで動き、3指標が数字で出ること

4. **記録**
   - `docs/accuracy_log.md` に1行目（今夜の数字。悪くてよい）
   - `docs/progress.md` に今夜の結果と未了を3行
   - 困ったことは `docs/questions.md` へ

## やらないこと（今夜は）
- ②確認画面、③MCPサーバー、④⑤
- 精度のチューニング（しきい値いじり）
- Windows CI（明日。ただしパスは最初から `pathlib` で書く）

## 動作確認の目安
- `pytest` が通る（detect / pair / mask をフィクスチャで検証、OCRは呼ばない）
- `kurokofude mask data/raw/<1枚>.tif` が1分以内に終わり、`data/masked/` に `.tif` `.tfw` `.chiban.json` の3点が出る
- 塗った画像を開いて、地番が残り、氏名が塗られていることを目視で確認（Yuki）
