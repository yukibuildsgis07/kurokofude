# CLAUDE.md — 黒子筆（Kurokofude）

このリポジトリは Built with Claude: Japan 2026 の作品です。設計は `docs/design.md` に従ってください。設計を変えたいときは、先に `docs/design.md` に追記してからコードを変えます。

## 作業の役割
- Yuki：プロマネ。方針決定と現場目線の確認
- Fable：設計。難所（幾何計算、OCR方針、関所の設計）の判断
- Opus（あなた）：実装。ダミー図面、エンジン、画面、テスト、修正

## 絶対に守ること
- **`data/raw/` 配下のファイルは読まない・開かない・grepしない。** フックで止まります。塗る前の図面に触る必要があるときは、`kurokofude mask` を実行して `data/masked/` 側を見てください
- **既存コードの流用禁止。** 2026-11-02 以降に書いたコードだけ。外部ライブラリの利用はOK
- **氏名は読まない。** OCRが返した氏名の文字列を保存・表示・ログ出力しない。四角形の座標だけ使う
- 業務データ・会社名・自治体名を含めない。図面はすべて `tools/make_dummy` で生成したもの
- 精度の数値は期間中に測ったものだけを使い、`docs/accuracy_log.md` に追記する

## 技術の決めごと
- Python 3.11。パスは `pathlib`。Windowsでも動くコードを書く（`/` の直書き、`os.system`、シェル依存を避ける）
- PaddleOCR は `enable_mkldnn=False`。結果は `data/cache/` にキャッシュし、同じ画像で再実行しない
- OCRのボックスは傾いた四角形（4頂点）のまま扱う。水平化しない
- 塗りつぶしは `cv2.fillPoly`。画像サイズ・位置を変えない。`.tfw` はコピー
- 設定は `configs/default.yaml`。マジックナンバーをコードに書かず設定に出す
- 単体テストはOCRを呼ばない（OCR結果のJSONフィクスチャを使う）。`pytest` が数十秒で終わること

## 作業の進め方
- 作業開始時に `docs/design.md` の該当節を読む。終わったら「何ができて、何が未了か」を `docs/progress.md` に3行で追記
- 1つの機能ごとにコミット。コミットメッセージは日本語でよい
- 精度を測ったら `docs/accuracy_log.md` に「日付／設定の変更点／氏名カバー率／塗りすぎ率／地番生存率」を1行追記
- 判断に迷う設計上の問題（幾何の条件、しきい値の意味、関所の抜け道など）は、自分で決めず `docs/questions.md` に書いて止まる。Yuki と Fable が答える
- 「できれば」「余裕があれば」の機能（④⑤）は、①②③が動いてから

## よく使うコマンド
```
pip install -e ".[dev]"
python tools/make_dummy/make.py --out data/raw --count 5
kurokofude mask data/raw/xxx.tif
kurokofude eval --truth data/truth/xxx.truth.json --result data/masked/xxx.chiban.json
kurokofude review data/masked/xxx.tif
pytest
```
