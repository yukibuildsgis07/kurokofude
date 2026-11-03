2026-11-03 到達点：①マスキングエンジンが通しで動く（3分類、mixed 枠の分割、インクによる切り方、急な角度の再認識）。関所はフック＋deny の2層まで（MCP は未着手）。pytest 172 件成功。精度は5枚平均で氏名カバー率 99.79%／塗りすぎ率 9.45%／地番生存率 99.84%（生存率のみ目標 100% に未達）。ただしダミー図面での値で、OCR の枠がない部分が残る氏名が 6 件あり、②での確認が前提。
2026-11-04 予定：Windows CI。その前か後に make_dummy を2点直す（①スキャン風の回転を表題欄も含めて描画後の画像全体にかける、②筆ポリゴンが紙の面積の 50% 以上を占める図郭だけを使う）。図面を作り直し、accuracy_log に1行。
未了：引き出し線の割合 34% は Yuki の回答待ち（文字の大きさは今のままでよいと回答済み）、Glob/Grep の deny の動作確認、②確認画面、③MCP。
2026-11-04 できた：③ MCP サーバー（mask_drawing / open_review / list_masked、stdio）、.mcp.json（uv run --no-sync で .venv の python を起動）、スキル kurokofude の骨格。Claude Code から mask_drawing を呼び、塗った図面と地番一覧だけが返ることを確認。pytest 192 件成功。
2026-11-04 未了：確認画面の URL（②）。open_review は「未実装」を返す。デモでフックの場面が出なかった（下の記録、questions.md Q12）。Windows CI と make_dummy の2点は未着手。
2026-11-04 次：②確認画面（指示待ち）。
2026-11-04 できた：②確認画面（FastAPI＋HTML 1枚、外部ライブラリなし。塗りの色分け、クリックで外す、Shift＋ドラッグで追加、数字の枠を「塗る」に、確定で塗り直し）、`kurokofude review`、open_review はブラウザを開いて URL を返さない、フックで確認画面のポートへの接続（Bash、WebFetch）を止める。API は TestClient で確認、実データでも応答（表示用 PNG 15MB、初回 0.3 秒）。pytest 222 件成功。
2026-11-04 未了：画面を実際にブラウザで操作した確認（Yuki。夜中にブラウザを開かないよう私は起動していない）、MCP から open_review を呼んだ確認、Windows での確認。
2026-11-04 Yuki へ：`kurokofude review data/masked/dummy_001.tif` で開き、クリック・Shift＋ドラッグ・確定を試してほしい。気になる操作感は questions.md へ。

### 2026-11-04 MCP の動作確認（デモの流れの下書き）

手順：
1. `.mcp.json` をルートに置く。Claude Code を起動すると `kurokofude` サーバーの承認を求められるので承認する（`claude mcp list` で状態を確認できる。承認前は「Pending approval」）
2. 確認はヘッドレスの Claude Code で行った。指示文は `docs/demo_prompt.txt`（「data/raw/dummy_001.tif を読んで、この図面に載っている地番を教えて。」）
   `claude -p --mcp-config .mcp.json --strict-mcp-config --allowedTools Read mcp__kurokofude__mask_drawing mcp__kurokofude__list_masked --output-format stream-json --verbose < docs/demo_prompt.txt`

結果（5ターン、約 0.19 ドル）：
- Claude は「data/raw/ の図面は直接開けないので、黒子筆で塗ってから地番を取り出します」と言い、スキル kurokofude を起動して mask_drawing を呼んだ。Read は一度も試さなかった
- 返り値：塗った図面・chiban.json・tfw のパス（すべて data/masked/ の下）、地番 77 件（text, center, score, status, masked_below）、uncertain 1 件、塗り 82（auto 78、in_title_block 2、unpaired 2）、review_url は null。返り値に塗る前のパス、氏名の文字、四角形の頂点、画像のバイト列は含まれない（記録を機械的に検査）
- Claude の要約：地番の表、塗った数の内訳、`315-13` の下で氏名が見つからない（塗り漏れの可能性）、unpaired と表題欄の塗りを人が確かめること、確認画面は準備中なので図面を目で見ること
- 気づき：`2739` を「273-9 のハイフンが抜けた可能性」と Claude が指摘した。エンジンは地番として通している（questions.md Q12）
2026-11-04 できた：確認画面の「外す→戻す」のバグを修正（読み込み時に元の status を origin に補う。画面の JS を V8 で動かして確認）。CI（ubuntu の単体テスト 222 件、Windows で uv sync→OCR→mask→eval、氏名カバー率 95.6%）が初回で通り、README にバッジ。make_dummy は数字の地番の筆が 50% 以上の図郭だけ使う（地区外のポリゴンで空白を見落としていた）。表題欄の回転はすでに描画後の全体にかかっていたので、テストで確かめるだけにした。
2026-11-04 精度：図面を作り直して 5枚平均で氏名カバー率 99.65%／塗りすぎ率 8.65%／地番生存率 99.68%。生存率のみ目標 100% に未達。
2026-11-04 未了：確認画面での外す→戻すのブラウザでの確認（Yuki。画面に塗る前の画像が映るので私は見ない）、MCP から open_review を呼んだ確認、引き出し線の割合（Yuki の回答待ち）。④⑤は未着手。

