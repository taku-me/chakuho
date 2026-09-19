"""実画面 AX スナップショット(マスク済み)から GUI 要素選択ベンチを作る。
各タスクに作者(Claude)の期待答え(role:label の集合、または __none__)を付け、3 変種(原順 / シャッフル / 他アプリ要素の混入)を出す。"""
import json, random
import os
SNAP = os.environ["AX_SNAPSHOT"]  # マスク済み AX スナップショット(通常画面)
d = json.load(open(SNAP)); apps = d["apps"]
NONE = "__none__"
# (app, instruction, expected role:label set or NONE)
T = []
def add(app, instr, *exp): T.append((app, instr, list(exp)))
A = "Docker Desktop"
add(A, "設定画面を開く", "AXButton:Settings")
add(A, "イメージの一覧に切り替える", "AXMenuItem:Images")
add(A, "コンテナを名前で検索したい", "AXButton:Search ⌘K")
add(A, "通知を確認する", "AXButton:Notifications")
add(A, "サイドバーを折りたたむ", "AXButton:collapse sidebar")
add(A, "Docker Desktop を最新版に更新する", "AXButton:Update", "AXButton:This version of Docker Desktop is no longer supported. Version 4.91.0 is availab")
add(A, "実行中のコンテナだけを表示する", "AXCheckBox:Only show running containers")
add(A, "停止しているコンテナを起動する", "AXButton:Start")
add(A, "コンテナを削除する", "AXButton:Delete")
add(A, "コンテナ ID をクリップボードにコピーする", "AXButton:Copy to clipboard")
add(A, "Resource Saver モードを解除して通常動作に戻す", "AXButton:Resume")
add(A, "Docker アカウントにサインインする", "AXButton:Sign in")
add(A, "Kubernetes の画面へ移動する", "AXMenuItem:Kubernetes")
add(A, "拡張機能(Extensions)を開く", "AXMenuItem:Extensions")
add(A, "AI アシスタントに質問する", "AXMenuItem:Ask Gordon")
add(A, "トラブルシューティングを開く", "AXButton:Troubleshoot")
add(A, "ボリューム(Volumes)の一覧を見る", "AXMenuItem:Volumes")
add(A, "CPU とメモリのグラフを表示する", "AXButton:Show charts")
add(A, "ネットワークの一覧を開く", NONE)
add(A, "コンテナをファイルにエクスポートする", NONE)
A = "DaVinci Resolve"
add(A, "カラーページへ移動する", "AXCheckBox:Color")
add(A, "書き出し(デリバー)ページへ移動する", "AXCheckBox:Deliver")
add(A, "すぐに書き出す(クイックエクスポート)", "AXButton:Quick Export")
add(A, "タイムラインを再生する", "AXCheckBox:Play")
add(A, "逆方向に再生する", "AXCheckBox:Play Reverse")
add(A, "イン点を打つ", "AXButton:Mark In")
add(A, "アウト点を打つ", "AXButton:Mark Out")
add(A, "ブレード(カット)編集モードに切り替える", "AXCheckBox:Blade Edit Mode")
add(A, "タイムラインをズームインする", "AXButton:Zoom In")
add(A, "スナップのオン/オフを切り替える", "AXCheckBox:Snapping")
add(A, "インスペクタを開く", "AXCheckBox:Inspector")
add(A, "プロジェクト設定を開く", "AXButton:Project Settings")
add(A, "別のプロジェクトを開くためプロジェクトマネージャを出す", "AXButton:Project Manager")
add(A, "音声をミュートする", "AXCheckBox:Mute")
add(A, "ナレーションを録音する", "AXButton:Voiceover")
add(A, "クリップを上書きでタイムラインに置く", "AXButton:Overwrite Clip")
add(A, "マーカーを追加する", "AXButton:Markers", "AXMenuButton:Markers")
add(A, "次の編集点へ移動する", "AXButton:Go to Next Edit")
add(A, "Fusion ページへ移動する", "AXCheckBox:Fusion")
add(A, "エフェクトライブラリを開く", "AXCheckBox:Effects")
add(A, "レンダーキューを開く", NONE)
add(A, "字幕トラックを追加する", NONE)
A = "システム設定"
add(A, "キーのリピート速度を変える", "AXSlider:キーのリピート速度")
add(A, "キーボードショートカットを設定する", "AXButton:キーボードショートカット…")
add(A, "入力ソースを編集する", "AXButton:編集…")
add(A, "ユーザ辞書を開く", "AXButton:ユーザ辞書…")
add(A, "バックライトが消えるまでの時間を変更する", "AXPopUpButton:30秒後")
add(A, "音声入力のマイク入力元を変更する", "AXPopUpButton:MacBook Proのマイク")
add(A, "キーボードの輝度を上げる", "AXButton:輝度を上げる")
add(A, "前の画面に戻る", "AXButton:戻る")
add(A, "設定項目を検索する", "AXButton:検索")
add(A, "ディスプレイの設定へ移動する", "AXStaticText:ディスプレイ")
add(A, "プライバシーとセキュリティの設定へ移動する", "AXStaticText:プライバシーとセキュリティ")
add(A, "音声入力を開始するショートカットを変更する", "AXPopUpButton:🌐︎キーを2回押す")
add(A, "外付けキーボードを設定する", "AXButton:キーボードを設定…")
add(A, "ヘルプを開く", "AXButton:ヘルプ")
add(A, "サウンドの設定へ移動する", "AXStaticText:サウンド")
add(A, "Bluetooth の設定へ移動する", NONE)
add(A, "表示言語を英語に変更する", NONE)
A = "eqMac"
add(A, "エキスパートモードのイコライザに切り替える", "AXStaticText:エキスパート")
add(A, "ベーシックモードのイコライザに切り替える", "AXStaticText:ベーシック")
add(A, "イコライザのバンドを追加する", "AXStaticText:バンド追加")
add(A, "エフェクトを追加する", "AXStaticText:エフェクトの追加")
add(A, "空間オーディオの設定を開く", "AXStaticText:空間オーディオ")
add(A, "スーパープリセットを選ぶ", "AXStaticText:スーパープリセット", "AXStaticText:プリセットが選択されていない")
add(A, "アプリケーションミキサーを開く", "AXStaticText:アプリケーションミキサー")
add(A, "出力先(MacBook Pro のスピーカー)を選び直す", "AXStaticText:MacBook Proのスピーカー")
add(A, "音源の選択を変える", "AXStaticText:音源")
add(A, "録音を開始する", NONE)
A = "Obsidian"
add(A, "新しいノートを作る", "AXGroup:新規ノート")
add(A, "ノートを検索する", "AXGroup:検索")
add(A, "新しいフォルダを作る", "AXGroup:新規フォルダ")
add(A, "ブックマークを開く", "AXGroup:ブックマーク")
add(A, "同期の状態を確認する", "AXButton:Sync Status")
add(A, "フォルダをすべて折りたたむ", "AXGroup:すべて折りたたむ")
add(A, "ピン留めを解除する", "AXGroup:ピンの解除")
add(A, "クリップボードの内容を新しい thino として保存する", "AXButton:Paste clipboard content and save as new thino")
add(A, "一番上までスクロールする", "AXButton:Top ↑")
add(A, "グラフビューを開く", NONE)
A = "Adobe Lightroom"
add(A, "写真を 100% 表示にする", "AXCheckBox:100%")
add(A, "写真全体を表示する", "AXCheckBox:全体")
add(A, "「写真」メニューを開く", "AXMenuBarItem:写真")
add(A, "現像モジュールに切り替える", NONE)
A = "アクティビティモニタ"
add(A, "表示メニューを開く", "AXMenuBarItem:表示")
add(A, "CPU タブに切り替える", NONE)
A = "Karabiner-Updater"
add(A, "ヘルプメニューを開く", "AXMenuBarItem:Help")

def key(e): return f"{e['id']}: [{e['role']}] {e['label']}"
def crit(els): return {key(e): (f"window: {e.get('window') or '-'}") for e in els}
random.seed(20260919)
cases = []
for i, (app, instr, exp) in enumerate(T):
    base = apps[app]
    others = [(a, e) for a, es in apps.items() if a != app for e in es]
    for variant in ("orig", "shuffled", "distractors"):
        els = list(base)
        if variant == "shuffled": random.shuffle(els)
        if variant == "distractors":
            extra = random.sample(others, min(15, len(others)))
            els = els + [dict(e, id=1000 + j, window=f"(別アプリ {a}) " + (e.get("window") or "")) for j, (a, e) in enumerate(extra)]
            random.shuffle(els)
        c = crit(els); c[NONE] = "どの要素も指示に合わない(該当なし)"
        cases.append({"case_id": f"{i:03d}-{variant}", "task_id": i, "app": app, "variant": variant, "instruction": instr,
                      "expected": exp, "n_options": len(c),
                      "request": {"state": {"app": app, "window": base[0].get("window") or app, "note": "要素一覧は候補側に列挙されている"},
                                  "questions": {"target": {"type": "choice",
                                      "instructions": f"Mac のアプリを AX 木(アクセシビリティ要素)経由で操作する。今のステップ: {instr}。操作すべき要素を 1 つ選ぶ。該当する要素が無ければ __none__。",
                                      "criteria": c}}}})
json.dump(cases, open("cases.json", "w"), ensure_ascii=False)
print(len(T), "tasks ->", len(cases), "cases;", sum(1 for t in T if t[2] == [NONE]), "none-tasks")
