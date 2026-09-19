"""シート/ダイアログ 3 画面から追加タスク(criteria の説明に in: sheet 等の文脈を付ける)。"""
import json, random
import os
d = json.load(open(os.environ["AX_SNAPSHOT"]))  # マスク済み AX スナップショット(シート/ダイアログ画面); apps = d["apps"]
NONE = "__none__"; T = []
def add(app, instr, *exp): T.append((app, instr, list(exp)))
A = "テキストエディット(保存シート)"
add(A, "ファイルを保存する", "AXButton:保存")
add(A, "保存をやめてシートを閉じる", "AXButton:キャンセル")
add(A, "保存するファイル名を変更する", "AXTextField:名称未設定.rtf")
add(A, "保存先のフォルダを変える", "AXPopUpButton:場所:")
add(A, "ファイルにタグを付ける", "AXTextField:タグエディタ")
add(A, "ファイル形式をプレーンテキストに変える", "AXPopUpButton:リッチテキスト書類")
add(A, "保存オプションをもっと表示する", "AXDisclosureTriangle:表示するオプションを増やす")
add(A, "本文を太字にする", "AXCheckBox:ボールド")
add(A, "書類を印刷する", NONE)
A = "テキストエディット(破棄確認シート: 削除/キャンセル/保存)"
add(A, "変更を保存して閉じる", "AXButton:保存")
add(A, "この書類は要らないので保存せずに捨てる", "AXButton:削除")
add(A, "閉じるのをやめて編集に戻る", "AXButton:キャンセル")
add(A, "別の名前を付けて保存したい", "AXTextField:名称未設定2.rtf")
add(A, "削除せず、いったん何もしない", "AXButton:キャンセル")
add(A, "本文の文字を編集する", "AXTextArea:a")
add(A, "保存先を iCloud に変える", "AXPopUpButton:場所:")
add(A, "書類を複製する", NONE)
A = "システム設定(キーボードショートカットのシート)"
add(A, "Mission Control のショートカットを見る", "AXUnknown:Mission Controlのショートカット")
add(A, "スクリーンショットのショートカットを変更する", "AXUnknown:スクリーンショットのショートカット")
add(A, "設定を終えてシートを閉じる", "AXButton:完了")
add(A, "ショートカットをデフォルトに戻す", "AXButton:デフォルトに戻す")
add(A, "Spotlight のショートカットを変更する", "AXUnknown:Spotlightのショートカット")
add(A, "入力ソース切り替えのショートカットを見る", "AXUnknown:入力ソースのショートカット")
add(A, "アプリごとのショートカットを追加する", "AXUnknown:アプリケーションのショートカット")
add(A, "Dock のショートカットを見る", "AXUnknown:Dockのショートカット")
add(A, "マウスのポインタ速度を変える", NONE)
def key(e): return f"{e['id']}: [{e['role']}] {e['label']}"
def desc(e): return f"window: {e.get('window') or '-'}" + (f"; in: {e['in']}" if e.get("in") else "")
random.seed(20260920); cases = []
base_id = 1000
for i, (app, instr, exp) in enumerate(T):
    base = apps[app]; others = [(a, e) for a, es in apps.items() if a != app for e in es]
    for variant in ("orig", "shuffled", "distractors"):
        els = list(base)
        if variant == "shuffled": random.shuffle(els)
        if variant == "distractors":
            extra = random.sample(others, min(15, len(others)))
            els = els + [dict(e, id=2000 + j, window=f"(別アプリ {a}) " + (e.get("window") or "")) for j, (a, e) in enumerate(extra)]; random.shuffle(els)
        c = {key(e): desc(e) for e in els}; c[NONE] = "どの要素も指示に合わない(該当なし)"
        cases.append({"case_id": f"{base_id+i:04d}-{variant}", "task_id": base_id + i, "app": app, "variant": variant, "instruction": instr, "expected": exp, "n_options": len(c),
                      "request": {"state": {"app": app.split("(")[0], "window": base[0].get("window") or app, "note": "要素一覧は候補側に列挙されている。説明の in: はシート/ダイアログ内の要素"},
                                  "questions": {"target": {"type": "choice", "instructions": f"Mac のアプリを AX 木(アクセシビリティ要素)経由で操作する。今のステップ: {instr}。操作すべき要素を 1 つ選ぶ。該当する要素が無ければ __none__。", "criteria": c}}}})
json.dump(cases, open("cases2.json", "w"), ensure_ascii=False); print(len(T), "tasks ->", len(cases), "cases")
