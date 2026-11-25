#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把群組裡的回報行整理成簡報 P4 的比較表。

用法：
    1. 把群組裡所有回報行貼進一個純文字檔（一行一筆，中間夾雜別的聊天內容沒關係）
    2. python src/collect_reports.py reports/回報行.txt

輸出：
    - Round 0 一致性檢查表（含 checksum 核對與 ±0.02 判定）
    - Round 1 變因比較表（可直接貼進 docs/04_P4比較表.md 和簡報）

這支程式不依賴 config.py，可以單獨拿去 Kaggle 或別人的電腦上跑。
"""
import re
import sys

TOL = 0.02  # 容許誤差：同設定的兩個人差在這個範圍內算一致
ROUND0_MODEL = "yolov8n"  # Round 0 規定的模型。標 round0 卻不是這個的，
                          # 實際上是 Round 1 的模型大小變因，標籤打錯了。


def _utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 每個欄位各自抓，缺哪個就是 None —— 舊版回報行欄位比較少，也要能吃
PATTERNS = {
    "name":        r"^\s*\[([^\]]+)\]",
    "round":       r"\b(round[01])\b",
    "model":       r"\b(yolov8[nsmlx])\b",
    "imgsz":       r"imgsz=(\d+)",
    "ep":          r"\bep=(\d+)",
    "mosaic":      r"mosaic=([\d.]+)",
    "batch":       r"batch=(\d+)",
    "seed":        r"seed=(\d+)",
    "fixed_conf":  r"mAP@conf([\d.]+)=",
    "fixed_map":   r"(?:mAP@conf[\d.]+|val mAP)=([\d.]+)",
    "best_map":    r"best=([\d.]+)\(",
    "best_conf":   r"best=[\d.]+\(conf=([\d.]+)",
    "best_shrink": r"shrink=([\d.]+)\)",
    "maxconf":     r"maxconf=([\d.]+)",
    "checksum":    r"checksum=([0-9a-f]+)",
    "note":        r"//\s*(.+?)\s*$",
}


def parse_line(line):
    """認得出是回報行就回傳 dict，否則回傳 None。"""
    if "[" not in line or "->" not in line:
        return None
    row = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, line)
        row[key] = m.group(1) if m else None
    if row["name"] is None or row["fixed_map"] is None:
        return None
    row["raw"] = line.strip()
    return row


def f(row, key, fmt="{}"):
    v = row.get(key)
    return "—" if v is None else fmt.format(v)


def score(row):
    """這一筆真正的成績：有掃描就用 best，沒有就只能用固定門檻那個。"""
    if row["best_map"] is not None:
        return float(row["best_map"]), "best"
    return float(row["fixed_map"]), "fixed"


def round0_table(rows):
    print("## Round 0 一致性檢查（五個人同設定）")
    print()
    print("| 人 | 固定門檻 mAP | best | 最佳 conf | shrink | maxconf | checksum |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {} | `{}` |".format(
            r["name"],
            f(r, "fixed_map"),
            f(r, "best_map"),
            f(r, "best_conf"),
            f(r, "best_shrink"),
            f(r, "maxconf"),
            r["checksum"] or "缺",
        ))
    print()

    # checksum 必須全對上，不然後面所有比較都無效
    sums = {r["checksum"] for r in rows if r["checksum"]}
    if len(sums) > 1:
        print("- ❌ **checksum 不一致**：{} —— val split 不同，這些數字不能互相比。".format(
            "、".join(sorted(sums))))
        print("  有人改了 SEED 或 SUBSET_PATIENTS，先修好再重跑。")
    elif len(sums) == 1:
        # 只報「已回報的這幾個人」，不要講成五個人都對上了 ——
        # 沒回報的人可能正拿著不同的 split，說成全員一致會掩蓋這件事。
        n = len({r["name"] for r in rows if r["checksum"]})
        print("- ✅ checksum 一致（`{}`）—— 已回報的 {} 人用的是同一份 val split。".format(
            sums.pop(), n))
        if n < 5:
            print("  還有 {} 人沒回報，**還不能說全員對齊**。".format(5 - n))

    # 門檻修好（c1b3a21）之前跑的那些回報行：固定門檻 0.0000 而且沒有 best。
    # 它們的 0.0000 是門檻高過 maxconf 造成的，不是模型的成績 ——
    # 拿去跟修好之後的分數比全距，會得到一個看起來很嚴重的假警報。
    def is_stale(r):
        return r["best_map"] is None and float(r["fixed_map"]) == 0.0

    stale = [r for r in rows if is_stale(r)]
    fresh = [r for r in rows if not is_stale(r)]

    if stale:
        print("- ⚠️ {} 的回報行是**舊門檻（修好之前）跑的**，固定門檻 0.0000 且沒有掃描值。"
              .format("、".join(r["name"] for r in stale)))
        print("  那個 0.0000 是門檻高過 maxconf 造成的，不是成績，**不列入下面的比較**。")
        print("  請他們重跑評分那格與門檻掃描（不用重新訓練）。")

    # 同一個人用同一個設定跑兩次 -> 那是在量雜訊，不是不一致。
    # 這個數字很重要：它決定了多大的差距才算「真的有差」。
    by_name = {}
    for r in fresh:
        by_name.setdefault(r["name"], []).append(score(r)[0])
    repeats = {k: v for k, v in by_name.items() if len(v) >= 2}
    for name, vs in repeats.items():
        print("- 🔁 **{}** 同設定重跑 {} 次：{} —— 全距 **{:.4f}**。".format(
            name, len(vs), "、".join("{:.4f}".format(v) for v in vs),
            max(vs) - min(vs)))
        print("  這是**重跑變異**不是不一致。小於這個幅度的差距都不能當成結論。")

    # 跨人比較用各自的平均，免得被重跑的運氣左右
    means = {k: sum(v) / len(v) for k, v in by_name.items()}
    kinds = {score(r)[1] for r in fresh}
    if len(means) >= 2:
        spread = max(means.values()) - min(means.values())
        label = "best" if kinds == {"best"} else "各自可取得的最高分數"
        verdict = "✅ 一致" if spread <= TOL else "❌ 超出容許誤差"
        print("- {}：{}的全距 = **{:.4f}**（容許 ±{}，{} 人取平均）".format(
            verdict, label, spread, TOL, len(means)))
        if spread > TOL:
            print("  差最多的兩個人要一起看 CONFIG 和 YOLO 標註檔。")
    elif len(means) == 1:
        print("- ⏳ 只有 {} 一個人有可比較的分數，**還沒辦法做跨人一致性檢查**。"
              .format(next(iter(means))))
    elif not fresh:
        print("- ❌ 沒有任何一筆是修好門檻之後跑的，**這份檢查目前沒有結論**。")
    print()


def round1_table(rows, baselines):
    print("## Round 1 變因比較（簡報 P4 的主表）")
    print()
    print("| 人 | 變因 | 模型 | imgsz | ep | mosaic | best mAP | 最佳 conf | shrink | 對 Round 0 的差距 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        best, _ = score(r)
        _, kind = score(r)
        base = baselines.get(r["name"])
        if base is None:
            delta = "— (無 Round 0)"
        elif base[1] != kind:
            # 一邊是 best、一邊只有固定門檻，相減沒有意義
            delta = "— (基準需重跑)"
        else:
            delta = "{:+.4f}".format(best - base[0])
        print("| {} | {} | {} | {} | {} | {} | **{:.4f}** | {} | {} | {} |".format(
            r["name"],
            r["note"] or "（未註明）",
            f(r, "model"),
            f(r, "imgsz"),
            f(r, "ep"),
            f(r, "mosaic"),
            best,
            f(r, "best_conf"),
            f(r, "best_shrink"),
            delta,
        ))
    print()
    if rows:
        win = max(rows, key=lambda r: score(r)[0])
        print("- 目前最好的設定：**{}** 的 `{}`，best mAP = **{:.4f}**".format(
            win["name"], win["note"] or "（未註明）", score(win)[0]))
        print("- 提醒：差距在 ±{} 以內的兩個設定**不能說誰比較好**，那是雜訊。".format(TOL))
    print()


def main(argv):
    _utf8_stdout()
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    rows = [r for r in (parse_line(ln) for ln in text.splitlines()) if r]
    if not rows:
        print("沒有讀到任何回報行。")
        print("回報行長這樣（Notebook 第 6 節會自己印出來）：")
        print("[姓名] round0 yolov8n imgsz=512 ep=10 mosaic=1.0 batch=16 seed=42 "
              "-> mAP@conf0.1=0.1778 best=0.1862(conf=0.05,shrink=1.0) "
              "maxconf=0.187 checksum=2f22fdec3b1c")
        return 1

    r0 = [r for r in rows if r["round"] == "round0"]
    r1 = [r for r in rows if r["round"] == "round1"]
    other = [r for r in rows if r["round"] is None]

    # 標成 round0 但模型不是 yolov8n 的，實際上是 Round 1 的模型大小變因。
    # 直接搬過去，不然它們會被當成「同設定的人」去比，然後報出假的不一致。
    mislabeled = [r for r in r0 if r["model"] and r["model"] != ROUND0_MODEL]
    if mislabeled:
        r0 = [r for r in r0 if r not in mislabeled]
        for r in mislabeled:
            r["note"] = r["note"] or "模型大小 {}".format(r["model"])
        r1 = mislabeled + r1

    print("<!-- 由 src/collect_reports.py 產生，共讀到 {} 筆回報行 -->".format(len(rows)))
    print()
    if mislabeled:
        print("> ⚠️ 有 {} 筆標成 `round0` 但模型不是 `{}`，".format(
            len(mislabeled), ROUND0_MODEL))
        print("> 已當成 Round 1 的模型大小變因處理：{}。".format(
            "、".join("{} {}".format(r["name"], r["model"]) for r in mislabeled)))
        print("> 請回報的人把標籤改成 `round1`，不然每次都要這樣猜。")
        print()
    if r0:
        round0_table(r0)
    # 同一個人有多筆 Round 0 時取平均當基準。
    # 取最後一筆是錯的：兩次重跑差 0.0119，挑哪一筆會讓差距的正負號翻過來。
    _base_vals, _base_kinds = {}, {}
    for r in r0:
        v, k = score(r)
        _base_vals.setdefault(r["name"], []).append(v)
        _base_kinds.setdefault(r["name"], set()).add(k)
    baselines = {}
    for name, vs in _base_vals.items():
        kinds = _base_kinds[name]
        # 混到不同種類（best vs 固定門檻）就不給基準，交給下游印「基準需重跑」
        kind = kinds.pop() if len(kinds) == 1 else None
        baselines[name] = (sum(vs) / len(vs), kind)
    if r1:
        round1_table(r1, baselines)
    if other:
        print("## 看不出是哪一輪的回報行（{} 筆）".format(len(other)))
        print()
        for r in other:
            print("- `{}`".format(r["raw"]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
